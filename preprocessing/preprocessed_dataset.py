import os
import json
import torch
import random
import torchaudio
import numpy as np
from tqdm import tqdm
from preprocessing.feature_extraction import extract_features
from preprocessing.build_dataset import build_metadata_list
from preprocessing.split_dataset import split_metadata
from preprocessing.frame_utils import pad_or_truncate_feature
from preprocessing.label_utils import create_phoneme_label, create_error_label
from utils.phoneme_utils import phoneme2index

MAX_FRAMES = 800  # 고정 mel 길이 (frame 단위)
HOP_LENGTH = 160
SAMPLING_RATE = 16000

def is_valid_wav(wav_path):
    try:
        torchaudio.load(wav_path)
        return True
    except Exception as e:
        print(f"[Invalid WAV] {wav_path} - {e}")
        return False

def build_tensor_dataset(split_list, split_name, output_dir):
    skipped = []
    mel_list = []
    phoneme_list = []
    error_list = []
    lengths = []
    label_lengths = []
    meta_list = []

    with open("utils/error_class_map.json", encoding="utf-8") as f:
        error_map = json.load(f)

    for meta in tqdm(split_list, desc=f"Processing {split_name}"):
        try:
            # 손상된 WAV 파일 사전 필터링
            if not is_valid_wav(meta["wav"]):
                print(f"[SKIP] Invalid WAV: {meta['wav']}")
                skipped.append(meta)
                continue    

            mel = extract_features(meta["wav"])
            # 추출 실패 또는 NaN/Inf 포함된 경우 필터링
            if mel is None or not torch.isfinite(mel).all():
                print(f"[SKIP] Corrupted features: {meta['wav']}")
                skipped.append(meta)
                continue

            mel_np = mel.cpu().numpy()
            original_len = min(len(mel_np), MAX_FRAMES)
            mel_padded = pad_or_truncate_feature(mel_np, MAX_FRAMES, fill_value=0)

            with open(meta["json"], encoding="utf-8") as f:
                meta_json = json.load(f)

            phones = meta_json["RecordingMetadata"]["phonemic"]["phones"]
            errors = meta_json["RecordingMetadata"]["phonemic"].get("error_tags", [])

            phoneme_label = create_phoneme_label(phones, MAX_FRAMES, phoneme2index, SAMPLING_RATE, HOP_LENGTH)
            error_label = create_error_label(errors, MAX_FRAMES, error_map, SAMPLING_RATE, HOP_LENGTH)

            mel_list.append(torch.tensor(mel_padded, dtype=torch.float32))
            phoneme_list.append(torch.tensor(phoneme_label))
            error_list.append(torch.tensor(error_label))
            lengths.append(original_len)
            label_lengths.append(int((phoneme_label != 0).sum()))
            meta_list.append(meta)

        except Exception as e:
            print(f"[Error] {meta['wav']} - {e}")
            skipped.append(meta)

    if mel_list:
        torch.save({
            "mels": torch.stack(mel_list),
            "phonemes": torch.stack(phoneme_list),
            "errors": torch.stack(error_list),
            "input_lengths": lengths,
            "label_lengths": label_lengths,
            "metas": meta_list
        }, os.path.join(output_dir, f"{split_name}_dataset.pt"))
        print(f"[Saved] ({len(mel_list)} samples)")
    
    if skipped:
        with open(os.path.join(output_dir, f"{split_name}_skipped.json"), "w", encoding="utf-8") as f:
            json.dump(skipped, f, indent=2, ensure_ascii=False)
        print(f"[Saved] {split_name}_skipped.json → {len(skipped)} skipped samples")

def main():
    wav_dir = "/home/ubuntu/data"
    json_dir = "/home/ubuntu/data"
    output_dir = "/home/ubuntu/preprocessed"
    os.makedirs(output_dir, exist_ok=True)

    metadata_list = build_metadata_list(wav_dir, json_dir)
    random.shuffle(metadata_list)

    train, val, test = split_metadata(metadata_list)

    build_tensor_dataset(train, "train", output_dir)
    build_tensor_dataset(val, "val", output_dir)
    build_tensor_dataset(test, "test", output_dir)

if __name__ == "__main__":
    main()