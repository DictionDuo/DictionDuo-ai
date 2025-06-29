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
from utils.phoneme_utils import Korean, phoneme2index, convert_prompt_to_phoneme_sequence

MAX_FRAMES = 512  # 고정 mel 길이 (frame 단위)
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
    phones_actual_list = []
    error_list = []
    lengths = []
    label_lengths = []
    meta_list = []

    with open("utils/error_class_map.json", encoding="utf-8") as f:
        error_map = json.load(f)

    korean = Korean()

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
            original_len = len(mel_np)
            mel_padded = pad_or_truncate_feature(mel_np, MAX_FRAMES, fill_value=0)

            with open(meta["json"], encoding="utf-8") as f:
                meta_json = json.load(f)

            try:
                phones = meta_json["RecordingMetadata"]["phonemic"]["phones"]
            except KeyError:
                print(f"[SKIP] Missing phonemic/phones: {meta['json']}")
                skipped.append(meta)
                continue

            prompt = meta_json["RecordingMetadata"].get("prompt", "")
            phoneme_indices = convert_prompt_to_phoneme_sequence(prompt, phoneme2index, korean)
            
            if not phoneme_indices:
                skipped.append(meta)
                continue

            phoneme_padded = pad_or_truncate_feature(phoneme_indices, MAX_FRAMES, fill_value=0)
            phoneme_tensor = torch.tensor(phoneme_padded)

            phones_actual = create_phoneme_label(phones, MAX_FRAMES, phoneme2index, SAMPLING_RATE, HOP_LENGTH)
            errors = meta_json["RecordingMetadata"]["phonemic"].get("error_tags", [])
            error_label = create_error_label(errors, MAX_FRAMES, error_map, SAMPLING_RATE, HOP_LENGTH)

            input_length = min(original_len, MAX_FRAMES)
            label_length = int((phoneme_tensor != 0).sum().item())

            if input_length == 0 or label_length == 0:
                print(f"[SKIP] Zero length input/label: {meta['wav']}")
                skipped.append(meta)
                continue

            mel_list.append(torch.tensor(mel_padded, dtype=torch.float32))
            phoneme_list.append(phoneme_tensor)
            phones_actual_list.append(torch.tensor(phones_actual))
            error_list.append(torch.tensor(error_label))
            lengths.append(input_length)
            label_lengths.append(label_length)
            meta_list.append(meta["json"])

        except Exception as e:
            print(f"[Error] {meta['wav']} - {e}")
            skipped.append(meta)

    if mel_list:
        torch.save({
            "mels": torch.stack(mel_list),
            "phonemes": torch.stack(phoneme_list),
            "phones_actual": torch.stack(phones_actual_list),
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