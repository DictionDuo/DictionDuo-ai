import os
import json
import torch
import random
import torchaudio
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from preprocessing.feature_extraction import extract_features
from preprocessing.build_dataset import build_metadata_list
from preprocessing.split_dataset import split_metadata
from utils.phoneme_utils import Korean, phoneme2index
from utils.seed import set_seed

MAX_INPUT_LENGTH = 800  # 고정 mel 길이 (frame 단위)
HOP_LENGTH = 160
SAMPLING_RATE = 16000
FRAME_TO_SEC = lambda frame: frame * HOP_LENGTH / SAMPLING_RATE

def is_valid_wav(wav_path):
    try:
        torchaudio.load(wav_path)
        return True
    except Exception as e:
        print(f"[Invalid WAV] {wav_path} - {e}")
        return False

def process_split(split_list, split_name, out_dir):
    skipped_meta = []
    all_mels = []
    all_phonemes = []
    all_errors = []
    input_lengths = []
    label_lengths = []

    with open("error_class_map.json", encoding="utf-8") as f:
        error_class_map = json.load(f)

    for meta in tqdm(split_list, desc=f"Processing {split_name}"):
        try:
            # 손상된 WAV 파일 사전 필터링
            if not is_valid_wav(meta["wav"]):
                print(f"[SKIP] Invalid WAV: {meta['wav']}")
                skipped_meta.append(meta)
                continue    

            mel = extract_features(meta["wav"])
            # 추출 실패 또는 NaN/Inf 포함된 경우 필터링
            if mel is None or not torch.isfinite(mel).all():
                print(f"[SKIP] Corrupted features: {meta['wav']}")
                skipped_meta.append(meta)
                continue

            mel = mel[:MAX_INPUT_LENGTH, :]
            mel_end_sec = FRAME_TO_SEC(mel.shape[0])

            with open(meta["json"], encoding="utf-8") as f:
                data = json.load(f)

            phones = data["RecordingMetadata"]["phonemic"]["phones"]
            errors = data["RecordingMetadata"]["phonemic"].get("error_tags", [])

            phoneme_seq = []
            error_seq = []

            for p in phones:
                if p["start"] >= mel_end_sec:
                    break

                phoneme = p["phone"]
                phoneme_idx = phoneme2index.get(phoneme)
                if phoneme_idx is None:
                    continue
                phoneme_seq.append(phoneme_idx)

                matched_tag = None
                for e in errors:
                    if e["start"] <= p["start"] < e["end"]:
                        matched_tag = e["error_tag"]
                        break

                error_class = error_class_map.get(matched_tag, 0)
                error_seq.append(error_class)

            # CTC 조건 확인
            if not phoneme_seq or len(phoneme_seq) >= (mel.shape[0] // 2):
                skipped_meta.append(meta)
                continue

            all_mels.append(mel.cpu())
            all_phonemes.append(torch.tensor(phoneme_seq, dtype=torch.long))
            all_errors.append(torch.tensor(error_seq, dtype=torch.long))
            input_lengths.append(mel.shape[0])
            label_lengths.append(len(phoneme_seq))

        except Exception as e:
            print(f"[Error] {meta['wav']} - {e}")
            skipped_meta.append(meta)

    if all_mels:
        mels_padded = pad_sequence(all_mels, batch_first=True)
        phonemes_padded = pad_sequence(all_phonemes, batch_first=True)
        errors_padded = pad_sequence(all_errors, batch_first=True)

        save_path = os.path.join(out_dir, f"{split_name}_dataset.pt")

        torch.save({
            "mels": mels_padded,
            "phonemes": phonemes_padded,
            "errors": errors_padded,
            "input_lengths": input_lengths,
            "label_lengths": label_lengths,
        }, save_path)
        print(f"[Saved] {save_path} ({len(all_mels)} samples)")
    
    if skipped_meta:
        skipped_path = os.path.join(out_dir, f"{split_name}_skipped.json")
        with open(skipped_path, "w", encoding="utf-8") as f:
            json.dump(skipped_meta, f, indent=2, ensure_ascii=False)
        print(f"[Saved] {split_name}_skipped.json → {len(skipped_meta)} skipped samples")

def main():
    wav_dir = "/home/ubuntu/data"
    json_dir = "/home/ubuntu/data"
    output_dir = "/home/ubuntu/preprocessed"
    os.makedirs(output_dir, exist_ok=True)

    set_seed(42)
    metadata_list = build_metadata_list(wav_dir, json_dir)
    random.shuffle(metadata_list)

    train_list, val_list, test_list = split_metadata(metadata_list)

    process_split(train_list, "train", output_dir)
    process_split(val_list, "val", output_dir)
    process_split(test_list, "test", output_dir)

if __name__ == "__main__":
    main()