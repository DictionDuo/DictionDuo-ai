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
SEC_TO_FRAME = lambda sec: int(sec * SAMPLING_RATE / HOP_LENGTH)

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
    all_phoneme_frames = []
    all_error_frames = []
    input_lengths = []

    error_map_path = os.path.join(os.path.dirname(__file__), "../utils/error_class_map.json")
    with open(error_map_path, encoding="utf-8") as f:
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

            original_length = min(mel.shape[0], MAX_INPUT_LENGTH)
            mel = mel[:MAX_INPUT_LENGTH, :]
            if mel.shape[0] < MAX_INPUT_LENGTH:
                pad_len = MAX_INPUT_LENGTH - mel.shape[0]
                mel = torch.nn.functional.pad(mel, (0, 0, 0, pad_len))  # pad time dimension

            with open(meta["json"], encoding="utf-8") as f:
                data = json.load(f)

            phones = data["RecordingMetadata"]["phonemic"]["phones"]
            errors = data["RecordingMetadata"]["phonemic"].get("error_tags", [])

            phoneme_frames = torch.zeros(MAX_INPUT_LENGTH, dtype=torch.long)
            error_frames = torch.zeros(MAX_INPUT_LENGTH, dtype=torch.long)

            for p in phones:
                start_f = max(0, SEC_TO_FRAME(p["start"]))
                end_f = min(MAX_INPUT_LENGTH, SEC_TO_FRAME(p["end"]))
                phoneme = p["phone"]
                idx = phoneme2index.get(phoneme, 0)
                phoneme_frames[start_f:end_f] = idx

            for e in errors:
                start_f = max(0, SEC_TO_FRAME(e["start"]))
                end_f = min(MAX_INPUT_LENGTH, SEC_TO_FRAME(e["end"]))
                tag = e.get("error_tag")
                err_idx = error_class_map.get(tag, 0)
                error_frames[start_f:end_f] = err_idx

            all_mels.append(mel.cpu())
            all_phoneme_frames.append(phoneme_frames)
            all_error_frames.append(error_frames)
            input_lengths.append(original_length)

        except Exception as e:
            print(f"[Error] {meta['wav']} - {e}")
            skipped_meta.append(meta)

    if all_mels:
        mels_padded = pad_sequence(all_mels, batch_first=True)
        phonemes_padded = pad_sequence(all_phoneme_frames, batch_first=True)
        errors_padded = pad_sequence(all_error_frames, batch_first=True)

        save_path = os.path.join(out_dir, f"{split_name}_dataset.pt")

        torch.save({
            "mels": mels_padded,
            "phonemes": phonemes_padded,
            "errors": errors_padded,
            "input_lengths": input_lengths
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