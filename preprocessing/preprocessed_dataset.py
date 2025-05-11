import os, json, torch
import numpy as np
import torchaudio
from tqdm import tqdm
from preprocessing.feature_extraction import extract_features
from preprocessing.build_dataset import build_metadata_list
from preprocessing.split_dataset import split_metadata
from utils.phoneme_utils import Korean, phoneme2index

def is_valid_wav(wav_path):
    try:
        torchaudio.load(wav_path)
        return True
    except Exception as e:
        print(f"[Invalid WAV] {wav_path} - {e}")
        return False
    
def pad_mel(mel, max_len):
    if mel.shape[0] < max_len:
        pad_width = ((0, max_len - mel.shape[0]), (0, 0))
        return np.pad(mel, pad_width)
    return mel[:max_len]

def pad_label(label, max_len):
    if len(label) < max_len:
        return label + [0] * (max_len - len(label))
    return label[:max_len]

def get_max_lengths(metadata_list):
    max_mel = 0
    max_label = 0

    for meta in tqdm(metadata_list, desc="Calculating max lengths"):
        if not is_valid_wav(meta["wav"]):
            continue

        mel = extract_features(meta["wav"])
        if mel is not None:
            max_mel = max(max_mel, mel.shape[0])

        with open(meta["json"], encoding="utf-8") as f:
            data = json.load(f)

        text = data.get("transcription", {}).get("AnswerLabelText") or data.get("RecordingMetadata", {}).get("prompt")

        if text:
            label_seq = Korean.text_to_phoneme_sequence(text, phoneme2index)
            max_label = max(max_label, len(label_seq))

    return max_mel, max_label

def save_tensor(mel, label_seq, max_mel, max_label, save_path):
    mel_padded = pad_mel(mel, max_mel)
    label_padded = pad_label(label_seq, max_label)
    torch.save({
        "mel": torch.tensor(mel_padded, dtype=torch.float32),
        "label": torch.tensor(label_padded, dtype=torch.long),
        "input_length": mel.shape[0],
        "label_length": len(label_seq),
    }, save_path)

def process_split(split_list, split_name, out_dir, max_mel, max_label):
    split_dir = os.path.join(out_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    for idx, meta in enumerate(tqdm(split_list, desc=f"Processing {split_name}")):
        try:
            if not is_valid_wav(meta["wav"]):
                continue    

            mel = extract_features(meta["wav"])
            if mel is None:
                continue

            with open(meta["json"], encoding="utf-8") as f:
                data = json.load(f)
            text = data.get("transcription", {}).get("AnswerLabelText") or data.get("RecordingMetadata", {}).get("prompt")
            if not text:
                continue
            
            label_seq = Korean.text_to_phoneme_sequence(text, phoneme2index)
            save_path = os.path.join(split_dir, f"{idx:05d}.pt")
            save_tensor(mel, label_seq, max_mel, max_label, save_path)
        except Exception as e:
            print(f"[Error] {meta['wav']} - {e}")

def main():
    wav_dir = "/home/ec2-user/data"
    json_dir = "/home/ec2-user/data"
    output_dir = "./preprocessed"
    os.makedirs(output_dir, exist_ok=True)

    metadata_list = build_metadata_list(wav_dir, json_dir)
    train_list, val_list, test_list = split_metadata(metadata_list)

    max_mel, max_label = get_max_lengths(metadata_list)
    print(f"[max_mel={max_mel}], [max_label={max_label}]")

    process_split(train_list, "train", output_dir, max_mel, max_label)
    process_split(val_list, "val", output_dir, max_mel, max_label)
    process_split(test_list, "test", output_dir, max_mel, max_label)

if __name__ == "__main__":
    main()