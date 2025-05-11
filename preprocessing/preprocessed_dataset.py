import os, json, torch
import numpy as np
from tqdm import tqdm
from preprocessing.feature_extraction import extract_features
from preprocessing.build_dataset import build_metadata_list
from preprocessing.split_dataset import split_metadata
from utils.phoneme_utils import Korean, phoneme2index
from torch.utils.data import Dataset
import boto3

class PreprocessedDataset(Dataset):
    def __init__(self, metadata_list, max_mel_length, max_label_length):
        self.samples = []
        for meta in tqdm(metadata_list, desc="Preprocessing"):
            mel = extract_features(meta["wav"])
            if mel is None:
                continue

            with open(meta["json"], encoding="utf-8") as f:
                data = json.load(f)
            text = data.get("transcription", {}).get("AnswerLabelText") or \
                   data.get("RecordingMetadata", {}).get("prompt")
            if not text:
                continue

            label_seq = Korean.text_to_phoneme_sequence(text, phoneme2index)

            mel = self.pad_mel(mel, max_mel_length)
            label = self.pad_label(label_seq, max_label_length)

            self.samples.append({
                "mel": torch.tensor(mel, dtype=torch.float32),
                "label": torch.tensor(label, dtype=torch.long),
                "input_length": mel.shape[0],
                "label_length": len(label_seq),
            })

    def pad_mel(self, mel, max_len):
        if mel.shape[0] < max_len:
            pad_width = ((0, max_len - mel.shape[0]), (0, 0))
            return np.pad(mel, pad_width)
        return mel[:max_len]

    def pad_label(self, label, max_len):
        if len(label) < max_len:
            return label + [0] * (max_len - len(label))
        return label[:max_len]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return item["mel"], item["label"], torch.tensor(item["input_length"]), torch.tensor(item["label_length"])

def get_max_lengths(metadata_list):
    max_mel = 0
    max_label = 0

    for meta in tqdm(metadata_list, desc="Calculating max lengths"):
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

def upload_to_s3(file_path, bucket, s3_path):
    s3 = boto3.client("s3")
    s3.upload_file(file_path, bucket, s3_path)
    print(f"[S3 Uploaded] {file_path} → s3://{bucket}/{s3_path}")

def main():
    wav_dir = "/home/ec2-user/data"
    json_dir = "/home/ec2-user/data"
    output_dir = "./preprocessed"
    bucket_name = "dictionduo-bucket"
    s3_prefix = "preprocessed"

    os.makedirs(output_dir, exist_ok=True)
    metadata_list = build_metadata_list(wav_dir, json_dir)
    train_list, val_list, test_list = split_metadata(metadata_list)

    max_mel, max_label = get_max_lengths(metadata_list)
    print(f"[max_mel={max_mel}], [max_label={max_label}]")

    for name, split in {"train": train_list, "val": val_list, "test": test_list}.items():
        dataset = PreprocessedDataset(split, max_mel, max_label)
        path = os.path.join(output_dir, f"{name}_dataset.pt")
        torch.save(dataset, path)
        upload_to_s3(path, bucket_name, f"{s3_prefix}/{name}_dataset.pt")

if __name__ == "__main__":
    main()