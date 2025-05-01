import random
from preprocessing.build_dataset import build_metadata_list

def split_metadata(metadata_list, train_ratio=0.85, val_ratio=0.10, seed=42):
    random.seed(seed)
    random.shuffle(metadata_list)

    total = len(metadata_list)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_list = metadata_list[:train_end]
    val_list = metadata_list[train_end:val_end]
    test_list = metadata_list[val_end:]

    return train_list, val_list, test_list

def build_and_split(wav_dir, json_dir, train_ratio=0.85, val_ratio=0.10, seed=42):
    metadata_list = build_metadata_list(wav_dir, json_dir)
    return split_metadata(metadata_list, train_ratio, val_ratio, seed)