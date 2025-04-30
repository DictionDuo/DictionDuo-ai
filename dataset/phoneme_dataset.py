import torch
from torch.utils.data import Dataset
import numpy as np
import librosa
import json
from preprocessing.feature_extraction import extract_features
from preprocessing.build_dataset import build_metadata_list, get_max_lengths
from utils.phoneme_utils import Korean

class PhonemeDataset(Dataset):
    def __init__(self, wav_dir, json_dir, phoneme2index):
        self.metadata_list = build_metadata_list(wav_dir, json_dir)
        self.phoneme2index = phoneme2index
        self.max_mel_length, self.max_label_length = get_max_lengths(self.metadata_list, phoneme2index)

    def __len__(self):
        return len(self.metadata_list)

    def pad_mel(self, features):
        length = features.shape[0]
        if length < self.max_mel_length:
            pad_width = ((0, self.max_mel_length - length), (0, 0))
            return np.pad(features, pad_width, mode='constant')
        return features[:self.max_mel_length]

    def pad_label(self, label_seq):
        if len(label_seq) < self.max_label_length:
            return label_seq + [0] * (self.max_label_length - len(label_seq))
        return label_seq

    def __getitem__(self, idx):
        meta = self.metadata_list[idx]
        features = extract_features(meta["wav"])

        if features is None:
            features = np.zeros((self.max_mel_length, 80), dtype=np.float32)
        else:
            features = self.pad_mel(features)

        with open(meta["json"], 'r', encoding='utf-8') as f:
            data = json.load(f)
        text = data["transcription"]["AnswerLabelText"]
        labels = Korean.text_to_phoneme_sequence(text, self.phoneme2index)
        labels = self.pad_label(labels)

        return torch.tensor(features), torch.tensor(labels)