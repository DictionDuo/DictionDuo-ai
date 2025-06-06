import torch
from torch.utils.data import Dataset

class PhonemeTensorDataset(Dataset):
    def __init__(self, data_dict):
        self.mels = data_dict["mels"]
        self.phonemes = data_dict["phonemes"]
        self.errors = data_dict["errors"]
        self.input_lengths = data_dict["input_lengths"]
        self.label_lengths = data_dict["label_lengths"]

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, idx):
        return (
            self.mels[idx],
            self.phonemes[idx],
            self.errors[idx],
            torch.tensor(self.input_lengths[idx], dtype=torch.long),
            torch.tensor(self.label_lengths[idx], dtype=torch.long),
        )