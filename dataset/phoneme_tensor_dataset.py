import torch
from torch.utils.data import Dataset

class PhonemeTensorDataset(Dataset):
    def __init__(self, data_dict, meta_list=None):
        """
        data_dict: .pt로부터 로드된 사전 (mels, phonemes, input_lengths, label_lengths, metas 포함)
        meta_list: 각 샘플에 대응되는 JSON 경로 문자열 리스트
        """
        self.mels = data_dict["mels"]
        self.phonemes = data_dict["phonemes"]            # 정답 (prompt 기반)
        self.input_lengths = data_dict["input_lengths"]
        self.label_lengths = data_dict["label_lengths"]
        self.metas = meta_list if meta_list is not None else [None] * len(self.mels)

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, idx):
        return (
            self.mels[idx],
            self.phonemes[idx],
            torch.tensor(self.input_lengths[idx], dtype=torch.long),
            torch.tensor(self.label_lengths[idx], dtype=torch.long),
            self.metas[idx]
        )