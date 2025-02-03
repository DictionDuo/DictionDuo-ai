import torch
from torch.utils.data import Dataset, DataLoader
import preprocessing.feature_extraction as fx  # feature_extraction.py 전체 사용
import numpy as np
import os

# 자동으로 GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PronunciationDataset(Dataset):
    def __init__(self, audio_files, max_length):
        self.audio_files = audio_files
        self.max_length = max_length
    
    def pad_or_truncate(self, features):
        if features is None:  # None 값 처리
            return np.zeros((self.max_length, 1))  # 기본적으로 (max_length, 1) 형태로 반환

        length = features.shape[0]
        if len(features.shape) == 1:  # 1차원 배열이면 (T,1)로 변환
            features = features[:, np.newaxis]  # (T,) → (T,1)

        if length > self.max_length:
            return features[:self.max_length]
        elif length < self.max_length:
            pad_width = self.max_length - length
            padding = np.zeros((pad_width, features.shape[1]))  # 안전한 패딩 생성
            return np.concatenate([features, padding], axis=0)
        return features
    
    def resize_feature(self, feature, target_length):
        current_length = feature.shape[0]
        if current_length == target_length:
            return feature
        return np.interp(np.linspace(0, 1, target_length), np.linspace(0, 1, current_length), feature)
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        y, sr = fx.load_audio(audio_path)
        
        if y is None or sr is None:
            return None  # 로딩 실패 시 None 반환
        
        try:
            # 모든 피처 추출
            mel_spectrogram = fx.extract_mel_spectrogram(y, sr)
            f0, _ = fx.extract_f0(y, sr)
            energy = fx.extract_energy(y, sr)
            shimmer = fx.calculate_shimmer(y, sr, f0, _)
            formants = fx.extract_formants(audio_path)
            jitter_abs, jitter_rel = fx.extract_jitter(f0)

            # Mel Spectrogram 길이에 맞춰 다른 피처 크기 조정
            T_mel = mel_spectrogram.shape[0]
            f0 = self.resize_feature(f0, T_mel)[:, np.newaxis]
            energy = self.resize_feature(energy, T_mel)[:, np.newaxis]
            jitter_abs = np.full((T_mel, 1), jitter_abs)
            shimmer = np.full((T_mel, 1), shimmer)
            formants = self.resize_feature(formants, T_mel)
            
            # 패딩 적용 (모든 피처)
            mel_spectrogram = self.pad_or_truncate(mel_spectrogram)
            f0 = self.pad_or_truncate(f0)
            energy = self.pad_or_truncate(energy)
            jitter_abs = self.pad_or_truncate(jitter_abs)
            shimmer = self.pad_or_truncate(shimmer)
            formants = self.pad_or_truncate(formants)
            
            # 텐서 변환 및 GPU 로드
            mel_spectrogram = torch.tensor(mel_spectrogram, dtype=torch.float32).to(device)
            f0 = torch.tensor(f0, dtype=torch.float32).to(device)
            energy = torch.tensor(energy, dtype=torch.float32).to(device)
            shimmer = torch.tensor(shimmer, dtype=torch.float32).to(device)
            formants = torch.tensor(formants, dtype=torch.float32).to(device)
            jitter_abs = torch.tensor(jitter_abs, dtype=torch.float32).to(device)
            jitter_rel = torch.tensor(jitter_rel, dtype=torch.float32).to(device)
            
            return mel_spectrogram, f0, energy, shimmer, formants, jitter_abs, jitter_rel
        
        except Exception as e:
            print(f"Error processing file {audio_path}: {e}")
            return None

# DataLoader에서 None 값 필터링하는 collate_fn 추가
def collate_fn(batch):
    batch = [item for item in batch if item is not None]  # None 제거
    if len(batch) == 0:
        return None  # 모든 데이터가 None이면 DataLoader에서 처리
    return torch.utils.data.dataloader.default_collate(batch)

def create_dataloader(audio_files, max_length, batch_size, shuffle=True):
    dataset = PronunciationDataset(audio_files, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader