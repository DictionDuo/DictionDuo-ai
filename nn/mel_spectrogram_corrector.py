import torch
import torch.nn as nn

# Corrector Block (Residual Connection 포함)
class CorrectorBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.LayerNorm(dim)
        )
    
    def forward(self, x):
        return self.layer(x) + x  # Residual Connection 적용

# 전체 Mel Spectrogram Corrector 모델
class MelSpectrogramCorrector(nn.Module):
    def __init__(self, input_dim=136, output_dim=128, num_layers=3):  # 기본적으로 3개 블록 사용
        super().__init__()
        self.input_layer = nn.Linear(input_dim, 64)
        self.blocks = nn.ModuleList([CorrectorBlock(64) for _ in range(num_layers)])
        self.output_layer = nn.Sequential(
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, output_dim)  # 최종 보정된 Mel Spectrogram 출력
        )
    
    def forward(self, x):
        x = self.input_layer(x.mean(dim=1))  # 입력을 64차원으로 변환
        for block in self.blocks:
            x = block(x)  # 여러 블록을 통과
        return self.output_layer(x)  # 최종 보정된 Mel Spectrogram 출력