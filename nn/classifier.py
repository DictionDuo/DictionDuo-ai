import torch
import torch.nn as nn

# Classifier Block (Residual Connection 포함)
class ClassifierBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.LayerNorm(dim)
        )
    
    def forward(self, x):
        return self.layer(x) + x  # Residual Connection 적용

# 전체 Pronunciation Classifier 모델
class PronunciationClassifier(nn.Module):
    def __init__(self, input_dim=136, num_layers=3):  # 기본적으로 3개 블록 사용
        super().__init__()
        self.input_layer = nn.Linear(input_dim, 64)
        self.blocks = nn.ModuleList([ClassifierBlock(64) for _ in range(num_layers)])
        self.output_layer = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1)  # 최종 발음 정확도 출력
        )
    
    def forward(self, x):
        x = self.input_layer(x.mean(dim=1))  # 입력을 64차원으로 변환
        for block in self.blocks:
            x = block(x)  # 여러 블록을 통과
        return self.output_layer(x)  # 최종 출력