import torch
import torch.nn as nn
import torch.nn.functional as fn

# Feed Forward Module
class FeedForwardModule(nn.Module):
    def __init__(self, dim, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * expansion_factor)
        self.fc2 = nn.Linear(dim * expansion_factor, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = fn.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.norm(x + residual)

# Convolution Module
class ConvolutionModule(nn.Module):
    def __init__(self, dim, kernel_size=31):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)
        self.batch_norm = nn.BatchNorm1d(dim)
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        residual = x
        x = x.transpose(1, 2)  # Convert to (batch, channel, time)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = fn.silu(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)  # Convert back to (batch, time, channel)
        return self.norm(x + residual)

# Multi-Head Self-Attention Module
class MHSA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        residual = x
        x, _ = self.attn(x, x, x)
        return self.norm(x + residual)

# Conformer Block
class ConformerBlock(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=31, dropout=0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(dim, dropout=dropout)
        self.conv = ConvolutionModule(dim, kernel_size=kernel_size)
        self.attn = MHSA(dim, num_heads)
        self.ffn2 = FeedForwardModule(dim, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        x = self.ffn1(x)
        x = self.attn(x)
        x = self.conv(x)
        x = self.ffn2(x)
        return self.norm(x)

# Full Conformer Model
class Conformer(nn.Module):
    def __init__(self, input_dim=136, num_layers=6, num_heads=8, kernel_size=31, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            ConformerBlock(input_dim, num_heads, kernel_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x