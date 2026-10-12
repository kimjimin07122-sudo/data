# Ling et al., TinyTransformer4TS (MIT) models/fp32 기반
# 수정: (1) Encoder가 매 층에 원래 입력을 넣던 버그 수정 (2) config.DEVICE 의존 제거 (3) 정규화/집계 방식 선택 가능
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, attn_dropout=0.2):
        super().__init__()
        assert d_model % nhead == 0
        self.h = nhead
        self.q, self.k, self.v = (nn.Linear(d_model, d_model) for _ in range(3))
        self.out = nn.Linear(d_model, d_model)
        self.scale = 1.0 / math.sqrt(d_model // nhead)
        self.drop = nn.Dropout(attn_dropout)

    def forward(self, x):
        B, L, _ = x.shape
        q = self.q(x).view(B, L, self.h, -1)
        k = self.k(x).view(B, L, self.h, -1)
        v = self.v(x).view(B, L, self.h, -1)
        attn = self.drop(torch.softmax(self.scale * torch.einsum('blhe,bshe->bhls', q, k), dim=-1))
        return self.out(torch.einsum('bhls,bshd->blhd', attn, v).reshape(B, L, -1))


class SeqNorm(nn.Module):
    """'batch' = Ling 원본 (BatchNorm1d), 'layer' = 표준 Transformer (LayerNorm)"""
    def __init__(self, d_model, kind):
        super().__init__()
        self.kind = kind
        self.norm = nn.BatchNorm1d(d_model) if kind == 'batch' else nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2) if self.kind == 'batch' else self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, norm):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, nhead)
        self.norm1 = SeqNorm(d_model, norm)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model))
        self.norm2 = SeqNorm(d_model, norm)

    def forward(self, x):
        x = self.norm1(x + self.mha(x))
        return self.norm2(x + self.ffn(x))


class TransformerPredictor(nn.Module):
    """입력 (B, L, 17) → 출력 (B, 10)"""
    def __init__(self, window_size, d_model=32, nhead=4, num_layers=2,
                 n_in=17, n_out=10, norm='batch', pool='gap'):
        super().__init__()
        self.pool = pool
        self.input_linear = nn.Linear(n_in, d_model)
        self.pe = PositionalEncoding(d_model, window_size)
        self.layers = nn.ModuleList([EncoderLayer(d_model, nhead, norm) for _ in range(num_layers)])
        self.output_linear = nn.Linear(d_model, n_out)

    def forward(self, x):
        x = self.pe(self.input_linear(x))
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1) if self.pool == 'gap' else x[:, -1]
        return self.output_linear(x)
