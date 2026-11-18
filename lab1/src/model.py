"""Модель раннего фузирования: cross-attention (Q = текст, K = V = аудио).

Раннее фузирование здесь означает, что совместное представление двух
модальностей строится ДО классификатора: текстовые токены «опрашивают»
последовательность аудио-фреймов через механизм внимания, и полученное
joint representation подаётся на MLP-классификатор.
"""

from __future__ import annotations

import torch
from torch import nn


class CrossAttentionFusion(nn.Module):
    """Бинарный классификатор афазии на основе cross-attention фузирования."""

    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(
        self,
        audio: torch.Tensor,        # [B, T_a, D_a]
        text: torch.Tensor,         # [B, T_t, D_t]
        audio_mask: torch.Tensor,   # [B, T_a], True = валидный фрейм
        text_mask: torch.Tensor,    # [B, T_t], True = валидный токен
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Возвращает логиты [B, n_classes] и веса внимания [B, T_t, T_a]."""
        queries = self.text_proj(text)          # Q — текстовые токены
        keys_values = self.audio_proj(audio)    # K, V — аудио-фреймы

        attn_out, attn_weights = self.cross_attn(
            query=queries,
            key=keys_values,
            value=keys_values,
            key_padding_mask=~audio_mask,  # True = игнорировать (соглашение PyTorch)
            need_weights=True,
            average_attn_weights=True,
        )
        fused = self.norm(queries + attn_out)  # residual + LayerNorm

        # Masked mean-pooling по текстовым токенам
        mask = text_mask.unsqueeze(-1).float()
        pooled = (fused * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        return self.classifier(pooled), attn_weights
