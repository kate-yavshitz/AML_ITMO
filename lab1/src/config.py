"""Загрузка конфигурации эксперимента из YAML."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


@dataclass
class Config:
    """Типизированная конфигурация лабораторной работы."""

    # Пути
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    cache_dir: Path = Path("data/cache")
    checkpoint_dir: Path = Path("checkpoints")
    reports_dir: Path = Path("reports")

    # Данные
    language: str = "ru"
    target_sr: int = 16000
    max_audio_sec: int = 300   # кап длины записи для фузирования (у wav2vec2 нет жёсткого окна)
    max_text_tokens: int = 128
    val_size: float = 0.15
    test_size: float = 0.15
    n_folds: int = 5       # StratifiedGroupKFold по дикторам; 1 = одиночный сплит
    use_nt: bool = True  # False = абляция E1: обучение без собственных записей nt/

    # Энкодеры
    audio_model: str = "facebook/wav2vec2-xls-r-300m"  # акустический энкодер (заморожен)
    fragment_sec: int = 10     # длина фрагмента для прогона через wav2vec2 (лимит VRAM)
    audio_fps: int = 50        # фреймов эмбеддинга в секунду у wav2vec2 (шаг 20 мс)
    whisper_model: str = "small"  # используется ТОЛЬКО как ASR для транскриптов
    text_model: str = "cointegrated/rubert-tiny2"

    # Фузирование
    d_model: int = 256
    n_heads: int = 4
    dropout: float = 0.1

    # Обучение
    seed: int = 42
    batch_size: int = 8
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-2
    patience: int = 8
    device: str = "auto"

    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f)
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in raw.items() if k in known}
        extra = {k: v for k, v in raw.items() if k not in known}
        cfg = cls(**kwargs)
        for name in ("data_dir", "raw_dir", "processed_dir", "cache_dir",
                     "checkpoint_dir", "reports_dir"):
            setattr(cfg, name, Path(getattr(cfg, name)))
        cfg.extra = extra
        return cfg

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.raw_dir, self.processed_dir, self.cache_dir,
                  self.checkpoint_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    """Фиксирует все генераторы случайных чисел для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
