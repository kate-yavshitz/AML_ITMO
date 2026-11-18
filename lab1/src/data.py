"""Загрузка манифеста датасета и PyTorch-Dataset для кэшированных эмбеддингов.

Манифест ``data/manifest.csv`` собирается скриптом ``scripts/preprocess_audio.py``
и содержит колонки: audio_path, sample_id, label, speaker, source, gender,
age, aphasia_type, severity, duration_sec.

Если рядом с аудио лежит ``<имя>.txt``, он используется как готовый транскрипт.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """Читает манифест, собранный препроцессингом."""
    if not manifest_path.exists() or manifest_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Манифест {manifest_path} отсутствует или пуст (возможно, остался "
            f"от упавшего запуска). Удалите его и запустите заново:\n"
            f"  python scripts/preprocess_audio.py --config configs/default.yaml"
        )
    manifest = pd.read_csv(manifest_path)
    required = {"audio_path", "sample_id", "label", "speaker", "source"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"В манифесте нет колонок: {missing}")
    return manifest


class EmbeddingDataset(Dataset[dict[str, torch.Tensor]]):
    """Отдаёт кэшированные эмбеддинги двух модальностей + маски + метку.

    Аудио-эмбеддинги хранятся в float16 (экономия диска) и приводятся
    к float32 при загрузке. ``max_frames`` обрезает длинные записи до
    капа (max_audio_sec * audio_fps) — под VRAM обучающегося фузирования.
    """

    def __init__(self, manifest: pd.DataFrame, cache_dir: Path,
                 audio_subdir: str = "audio_w2v",
                 max_frames: int | None = None) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.audio_subdir = audio_subdir
        self.max_frames = max_frames

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.manifest.iloc[idx]
        sid = str(row["sample_id"])
        audio = np.load(self.cache_dir / self.audio_subdir / f"{sid}.npy")  # [T_a, D_a]
        text = np.load(self.cache_dir / "text" / f"{sid}.npy")              # [T_t, D_t]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if self.max_frames is not None:
            audio = audio[: self.max_frames]
        return {
            "audio": torch.from_numpy(audio),
            "text": torch.from_numpy(text),
            "audio_len": torch.tensor(audio.shape[0]),
            "text_len": torch.tensor(text.shape[0]),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "sample_id": sid,
        }


def collate_embeddings(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Паддит последовательности до максимальной длины в батче и строит маски."""
    audio_lens = torch.stack([b["audio_len"] for b in batch])
    text_lens = torch.stack([b["text_len"] for b in batch])
    max_a, max_t = int(audio_lens.max()), int(text_lens.max())
    d_audio = batch[0]["audio"].shape[1]
    d_text = batch[0]["text"].shape[1]

    bs = len(batch)
    audio = torch.zeros(bs, max_a, d_audio)
    text = torch.zeros(bs, max_t, d_text)
    audio_mask = torch.zeros(bs, max_a, dtype=torch.bool)
    text_mask = torch.zeros(bs, max_t, dtype=torch.bool)
    for i, b in enumerate(batch):
        la, lt = int(b["audio_len"]), int(b["text_len"])
        audio[i, :la] = b["audio"]
        text[i, :lt] = b["text"]
        audio_mask[i, :la] = True
        text_mask[i, :lt] = True

    return {
        "audio": audio,
        "text": text,
        "audio_mask": audio_mask,
        "text_mask": text_mask,
        "label": torch.stack([b["label"] for b in batch]),
        "sample_id": [b["sample_id"] for b in batch],
    }
