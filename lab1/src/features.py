"""Извлечение и кэширование эмбеддингов двух модальностей.

Аудио: wav2vec 2.0 XLS-R-300m (заморожен) -> последовательность фреймов
[T, 1024], ~50 фреймов/с. В отличие от Whisper, жёсткого окна 30 с нет:
запись режется на фрагменты ``fragment_sec`` (лимит VRAM: self-attention
трансформера квадратичен по длине), каждый фрагмент прогоняется через
энкодер, а фреймы склеиваются обратно в полную временну́ю ось записи
(до капа ``max_audio_sec``). Нормализация входа делается ОДИН раз на всю
запись (до нарезки), чтобы амплитудная динамика между фрагментами
(громкая речь vs паузы) не стиралась пофрагментной нормировкой.

Текст: RuBERT-tiny2 (заморожен) -> последовательность токенов [L, D_t].

Кэширование на диск принципиально: корпус маленький, энкодеры заморожены,
поэтому обучение головы фузирования занимает секунды и не требует большой VRAM.
Аудио-эмбеддинги хранятся в float16 (объём кэша вдвое меньше) и приводятся
к float32 при загрузке в Dataset.

Запись кэша атомарная: сначала во временный файл, затем os.replace.
Это защищает от ситуации, когда запись оборвалась посередине (нехватка
места, сбой смонтированной ФС) и битый .npy остался лежать в кэше —
иначе следующий запуск принял бы его за готовый.
"""

from __future__ import annotations

import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer

from src.config import Config

AUDIO_CACHE_SUBDIR = "audio_w2v"  # отдельно от старого кэша Whisper ("audio")


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    """Атомарно сохраняет массив: временный файл -> переименование."""
    tmp_path = path.with_suffix(".npy.tmp")
    with open(tmp_path, "wb") as f:  # np.save(f, ...) не дописывает расширение
        np.save(f, array)
    os.replace(tmp_path, path)


def _is_valid_npy(path: Path) -> bool:
    """Проверяет, что кэш-файл читается целиком (не битый от обрыва записи)."""
    if not path.exists():
        return False
    try:
        arr = np.load(path)
        return arr.size > 0
    except Exception:
        return False


def _find_todo(manifest: pd.DataFrame, out_dir: Path) -> list[pd.Series]:
    """Возвращает записи без валидного кэша; битые файлы удаляет."""
    todo = []
    for _, row in manifest.iterrows():
        cache_path = out_dir / f"{row['sample_id']}.npy"
        if _is_valid_npy(cache_path):
            continue
        if cache_path.exists():
            print(f"Битый кэш, пересчитаю: {cache_path}")
            cache_path.unlink()
        todo.append(row)
    return todo


@torch.no_grad()
def extract_audio_embeddings(manifest: pd.DataFrame, cfg: Config,
                             device: torch.device) -> int:
    """Считает эмбеддинги wav2vec 2.0 для всех записей без кэша.

    Запись обрезается до ``max_audio_sec``, нормализуется целиком и
    прогоняется фрагментами по ``fragment_sec`` секунд; фреймы фрагментов
    склеиваются в общую временну́ю ось [T, hidden].

    Returns:
        Размерность аудио-эмбеддинга.
    """
    out_dir = cfg.cache_dir / AUDIO_CACHE_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = _find_todo(manifest, out_dir)

    extractor = AutoFeatureExtractor.from_pretrained(cfg.audio_model)
    model = AutoModel.from_pretrained(cfg.audio_model).to(device).eval()
    audio_dim: int = model.config.hidden_size
    if not todo:
        return audio_dim

    max_samples = cfg.max_audio_sec * cfg.target_sr
    frag_samples = cfg.fragment_sec * cfg.target_sr
    for row in tqdm(todo, desc="Audio embeddings (wav2vec2)"):
        wav, _ = librosa.load(str(row["audio_path"]), sr=cfg.target_sr, mono=True)
        wav = wav[:max_samples]
        # Нормализация ОДИН раз на всю запись — до нарезки на фрагменты
        values = extractor(wav, sampling_rate=cfg.target_sr,
                           return_tensors="pt").input_values[0]
        chunks: list[np.ndarray] = []
        for start in range(0, values.shape[0], frag_samples):
            frag = values[start:start + frag_samples].unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out = model(frag).last_hidden_state  # [1, t, D]
            chunks.append(out.squeeze(0).float().cpu().numpy())
        emb = np.concatenate(chunks, axis=0)         # [T, D]
        _atomic_save_npy(out_dir / f"{row['sample_id']}.npy", emb.astype(np.float16))
    return audio_dim


@torch.no_grad()
def extract_text_embeddings(manifest: pd.DataFrame, cfg: Config,
                            device: torch.device) -> int:
    """Считает токен-эмбеддинги RuBERT для всех транскриптов без кэша.

    Returns:
        Размерность текстового эмбеддинга.
    """
    out_dir = cfg.cache_dir / "text"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = _find_todo(manifest, out_dir)

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model)
    model = AutoModel.from_pretrained(cfg.text_model).to(device).eval()
    text_dim: int = model.config.hidden_size
    if not todo:
        return text_dim

    for row in tqdm(todo, desc="Text embeddings"):
        tokens = tokenizer(
            str(row["transcript"]),
            max_length=cfg.max_text_tokens,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        emb = model(**tokens).last_hidden_state  # [1, L, D_t]
        _atomic_save_npy(out_dir / f"{row['sample_id']}.npy",
                         emb.squeeze(0).float().cpu().numpy())
    return text_dim


def ensure_features(manifest: pd.DataFrame, cfg: Config) -> tuple[int, int]:
    """Строит недостающие кэши и возвращает (audio_dim, text_dim)."""
    device = cfg.resolve_device()
    audio_dim = extract_audio_embeddings(manifest, cfg, device)
    text_dim = extract_text_embeddings(manifest, cfg, device)
    print(f"Эмбеддинги готовы: audio_dim={audio_dim}, text_dim={text_dim}")
    return audio_dim, text_dim
