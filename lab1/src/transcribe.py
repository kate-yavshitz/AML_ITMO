"""Транскрибация фонограмм Whisper'ом с кэшированием результатов.

Если рядом с аудио уже лежит ``<имя>.txt``, он используется как есть —
Whisper не запускается лишний раз.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

import whisper
from src.config import Config

EMPTY_TRANSCRIPT = "[нет речи]"


def ensure_transcripts(manifest: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Гарантирует наличие транскрипта для каждой записи манифеста.

    Возвращает манифест с дополнительной колонкой ``transcript``.
    """
    cache_dir = cfg.cache_dir / "transcripts"
    cache_dir.mkdir(parents=True, exist_ok=True)

    texts: list[str] = []
    missing: list[int] = []
    for i, row in manifest.iterrows():
        sid = str(row["sample_id"])
        cached = cache_dir / f"{sid}.txt"
        sidecar = Path(str(row["audio_path"])).with_suffix(".txt")
        if cached.exists():
            texts.append(cached.read_text(encoding="utf-8"))
        elif sidecar.exists():
            text = sidecar.read_text(encoding="utf-8").strip()
            cached.write_text(text, encoding="utf-8")
            texts.append(text)
        else:
            texts.append("")
            missing.append(i)

    if missing:
        device = str(cfg.resolve_device())
        print(f"Транскрибация {len(missing)} фонограмм Whisper-{cfg.whisper_model} ({device})...")
        model = whisper.load_model(cfg.whisper_model, device=device)
        for i in tqdm(missing, desc="ASR"):
            row = manifest.iloc[i]
            audio_path = str(row["audio_path"])
            result = model.transcribe(audio_path, language=cfg.language, fp16=(device == "cuda"))
            text = result["text"].strip() or EMPTY_TRANSCRIPT
            (cache_dir / f"{row['sample_id']}.txt").write_text(text, encoding="utf-8")
            texts[i] = text

    manifest = manifest.copy()
    manifest["transcript"] = texts
    return manifest
