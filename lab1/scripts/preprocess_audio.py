"""Приведение всех фонограмм к единому формату + сборка манифеста.

Вход:  data/raw/{aphasia,nt_train,nt}/*.wav  (произвольные SR/каналы/громкость)
Выход: data/processed/{aphasia,control}/*.wav — моно, 16 кГц, float32,
       тишина по краям обрезана, пиковая нормализация до -1 dBFS;
       data/manifest.csv — метаданные каждой записи.

Запуск:
    python scripts/preprocess_audio.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.metadata import parse_filename

SOURCE_FOLDERS = ("aphasia", "nt_train", "nt")
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
PEAK_LEVEL = 0.89  # -1 dBFS
TOP_DB = 25.0      # порог обрезки тишины


def iter_audio(src_dir: Path) -> list[Path]:
    """Находит аудиофайлы независимо от регистра расширения (.wav/.WAV)."""
    return [p for p in sorted(src_dir.rglob("*"))
            if p.suffix.lower() in AUDIO_EXTENSIONS]


def preprocess_file(src: Path, dst: Path, target_sr: int) -> float:
    """Конвертирует один файл: моно 16 кГц -> трим тишины -> пик-нормализация.

    Returns:
        Длительность обработанной записи в секундах.
    """
    import librosa  # импорт здесь: парсинг метаданных не требует librosa
    import numpy as np
    import soundfile as sf

    y, _ = librosa.load(str(src), sr=target_sr, mono=True)
    y_trimmed, _ = librosa.effects.trim(y, top_db=TOP_DB)
    if len(y_trimmed) < target_sr // 2:  # меньше 0.5 с после трима — не трогаем
        y_trimmed = y
    peak = float(np.max(np.abs(y_trimmed)))
    if peak > 0:
        y_trimmed = y_trimmed / peak * PEAK_LEVEL
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), y_trimmed, target_sr, subtype="PCM_16")
    return len(y_trimmed) / target_sr


def main() -> None:
    parser = argparse.ArgumentParser(description="Препроцессинг аудио и сборка манифеста")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int, default=0,
                        help="Обработать не более N файлов (для smoke-теста)")
    args = parser.parse_args()
    cfg = Config.from_yaml(args.config)

    rows: list[dict[str, object]] = []
    errors: list[str] = []
    processed = 0

    for folder in SOURCE_FOLDERS:
        src_dir = cfg.raw_dir / folder
        if not src_dir.is_dir():
            print(f"Папка {src_dir} не найдена — пропускаю")
            continue
        for wav in iter_audio(src_dir):
            try:
                meta = parse_filename(wav, folder)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if args.limit and processed >= args.limit:
                break
            class_dir = "aphasia" if meta.label == 1 else "control"
            dst = cfg.processed_dir / class_dir / f"{meta.sample_id}.wav"
            if dst.exists():
                duration = -1.0  # уже обработан, длительность не пересчитываем
            else:
                duration = preprocess_file(wav, dst, cfg.target_sr)
            processed += 1
            rows.append({
                "audio_path": str(dst),
                "sample_id": meta.sample_id,
                "label": meta.label,
                "speaker": meta.speaker,
                "source": meta.source,
                "gender": meta.gender,
                "age": meta.age,
                "aphasia_type": meta.aphasia_type,
                "severity": meta.severity,
                "duration_sec": round(duration, 2),
            })

    if not rows:
        # Диагностика вместо traceback: покажем, что контейнер реально видит
        print(f"ОШИБКА: в {cfg.raw_dir} не найдено ни одного аудиофайла.")
        print(f"Содержимое {cfg.raw_dir} (первые 50 объектов):")
        contents = list(cfg.raw_dir.rglob("*")) if cfg.raw_dir.exists() else []
        for p in contents[:50]:
            print(f"  {p}")
        if not contents:
            print("  <пусто или папка не смонтирована>")
        raise SystemExit(
            "\nПроверьте монтирование: папка sound должна быть видна в контейнере "
            "как /app/data/raw и содержать aphasia/, nt_train/, nt/ с .wav внутри. "
            "Команда для проверки: docker run --rm -v <путь_к_sound>:/app/data/raw "
            "lab-aphasia ls -R /app/data/raw | head"
        )

    manifest = pd.DataFrame(rows)
    manifest_path = cfg.data_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print(f"\nМанифест: {manifest_path} ({len(manifest)} записей)")
    print(f"Классы: {manifest['label'].value_counts().to_dict()}")
    print(f"Дикторы: {manifest.groupby('source')['speaker'].nunique().to_dict()}")
    if errors:
        print(f"\nОШИБКИ парсинга ({len(errors)}):")
        for err in errors:
            print(f"  {err}")


if __name__ == "__main__":
    main()
