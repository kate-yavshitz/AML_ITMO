"""Генерация синтетического датасета для smoke-теста пайплайна.

Создаёт data/raw/aphasia и data/raw/nt_train в формате имён RuAphasiaBank
(чтобы препроцессинг и парсер метаданных работали как на боевых данных):
«control» (nt_train) — ровный гармонический сигнал (имитация регулярной
фонации), «aphasia» — с паузами и дрожанием частоты.

Нужен только для проверки, что весь пайплайн работает end-to-end,
не для получения осмысленных метрик.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000


def synth_control(rng: np.random.Generator, duration: float = 3.0) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * duration)) / SAMPLE_RATE
    f0 = rng.uniform(100, 200)
    signal = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    return (0.3 * signal).astype(np.float32)


def synth_aphasia(rng: np.random.Generator, duration: float = 3.0) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * duration)) / SAMPLE_RATE
    f0 = rng.uniform(100, 200)
    jitter = 1 + 0.05 * np.sin(2 * np.pi * 7 * t)          # дрожание частоты
    signal = sum(np.sin(2 * np.pi * f0 * k * t * jitter) / k for k in range(1, 6))
    gate = (np.sin(2 * np.pi * 0.8 * t) > -0.2).astype(np.float32)  # паузы
    return (0.3 * signal * gate).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Синтетический датасет для smoke-теста")
    parser.add_argument("--n", type=int, default=6, help="Дикторов на класс")
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    # Формат имён RuAphasiaBank: sp{ID}_{пол}_{возраст}_{тип}_{тяжесть}_{№}
    for i in range(args.n):
        for j in range(2):  # по 2 записи на диктора — проверка группового сплита
            aph = Path(f"data/raw/aphasia/sp{i + 1}_m_40_cma_m_{j + 1}.wav")
            aph.parent.mkdir(parents=True, exist_ok=True)
            sf.write(aph, synth_aphasia(rng), SAMPLE_RATE)
            nt = Path(f"data/raw/nt_train/sp{i + 100}_m_60_nt_{j + 1}.wav")
            nt.parent.mkdir(parents=True, exist_ok=True)
            sf.write(nt, synth_control(rng), SAMPLE_RATE)
    print(f"Синтетика готова: {args.n} дикторов на класс в data/raw/")


if __name__ == "__main__":
    main()
