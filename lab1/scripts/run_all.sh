#!/usr/bin/env bash
# Полный цикл лабораторной внутри контейнера:
#   1. скачивание данных (если задан DATA_URL и data/raw пуст)
#   2. обучение с ранним фузированием
#   3. визуализация итогового отчёта (EDA + кривые + confusion matrix + attention)
set -e

if [ -n "$DATA_URL" ] && [ -z "$(ls -A data/raw 2>/dev/null)" ]; then
    echo "=== Скачивание данных ==="
    python scripts/download_data.py --url "$DATA_URL"
fi

# -s: файл существует И непустой; пустой manifest.csv (остаток упавшего
# запуска) не считается — препроцессинг запустится заново
if [ ! -s data/manifest.csv ]; then
    rm -f data/manifest.csv
    echo "=== Препроцессинг аудио (librosa: моно 16 кГц, трим тишины, нормализация) ==="
    python scripts/preprocess_audio.py --config configs/default.yaml
fi

echo "=== Обучение ==="
python -m src.train --config configs/default.yaml

echo "=== Визуализация отчёта ==="
python -m src.report --config configs/default.yaml

echo "=== Готово. Результаты: reports/, чекпоинты: checkpoints/best_fold*.pt ==="
