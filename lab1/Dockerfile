# Базовый образ с CUDA 11.6 — совместим с драйвером 512.78 (WSL2, 6 ГБ VRAM)
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Europe/Moscow \
    # Кэши скачанных моделей (Whisper, HF) кладём в data/.cache: если data/
    # смонтирована с хоста, модели не качаются заново при каждом запуске
    XDG_CACHE_HOME=/app/data/.cache

# ffmpeg нужен Whisper'у для декодирования аудио
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# openai-whisper собирается из исходников, и его setup.py импортирует
# pkg_resources, которого нет в свежем setuptools. Ставим setuptools<81
# и собираем пакеты без изоляции (--no-build-isolation), чтобы setup.py
# использовал именно его.
RUN pip install --no-cache-dir "setuptools<81" wheel && \
    pip install --no-cache-dir --no-build-isolation -r requirements.txt

COPY . .

# Точка входа по умолчанию: полный цикл — обучение + визуализация отчёта.
# Данные и результаты монтируются снаружи через -v (см. README).
CMD ["bash", "scripts/run_all.sh"]
