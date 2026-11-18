"""Скачивание датасета внутри контейнера.

Источник задаётся переменной окружения DATA_URL или аргументом --url.
Поддерживаются:
  * прямая ссылка на .zip / .tar(.gz);
  * публичная ссылка Яндекс.Диска (https://disk.yandex.ru/d/...);
  * публичная ссылка Google Drive (через gdown).

Пример:
    DATA_URL="https://disk.yandex.ru/d/XXXX" python scripts/download_data.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import tarfile
import zipfile
from pathlib import Path

import requests

RAW_DIR = Path("data/raw")


def resolve_yandex_disk(public_url: str) -> str:
    """Получает прямую ссылку на скачивание по публичной ссылке Яндекс.Диска."""
    api = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
    response = requests.get(api, params={"public_key": public_url}, timeout=60)
    response.raise_for_status()
    return str(response.json()["href"])


def download(url: str, dest: Path) -> None:
    """Качает файл потоково с прогрессом."""
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r{downloaded / 1e6:.1f}/{total / 1e6:.1f} МБ", end="", flush=True)
    print()


def extract(archive: Path, dest: Path) -> None:
    """Распаковывает zip/tar-архив в папку с сырыми данными."""
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif archive.suffixes[-2:] in ([".tar", ".gz"], [".tar", ".xz"]) or archive.suffix == ".tar":
        with tarfile.open(archive) as tf:
            tf.extractall(dest)
    else:
        raise ValueError(f"Неизвестный формат архива: {archive.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачивание датасета")
    parser.add_argument("--url", default=os.getenv("DATA_URL", ""))
    args = parser.parse_args()

    if not args.url:
        raise SystemExit(
            "Не задан источник данных. Укажите --url или переменную DATA_URL.\n"
            "Альтернатива: положите данные в data/raw/{aphasia,control}/ и "
            "смонтируйте папку в контейнер через -v."
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path("data/_download")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    url = args.url
    if "disk.yandex" in url:
        print("Яндекс.Диск: получаю прямую ссылку...")
        url = resolve_yandex_disk(url)
        archive = tmp / "dataset.zip"
    elif "drive.google" in url or "docs.google" in url:
        import gdown
        archive = tmp / "dataset"
        gdown.download(url, str(archive), fuzzy=True)
    else:
        name = url.split("?")[0].rstrip("/").rsplit("/", 1)[-1] or "dataset.zip"
        archive = tmp / name

    if not archive.exists():
        print(f"Скачиваю {url} ...")
        download(url, archive)

    print(f"Распаковываю в {RAW_DIR} ...")
    if zipfile.is_zipfile(archive) or tarfile.is_tarfile(archive):
        extract(archive, RAW_DIR)
    else:  # одиночный файл (например, labels.csv)
        shutil.copy(archive, RAW_DIR / archive.name)

    shutil.rmtree(tmp, ignore_errors=True)
    n_audio = len([p for p in RAW_DIR.rglob("*")
                   if p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}])
    print(f"Готово: найдено {n_audio} аудиофайлов в {RAW_DIR}")


if __name__ == "__main__":
    main()
