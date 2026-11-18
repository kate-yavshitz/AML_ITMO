"""Парсинг метаданных из имён файлов трёх источников корпуса.

Схемы именования:

* ``aphasia/``  — RuAphasiaBank, пациенты:
      sp10_m_0_cma_s_22.wav  ->  sp{ID}_{пол}_{возраст}_{тип}_{тяжесть}_{№}
* ``nt_train/`` — RuAphasiaBank, контроль:
      sp57_m_66_nt_167.wav   ->  sp{ID}_{пол}_{возраст}_nt_{№}
* ``nt/``       — собственные записи, свободный формат:
      aleksinov_f_56(2).wav, Shirina_f_40_dialog.wav,
      ГильманшинаГМ_59_Ж_домохозяйка_диалог_21.08.2026.wav

Возраст 0 означает «возраст неизвестен»; по словам автора корпуса, такие
записи тоже сделаны с людьми 40+ — молодых участников в эксперименте нет,
за редкими исключениями (например, sp20 с полем 24 — один из немногих
молодых пациентов; часть его файлов размечена возрастом 0).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

RUBANK_APHASIA_RE = re.compile(
    r"^sp(?P<speaker>\d+)_(?P<gender>[mf])_(?P<age>\d+)_(?P<atype>[a-z]+)_(?P<severity>[a-z]+)_(?P<num>\d+)$"
)
RUBANK_NT_RE = re.compile(
    r"^sp(?P<speaker>\d+)_(?P<gender>[mf])_(?P<age>\d+)_nt_(?P<num>\d+)$"
)

GENDER_MAP = {"male": "m", "female": "f", "м": "m", "ж": "f"}


@dataclass
class FileMeta:
    """Метаданные одной фонограммы."""

    audio_path: Path
    sample_id: str
    label: int            # 1 = афазия, 0 = контроль
    speaker: str
    source: str           # rubank | collected
    gender: str           # m | f | unknown
    age: int              # 0 = неизвестен (но все участники 40+, см. docstring)
    aphasia_type: str     # cma | efa | ... | nt
    severity: str         # s | m | mld | "" (для контроля)


def _slugify(text: str) -> str:
    """Безопасный sample_id: латиница/кириллица/цифры, без пробелов и скобок."""
    text = unicodedata.normalize("NFKC", text).strip()
    text = re.sub(r"\((\d+)\)", r"_\1", text)   # name(2) -> name_2
    text = re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE)
    return re.sub(r"_+", "_", text).strip("_").lower()


def _parse_rubank(stem: str, label: int) -> FileMeta | None:
    """Парсит форматы RuAphasiaBank (aphasia и nt_train)."""
    match = RUBANK_APHASIA_RE.match(stem)
    if match:
        d = match.groupdict()
        return FileMeta(Path(), "", label, f"sp{d['speaker']}", "rubank",
                        d["gender"], int(d["age"]), d["atype"], d["severity"])
    match = RUBANK_NT_RE.match(stem)
    if match:
        d = match.groupdict()
        return FileMeta(Path(), "", 0, f"sp{d['speaker']}", "rubank",
                        d["gender"], int(d["age"]), "nt", "")
    return None


def _parse_collected(stem: str) -> FileMeta:
    """Парсит свободный формат папки nt/.

    Диктор = первые два токена для латиницы (aleksinov_f, kolpakova_n,
    shirina_f), первый токен для кириллических фамилий с инициалами.
    Возраст и пол извлекаются, когда зашиты в имени.
    """
    tokens = [t for t in _slugify(stem).split("_") if t]
    # Нормализуем маркеры пола, чтобы Shirina_m_44 и Shirina_male_44
    # давали одного диктора
    tokens = [GENDER_MAP.get(t, t) for t in tokens]
    has_cyrillic = bool(re.search(r"[а-яё]", stem.lower()))

    # Два известных исключения: kolpakova_f и kolpakova_n — разные люди,
    # поэтому для латиницы берём первые ДВА токена (фамилия + маркер).
    n_speaker_tokens = 1 if has_cyrillic else 2
    speaker = "_".join(tokens[:n_speaker_tokens]) or _slugify(stem)

    gender, age = "unknown", 0
    for i, token in enumerate(tokens):
        norm = GENDER_MAP.get(token, token)
        if norm in ("m", "f") and gender == "unknown":
            gender = norm
            # возраст обычно идёт сразу после пола
            if i + 1 < len(tokens) and tokens[i + 1].isdigit():
                age = int(tokens[i + 1])
        elif token.isdigit() and age == 0 and 10 <= int(token) <= 100:
            age = int(token)

    return FileMeta(Path(), "", 0, speaker, "collected", gender, age, "nt", "")


def parse_filename(path: Path, folder: str) -> FileMeta:
    """Определяет метаданные файла по имени и папке-источнику.

    Args:
        path: путь к аудиофайлу;
        folder: имя родительской папки (aphasia | nt_train | nt).

    Raises:
        ValueError: если папка неизвестна или формат RuAphasiaBank не распознан.
    """
    stem = path.stem
    # Чистим технические дефекты имён: двойное расширение, пробелы
    if stem.lower().endswith(".wav"):
        stem = stem[:-4]
    stem = stem.strip()

    if folder == "aphasia":
        meta = _parse_rubank(stem, label=1)
        if meta is None:
            raise ValueError(f"Не распознан формат RuAphasiaBank: {path.name}")
    elif folder == "nt_train":
        meta = _parse_rubank(stem, label=0)
        if meta is None:
            raise ValueError(f"Не распознан формат RuAphasiaBank: {path.name}")
    elif folder == "nt":
        meta = _parse_collected(stem)
    else:
        raise ValueError(f"Неизвестная папка-источник: {folder}")

    meta.audio_path = path
    meta.sample_id = _slugify(f"{folder}_{stem}")
    return meta
