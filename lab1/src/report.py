"""Визуализация итогового отчёта: EDA + результаты обучения.

Все графики сохраняются в ``reports/`` и могут быть построены внутри контейнера:
    python -m src.report --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-режим для контейнера

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix

from src.config import Config
from src.data import EmbeddingDataset, collate_embeddings, load_manifest
from src.model import CrossAttentionFusion


def plot_eda(manifest: pd.DataFrame, cfg: Config) -> None:
    """EDA: баланс классов, источники, длительности, возраст, типы афазии.

    Возрастной график важен для проверки конфаунда: если классы заметно
    различаются по возрасту, классификатор может выучить возраст вместо патологии.
    """
    df = manifest.copy()
    df["class"] = df["label"].map({1: "aphasia", 0: "control"})

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # 1. Баланс классов по записям и по дикторам
    counts = df["class"].value_counts()
    speakers = df.groupby("speaker")["class"].first().value_counts()
    x = np.arange(len(counts.index))
    axes[0, 0].bar(x - 0.2, [counts.get(c, 0) for c in counts.index], 0.4, label="записи",
                   color="#2e86ab")
    axes[0, 0].bar(x + 0.2, [speakers.get(c, 0) for c in counts.index], 0.4, label="дикторы",
                   color="#d1495b")
    axes[0, 0].set_xticks(x, counts.index)
    axes[0, 0].set_title("Баланс классов")
    axes[0, 0].set_xlabel("Класс")
    axes[0, 0].set_ylabel("Количество")
    axes[0, 0].legend()

    # 2. Источники данных
    src_counts = df.groupby(["source", "class"]).size().unstack(fill_value=0)
    src_counts.plot(kind="bar", ax=axes[0, 1], color=["#2e86ab", "#d1495b"])
    axes[0, 1].set_title("Источники (rubank = RuAphasiaBank)")
    axes[0, 1].set_xlabel("Источник данных")
    axes[0, 1].set_ylabel("Число записей")
    axes[0, 1].tick_params(axis="x", rotation=0)

    # 3. Длительности
    if "duration_sec" in df.columns:
        for cls, color in (("aphasia", "#d1495b"), ("control", "#2e86ab")):
            axes[0, 2].hist(df.loc[df["class"] == cls, "duration_sec"].dropna(),
                            bins=20, alpha=0.6, label=cls, color=color)
        axes[0, 2].set_title("Длительность фонограмм")
        axes[0, 2].set_xlabel("Длительность, с")
        axes[0, 2].set_ylabel("Число записей")
        axes[0, 2].legend()

    # 4. Возраст по классам (0 = неизвестен — исключаем из графика)
    known_age = df[df["age"] > 0]
    for cls, color in (("aphasia", "#d1495b"), ("control", "#2e86ab")):
        axes[1, 0].hist(known_age.loc[known_age["class"] == cls, "age"],
                        bins=15, alpha=0.6, label=cls, color=color)
    axes[1, 0].set_title(f"Возраст (возраст неизвестен у {(df['age'] == 0).sum()} записей)")
    axes[1, 0].set_xlabel("Возраст, лет")
    axes[1, 0].set_ylabel("Число записей")
    axes[1, 0].legend()

    # 5. Типы афазии
    aph = df[df["label"] == 1]
    if not aph.empty:
        aph["aphasia_type"].value_counts().plot(kind="bar", ax=axes[1, 1], color="#d1495b")
        axes[1, 1].set_title("Типы афазии (записи)")
        axes[1, 1].set_xlabel("Тип афазии")
        axes[1, 1].set_ylabel("Число записей")
        axes[1, 1].tick_params(axis="x", rotation=45)

    # 6. Записей на диктора
    per_speaker = df.groupby(["speaker", "class"]).size()
    for cls, color in (("aphasia", "#d1495b"), ("control", "#2e86ab")):
        vals = per_speaker.xs(cls, level="class") if cls in per_speaker.index.get_level_values(1) else []
        axes[1, 2].hist(vals, bins=15, alpha=0.6, label=cls, color=color)
    axes[1, 2].set_title("Записей на диктора")
    axes[1, 2].set_xlabel("Число записей одного диктора")
    axes[1, 2].set_ylabel("Число дикторов")
    axes[1, 2].legend()

    fig.tight_layout()
    out = cfg.reports_dir / "eda.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    _check_saved(out)


def _check_saved(path: Path) -> None:
    """Проверяет, что картинка реально записалась и не пустая.

    На смонтированных ФС (Windows-диск через /mnt/d) запись иногда
    «теряется» — файл виден, но пуст. Лучше узнать об этом сразу.
    """
    size_kb = path.stat().st_size // 1024 if path.exists() else 0
    if size_kb == 0:
        print(f"ВНИМАНИЕ: {path} пуст или не записался — "
              f"перезапустите шаг отчёта (python -m src.report)")
    else:
        print(f"Сохранено: {path} ({size_kb} КБ)")


def plot_test_results(cfg: Config) -> None:
    """Confusion matrix + ROC по out-of-fold предсказаниям.

    Каждый диктор RuAphasiaBank предсказан ровно один раз — моделью,
    которая его не видела при обучении.
    """
    pred_path = cfg.reports_dir / "predictions.csv"
    if not pred_path.exists():
        print("predictions.csv не найден — сначала запустите обучение")
        return
    preds = pd.read_csv(pred_path)
    y_true = preds["y_true"].to_numpy()
    probs = preds["prob_aphasia"].to_numpy()
    y_pred = (probs >= 0.5).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred),
                           display_labels=["control", "aphasia"]).plot(ax=axes[0], colorbar=False)
    axes[0].set_title("Confusion matrix (out-of-fold)")
    if len(np.unique(y_true)) > 1:
        # from_predictions сам считает AUC и выводит его в легенде
        RocCurveDisplay.from_predictions(y_true, probs, ax=axes[1],
                                         name="cross-attention (OOF)")
        axes[1].set_title("ROC (out-of-fold)")
    fig.tight_layout()
    out = cfg.reports_dir / "test_results.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    _check_saved(out)


@torch.no_grad()
def plot_attention_example(cfg: Config) -> None:
    """Визуализация cross-attention: каким моментам записи «уделяет внимание» текст.

    Две панели: тепловая карта (текстовые токены x время) и усреднённый
    профиль внимания по времени. Веса softmax по тысячам фреймов крошечные
    (~1/T), поэтому верх шкалы задаём по 99-му перцентилю — иначе
    автомасштаб imshow схлопывает карту в чёрный прямоугольник.
    """
    best_path = cfg.checkpoint_dir / "best_fold0.pt"
    if not best_path.exists():
        return
    device = cfg.resolve_device()
    manifest = load_manifest(cfg.data_dir / "manifest.csv")
    sample = manifest.sample(n=1, random_state=cfg.seed).iloc[0]

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model = CrossAttentionFusion(
        audio_dim=int(checkpoint["audio_dim"]), text_dim=int(checkpoint["text_dim"]),
        d_model=cfg.d_model, n_heads=cfg.n_heads, dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    item = EmbeddingDataset(manifest.iloc[[sample.name]], cfg.cache_dir,
                            max_frames=cfg.max_audio_sec * cfg.audio_fps)[0]
    batch = collate_embeddings([item])
    _, attn = model(batch["audio"].to(device), batch["text"].to(device),
                    batch["audio_mask"].to(device), batch["text_mask"].to(device))

    attn_map = np.nan_to_num(attn[0].cpu().numpy())      # [T_t, T_a]
    real_tokens = int(batch["text_mask"][0].sum())
    attn_map = attn_map[:real_tokens]                    # только настоящие токены
    t_frames = attn_map.shape[1]
    frame_sec = 1.0 / cfg.audio_fps                      # wav2vec2: 20 мс на фрейм
    print(f"Attention stats: min={attn_map.min():.2e}, max={attn_map.max():.2e}, "
          f"mean={attn_map.mean():.2e}, токенов={real_tokens}, фреймов={t_frames}")

    time_axis = np.arange(t_frames) * frame_sec
    profile = attn_map.mean(axis=0)                      # усреднение по токенам
    vmax = float(np.percentile(attn_map, 99))

    fig, (ax_map, ax_prof) = plt.subplots(
        2, 1, figsize=(12, 7), height_ratios=[2, 1], sharex=True)

    im = ax_map.imshow(attn_map, aspect="auto", origin="upper", cmap="magma",
                       vmin=0.0, vmax=vmax,
                       extent=[0, t_frames * frame_sec, real_tokens - 0.5, -0.5])
    ax_map.set_ylabel("Текстовые токены (Q)")
    ax_map.set_title(f"Cross-attention: {sample['sample_id']} "
                     f"(метка: {'aphasia' if sample['label'] == 1 else 'control'})")
    fig.colorbar(im, ax=ax_map, label="вес внимания")

    ax_prof.plot(time_axis, profile, color="#2e86ab")
    ax_prof.set_xlabel("Время в записи, с")
    ax_prof.set_ylabel("Средний вес")
    ax_prof.grid(alpha=0.3)
    # Вертикальная линия — конец реального звука. Правее неё у wav2vec2
    # фреймов просто нет (паддинга, как у Whisper, больше не добавляем),
    # но при капе max_audio_sec запись могла быть обрезана.
    duration = float(sample.get("duration_sec", 0) or 0)
    if duration > 0:
        end_sec = min(duration, t_frames * frame_sec)
        ax_prof.axvline(end_sec, color="#d1495b", linestyle="--",
                        label=f"конец реальной записи ({end_sec:.0f} с)")
        ax_prof.legend()

    fig.tight_layout()
    out = cfg.reports_dir / "attention_example.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    _check_saved(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Визуализация отчёта лабораторной")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = Config.from_yaml(args.config)
    cfg.ensure_dirs()

    manifest = load_manifest(cfg.data_dir / "manifest.csv")
    transcripts_dir = cfg.cache_dir / "transcripts"
    if transcripts_dir.exists():
        manifest["transcript"] = [
            (transcripts_dir / f"{sid}.txt").read_text(encoding="utf-8")
            if (transcripts_dir / f"{sid}.txt").exists() else ""
            for sid in manifest["sample_id"]
        ]
    plot_eda(manifest, cfg)
    plot_test_results(cfg)
    plot_attention_example(cfg)


if __name__ == "__main__":
    main()
