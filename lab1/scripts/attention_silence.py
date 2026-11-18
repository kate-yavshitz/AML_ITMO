"""Проверка гипотезы «модель смотрит на паузы».

Идея: если концентрация внимания на тихих фреймах — клинический маркер
(паузы-хезитации при афазии), то доля внимания на тишину должна быть
систематически выше у пациентов, чем у контроля. Если же это артефакт
«attention sink» (парковка softmax-массы на неинформативных фреймах),
доля будет примерно одинаковой в обоих классах.

Протокол честный: каждая запись анализируется моделью СВОЕГО фолда
(та, что не видела этого диктора при обучении) — маппинг берём из
reports/predictions.csv.

Для каждой записи:
  1. профиль внимания = усреднение весов cross-attention по токенам;
  2. RMS-энергия аудио по тем же 20-мс фреймам (1500 шт. на 30 с окна);
  3. «тихие» фреймы = нижние 30% по энергии внутри РЕАЛЬНОЙ длины записи
     (нулевой паддинг Whisper за концом записи в анализ не берём);
  4. attn_quiet_share = доля внимания на тихих фреймах
     (внимание нормируется на реальные фреймы, без паддинга).

Запуск:
    python scripts/attention_silence.py --config configs/default.yaml

Выход: reports/attention_silence.csv, reports/attention_silence.png,
сравнение групп (t-тест + Mann-Whitney) в консоль.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data import load_manifest
from src.model import CrossAttentionFusion

QUIET_PERCENTILE = 30.0  # нижние 30% фреймов по энергии считаем «тихими»


def frame_rms(wav_path: str, sr: int, n_frames: int, max_sec: int,
              fps: int = 50) -> np.ndarray:
    """RMS-энергия по 20-мс фреймам внутри реальной записи (без паддинга).

    Шаг фрейма ФИКСИРОВАН (sr // fps отсчётов): он должен совпадать с
    временны́м шагом эмбеддингов wav2vec2, а не выводиться из длины записи.

    Returns:
        Массив [n_frames]; фреймы за пределами реальной длины = NaN.
    """
    y, _ = librosa.load(wav_path, sr=sr)
    y = y[: max_sec * sr]
    hop = sr // fps                         # 320 отсчётов = 20 мс при 16 кГц
    n_real = min(len(y) // hop, n_frames)
    rms = np.full(n_frames, np.nan)
    if n_real > 0:
        frames = y[: n_real * hop].reshape(n_real, hop)
        rms[:n_real] = np.sqrt((frames**2).mean(axis=1))
    return rms


@torch.no_grad()
def attention_profile(model: CrossAttentionFusion, audio: np.ndarray,
                      text: np.ndarray, device: torch.device) -> np.ndarray:
    """Средний по токенам профиль внимания: [T_a]."""
    a = torch.from_numpy(audio).unsqueeze(0).to(device)
    t = torch.from_numpy(text).unsqueeze(0).to(device)
    a_mask = torch.ones(1, a.shape[1], dtype=torch.bool, device=device)
    t_mask = torch.ones(1, t.shape[1], dtype=torch.bool, device=device)
    _, attn = model(a, t, a_mask, t_mask)
    return attn[0].cpu().numpy().mean(axis=0)  # [T_a]


def main() -> None:
    parser = argparse.ArgumentParser(description="Анализ внимания на тихих фреймах")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = Config.from_yaml(args.config)
    cfg.ensure_dirs()
    device = cfg.resolve_device()

    manifest = load_manifest(cfg.data_dir / "manifest.csv").set_index("sample_id")
    preds = pd.read_csv(cfg.reports_dir / "predictions.csv")  # fold per OOF-записи

    # Загружаем чекпоинты всех фолдов
    models: dict[int, CrossAttentionFusion] = {}
    for fold in sorted(preds["fold"].unique()):
        ckpt = torch.load(cfg.checkpoint_dir / f"best_fold{fold}.pt",
                          map_location=device, weights_only=False)
        model = CrossAttentionFusion(
            audio_dim=int(ckpt["audio_dim"]), text_dim=int(ckpt["text_dim"]),
            d_model=cfg.d_model, n_heads=cfg.n_heads, dropout=cfg.dropout,
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        models[int(fold)] = model

    rows: list[dict[str, float | int | str]] = []
    for _, pred in tqdm(list(preds.iterrows()), desc="Attention on silence"):
        sid = str(pred["sample_id"])
        if sid not in manifest.index:
            continue
        row = manifest.loc[sid]
        audio = np.load(cfg.cache_dir / "audio_w2v" / f"{sid}.npy").astype(np.float32)
        text = np.load(cfg.cache_dir / "text" / f"{sid}.npy")
        profile = attention_profile(models[int(pred["fold"])], audio, text, device)
        profile = profile / profile.sum()  # на всякий случай перенормируем

        rms = frame_rms(str(row["audio_path"]), cfg.target_sr,
                        n_frames=len(profile), max_sec=cfg.max_audio_sec,
                        fps=cfg.audio_fps)
        real = ~np.isnan(rms)                # только реальный звук, без паддинга
        if real.sum() < 50:                  # совсем короткие записи пропускаем
            continue
        thr = np.nanpercentile(rms[real], QUIET_PERCENTILE)
        quiet = real & (rms <= thr)

        attn_real = profile[real] / profile[real].sum()
        quiet_share_of_frames = float(quiet[real].sum() / real.sum())
        attn_quiet_share = float(attn_real[quiet[real]].sum())
        rows.append({
            "sample_id": sid,
            "label": int(pred["y_true"]),
            "fold": int(pred["fold"]),
            "duration_sec": float(row.get("duration_sec", 0) or 0),
            "quiet_share_of_frames": quiet_share_of_frames,
            "attn_quiet_share": attn_quiet_share,
            # >1: внимания на тишине больше, чем «положено» при равномерном
            "attention_concentration": attn_quiet_share / max(quiet_share_of_frames, 1e-9),
        })

    df = pd.DataFrame(rows)
    out_csv = cfg.reports_dir / "attention_silence.csv"
    df.to_csv(out_csv, index=False)
    print(f"Сохранено: {out_csv} ({len(df)} записей)")

    # --- Сравнение групп ---
    aph = df.loc[df["label"] == 1, "attn_quiet_share"].to_numpy()
    ctl = df.loc[df["label"] == 0, "attn_quiet_share"].to_numpy()
    t_stat, t_p = stats.ttest_ind(aph, ctl, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(aph, ctl, alternative="two-sided")
    print("\n=== Доля внимания на тихих фреймах (нижние "
          f"{QUIET_PERCENTILE:.0f}% по энергии) ===")
    print(f"aphasia (n={len(aph)}): mean={aph.mean():.3f} ± {aph.std():.3f}")
    print(f"control (n={len(ctl)}): mean={ctl.mean():.3f} ± {ctl.std():.3f}")
    print(f"разница средних: {aph.mean() - ctl.mean():+.3f}")
    print(f"t-тест Уэлча: t={t_stat:.2f}, p={t_p:.4f}")
    print(f"Mann-Whitney:   U={u_stat:.0f}, p={u_p:.4f}")
    print("\nИнтерпретация: если разница значимо > 0 — внимание к тишине "
          "сильнее у пациентов (маркер пауз). Если разницы нет — "
          "похоже на артефакт attention sink.")

    # --- График ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    data = [ctl, aph]
    bp = axes[0].boxplot(data, patch_artist=True, widths=0.5)
    axes[0].set_xticks([1, 2])
    axes[0].set_xticklabels(["control", "aphasia"])
    for patch, color in zip(bp["boxes"], ["#2e86ab", "#d1495b"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for i, vals in enumerate(data):
        jitter = np.random.default_rng(0).normal(0, 0.03, len(vals))
        axes[0].scatter(np.full(len(vals), i + 1) + jitter, vals,
                        s=10, alpha=0.5, color="black")
    axes[0].set_ylabel("Доля внимания на тихих фреймах")
    axes[0].set_xlabel("Класс")
    axes[0].set_title(f"Внимание к тишине по классам (p={u_p:.4f})")

    for label, color, name in ((0, "#2e86ab", "control"), (1, "#d1495b", "aphasia")):
        sub = df[df["label"] == label]
        axes[1].scatter(sub["duration_sec"], sub["attn_quiet_share"],
                        s=14, alpha=0.6, color=color, label=name)
    axes[1].axhline(QUIET_PERCENTILE / 100.0, color="k", linestyle="--",
                    alpha=0.5, label="равномерное внимание")
    axes[1].set_xlabel("Длительность записи, с")
    axes[1].set_ylabel("Доля внимания на тихих фреймах")
    axes[1].set_title("Зависит ли внимание к тишине от длительности")
    axes[1].legend()
    fig.tight_layout()
    out_png = cfg.reports_dir / "attention_silence.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {out_png}")


if __name__ == "__main__":
    main()
