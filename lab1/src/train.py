"""Обучение классификатора афазии на раннем фузировании.

Пайплайн:
    манифест -> транскрипты (Whisper ASR при необходимости) ->
    кэшированные эмбеддинги -> кросс-валидация по дикторам ->
    метрики mean±std + чекпоинты + out-of-fold предсказания.

Дизайн сплита:
    * test/val — ТОЛЬКО RuAphasiaBank (условия записи согласованы с классом
      афазии); собственные записи nt/ идут только в train, иначе модель могла
      бы выучить канал записи вместо патологии;
    * все записи одного диктора — в одном подмножестве (нет утечек голоса);
    * StratifiedGroupKFold: контрольных дикторов в RuAphasiaBank всего ~20,
      поэтому одиночный сплит дал бы в test лишь 3 дикторов контроля.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config, set_seed
from src.data import EmbeddingDataset, collate_embeddings, load_manifest
from src.features import ensure_features
from src.model import CrossAttentionFusion
from src.transcribe import ensure_transcripts


def make_folds(manifest: pd.DataFrame, cfg: Config) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Возвращает список (train, val, test) — по одному на фолд.

    Внутри каждого фолда: test — дикторы RuAphasiaBank фолда; val — часть
    оставшихся дикторов RuAphasiaBank; train — остальные + все записи nt/.
    """
    rubank = manifest[manifest["source"] == "rubank"].reset_index(drop=True)
    collected = manifest[manifest["source"] != "rubank"]

    inconsistent = rubank.groupby("speaker")["label"].nunique()
    if (inconsistent > 1).any():
        raise ValueError(f"Дикторы с противоречивыми метками: "
                         f"{inconsistent[inconsistent > 1].index.tolist()}")

    n_folds = max(cfg.n_folds, 1)
    if n_folds == 1:  # одиночный сплит: эквивалентно фолду с долей test_size
        n_folds = round(1 / cfg.test_size)

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=cfg.seed)
    folds: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    for train_idx, test_idx in sgkf.split(rubank, rubank["label"], groups=rubank["speaker"]):
        rb_train, rb_test = rubank.iloc[train_idx], rubank.iloc[test_idx]

        speakers = rb_train.groupby("speaker").agg(label=("label", "first")).reset_index()
        rel_val = cfg.val_size / (1 - cfg.test_size)
        spk_tr, spk_val = train_test_split(
            speakers, test_size=rel_val, stratify=speakers["label"], random_state=cfg.seed,
        )
        train_df = pd.concat([rb_train[rb_train["speaker"].isin(spk_tr["speaker"])], collected])
        val_df = rb_train[rb_train["speaker"].isin(spk_val["speaker"])]
        folds.append((train_df, val_df, rb_test))
    return folds


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Accuracy / F1 / ROC-AUC по вероятностям класса «афазия»."""
    y_pred = (probs >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
    return metrics


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader,
                  device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Возвращает (вероятности, истинные метки, sample_id) для выборки."""
    model.eval()
    probs_all, labels_all, ids_all = [], [], []
    for batch in loader:
        logits, _ = model(
            batch["audio"].to(device), batch["text"].to(device),
            batch["audio_mask"].to(device), batch["text_mask"].to(device),
        )
        probs_all.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        labels_all.append(batch["label"].numpy())
        ids_all.extend(batch["sample_id"])
    return np.concatenate(probs_all), np.concatenate(labels_all), ids_all


def make_loader(df: pd.DataFrame, cfg: Config, shuffle: bool) -> DataLoader:
    return DataLoader(
        EmbeddingDataset(df, cfg.cache_dir,
                         max_frames=cfg.max_audio_sec * cfg.audio_fps),
        batch_size=cfg.batch_size, shuffle=shuffle, collate_fn=collate_embeddings,
    )


def train_fold(train_df: pd.DataFrame, val_df: pd.DataFrame, cfg: Config,
               device: torch.device, audio_dim: int, text_dim: int,
               ckpt_path: Path) -> None:
    """Обучает модель на одном фолде, сохраняет лучший чекпоинт по val F1."""
    train_loader = make_loader(train_df, cfg, shuffle=True)
    val_loader = make_loader(val_df, cfg, shuffle=False)

    # Взвешенный лосс — компенсация дисбаланса классов
    counts = train_df["label"].value_counts().reindex([0, 1], fill_value=0).to_numpy(dtype=np.float32)
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1.0), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = CrossAttentionFusion(
        audio_dim=audio_dim, text_dim=text_dim,
        d_model=cfg.d_model, n_heads=cfg.n_heads, dropout=cfg.dropout,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val_f1, epochs_without_improvement = -1.0, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            logits, _ = model(
                batch["audio"].to(device), batch["text"].to(device),
                batch["audio_mask"].to(device), batch["text_mask"].to(device),
            )
            loss = criterion(logits, batch["label"].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(batch["label"])
        scheduler.step()

        val_probs, val_true, _ = predict_probs(model, val_loader, device)
        val_metrics = compute_metrics(val_true, val_probs)
        train_loss /= max(len(train_df), 1)
        print(f"  epoch {epoch}: loss={train_loss:.4f} | "
              + " ".join(f"val_{k}={v:.4f}" for k, v in val_metrics.items()))

        if val_metrics["f1"] > best_val_f1:
            best_val_f1, epochs_without_improvement = val_metrics["f1"], 0
            torch.save({"model_state": model.state_dict(),
                        "audio_dim": audio_dim, "text_dim": text_dim,
                        "epoch": epoch, "val_f1": best_val_f1}, ckpt_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                print(f"  early stopping на эпохе {epoch}")
                break


def train(cfg: Config) -> dict[str, float]:
    set_seed(cfg.seed)
    cfg.ensure_dirs()
    device = cfg.resolve_device()
    print(f"Device: {device}")

    manifest = load_manifest(cfg.data_dir / "manifest.csv")
    if not cfg.use_nt:  # абляция E1: обучение только на RuAphasiaBank
        manifest = manifest[manifest["source"] == "rubank"].reset_index(drop=True)
        print("Абляция E1: собственные записи nt/ исключены из обучения")
    print(f"Записей: {len(manifest)} | "
          f"aphasia={int((manifest['label'] == 1).sum())}, "
          f"control={int((manifest['label'] == 0).sum())} | "
          f"дикторов: {manifest['speaker'].nunique()}")

    manifest = ensure_transcripts(manifest, cfg)
    audio_dim, text_dim = ensure_features(manifest, cfg)

    folds = make_folds(manifest, cfg)
    print(f"Фолдов: {len(folds)}")

    predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, float]] = []
    for fold_idx, (train_df, val_df, test_df) in enumerate(folds):
        print(f"\n=== Фолд {fold_idx}: train={len(train_df)} "
              f"({train_df['speaker'].nunique()} spk), val={len(val_df)} "
              f"({val_df['speaker'].nunique()} spk), test={len(test_df)} "
              f"({test_df['speaker'].nunique()} spk) ===")
        ckpt_path = cfg.checkpoint_dir / f"best_fold{fold_idx}.pt"
        train_fold(train_df, val_df, cfg, device, audio_dim, text_dim, ckpt_path)

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = CrossAttentionFusion(
            audio_dim=int(checkpoint["audio_dim"]), text_dim=int(checkpoint["text_dim"]),
            d_model=cfg.d_model, n_heads=cfg.n_heads, dropout=cfg.dropout,
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])

        probs, y_true, ids = predict_probs(model, make_loader(test_df, cfg, False), device)
        fold_metric = compute_metrics(y_true, probs)
        fold_metrics.append(fold_metric)
        predictions.append(pd.DataFrame({
            "fold": fold_idx, "sample_id": ids, "y_true": y_true, "prob_aphasia": probs,
        }))
        print(f"Фолд {fold_idx} test: "
              + " ".join(f"{k}={v:.4f}" for k, v in fold_metric.items()))

    # Out-of-fold предсказания: каждый диктор RuAphasiaBank протестирован ровно раз
    oof = pd.concat(predictions, ignore_index=True)
    oof.to_csv(cfg.reports_dir / "predictions.csv", index=False)
    oof_metrics = compute_metrics(oof["y_true"].to_numpy(), oof["prob_aphasia"].to_numpy())

    keys = fold_metrics[0].keys()
    summary = {
        "use_nt": cfg.use_nt,
        "n_folds": len(folds),
        "oof_metrics": oof_metrics,
        "folds_mean": {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys},
        "folds_std": {k: float(np.std([m[k] for m in fold_metrics])) for k in keys},
        "folds": fold_metrics,
    }
    (cfg.reports_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== ИТОГ (out-of-fold по всем дикторам RuAphasiaBank) ===")
    for k, v in oof_metrics.items():
        print(f"{k}: {v:.4f}")
    print("=== По фолдам (mean ± std) ===")
    for k in keys:
        print(f"{k}: {summary['folds_mean'][k]:.4f} ± {summary['folds_std'][k]:.4f}")
    return oof_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение классификатора афазии")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(Config.from_yaml(args.config))


if __name__ == "__main__":
    main()
