"""Инференс: одна фонограмма -> вероятность афазии.

Аудио-модальность — wav2vec 2.0 (фрагменты, как при обучении),
транскрипт — Whisper (только ASR), текстовые эмбеддинги — RuBERT.

Пример:
    python -m src.inference --wav data/raw/aphasia/patient_001.wav \
        --checkpoint checkpoints/best_fold0.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer

import whisper
from src.config import Config
from src.model import CrossAttentionFusion


@torch.no_grad()
def predict(wav_path: Path, cfg: Config, checkpoint_path: Path) -> dict[str, float | str]:
    """Возвращает метку и вероятность афазии для одной фонограммы."""
    device = cfg.resolve_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = CrossAttentionFusion(
        audio_dim=int(checkpoint["audio_dim"]), text_dim=int(checkpoint["text_dim"]),
        d_model=cfg.d_model, n_heads=cfg.n_heads, dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Аудио-модальность: wav2vec2 по фрагментам (как в features.py)
    extractor = AutoFeatureExtractor.from_pretrained(cfg.audio_model)
    w2v = AutoModel.from_pretrained(cfg.audio_model).to(device).eval()
    wav, _ = librosa.load(str(wav_path), sr=cfg.target_sr, mono=True)
    wav = wav[: cfg.max_audio_sec * cfg.target_sr]
    values = extractor(wav, sampling_rate=cfg.target_sr,
                       return_tensors="pt").input_values[0]
    frag_samples = cfg.fragment_sec * cfg.target_sr
    chunks: list[np.ndarray] = []
    for start in range(0, values.shape[0], frag_samples):
        frag = values[start:start + frag_samples].unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            out = w2v(frag).last_hidden_state
        chunks.append(out.squeeze(0).float().cpu().numpy())
    audio_emb = torch.from_numpy(np.concatenate(chunks, axis=0)).unsqueeze(0).to(device)

    # Текстовая модальность: Whisper только как ASR
    asr = whisper.load_model(cfg.whisper_model, device=str(device))
    text = asr.transcribe(str(wav_path), language=cfg.language,
                          fp16=(device.type == "cuda"))["text"].strip()
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model)
    text_model = AutoModel.from_pretrained(cfg.text_model).to(device).eval()
    tokens = tokenizer(text or "[нет речи]", max_length=cfg.max_text_tokens,
                       truncation=True, return_tensors="pt").to(device)
    text_emb = text_model(**tokens).last_hidden_state.float()  # [1, L, D_t]

    audio_mask = torch.ones(1, audio_emb.shape[1], dtype=torch.bool, device=device)
    text_mask = tokens["attention_mask"].bool()

    logits, _ = model(audio_emb, text_emb, audio_mask, text_mask)
    prob = float(torch.softmax(logits, dim=-1)[0, 1].cpu())
    return {
        "label": "aphasia" if prob >= 0.5 else "control",
        "prob_aphasia": prob,
        "transcript": text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Инференс классификатора афазии")
    parser.add_argument("--wav", required=True, type=Path, help="Путь к фонограмме")
    parser.add_argument("--checkpoint", default=Path("checkpoints/best_fold0.pt"), type=Path)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    result = predict(args.wav, Config.from_yaml(args.config), args.checkpoint)
    print(f"Транскрипт : {result['transcript']}")
    print(f"Предсказание: {result['label']} (P(aphasia) = {result['prob_aphasia']:.3f})")


if __name__ == "__main__":
    main()
