"""Shared loading and batch-reencoding utilities for extension experiments."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src import dataset as dataset_module
from src.models.unwrapper import PhaseUnwrapper


def load_edge_model(cfg, device):
    path = str(cfg.checkpoint.resume or "")
    if not path or not os.path.isfile(path):
        candidates = sorted(Path("checkpoints").rglob("*.pt")) if Path("checkpoints").is_dir() else []
        listing = "\n".join(f"  {item}" for item in candidates[:30]) or "  (none found under ./checkpoints)"
        raise SystemExit(
            f"Learned edge-head checkpoint not found: {path!r}\n"
            f"Current directory: {os.getcwd()}\n"
            f"Available .pt files:\n{listing}\n"
            "Set checkpoint.resume to the trained edge-head checkpoint used by "
            "scripts.eval_learned_mcf. ext_dl baseline checkpoints are not compatible."
        )
    checkpoint = torch.load(path, map_location=device)
    saved = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    if saved and "model" in saved:
        cfg.model = OmegaConf.merge(cfg.model, OmegaConf.create(saved["model"]))
        if "data" in saved and "input_grad" in saved["data"]:
            cfg.data.input_grad = saved["data"]["input_grad"]
    cfg.model.backbone.in_chans = dataset_module.input_channels(cfg.data)
    model = PhaseUnwrapper(model_cfg=cfg.model, img_size=cfg.data.get("img_size", 256)).to(device)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 5 or len(unexpected) > 5:
        raise SystemExit(f"Checkpoint/model mismatch: missing={len(missing)}, unexpected={len(unexpected)}")
    if getattr(model, "head_type", None) != "edge":
        raise SystemExit("These experiments require an edge-head checkpoint.")
    model.eval(); return model


def probabilities(model, batch):
    """Return edge probabilities without assuming a private head implementation."""
    out = model(batch, hard=False)
    if "probs_h" not in out or "probs_v" not in out:
        raise KeyError("Edge model output must expose probs_h and probs_v.")
    return out["probs_h"], out["probs_v"]
