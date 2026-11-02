"""Auditable benchmark for learned MCF, DENet, and SNAPHU.

The script makes the information supplied to SNAPHU explicit.  InSAR-DLPU does
not ship coherence for the 1,000-scene simulated split, so its primary rows are
``uniform`` (the historical benchmark) and ``phase_proxy`` (derived only from
the same wrapped phase seen by the network).  A ``supplied`` row is permitted
only for datasets that actually provide coherence.  Every method receives the
same scenes and produces the complete error decomposition, runtime, per-scene
CSV, summary CSV, and a protocol manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from ext_dl.denet import DENetBaseline, denet_edge_costs
from scripts.ext_experiment_common import load_edge_model
from src import baselines, dataset as dataset_module, mcf, utils
from src.metrics import decomposition_metrics, summarize_method_rows


PROTOCOL_VERSION = 3
SCENE_FIELDS = [
    "protocol", "split", "scene_index", "scene_name", "method", "information",
    "snaphu_cost", "nlooks", "rmse", "slip", "eps", "residues", "runtime_s",
]
SUMMARY_FIELDS = [
    "split", "method", "information", "snaphu_cost", "nlooks", "mean", "median",
    "p90", "p95", "p99", "cvar95", "max", "cat_rmse_2pi", "slip",
    "cat_slip_005", "eps", "residues",
    "runtime_s", "runtime_median_s", "runtime_p95_s", "n",
]


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _unsigned_costs(logits6: np.ndarray):
    logits6 = np.asarray(logits6, dtype=np.float64)
    ph = np.exp(logits6[:3] - logits6[:3].max(axis=0, keepdims=True))
    pv = np.exp(logits6[3:] - logits6[3:].max(axis=0, keepdims=True))
    ph /= ph.sum(axis=0, keepdims=True)
    pv /= pv.sum(axis=0, keepdims=True)
    pcut_x = np.clip(ph[0, :, :-1] + ph[2, :, :-1], 1e-8, 1.0)
    pcut_y = np.clip(pv[0, :-1, :] + pv[2, :-1, :], 1e-8, 1.0)
    return -np.log(pcut_x), -np.log(pcut_y)


def _load_denet(cfg: DictConfig, device: torch.device):
    ext = cfg.get("ext", {})
    path = str(ext.get("denet_checkpoint", ""))
    if not path:
        return None, None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"DENet checkpoint not found: {path}")
    model = DENetBaseline().to(device)
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model, path


def _snaphu_specs(ext) -> list[dict]:
    sources = [str(x) for x in ext.get("fair_snaphu_sources", ["uniform", "phase_proxy"])]
    costs = [str(x) for x in ext.get("fair_snaphu_costs", ["defo"])]
    looks = [float(x) for x in ext.get("fair_snaphu_nlooks", [8.0])]
    allowed = {"uniform", "phase_proxy", "supplied"}
    if set(sources) - allowed:
        raise ValueError(f"Unknown SNAPHU coherence source: {sorted(set(sources)-allowed)}")
    return [
        {
            "key": f"snaphu:{source}:{cost}:{nlooks:g}",
            "method": f"SNAPHU ({source}, {cost}, L={nlooks:g})",
            "information": source,
            "cost": cost,
            "nlooks": nlooks,
        }
        for source in sources for cost in costs for nlooks in looks
    ]


def _manifest(cfg, split, methods, snaphu_specs, edge_path, denet_path):
    def identity(path):
        return None if not path else {
            "path": str(Path(path).resolve()), "size": os.path.getsize(path), "sha256": _sha256(path)
        }
    git_error = None
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        git_dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip())
    except (OSError, subprocess.CalledProcessError) as exc:
        git_commit, git_dirty = None, None
        git_error = f"{type(exc).__name__}: {exc}"
    device = utils.get_device(cfg.device)
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    data_config = OmegaConf.to_container(cfg.data, resolve=True)
    augment_config = data_config.get("augment") or {}
    effective_data = {
        "split": split,
        "augmentation_enabled": bool(split == "train" and augment_config.get("enabled", False)),
        "source": data_config.get("source"),
        "limit": data_config.get("limit"),
        "sample_strategy": data_config.get("sample_strategy", "head"),
        "sample_seed": data_config.get("sample_seed", 0),
    }
    return {
        "protocol": PROTOCOL_VERSION,
        "split": split,
        "methods": methods,
        "snaphu": snaphu_specs,
        "uniform_coherence": float(cfg.get("ext", {}).get("fair_uniform_coherence", 1.0)),
        "data": data_config,
        "effective_data": effective_data,
        "edge_checkpoint": identity(edge_path),
        "denet_checkpoint": identity(denet_path),
        "runtime_device": str(device),
        "timing_definition": (
            "amortized wall time per scene at configured batch size; neural rows include "
            "their share of batched inference plus their own MCF solve"
        ),
        "code": {
            "git_commit": git_commit,
            "dirty_worktree": git_dirty,
            "provenance_error": git_error,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "gpu": gpu,
        },
    }


def _read_done(path: str) -> set[tuple[int, str]]:
    if not os.path.isfile(path):
        return set()
    with open(path, newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != SCENE_FIELDS:
            raise ValueError(f"Incompatible per-scene CSV: {path}")
        return {(int(row["scene_index"]), row["method"]) for row in reader}


def _write_summary(scene_path: str, summary_path: str) -> None:
    grouped = defaultdict(list)
    with open(scene_path, newline="") as stream:
        for row in csv.DictReader(stream):
            grouped[row["method"]].append(row)
    with open(summary_path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(SUMMARY_FIELDS)
        for method, rows in sorted(grouped.items()):
            stats = summarize_method_rows(rows)
            first = rows[0]
            writer.writerow([
                first["split"], method, first["information"], first["snaphu_cost"],
                first["nlooks"], *[f"{stats[key]:.8g}" for key in
                ("mean", "median", "p90", "p95", "p99", "cvar95", "max",
                "cat_rmse_2pi", "slip", "cat_slip_005", "eps", "residues",
                "runtime_s", "runtime_median_s", "runtime_p95_s", "n")],
            ])


def _row(split, scene_index, scene_name, method, information, cost, nlooks,
         pred, gt, psi, wrapped_residues, runtime_s):
    values = decomposition_metrics(pred, gt, psi, wrapped_residues)
    return [
        PROTOCOL_VERSION, split, scene_index, scene_name, method, information, cost,
        nlooks, values["rmse"], values["slip"], values["eps"], values["residues"],
        runtime_s,
    ]


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    ext = cfg.get("ext", {})
    methods = [str(x) for x in ext.get(
        "fair_methods", ["learned_signed", "learned_unsigned", "unit_mcf", "snaphu"]
    )]
    allowed = {"learned_signed", "learned_unsigned", "unit_mcf", "denet", "snaphu"}
    if not methods or set(methods) - allowed:
        raise ValueError(f"fair_methods must be drawn from {sorted(allowed)}")
    device = utils.get_device(cfg.device)
    split = str(cfg.eval.get("split", "test"))
    max_batches = cfg.eval.get("max_batches", None)
    specs = _snaphu_specs(ext) if "snaphu" in methods else []
    if specs and not baselines.snaphu_available():
        raise RuntimeError("SNAPHU rows requested but the `snaphu` package is unavailable")

    edge_path = str(cfg.checkpoint.resume or "")
    need_edge = bool({"learned_signed", "learned_unsigned"} & set(methods))
    model = load_edge_model(cfg, device) if need_edge else None
    denet, denet_path = _load_denet(cfg, device) if "denet" in methods else (None, None)
    if "denet" in methods and denet is None:
        raise ValueError("fair_methods includes denet; set +ext.denet_checkpoint=...")

    out_dir = str(ext.get("out_dir", "results/fair_benchmark"))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"fair_benchmark_{split}")
    scene_path, summary_path, manifest_path = stem + "_per_scene.csv", stem + ".csv", stem + "_manifest.json"
    manifest = _manifest(cfg, split, methods, specs, edge_path if need_edge else None, denet_path)
    if os.path.isfile(manifest_path):
        with open(manifest_path) as stream:
            if json.load(stream) != manifest:
                raise ValueError(f"Protocol differs from {manifest_path}; choose a new ext.out_dir")
    elif os.path.isfile(scene_path):
        raise ValueError(f"Per-scene rows exist without manifest: {scene_path}")
    else:
        with open(manifest_path, "w") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
    if not os.path.isfile(scene_path):
        with open(scene_path, "w", newline="") as stream:
            csv.writer(stream).writerow(SCENE_FIELDS)
    done = _read_done(scene_path)

    loader = dataset_module.build_dataloader(cfg.data, split=split, shuffle=False, device=device)
    uniform_value = float(ext.get("fair_uniform_coherence", 1.0))
    scene_offset = 0
    try:
        with open(scene_path, "a", newline="") as stream, torch.no_grad():
            writer = csv.writer(stream)
            for batch_index, batch in enumerate(tqdm(loader, desc=f"fair benchmark [{split}]")):
                if max_batches is not None and batch_index >= int(max_batches):
                    break
                batch = utils.move_batch_to_device(batch, device)
                psi_batch = batch["wrapped"][:, 0].detach().cpu().numpy()
                gt_batch = batch["absolute"][:, 0].detach().cpu().numpy()
                names = list(batch.get("name", [str(scene_offset+i) for i in range(len(psi_batch))]))
                coherence = batch["coherence"][:, 0].detach().cpu().numpy()
                available = batch.get("coherence_available", torch.zeros(len(psi_batch), dtype=torch.bool))
                if isinstance(available, torch.Tensor):
                    available = available.detach().cpu().numpy().astype(bool)

                logits = None
                inference_share = 0.0
                if need_edge:
                    _sync(device); start = time.perf_counter()
                    logits_t, _ = model.encode(batch)
                    _sync(device); inference_share = (time.perf_counter() - start) / len(psi_batch)
                    logits = logits_t.detach().float().cpu().numpy()
                denet_prob = None
                denet_share = 0.0
                if denet is not None:
                    _sync(device); start = time.perf_counter()
                    denet_prob = denet(batch)["p_disc"].detach().float().cpu().numpy()
                    _sync(device); denet_share = (time.perf_counter() - start) / len(psi_batch)

                for index_in_batch, (psi, gt) in enumerate(zip(psi_batch, gt_batch)):
                    scene_index = scene_offset + index_in_batch
                    wrapped_residues = mcf.residues(psi)
                    predictions = []
                    if "learned_signed" in methods:
                        start = time.perf_counter()
                        pred = mcf.unwrap_mcf(psi, edge_costs=mcf.edge_costs_from_logits(logits[index_in_batch]))
                        predictions.append(("learned signed cost + MCF", "wrapped phase", "", "", pred,
                                            inference_share + time.perf_counter() - start))
                    if "learned_unsigned" in methods:
                        wx, wy = _unsigned_costs(logits[index_in_batch])
                        start = time.perf_counter(); pred = mcf.unwrap_mcf(psi, wx=wx, wy=wy)
                        predictions.append(("learned unsigned cost + MCF", "wrapped phase", "", "", pred,
                                            inference_share + time.perf_counter() - start))
                    if "unit_mcf" in methods:
                        start = time.perf_counter(); pred = mcf.unwrap_mcf(psi)
                        predictions.append(("unit-cost MCF", "wrapped phase", "", "", pred,
                                            time.perf_counter() - start))
                    if denet is not None:
                        wx, wy = denet_edge_costs(denet_prob[index_in_batch])
                        start = time.perf_counter(); pred = mcf.unwrap_mcf(psi, wx=wx, wy=wy)
                        predictions.append(("DENet cost + shared MCF", "wrapped phase", "", "", pred,
                                            denet_share + time.perf_counter() - start))
                    for spec in specs:
                        if spec["information"] == "uniform":
                            corr = np.full_like(psi, uniform_value, dtype=np.float32)
                        elif spec["information"] == "phase_proxy":
                            corr = mcf.quality_map(psi).astype(np.float32)
                        else:
                            if not bool(available[index_in_batch]):
                                raise ValueError(
                                    "SNAPHU supplied-coherence requested, but this dataset marks coherence unavailable"
                                )
                            corr = coherence[index_in_batch]
                        start = time.perf_counter()
                        pred = baselines.snaphu_unwrap(
                            psi, coherence=corr, nlooks=spec["nlooks"], cost=spec["cost"]
                        )
                        predictions.append((spec["method"], spec["information"], spec["cost"],
                                            spec["nlooks"], pred, time.perf_counter() - start))

                    for method, information, cost, nlooks, pred, runtime_s in predictions:
                        if (scene_index, method) in done:
                            continue
                        writer.writerow(_row(split, scene_index, names[index_in_batch], method,
                                             information, cost, nlooks, pred, gt, psi,
                                             wrapped_residues, runtime_s))
                        done.add((scene_index, method))
                    stream.flush()
                scene_offset += len(psi_batch)
    finally:
        _write_summary(scene_path, summary_path)
        print(f"summary: {summary_path}\nper-scene: {scene_path}\nmanifest: {manifest_path}")


if __name__ == "__main__":
    main()
