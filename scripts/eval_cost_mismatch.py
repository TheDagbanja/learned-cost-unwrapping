"""Paired, resumable synthetic cost-model-mismatch evaluation.

Each scene reuses the same standardized complex noise and patch locations
across all regimes and severities. Per-scene rows are flushed during the run.
SNAPHU true versus uniform coherence isolates a classical-cost assumption.
This is a synthetic stress test, not measured-coherence evaluation.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src import baselines, dataset as dataset_module, mcf, mcf_infer, utils
from scripts.eval_learned_mcf import _metrics
from scripts.ext_experiment_common import load_edge_model
from scripts.mismatch_protocol import (
    PROTOCOL_VERSION, REGIMES, coherence_field, make_scene_latents, noisy_wrapped,
    snaphu_default_coherence_threshold,
)
from ext_dl.wls import unwrap_wls


METHODS = {
    "learned_mcf": "learned-cost MCF",
    "unit_mcf": "unit-cost MCF",
    "snaphu_true": "SNAPHU true coherence",
    "snaphu_uniform": "SNAPHU uniform coherence",
    "wls_true": "WLS true coherence",
}
SCENE_FIELDS = ["protocol", "split", "scene_index", "regime", "severity", "method",
                "gamma_mean", "gamma_std", "rmse", "slip", "eps", "residues"]
SUMMARY_FIELDS = ["split", "regime", "severity", "method", "rmse", "slip", "eps",
                  "residues", "n", "rmse_sd", "slip_sd", "gamma_mean", "gamma_std"]


def reencode(batch, psi: torch.Tensor, gamma: torch.Tensor, cfg):
    """Rebuild observation-dependent fields; do not reuse original weights."""
    result = dict(batch)
    result["wrapped"] = psi.unsqueeze(1)
    result["input"] = utils.encode_input(psi, include_grad=bool(cfg.data.get("input_grad", True)))
    result["coherence"] = gamma.unsqueeze(1)
    result["weights_x"], result["weights_y"] = utils.coherence_to_edge_weights(gamma, pad=True)
    return result


def _subset_batch(batch, indices):
    count = len(batch["wrapped"])
    result = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == count:
            result[key] = value[indices]
        elif isinstance(value, (list, tuple)) and len(value) == count:
            result[key] = [value[i] for i in indices]
        else:
            result[key] = value
    return result


def _manifest(cfg, checkpoint, split, seed, stride, regimes, severities,
              noise_looks, snaphu_nlooks, snaphu_cost, wls_rtol, wls_maxiter):
    stat = os.stat(checkpoint)
    return {
        "protocol": PROTOCOL_VERSION, "split": split, "seed": seed,
        "scene_stride": stride, "regimes": regimes, "severities": severities,
        "noise_looks": noise_looks, "snaphu_nlooks": snaphu_nlooks,
        "snaphu_default_coherence_threshold": snaphu_default_coherence_threshold(snaphu_nlooks),
        "snaphu_cost": snaphu_cost,
        "wls_rtol": wls_rtol, "wls_maxiter": wls_maxiter,
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_size": stat.st_size, "checkpoint_mtime_ns": stat.st_mtime_ns,
        "data": OmegaConf.to_container(cfg.data, resolve=True),
    }


def _existing_rows(path):
    if not os.path.isfile(path):
        return set()
    with open(path, newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != SCENE_FIELDS:
            raise ValueError(f"Incompatible per-scene CSV: {path}")
        return {(int(row["scene_index"]), row["regime"], float(row["severity"]),
                 row["method"]) for row in reader}


def _write_summary(scene_path, summary_path):
    groups = defaultdict(list)
    if os.path.isfile(scene_path):
        with open(scene_path, newline="") as stream:
            for row in csv.DictReader(stream):
                groups[(row["split"], row["regime"], float(row["severity"]), row["method"])].append(row)
    with open(summary_path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(SUMMARY_FIELDS)
        for (split, regime, severity, method), rows in sorted(groups.items()):
            values = np.asarray([[float(row[key]) for key in
                                  ("rmse", "slip", "eps", "residues", "gamma_mean", "gamma_std")]
                                 for row in rows], dtype=np.float64)
            mean = values.mean(axis=0)
            sd = values.std(axis=0, ddof=1) if len(rows) > 1 else np.zeros(6)
            writer.writerow([split, regime, severity, method,
                             *[f"{v:.6f}" for v in mean[:4]], len(rows),
                             f"{sd[0]:.6f}", f"{sd[1]:.6f}",
                             f"{mean[4]:.6f}", f"{mean[5]:.6f}"])


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    ext = cfg.get("ext", {})
    split = str(cfg.eval.get("split", "test"))
    regimes = list(ext.get("mismatch_regimes", REGIMES))
    severities = [float(v) for v in ext.get("mismatch_severities", [0.25, 0.5, 0.75, 1.0])]
    method_keys = list(ext.get("mismatch_methods", ["learned_mcf", "unit_mcf"]))
    if not method_keys or len(method_keys) != len(set(method_keys)) or set(method_keys) - set(METHODS):
        raise ValueError(f"Invalid mismatch_methods: {method_keys}; allowed={list(METHODS)}")
    if not regimes or any(r not in REGIMES for r in regimes):
        raise ValueError(f"Invalid mismatch_regimes: {regimes}")
    if not severities or any(not 0.0 <= s <= 1.0 for s in severities):
        raise ValueError("mismatch_severities must be nonempty and in [0, 1]")
    stride = int(ext.get("mismatch_scene_stride", 1))
    max_scenes = ext.get("mismatch_max_scenes", None)
    max_scenes = None if max_scenes is None else int(max_scenes)
    noise_looks = int(ext.get("mismatch_looks", 8))
    snaphu_nlooks = float(ext.get("mismatch_snaphu_nlooks", noise_looks))
    snaphu_cost = str(ext.get("mismatch_snaphu_cost", "smooth"))
    wls_rtol = float(ext.get("mismatch_wls_rtol", 1e-4))
    wls_maxiter = int(ext.get("mismatch_wls_maxiter", 300))
    if stride < 1 or (max_scenes is not None and max_scenes < 1) or noise_looks < 1 or snaphu_nlooks < 1:
        raise ValueError("scene stride, max scenes and looks must be positive")
    if any(key.startswith("snaphu_") for key in method_keys):
        threshold = snaphu_default_coherence_threshold(snaphu_nlooks)
        if threshold >= 0.98:
            raise ValueError(
                f"SNAPHU nlooks={snaphu_nlooks:g} gives default coherence cutoff "
                f"{threshold:.3f} >= the experiment's maximum gamma 0.98; "
                "true and uniform coherence become indistinguishable. "
                "Use mismatch_looks=8 and mismatch_snaphu_nlooks=8 for protocol v3."
            )
        if not baselines.snaphu_available():
            raise RuntimeError("SNAPHU requested but not installed. Install `snaphu` or omit those methods.")
        print(f"SNAPHU equivalent coherence looks={snaphu_nlooks:g}; "
              f"default correlation cutoff≈{threshold:.3f}")

    checkpoint = str(cfg.checkpoint.resume or "")
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Learned edge checkpoint not found: {checkpoint!r}")
    seed = int(cfg.get("seed", 0))
    out_dir = str(ext.get("out_dir", cfg.checkpoint.dir))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"cost_mismatch_{split}")
    scene_path, summary_path, manifest_path = stem + "_per_scene.csv", stem + ".csv", stem + "_manifest.json"
    manifest = _manifest(cfg, checkpoint, split, seed, stride, regimes, severities,
                         noise_looks, snaphu_nlooks, snaphu_cost, wls_rtol, wls_maxiter)
    if os.path.isfile(manifest_path):
        with open(manifest_path) as stream:
            old = json.load(stream)
        if old != manifest:
            raise ValueError(f"Run protocol differs from {manifest_path}; use a new ext.out_dir")
    elif os.path.isfile(scene_path):
        raise ValueError(f"Per-scene CSV exists without a manifest: {scene_path}")
    else:
        with open(manifest_path, "w") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
    done = _existing_rows(scene_path)
    if not os.path.isfile(scene_path):
        with open(scene_path, "w", newline="") as stream:
            csv.writer(stream).writerow(SCENE_FIELDS)

    device = utils.get_device(cfg.device)
    model = load_edge_model(cfg, device) if "learned_mcf" in method_keys else None
    loader = dataset_module.build_dataloader(cfg.data, split=split, shuffle=False, device=device)
    max_batches = cfg.eval.get("max_batches", None)
    selected_total, scene_offset, written = 0, 0, 0
    try:
        with open(scene_path, "a", newline="") as stream, torch.no_grad():
            writer = csv.writer(stream)
            for batch_index, batch in enumerate(tqdm(loader, desc=f"cost mismatch [{split}]", dynamic_ncols=True)):
                if max_batches is not None and batch_index >= int(max_batches):
                    break
                batch_size = len(batch["wrapped"])
                indices = [i for i in range(batch_size) if (scene_offset + i) % stride == 0]
                if max_scenes is not None:
                    indices = indices[:max(0, max_scenes - selected_total)]
                global_indices = [scene_offset + i for i in indices]
                scene_offset += batch_size
                if not indices:
                    if max_scenes is not None and selected_total >= max_scenes:
                        break
                    continue
                selected_total += len(indices)
                subset = utils.move_batch_to_device(_subset_batch(batch, indices), device)
                if "absolute" not in subset:
                    raise ValueError("Mismatch evaluation requires ground-truth absolute phase")
                gt = subset["absolute"][:, 0].detach().cpu().numpy()
                latents = [make_scene_latents(x.shape, seed, scene_id)
                           for x, scene_id in zip(gt, global_indices)]
                for regime in regimes:
                    for severity in severities:
                        missing = {(j, key) for j, scene_id in enumerate(global_indices)
                                   for key in method_keys
                                   if (scene_id, regime, severity, METHODS[key]) not in done}
                        if not missing:
                            continue
                        gammas = [coherence_field(x.shape, regime, severity, latent)
                                  for x, latent in zip(gt, latents)]
                        psis = [noisy_wrapped(x, gamma, latent, noise_looks)
                                for x, gamma, latent in zip(gt, gammas, latents)]
                        preds = [dict() for _ in gt]
                        need_learned = any(key == "learned_mcf" for _, key in missing)
                        need_unit = any(key == "unit_mcf" for _, key in missing)
                        if need_learned:
                            psi_t = torch.from_numpy(np.stack(psis)).to(device)
                            gamma_t = torch.from_numpy(np.stack(gammas)).to(device)
                            stress = reencode(subset, psi_t, gamma_t, cfg)
                            if need_unit:
                                learned, unit = mcf_infer.unwrap_batch(model, stress, also_unit=True)
                            else:
                                learned = mcf_infer.unwrap_batch(model, stress, also_unit=False)
                            for j in range(len(gt)):
                                preds[j]["learned_mcf"] = learned[j]
                                if need_unit:
                                    preds[j]["unit_mcf"] = unit[j]
                        elif need_unit:
                            for j, psi in enumerate(psis):
                                if (j, "unit_mcf") in missing:
                                    preds[j]["unit_mcf"] = mcf.unwrap_mcf(psi)
                        for j, (psi, gamma) in enumerate(zip(psis, gammas)):
                            if (j, "snaphu_true") in missing:
                                preds[j]["snaphu_true"] = baselines.snaphu_unwrap(
                                    psi, coherence=gamma, nlooks=snaphu_nlooks, cost=snaphu_cost)
                            if (j, "snaphu_uniform") in missing:
                                if "snaphu_true" in preds[j] and np.ptp(gamma) < 1e-7:
                                    preds[j]["snaphu_uniform"] = preds[j]["snaphu_true"]
                                else:
                                    uniform = np.full_like(gamma, float(gamma.mean()))
                                    preds[j]["snaphu_uniform"] = baselines.snaphu_unwrap(
                                        psi, coherence=uniform, nlooks=snaphu_nlooks, cost=snaphu_cost)
                            if (j, "wls_true") in missing:
                                preds[j]["wls_true"] = unwrap_wls(
                                    psi, coherence=gamma, rtol=wls_rtol, maxiter=wls_maxiter)
                            for key in method_keys:
                                if (j, key) not in missing:
                                    continue
                                rmse, slip, eps, residues = _metrics(preds[j][key], gt[j], psi)
                                writer.writerow([PROTOCOL_VERSION, split, global_indices[j], regime,
                                                 severity, METHODS[key], float(gamma.mean()),
                                                 float(gamma.std()), rmse, slip, eps, residues])
                                done.add((global_indices[j], regime, severity, METHODS[key]))
                                written += 1
                        stream.flush()
    finally:
        _write_summary(scene_path, summary_path)
        print(f"wrote {written} new scene-method rows; selected {selected_total} scenes; "
              f"summary: {summary_path}; per-scene: {scene_path}")


if __name__ == "__main__":
    main()
