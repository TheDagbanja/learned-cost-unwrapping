"""Evaluate learned baselines and WLS with the manuscript metric decomposition."""

from __future__ import annotations

import csv
import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src import dataset as dataset_module
from src import mcf, utils
from scripts.eval_learned_mcf import _metrics
from .denet import denet_edge_costs
from .registry import BUILDERS
from .wls import batch_coherence, batch_edge_weights, unwrap_wls


ALL = list(BUILDERS) + ["wls"]


def _load_existing(path, split):
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path, newline="") as handle:
        for item in csv.DictReader(handle):
            if item.get("split") != split:
                continue
            rows[item["name"]] = (split, item["name"], float(item["rmse"]),
                                  100.0*float(item["slip"]), float(item["eps"]),
                                  float(item["residues"]), item["family"])
    return rows


def _write_rows(path, rows):
    ordered = sorted(rows.values(), key=lambda row: row[2])
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split","name","rmse","slip","eps","residues","family"])
        for row in ordered:
            writer.writerow([row[0],row[1],f"{row[2]:.4f}",f"{row[3]/100:.4f}",
                             f"{row[4]:.4f}",f"{row[5]:.1f}",row[6]])


@torch.no_grad()
def _phi_hat_batch(name, model, batch):
    psi = batch["wrapped"][:, 0].detach().cpu().numpy()
    if name == "wls":
        return np.stack([unwrap_wls(psi[b], coherence=batch_coherence(batch, b),
                                    weights=batch_edge_weights(batch, b, psi[b].shape))
                         for b in range(len(psi))])
    if name == "denet":
        p = model(batch)["p_disc"].detach().cpu().numpy(); result = np.empty_like(psi, dtype=np.float64)
        for b in range(len(psi)):
            wx, wy = denet_edge_costs(p[b]); result[b] = mcf.unwrap_mcf(psi[b].astype(np.float64), wx=wx, wy=wy)
        return result
    return model(batch, hard=True)["phi_hat"][:, 0].detach().cpu().numpy()


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    device = utils.get_device(cfg.device); split = cfg.eval.get("split", "test")
    max_batches = cfg.eval.get("max_batches", None); ckpt_dir = cfg.checkpoint.dir
    methods = list(cfg.get("ext", {}).get("methods", ALL)); in_chans = dataset_module.input_channels(cfg.data)
    loader = dataset_module.build_dataloader(cfg.data, split=split, shuffle=False, device=device)
    os.makedirs(ckpt_dir, exist_ok=True)
    out_csv = os.path.join(ckpt_dir, f"dl_leaderboard_{split}.csv")
    rows = _load_existing(out_csv, split)
    for name in methods:
        if name not in ALL:
            raise SystemExit(f"Unknown method {name!r}; choose from {ALL}")
        model = None
        if name != "wls":
            path = os.path.join(ckpt_dir, f"{name}_last.pt")
            if not os.path.exists(path): print(f"[skip] {name}: no checkpoint at {path}"); continue
            model = BUILDERS[name](cfg, in_chans).to(device)
            state = torch.load(path, map_location=device); model.load_state_dict(state.get("model", state)); model.eval()
        acc = []
        for i, batch in enumerate(tqdm(loader, desc=f"[{name}] {split}", dynamic_ncols=True)):
            if max_batches is not None and i >= int(max_batches): break
            batch = utils.move_batch_to_device(batch, device)
            if "absolute" not in batch: raise SystemExit(f"split={split!r} has no ground truth")
            pred = _phi_hat_batch(name, model, batch)
            gt = batch["absolute"][:, 0].detach().cpu().numpy(); psi = batch["wrapped"][:, 0].detach().cpu().numpy()
            acc.extend(_metrics(pred[b], gt[b], psi[b].astype(np.float64)) for b in range(len(gt)))
        if not acc: continue
        mean = np.asarray(acc).mean(0); family = "classical" if name == "wls" else ("hybrid" if name == "denet" else "dl")
        rows[name] = (split, name, mean[0], mean[1]*100, mean[2], mean[3], family)
        _write_rows(out_csv, rows)  # checkpoint evaluation progress method-by-method
        print(f"{name:16s} RMSE={mean[0]:.3f} slip={mean[1]*100:5.1f}% slip-free={mean[2]:.3f} residues={mean[3]:.1f}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
