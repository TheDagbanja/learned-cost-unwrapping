"""Controlled solver/cost ablation for the learned edge model.

Rows isolate: signed learned costs + MCF, unsigned learned costs + MCF,
unit-cost MCF, and local learned edge decisions + least-squares integration.
An optional trained direct predictor is included as a contextual row, not
misrepresented as a same-network solver ablation.
"""

from __future__ import annotations

import csv
import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src import dataset as dataset_module, mcf, mcf_infer, utils
from scripts.ext_experiment_common import load_edge_model, probabilities
from ext_dl.registry import BUILDERS
from ext_dl.wls import integrate_gradients
from src.metrics import decomposition_metrics, summarize_method_rows


def _interior(prob, horizontal: bool):
    return prob[..., :, :-1] if horizontal else prob[..., :-1, :]


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    device = utils.get_device(cfg.device); model = load_edge_model(cfg, device)
    split = cfg.eval.get("split", "test"); max_batches = cfg.eval.get("max_batches", None)
    loader = dataset_module.build_dataloader(cfg.data, split=split, shuffle=False, device=device)
    methods = {name: [] for name in ("signed learned cost + MCF", "unsigned learned cost + MCF",
                                      "unit cost + MCF", "learned local argmax + LS")}

    direct_name = str(cfg.get("ext", {}).get("direct_baseline", ""))
    direct = None
    if direct_name:
        path = os.path.join(cfg.checkpoint.dir, f"{direct_name}_last.pt")
        if os.path.exists(path):
            direct = BUILDERS[direct_name](cfg, dataset_module.input_channels(cfg.data)).to(device)
            state = torch.load(path, map_location=device); direct.load_state_dict(state.get("model", state)); direct.eval()
            methods[f"direct predictor ({direct_name})"] = []
        else: print(f"[skip] optional direct predictor checkpoint not found: {path}")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc=f"solver ablations [{split}]", dynamic_ncols=True)):
            if max_batches is not None and i >= int(max_batches): break
            batch = utils.move_batch_to_device(batch, device)
            signed, unit = mcf_infer.unwrap_batch(model, batch, also_unit=True)
            ph, pv = probabilities(model, batch); ph = ph.detach().cpu().numpy(); pv = pv.detach().cpu().numpy()
            psi = batch["wrapped"][:,0].detach().cpu().numpy(); gt = batch["absolute"][:,0].detach().cpu().numpy()
            direct_pred = None if direct is None else direct(batch, hard=True)["phi_hat"][:,0].detach().cpu().numpy()
            names = list(batch.get("name", [f"{i}_{b}" for b in range(len(psi))]))
            for b in range(len(psi)):
                pcut_x = _interior(ph[b,0] + ph[b,-1], True); pcut_y = _interior(pv[b,0] + pv[b,-1], False)
                wx = -np.log(np.clip(pcut_x, 1e-8, 1.0)); wy = -np.log(np.clip(pcut_y, 1e-8, 1.0))
                unsigned = mcf.unwrap_mcf(psi[b].astype(np.float64), wx=wx, wy=wy)
                kx = _interior(ph[b].argmax(0) - (ph.shape[1]//2), True)
                ky = _interior(pv[b].argmax(0) - (pv.shape[1]//2), False)
                gx = mcf.wrap(psi[b,:,1:] - psi[b,:,:-1]) + mcf.TWO_PI * kx
                gy = mcf.wrap(psi[b,1:,:] - psi[b,:-1,:]) + mcf.TWO_PI * ky
                local = integrate_gradients(gx, gy)
                candidates = {"signed learned cost + MCF": signed[b], "unsigned learned cost + MCF": unsigned,
                              "unit cost + MCF": unit[b], "learned local argmax + LS": local}
                if direct_pred is not None: candidates[f"direct predictor ({direct_name})"] = direct_pred[b]
                wrapped_residues = mcf.residues(psi[b])
                for name, pred in candidates.items():
                    metrics = decomposition_metrics(pred, gt[b], psi[b], wrapped_residues)
                    methods[name].append({
                        "scene_name": names[b], **metrics,
                    })

    out_dir = str(cfg.get("ext", {}).get("out_dir", cfg.checkpoint.dir)); os.makedirs(out_dir, exist_ok=True)
    per_scene_path = os.path.join(out_dir, f"solver_ablations_{split}_per_scene.csv")
    with open(per_scene_path, "w", newline="") as f:
        fields = ["split", "method", "scene_name", "rmse", "slip", "eps", "residues"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for name, rows in methods.items():
            for row in rows:
                writer.writerow({"split": split, "method": name, **{k: row[k] for k in fields[2:]}})

    out_path = os.path.join(out_dir, f"solver_ablations_{split}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f); writer.writerow([
            "split", "method", "mean", "median", "p95", "cvar95", "max",
            "slip", "eps", "residues", "n",
        ])
        for name, values in methods.items():
            if not values: continue
            stats = summarize_method_rows(values)
            writer.writerow([split, name, *[f"{stats[k]:.6f}" for k in
                ("mean", "median", "p95", "cvar95", "max", "slip", "eps", "residues")], len(values)])
            print(f"{name:34s} RMSE={stats['mean']:.3f} P95={stats['p95']:.3f} "
                  f"slip={100*stats['slip']:5.1f}% eps={stats['eps']:.3f}")
    print(f"wrote {out_path}\nwrote {per_scene_path}")


if __name__ == "__main__": main()
