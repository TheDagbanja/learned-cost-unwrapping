"""Dump per-pixel arrays for Fig. 6 (reliability ROC) and Fig. 7 (spatial slip map).

``eval_learned_mcf.py`` and ``ext_dl/eval_dl.py`` collapse every scene to a
scalar (RMSE, slip%, ...). Figs. 6-7 need the underlying PER-PIXEL arrays those
scripts throw away. This script reuses the identical model-loading and unwrap
path (edge head -> logits -> learned MCF costs -> solve; DL baseline -> phi_hat)
and keeps the arrays instead, writing exactly what
``scripts/paper_figures.py --figure roc`` and ``--figure slipmap`` expect:

  <out>/calib.npz  pooled over the first N scenes of eval.split (N=500 by
                   default, matching the manuscript's "over 500 simulated
                   scenes"): entropy, is_error, grad_proxy, err_value.
                   entropy/is_error/err_value come from OUR edge head (the
                   reliability map is a property of our method, not the DL
                   baselines); grad_proxy is the classical Ghiglia/Pritt
                   phase-derivative-variance proxy (mcf.quality_map), inverted
                   so higher = noisier = more error-prone, for the ROC comparison.
  <out>/scene.npz  ONE scene: psi, gt, ours, dl. Chosen automatically as the
                   scanned scene with the largest (DL slip% - ours slip%) gap,
                   so the qualitative figure isn't hand-picked.

Run from the repo root (edge-head checkpoint from src.train; DL checkpoint from
ext_dl.train_dl). New keys need the Hydra ``+`` prefix (same pattern as
``+baseline=``):

    python -m scripts.dump_fig_arrays \\
        checkpoint.resume=checkpoints/edge_last.pt \\
        +dl_checkpoint=checkpoints/dlpu/phasenet2_last.pt \\
        +dl_name=phasenet2 eval.split=test +n_scenes=500 \\
        +out=manuscript/Extension/figs logging.backend=none

Then, locally or on the server:
    python -m scripts.paper_figures --figure roc \\
        --calib manuscript/Extension/figs/calib.npz --out manuscript/Extension/figs
    python -m scripts.paper_figures --figure slipmap \\
        --scene manuscript/Extension/figs/scene.npz --out manuscript/Extension/figs
"""

from __future__ import annotations

import os

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src import dataset as dataset_module
from src import mcf
from src import utils
from src.models.unwrapper import PhaseUnwrapper


def _load_edge_model(cfg, device):
    if not cfg.checkpoint.resume or not os.path.exists(cfg.checkpoint.resume):
        raise SystemExit(f"Need a trained edge-head checkpoint at checkpoint.resume "
                         f"(got {cfg.checkpoint.resume!r}).")
    ckpt = torch.load(cfg.checkpoint.resume, map_location=device)
    saved = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    if saved and "model" in saved:
        cfg.model = OmegaConf.merge(cfg.model, OmegaConf.create(saved["model"]))
        if "data" in saved and "input_grad" in saved["data"]:
            cfg.data.input_grad = saved["data"]["input_grad"]
    cfg.model.backbone.in_chans = dataset_module.input_channels(cfg.data)
    model = PhaseUnwrapper(model_cfg=cfg.model, img_size=cfg.data.get("img_size", 256)).to(device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 5 or len(unexpected) > 5:
        raise SystemExit(f"Checkpoint/arch mismatch (missing={len(missing)}, "
                         f"unexpected={len(unexpected)}). Wrong head_type?")
    if model.head_type != "edge":
        raise SystemExit(f"Needs head_type=edge; checkpoint is {model.head_type!r}.")
    model.eval()
    return model


def _load_dl_model(cfg, name, path, in_chans, device):
    from ext_dl.train_dl import BUILDERS
    if name not in BUILDERS:
        raise SystemExit(f"dl_name={name!r} must be one of {list(BUILDERS)}.")
    if name == "denet":
        raise SystemExit("dl_name=denet has no phi_hat (it predicts MCF costs); "
                         "the slipmap panel needs a per-pixel baseline, e.g. phasenet2 "
                         "(the strongest pure deep net in Table I).")
    model = BUILDERS[name](cfg, in_chans).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def _edge_forward(model, batch):
    """logits (B,6,H,W) -> per-scene (phi_learned, entropy) numpy lists."""
    logits, _ = model.encode(batch)
    logits = logits.detach().float().cpu().numpy()
    psi = batch["wrapped"][:, 0].detach().cpu().numpy()
    out_phi, out_ent = [], []
    for b in range(psi.shape[0]):
        ec = mcf.edge_costs_from_logits(logits[b])
        out_phi.append(mcf.unwrap_mcf(psi[b], edge_costs=ec))
        out_ent.append(mcf.uncertainty_from_logits(logits[b]))
    return out_phi, out_ent


@torch.no_grad()
def _dl_forward(model, batch):
    out = model(batch, hard=True)
    return out["phi_hat"][:, 0].detach().float().cpu().numpy()


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = str(cfg.get("out", "manuscript/Extension/figs"))
    os.makedirs(out_dir, exist_ok=True)
    n_scenes = int(cfg.get("n_scenes", 500))

    edge_model = _load_edge_model(cfg, device)
    print(f"Loaded edge head: {cfg.model.backbone.name}")

    dl_path = cfg.get("dl_checkpoint", None)
    dl_name = str(cfg.get("dl_name", "phasenet2"))
    dl_model = None
    if dl_path:
        in_chans = dataset_module.input_channels(cfg.data)
        dl_model = _load_dl_model(cfg, dl_name, dl_path, in_chans, device)
        print(f"Loaded DL baseline: {dl_name} ({dl_path})")
    else:
        print("No +dl_checkpoint given: scene.npz (slipmap) will be skipped, "
              "calib.npz (roc) is unaffected.")

    split = cfg.eval.get("split", "test")
    loader = dataset_module.build_dataloader(cfg.data, split=split, shuffle=False, device=device)

    ent_all, err_all, proxy_all, errval_all, scene_index_all = [], [], [], [], []
    best = {"score": -1.0}
    n_done = 0
    bar = tqdm(loader, desc=f"dump_fig_arrays [{split}]", dynamic_ncols=True)
    for batch in bar:
        if n_done >= n_scenes:
            break
        batch = utils.move_batch_to_device(batch, device)
        if "absolute" not in batch:
            raise SystemExit(f"split={split!r} has no ground truth; cannot score.")
        psi = batch["wrapped"][:, 0].detach().cpu().numpy()
        gt = batch["absolute"][:, 0].detach().cpu().numpy()

        phi_learned_list, ent_list = _edge_forward(edge_model, batch)
        phi_dl_list = _dl_forward(dl_model, batch) if dl_model is not None else None

        for b in range(gt.shape[0]):
            if n_done >= n_scenes:
                break
            phi_l, ent = phi_learned_list[b], ent_list[b]
            d = phi_l - gt[b]
            d = d - d.mean()
            r = np.round(d / mcf.TWO_PI)
            is_err = (r != 0)
            err_val = d - mcf.TWO_PI * r
            proxy = 1.0 - mcf.quality_map(psi[b])

            ent_all.append(ent.ravel())
            err_all.append(is_err.ravel())
            proxy_all.append(proxy.ravel())
            errval_all.append(err_val.ravel())
            scene_index_all.append(np.full(ent.size, n_done, dtype=np.int32))

            if phi_dl_list is not None:
                phi_dl = phi_dl_list[b]
                ours_slip = float(is_err.mean())
                d_dl = phi_dl - gt[b]
                d_dl = d_dl - d_dl.mean()
                dl_slip = float((np.round(d_dl / mcf.TWO_PI) != 0).mean())
                score = dl_slip - ours_slip
                if score > best["score"]:
                    best = dict(score=score, psi=psi[b].copy(), gt=gt[b].copy(),
                               ours=phi_l.copy(), dl=phi_dl.copy(),
                               ours_slip=ours_slip, dl_slip=dl_slip)
            n_done += 1
        bar.set_postfix(scenes=n_done)
    bar.close()

    calib_path = os.path.join(out_dir, "calib.npz")
    np.savez(calib_path,
             entropy=np.concatenate(ent_all).astype(np.float32),
             is_error=np.concatenate(err_all).astype(bool),
             grad_proxy=np.concatenate(proxy_all).astype(np.float32),
             err_value=np.concatenate(errval_all).astype(np.float32),
             scene_index=np.concatenate(scene_index_all))
    print(f"wrote {calib_path}  (pooled over {n_done} scenes, "
          f"{sum(a.size for a in ent_all):,} pixels)")

    if dl_model is not None and "psi" in best:
        scene_path = os.path.join(out_dir, "scene.npz")
        np.savez(scene_path, psi=best["psi"], gt=best["gt"], ours=best["ours"], dl=best["dl"])
        print(f"wrote {scene_path}  (ours slip={best['ours_slip']*100:.1f}%, "
              f"{dl_name} slip={best['dl_slip']*100:.1f}%, gap={best['score']*100:.1f}pp)")


if __name__ == "__main__":
    main()
