"""Evaluation metrics and entry point (proposal §4.4).

Reports, after reference alignment between prediction and ground truth:

* **RMSE / MAE** in radians.
* **2pi-jump rate** -- fraction of pixels off by at least one whole cycle
  (``round((phi_hat - phi_gt)/2pi) != 0`` after alignment).
* **residues** -- number of Goldstein residues remaining in the predicted
  corrected-gradient field (loop curl ``!= 0``); a perfect set of integer
  ambiguities yields zero. The wrapped-input residue count is reported for
  reference.
* **runtime** -- mean seconds per image.

The same functions are reused by the K-shot routine in :mod:`src.train`.

Standalone use (zero-shot metrics for a checkpoint)::

    python -m src.evaluate checkpoint.resume=checkpoints/pretrain_last.pt \\
                           eval.split=test_real logging.backend=none
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict

import torch
import torch.nn.functional as F

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from . import utils
from . import dataset as dataset_module
from .models.unwrapper import PhaseUnwrapper

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------
@torch.no_grad()
def phase_metrics(
    phi_hat: torch.Tensor, phi_gt: torch.Tensor
) -> Dict[str, float]:
    """RMSE, MAE and 2pi-jump rate after removing the per-sample reference.

    ``phi_hat`` / ``phi_gt`` are ``(B, 1, H, W)`` or ``(B, H, W)``. Both are
    zero-meaned (the integrator's additive constant is unidentifiable), then
    compared. Returned values are averaged over the batch.
    """
    if phi_hat.dim() == 4:
        phi_hat = phi_hat.squeeze(1)
    if phi_gt.dim() == 4:
        phi_gt = phi_gt.squeeze(1)
    pred = utils.zero_mean(phi_hat)
    gt = utils.zero_mean(phi_gt)
    resid = pred - gt
    rmse = torch.sqrt((resid ** 2).mean(dim=(-2, -1)))
    mae = resid.abs().mean(dim=(-2, -1))
    jumps = (torch.round(resid / utils.TWO_PI) != 0).float().mean(dim=(-2, -1))
    return {
        "rmse": rmse.mean().item(),
        "mae": mae.mean().item(),
        "jump_rate": jumps.mean().item(),
    }


@torch.no_grad()
def residue_charge(g_x: torch.Tensor, g_y: torch.Tensor) -> torch.Tensor:
    r"""Goldstein residue charge per 2x2 loop from a gradient field.

    ``g_x``/``g_y`` are full-resolution padded edge maps ``(..., H, W)`` (the
    horizontal/vertical gradients). The loop curl around cell ``(i, j)`` is

        curl = g_x[i,j] + g_y[i,j+1] - g_x[i+1,j] - g_y[i,j]

    and the charge is ``round(curl / 2pi) in {..., -1, 0, +1, ...}``. Returns a
    ``(..., H-1, W-1)`` integer-valued tensor.
    """
    curl = (
        g_x[..., :-1, :-1]
        + g_y[..., :-1, 1:]
        - g_x[..., 1:, :-1]
        - g_y[..., :-1, :-1]
    )
    return torch.round(curl / utils.TWO_PI)


@torch.no_grad()
def count_residues(g_x: torch.Tensor, g_y: torch.Tensor) -> float:
    """Mean number of non-zero residues per image in a gradient field."""
    charge = residue_charge(g_x, g_y)
    per_image = (charge != 0).flatten(start_dim=-2).sum(dim=-1).float()
    return per_image.mean().item()


# ---------------------------------------------------------------------------
# Dataset-level evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_outputs(
    outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """Compute all metrics for a single forward pass (requires ``absolute``)."""
    metrics: Dict[str, float] = {}
    if "absolute" in batch:
        metrics.update(phase_metrics(outputs["phi_hat"], batch["absolute"]))
    metrics["residues_pred"] = count_residues(outputs["g_x"], outputs["g_y"])
    psi = batch["wrapped"].squeeze(1)
    wgx, wgy = utils.wrapped_gradients(psi, pad=True)
    metrics["residues_input"] = count_residues(wgx, wgy)

    # --- Per-pixel wrap-count head: report K accuracy, skip edge-cut metrics ---
    if "k_reg" in outputs:
        if "k_pixel" in batch:
            kp = outputs["k_pred"]
            kg = batch["k_pixel"]
            # offset-invariant: K is defined up to a global integer constant.
            off = torch.round((kg - kp).mean(dim=(-2, -1), keepdim=True))
            metrics["k_accuracy"] = (kp + off == kg).float().mean().item()
        return metrics

    # Applied-cut rate: fraction of edges where a non-zero correction was actually
    # integrated (i.e. after any cut_threshold gating). Derived from the corrected
    # gradients so it reflects what the integrator really used, not raw argmax.
    applied_h = torch.round((outputs["g_x"] - wgx) / utils.TWO_PI)
    applied_v = torch.round((outputs["g_y"] - wgy) / utils.TWO_PI)
    metrics["applied_cut_rate"] = 0.5 * (
        (applied_h[..., :, :-1] != 0).float().mean()
        + (applied_v[..., :-1, :] != 0).float().mean()
    ).item()
    if "k_x" in batch and "k_y" in batch:
        gt_nz = 0.5 * (
            (batch["k_x"][..., :, :-1] != utils.ZERO_AMBIGUITY_CLASS).float().mean()
            + (batch["k_y"][..., :-1, :] != utils.ZERO_AMBIGUITY_CLASS).float().mean()
        )
        metrics["gt_nonzero_rate"] = gt_nz.item()

        # Cut-detection precision/recall/F1 (binary cut vs no-cut on the raw
        # argmax, over valid edges). This isolates "can the model localise cuts"
        # from the integration's RMSE sensitivity.
        z = utils.ZERO_AMBIGUITY_CLASS
        tp = fp = fn = 0.0
        for probs, kc, valid in (
            (outputs["probs_h"], batch["k_x"], batch.get("valid_x")),
            (outputs["probs_v"], batch["k_y"], batch.get("valid_y")),
        ):
            pred = probs.argmax(1) != z
            gt = kc != z
            if valid is not None:
                m = valid.bool()
                pred, gt = pred[m], gt[m]
            tp += float((pred & gt).sum())
            fp += float((pred & ~gt).sum())
            fn += float((~pred & gt).sum())
        metrics["cut_precision"] = tp / (tp + fp + 1e-9)
        metrics["cut_recall"] = tp / (tp + fn + 1e-9)
        metrics["cut_f1"] = 2 * tp / (2 * tp + fp + fn + 1e-9)
    return metrics


@torch.no_grad()
def oracle_outputs(model: PhaseUnwrapper, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Integrate the *ground-truth* ambiguity through the real torch integrator.

    A correctness probe: if the implementation is right, this matches the NumPy
    oracle (~0.87 RMSE on DLPU ``test``). It isolates the integrator/convention
    code from the (separate) question of prediction quality.
    """
    psi = batch["wrapped"].squeeze(1)
    wgx, wgy = utils.wrapped_gradients(psi, pad=True)
    kc_h, kc_v = batch["k_x"], batch["k_y"]               # class indices (B,H,W)
    g_x = wgx + utils.TWO_PI * (kc_h.float() - 1.0)
    g_y = wgy + utils.TWO_PI * (kc_v.float() - 1.0)
    phi = model.integrator(g_x, g_y, batch.get("weights_x"), batch.get("weights_y"))
    probs_h = F.one_hot(kc_h, 3).permute(0, 3, 1, 2).float()
    probs_v = F.one_hot(kc_v, 3).permute(0, 3, 1, 2).float()
    return {"phi_hat": phi.unsqueeze(1), "g_x": g_x, "g_y": g_y,
            "probs_h": probs_h, "probs_v": probs_v}


def _d4_ops():
    """The 8 dihedral (D4) transforms on the last two dims, as (apply, invert)."""
    f, r, t = torch.flip, torch.rot90, (lambda x: torch.transpose(x, -2, -1))
    return [
        (lambda x: x, lambda x: x),
        (lambda x: f(x, [-1]), lambda x: f(x, [-1])),
        (lambda x: f(x, [-2]), lambda x: f(x, [-2])),
        (lambda x: f(x, [-2, -1]), lambda x: f(x, [-2, -1])),
        (lambda x: r(x, 1, (-2, -1)), lambda x: r(x, -1, (-2, -1))),
        (lambda x: r(x, -1, (-2, -1)), lambda x: r(x, 1, (-2, -1))),
        (t, t),
        (lambda x: f(t(x), [-1]), lambda x: t(f(x, [-1]))),
    ]


@torch.no_grad()
def tta_pixel_outputs(model: PhaseUnwrapper, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Test-time augmentation for the pixel heads: average K over the 8 D4 views.

    Each view transforms the wrapped phase, re-encodes the input, predicts the
    (soft) wrap count, and maps it back; the averaged K is rounded once. Reduces
    boundary/regional errors with no retraining. Square inputs only.
    """
    psi = batch["wrapped"]
    psi2 = psi.squeeze(1) if psi.dim() == 4 else psi
    acc, n = None, 0
    for fwd, inv in _d4_ops():
        psi_t = fwd(psi2)
        bt = {"input": utils.encode_input(psi_t, include_grad=True), "wrapped": psi_t.unsqueeze(1)}
        k_t = inv(model(bt, hard=False)["k_reg"].squeeze(1))
        acc = k_t if acc is None else acc + k_t
        n += 1
    k_int = torch.round(acc / n)
    phi_hat = psi2 + utils.TWO_PI * k_int
    g_x = utils.pad_edge_x(utils.edge_diff_x(phi_hat), 0.0)
    g_y = utils.pad_edge_y(utils.edge_diff_y(phi_hat), 0.0)
    return {"phi_hat": phi_hat.unsqueeze(1), "k_reg": (acc / n).unsqueeze(1),
            "k_pred": k_int.unsqueeze(1), "g_x": g_x, "g_y": g_y}


@torch.no_grad()
def evaluate_dataset(
    model: PhaseUnwrapper,
    loader,
    device: torch.device,
    max_batches: int | None = None,
    desc: str = "Evaluate",
    cut_threshold: float = 0.0,
    oracle: bool = False,
    tta: bool = False,
) -> Dict[str, float]:
    """Run the model over ``loader`` and return batch-size-weighted mean metrics.

    With ``oracle=True`` the ground-truth ambiguity is integrated instead of the
    model prediction (a correctness probe; see :func:`oracle_outputs`).
    """
    model.eval()
    sums: Dict[str, float] = {}
    total, runtime = 0, 0.0
    per_image_rmse: list = []
    bar = tqdm(loader, desc=desc, leave=True, dynamic_ncols=True)
    for i, batch in enumerate(bar):
        if max_batches is not None and i >= max_batches:
            break
        batch = utils.move_batch_to_device(batch, device)
        bs = batch["wrapped"].shape[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if oracle:
            outputs = oracle_outputs(model, batch)
        elif tta and getattr(model, "head_type", "edge") in ("pixel", "pixel_cls"):
            outputs = tta_pixel_outputs(model, batch)
        else:
            outputs = model(batch, hard=True, cut_threshold=cut_threshold)
        if device.type == "cuda":
            torch.cuda.synchronize()
        runtime += time.perf_counter() - t0

        m = evaluate_outputs(outputs, batch)
        for k, v in m.items():
            sums[k] = sums.get(k, 0.0) + v * bs
        total += bs
        # Per-image RMSE for a distribution-aware summary (the mean is dominated
        # by a few pathological dense-fringe scenes; median is more honest).
        if "absolute" in batch:
            ph = utils.zero_mean(outputs["phi_hat"].squeeze(1))
            gt = utils.zero_mean(batch["absolute"].squeeze(1))
            per_image_rmse.extend(
                torch.sqrt(((ph - gt) ** 2).mean(dim=(-2, -1))).tolist()
            )
        if "rmse" in m:
            bar.set_postfix(rmse=f"{sums['rmse'] / total:.4f}")

    metrics = {k: v / max(total, 1) for k, v in sums.items()}
    metrics["runtime_s_per_image"] = runtime / max(total, 1)
    if per_image_rmse:
        arr = torch.tensor(per_image_rmse)
        metrics["rmse_median"] = arr.median().item()
        metrics["rmse_p90"] = arr.quantile(0.90).item()
    return metrics


def format_metrics_table(metrics: Dict[str, float], title: str = "Metrics") -> str:
    """Render a metrics dict as a small aligned text table."""
    width = max((len(k) for k in metrics), default=10)
    lines = [f"=== {title} ==="]
    for k, v in metrics.items():
        lines.append(f"  {k:<{width}} : {v:.6f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    if not cfg.logging.get("verbose", False):
        logging.getLogger().setLevel(logging.WARNING)
    utils.set_seed(cfg.seed, deterministic=cfg.deterministic)
    device = utils.get_device(cfg.device)

    ckpt = None
    if cfg.checkpoint.resume:
        if not os.path.exists(cfg.checkpoint.resume):
            raise SystemExit(
                f"Checkpoint not found: {cfg.checkpoint.resume!r}\n"
                f"Pretrain first (python -m src.train regime=pretrain) or point "
                f"checkpoint.resume at an existing .pt file."
            )
        ckpt = torch.load(cfg.checkpoint.resume, map_location=device)
        # Rebuild the architecture the checkpoint was trained with (backbone,
        # head_type, channels, ...) so eval can't silently load into the wrong model.
        saved = ckpt.get("config") if isinstance(ckpt, dict) else None
        if saved and "model" in saved:
            cfg.model = OmegaConf.merge(cfg.model, OmegaConf.create(saved["model"]))
            if "data" in saved and "input_grad" in saved["data"]:
                cfg.data.input_grad = saved["data"]["input_grad"]

    cfg.model.backbone.in_chans = dataset_module.input_channels(cfg.data)
    model = PhaseUnwrapper(model_cfg=cfg.model, img_size=cfg.data.get("img_size", 256)).to(device)
    if ckpt is not None:
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(missing) > 5 or len(unexpected) > 5:
            raise SystemExit(
                f"Checkpoint architecture mismatch ({len(missing)} missing, "
                f"{len(unexpected)} unexpected keys) -- the checkpoint was trained "
                f"with a different model config. The saved config is applied "
                f"automatically; if you still see this, the checkpoint predates "
                f"config-saving: pass the matching settings (e.g. "
                f"model.backbone.name=..., model.decoder.head_type=...)."
            )
        print(f"Loaded checkpoint: {cfg.checkpoint.resume} "
              f"(backbone={cfg.model.backbone.name}, head={cfg.model.decoder.get('head_type', 'edge')})")
    else:
        print("WARNING: no checkpoint.resume given; evaluating an untrained model.")

    split = cfg.eval.get("split", cfg.data.get("split", "test_real"))
    model.set_adaptation_mode("frozen")
    loader = dataset_module.build_dataloader(cfg.data, split=split, shuffle=False, device=device)
    oracle = bool(cfg.eval.get("oracle", False))
    tta = bool(cfg.eval.get("tta", False))
    metrics = evaluate_dataset(
        model, loader, device,
        max_batches=cfg.eval.get("max_batches", None),
        desc=f"{'Oracle' if oracle else 'Evaluate'} [{split}]{' +TTA' if tta else ''}",
        cut_threshold=float(cfg.eval.get("cut_threshold", 0.0)),
        oracle=oracle,
        tta=tta,
    )
    title = f"{'ORACLE (GT k)' if oracle else 'Zero-shot'} metrics on '{split}'{' (TTA)' if tta else ''}"
    print(format_metrics_table(metrics, title=title))


if __name__ == "__main__":
    main()
