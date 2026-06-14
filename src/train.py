"""Training / adaptation entry point (Hydra) for the InSAR phase unwrapper.

Dispatches on ``cfg.regime``:

* ``pretrain``  -- supervised pretraining on InSAR-DLPU with
  ``L = L_CE + L_cong + 0.05 L_sparse`` (proposal Eq. 9).
* ``zero_shot`` -- frozen forward pass over a target split; logs the physics
  consistency (congruence) and an aligned RMSE diagnostic.
* ``tta``       -- unsupervised per-scene test-time adaptation updating **only**
  the LoRA factors with ``L = L_cong + 0.05 L_sparse + lambda3 L_ent``
  (Eq. 10), logging the adaptation-delta curve (congruence vs step).

K-shot prototypical refinement lands in Phase 4.

Run remotely via ``./run_train.sh`` or directly::

    python -m src.train regime=pretrain data.batch_size=16
    python -m src.train regime=tta checkpoint.resume=checkpoints/pretrain.pt \\
                        data.split=test_real data.batch_size=1
"""

from __future__ import annotations

import itertools
import logging
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

from . import utils
from . import dataset as dataset_module
from . import augment as augment_module
from . import mcf_infer
from .losses import (
    PhysicsGuidedLoss,
    compute_class_weights,
    congruence_loss,
    cross_entropy_loss,
    sparsity_loss,
)
from .models.unwrapper import PhaseUnwrapper, compute_prototypes, prototypical_logits
from .evaluate import evaluate_dataset, evaluate_outputs, format_metrics_table

log = logging.getLogger(__name__)


# ===========================================================================
# Logging backends (wandb / tensorboard / none)
# ===========================================================================
class Logger:
    """Thin wrapper over Weights & Biases or TensorBoard (or a no-op)."""

    def __init__(self, cfg: DictConfig) -> None:
        self.backend = str(cfg.logging.backend).lower()
        self.writer = None
        self._wandb = None
        if self.backend == "wandb":
            try:
                import wandb

                wandb.init(
                    project=cfg.logging.project,
                    name=cfg.logging.run_name,
                    config=OmegaConf.to_container(cfg, resolve=True),
                    dir=cfg.logging.log_dir,
                )
                self._wandb = wandb
            except Exception as exc:  # pragma: no cover - environment dependent
                log.warning("wandb unavailable (%s); disabling logging.", exc)
                self.backend = "none"
        elif self.backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=cfg.logging.log_dir)

    def log(self, metrics: Dict[str, float], step: int) -> None:
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)
        elif self.writer is not None:
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, step)

    def log_image(self, tag: str, image: np.ndarray, step: int) -> None:
        """``image`` is HWC uint8."""
        if self._wandb is not None:
            self._wandb.log({tag: self._wandb.Image(image)}, step=step)
        elif self.writer is not None:
            self.writer.add_image(tag, image, step, dataformats="HWC")

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
        elif self.writer is not None:
            self.writer.close()


# ===========================================================================
# Visualization
# ===========================================================================
def make_phase_figure(
    psi: torch.Tensor, phi_hat: torch.Tensor, phi_gt: Optional[torch.Tensor] = None
) -> np.ndarray:
    """Render wrapped / predicted (/ ground-truth + error) phase as an HWC array."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    psi = psi.detach().float().cpu().numpy()
    phi_hat = phi_hat.detach().float().cpu().numpy()
    panels = [("wrapped psi", psi, "twilight"), ("pred phi_hat", phi_hat, "viridis")]
    if phi_gt is not None:
        gt = phi_gt.detach().float().cpu().numpy()
        gt = gt - gt.mean()
        pred = phi_hat - phi_hat.mean()
        panels.append(("gt phi", gt, "viridis"))
        panels.append(("|pred - gt|", np.abs(pred - gt), "magma"))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, data, cmap) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img = buf[..., :3].copy()
    plt.close(fig)
    return img


# ===========================================================================
# Checkpoints
# ===========================================================================
def save_checkpoint(path: str, model: torch.nn.Module, optimizer, epoch: int, cfg: DictConfig) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )
    log.info("Saved checkpoint to %s", path)


def load_checkpoint(path: str, model: torch.nn.Module, map_location="cpu") -> Dict[str, Any]:
    if not os.path.exists(path):
        raise SystemExit(
            f"Checkpoint not found: {path!r}\n"
            f"Pretrain first (python -m src.train regime=pretrain) or point "
            f"checkpoint.resume at an existing .pt file."
        )
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log.warning("Missing keys when loading checkpoint: %d", len(missing))
    if unexpected:
        log.warning("Unexpected keys when loading checkpoint: %d", len(unexpected))
    log.info("Loaded checkpoint from %s", path)
    return ckpt if isinstance(ckpt, dict) else {}


# ===========================================================================
# Small diagnostics (full metrics live in evaluate.py, Phase 4)
# ===========================================================================
@torch.no_grad()
def aligned_rmse(phi_hat: torch.Tensor, phi_gt: torch.Tensor) -> float:
    """RMSE after removing the per-sample mean offset (reference alignment)."""
    p = utils.zero_mean(phi_hat.squeeze(1))
    g = utils.zero_mean(phi_gt.squeeze(1))
    return torch.sqrt(((p - g) ** 2).mean()).item()


def _build_optimizer(cfg: DictConfig, params: List[torch.nn.Parameter], lr: float):
    name = str(cfg.train.optimizer).lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=cfg.train.weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=cfg.train.weight_decay)
    raise ValueError(f"Unknown optimizer {name!r}.")


def _format_summary(avg: Dict[str, float]) -> str:
    """Format averaged loss terms as a compact one-line epoch summary."""
    order = ("total", "kce", "kreg", "ktv", "ce", "curl", "cong", "sparse", "ent")
    keys = [k for k in order if k in avg] + [k for k in avg if k not in order]
    return "  ".join(f"{k}={avg[k]:.4f}" for k in keys)


# ===========================================================================
# Regime: supervised pretraining
# ===========================================================================
def run_pretrain(cfg: DictConfig, model: PhaseUnwrapper, device: torch.device, logger: Logger) -> None:
    model.set_adaptation_mode("pretrain")

    # Curriculum (synthetic only): ramp data difficulty from easy -> target.
    source = str(cfg.data.get("source", "dlpu"))
    target_diff = float((cfg.data.get("synthetic", {}) or {}).get("difficulty", 0.5)) if source == "synthetic" else 1.0
    curriculum = bool(cfg.train.get("curriculum", False)) and source == "synthetic"
    cur_start = float(cfg.train.get("curriculum_start", 0.25))
    cur_epochs = int(cfg.train.get("curriculum_epochs", 20))

    def _difficulty_for(epoch: int) -> Optional[float]:
        if not curriculum:
            return None
        frac = min(1.0, epoch / max(1, cur_epochs))
        return cur_start + (target_diff - cur_start) * frac

    loader = dataset_module.build_dataloader(
        cfg.data, split="train", device=device, difficulty=_difficulty_for(0)
    )
    loss_fn = PhysicsGuidedLoss(
        lambda_cong=cfg.train.lambda_cong,
        lambda_sparse=cfg.train.lambda_sparse,
        lambda_curl=cfg.train.get("lambda_curl", 0.0),
        lambda_ktv=cfg.train.get("lambda_ktv", 0.01),
        lambda_kreg=cfg.train.get("lambda_kreg", 1.0),
        k_max=int(cfg.model.decoder.get("k_max", 16)),
        ce_class_weights=cfg.train.get("ce_class_weights", "inverse_sqrt"),
        cut_loss=cfg.train.get("cut_loss", "ce"),
        focal_gamma=cfg.train.get("focal_gamma", 2.0),
    )
    params = list(model.trainable_parameters())
    optimizer = _build_optimizer(cfg, params, cfg.train.lr)
    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    try:  # torch>=2.3 API; fall back for older versions
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # LR schedule: linear warmup then cosine decay (stabilises full fine-tuning).
    epochs = cfg.train.epochs
    steps_per_epoch = len(loader)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = int(cfg.train.get("warmup_epochs", 1) * steps_per_epoch)

    def _lr_factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_factor)

    counts = model.count_parameters()
    print(
        f"Trainable {counts['trainable']:,}/{counts['total']:,} params "
        f"({100 * counts['trainable'] / counts['total']:.3f}%); LoRA {counts['lora']:,}"
    )
    straight_through = bool(cfg.train.get("straight_through", True))

    # Optional per-epoch validation so the unwrapping RMSE can be watched live.
    val_every = int(cfg.train.get("val_every", 0))
    val_loader = None
    if val_every > 0:
        val_loader = dataset_module.build_dataloader(
            cfg.data, split=cfg.train.get("val_split", "test"), shuffle=False, device=device
        )
        val_batches = cfg.train.get("val_batches", 20)

    step = 0
    for epoch in range(epochs):
        model.train()
        if curriculum and epoch > 0:                      # rebuild with ramped difficulty
            loader = dataset_module.build_dataloader(
                cfg.data, split="train", device=device, difficulty=_difficulty_for(epoch)
            )
        sums: Dict[str, float] = {}
        nb = 0
        diff_tag = f" diff={_difficulty_for(epoch):.2f}" if curriculum else ""
        bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}{diff_tag}", leave=True, dynamic_ncols=True)
        for batch in bar:
            batch = utils.move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(batch, straight_through=straight_through)
                terms = loss_fn(outputs, batch, regime="pretrain")
                loss = terms["total"]
            scaler.scale(loss).backward()
            if cfg.train.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # Only advance the LR schedule when the optimizer actually stepped
            # (AMP skips the step + lowers the scale when grads are inf/NaN).
            if scaler.get_scale() >= scale_before:
                scheduler.step()

            for k, v in terms.items():
                sums[k] = sums.get(k, 0.0) + v.item()
            nb += 1
            bar.set_postfix(loss=f"{loss.item():.4f}")

            if step % cfg.logging.log_every == 0:
                logger.log({f"train/{k}": v.item() for k, v in terms.items()}, step)
            if cfg.logging.vis_every and step % cfg.logging.vis_every == 0:
                img = make_phase_figure(
                    batch["wrapped"][0, 0], outputs["phi_hat"][0, 0],
                    batch["absolute"][0, 0] if "absolute" in batch else None,
                )
                logger.log_image("train/sample", img, step)
            step += 1
        bar.close()

        avg = {k: sums[k] / max(nb, 1) for k in sums}
        # Always refresh a rolling 'last' checkpoint so an interrupted run is usable.
        save_checkpoint(os.path.join(cfg.checkpoint.dir, "pretrain_last.pt"), model, optimizer, epoch, cfg)
        saved = ""
        if cfg.checkpoint.save_every and (epoch + 1) % cfg.checkpoint.save_every == 0:
            save_checkpoint(
                os.path.join(cfg.checkpoint.dir, f"pretrain_epoch{epoch + 1}.pt"),
                model, optimizer, epoch, cfg,
            )
            saved = "  [ckpt saved]"
        print(f"Epoch {epoch + 1}/{epochs} summary | {_format_summary(avg)}{saved}")
        logger.log({**{f"epoch/{k}": v for k, v in avg.items()}, "epoch": epoch + 1}, step)

        if val_loader is not None and (epoch + 1) % val_every == 0:
            val = evaluate_dataset(
                model, val_loader, device, max_batches=val_batches,
                desc=f"  val (epoch {epoch + 1})",
                cut_threshold=float(cfg.eval.get("cut_threshold", 0.0)),
            )
            if "k_accuracy" in val:                       # pixel head
                print(
                    f"  val | rmse={val.get('rmse', float('nan')):.4f} "
                    f"mae={val.get('mae', float('nan')):.4f} "
                    f"jump={val.get('jump_rate', float('nan')):.4f} "
                    f"residues={val.get('residues_pred', float('nan')):.0f}"
                    f"(in {val.get('residues_input', float('nan')):.0f}) "
                    f"K_acc={val.get('k_accuracy', float('nan')):.4f}"
                )
            else:                                         # edge head
                print(
                    f"  val | rmse={val.get('rmse', float('nan')):.4f} "
                    f"residues={val.get('residues_pred', float('nan')):.0f}"
                    f"(in {val.get('residues_input', float('nan')):.0f}) "
                    f"cut[P={val.get('cut_precision', float('nan')):.2f} "
                    f"R={val.get('cut_recall', float('nan')):.2f} "
                    f"F1={val.get('cut_f1', float('nan')):.2f}] "
                    f"applied={val.get('applied_cut_rate', float('nan')):.4f}"
                    f"(gt {val.get('gt_nonzero_rate', float('nan')):.4f})"
                )
            logger.log({f"val/{k}": v for k, v in val.items()}, step)
            model.train()

    last_path = os.path.join(cfg.checkpoint.dir, "pretrain_last.pt")
    save_checkpoint(last_path, model, optimizer, epochs, cfg)
    print(f"Training complete. Final checkpoint: {last_path}")


# ===========================================================================
# Regime: zero-shot evaluation
# ===========================================================================
@torch.no_grad()
def run_zero_shot(cfg: DictConfig, model: PhaseUnwrapper, device: torch.device, logger: Logger) -> None:
    model.set_adaptation_mode("frozen")
    model.eval()
    split = cfg.data.get("split", "test_real")
    loader = dataset_module.build_dataloader(cfg.data, split=split, shuffle=False, device=device)
    loss_fn = PhysicsGuidedLoss()

    cong_sum, rmse_sum, n = 0.0, 0.0, 0
    max_batches = cfg.get("eval", {}).get("max_batches", None)
    bar = tqdm(loader, desc=f"Zero-shot [{split}]", leave=True, dynamic_ncols=True)
    for i, batch in enumerate(bar):
        if max_batches is not None and i >= max_batches:
            break
        batch = utils.move_batch_to_device(batch, device)
        outputs = model(batch, hard=True)
        terms = loss_fn(outputs, batch, regime="zero_shot")
        cong_sum += terms["cong"].item()
        n += 1
        if "absolute" in batch:
            rmse_sum += aligned_rmse(outputs["phi_hat"], batch["absolute"])
        bar.set_postfix(cong=f"{cong_sum / n:.4f}")
        if i == 0:
            img = make_phase_figure(
                batch["wrapped"][0, 0], outputs["phi_hat"][0, 0],
                batch["absolute"][0, 0] if "absolute" in batch else None,
            )
            logger.log_image("zero_shot/sample", img, 0)
    bar.close()

    metrics = {"zero_shot/congruence": cong_sum / max(n, 1)}
    summary = f"congruence={metrics['zero_shot/congruence']:.4f}"
    if rmse_sum > 0:
        metrics["zero_shot/aligned_rmse"] = rmse_sum / max(n, 1)
        summary += f"  aligned_rmse={metrics['zero_shot/aligned_rmse']:.4f}"
    logger.log(metrics, 0)
    print(f"Zero-shot [{split}] summary | {summary}  (n={n})")


# ===========================================================================
# Regime: unsupervised test-time adaptation
# ===========================================================================
def _snapshot_lora(model: PhaseUnwrapper) -> List[torch.Tensor]:
    return [p.detach().clone() for p in model.lora_parameters()]


@torch.no_grad()
def _restore_lora(model: PhaseUnwrapper, snapshot: List[torch.Tensor]) -> None:
    for p, s in zip(model.lora_parameters(), snapshot):
        p.copy_(s)


def run_tta(cfg: DictConfig, model: PhaseUnwrapper, device: torch.device, logger: Logger) -> None:
    """Per-scene unsupervised adaptation of the LoRA factors (proposal §3.6).

    Each batch (ideally one scene; set ``data.batch_size=1``) is adapted
    independently; LoRA is reset to the pretrained snapshot before each scene so
    measurements are comparable. The unsupervised objective is head-specific:

    * ``edge`` head  -> congruence + sparsity + entropy (Eq. 10).
    * ``pixel_cls``  -> prediction entropy (TENT-style) + smoothness.

    Entropy alone is near-useless under a large shift (the model is *confidently
    wrong*, so the entropy gradient ~vanishes). The dominant signal here is
    **augmentation consistency**: ``phi_hat`` must be invariant to a D4 symmetry
    of the input and to added wrapped-phase noise (both reconstruct the same
    absolute field up to a constant). That has gradient exactly where the model
    is unstable -- i.e. wrong -- so it adapts even when entropy is flat. It is
    head-agnostic (both heads emit ``phi_hat``) and controlled by
    ``tta.lambda_consistency`` / ``tta.aug_noise_std`` (0 disables).

    The reported adaptation delta is the change in unwrapping RMSE (when ground
    truth is available), i.e. what adapting actually buys.
    """
    model.set_adaptation_mode("tta")
    model.eval()  # freeze BatchNorm stats; only LoRA adapts
    split = cfg.data.get("split", "test_real")
    loader = dataset_module.build_dataloader(cfg.data, split=split, shuffle=False, device=device)
    loss_fn = PhysicsGuidedLoss(
        lambda_cong=1.0, lambda_sparse=cfg.tta.lambda_sparse, lambda_ent=cfg.tta.lambda_ent,
        lambda_ktv=cfg.train.get("lambda_ktv", 0.01), k_max=int(cfg.model.decoder.get("k_max", 16)),
    )
    base_snapshot = _snapshot_lora(model)

    # Augmentation-consistency settings (the real unsupervised signal here).
    lambda_cons = float(cfg.tta.get("lambda_consistency", 1.0))
    aug_noise_std = float(cfg.tta.get("aug_noise_std", 0.3))
    use_cons = lambda_cons > 0.0
    cons_rng = np.random.default_rng(int(cfg.seed))

    rmse_before: List[float] = []
    rmse_after: List[float] = []
    bar = tqdm(loader, desc=f"TTA [{split}]", leave=True, dynamic_ncols=True)
    for scene_idx, batch in enumerate(bar):
        batch = utils.move_batch_to_device(batch, device)
        _restore_lora(model, base_snapshot)

        with torch.no_grad():                              # metrics before adaptation
            m0 = evaluate_outputs(model(batch, hard=True), batch)

        optimizer = _build_optimizer(cfg, list(model.lora_parameters()), cfg.tta.lr)
        for s in range(cfg.tta.steps):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch, straight_through=True)  # ignored by pixel heads
            terms = loss_fn(outputs, batch, regime="tta")
            total = terms["total"]
            if use_cons:
                # phi_hat must be invariant to a D4 symmetry + added phase noise.
                op = augment_module.sample_tta_ops(batch["wrapped"].shape, cons_rng)[0]
                aug = augment_module.augment_batch_view(batch, op, noise_std=aug_noise_std)
                out_s = model(aug, straight_through=True)
                cons = augment_module.consistency_loss(
                    out_s["phi_hat"], outputs["phi_hat"].detach(), op
                )
                total = total + lambda_cons * cons
            total.backward()
            optimizer.step()

        with torch.no_grad():                              # metrics after adaptation
            out1 = model(batch, hard=True)
            m1 = evaluate_outputs(out1, batch)

        if "rmse" in m0 and "rmse" in m1:
            rmse_before.append(m0["rmse"])
            rmse_after.append(m1["rmse"])
            bar.set_postfix(rmse=f"{m0['rmse']:.3f}->{m1['rmse']:.3f}",
                            mean_after=f"{float(np.mean(rmse_after)):.3f}")
            logger.log({f"tta/scene{scene_idx}/rmse_before": m0["rmse"],
                        f"tta/scene{scene_idx}/rmse_after": m1["rmse"]}, scene_idx)
        if scene_idx == 0:
            img = make_phase_figure(
                batch["wrapped"][0, 0], out1["phi_hat"][0, 0],
                batch["absolute"][0, 0] if "absolute" in batch else None,
            )
            logger.log_image("tta/scene0_adapted", img, 0)
    bar.close()

    if rmse_before:
        b, a = float(np.mean(rmse_before)), float(np.mean(rmse_after))
        logger.log({"tta/rmse_before": b, "tta/rmse_after": a, "tta/rmse_delta": b - a}, 0)
        print(f"TTA [{split}] summary | rmse {b:.4f} -> {a:.4f}  "
              f"(delta {b - a:+.4f}) over {len(rmse_before)} scenes")
    else:
        print(f"TTA [{split}] complete ({scene_idx + 1} scenes; no ground truth for RMSE).")


# ===========================================================================
# Regime: K-shot prototypical refinement
# ===========================================================================
def _run_kshot_pixel(cfg: DictConfig, model: PhaseUnwrapper, device: torch.device, logger: Logger) -> None:
    """K-shot for the pixel(_cls) head: supervised LoRA fine-tune on K patches.

    The prototypical mechanism is edge-head-specific; for a wrap-count head the
    natural K-shot is to fine-tune the LoRA factors on the few labelled support
    patches (the proposal's 'LoRA-K-shot' variant), then report the query metrics
    before vs after adaptation.
    """
    from torch.utils.data import DataLoader, Subset

    split = cfg.data.get("split", "test_real")
    full = dataset_module.build_dataset(cfg.data, split)
    K = int(cfg.kshot.k)
    n = len(full)
    if n <= K:
        raise ValueError(f"Need more than K={K} samples in split {split!r} (have {n}).")
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(cfg.seed)).tolist()
    support_idx, query_idx = perm[:K], perm[K:]
    bs = int(cfg.data.batch_size)
    # Materialise the support as mini-batches (<= bs each) so the fit memory is
    # bounded by bs, not K. At k_max=32 the (K, 2*k_max+1, H, W) logits would OOM
    # for K>~20 if forwarded all at once.
    support_batches = [
        utils.move_batch_to_device(b, device)
        for b in DataLoader(Subset(full, support_idx), batch_size=bs, shuffle=False)
    ]
    query_loader = DataLoader(
        Subset(full, query_idx), batch_size=bs, shuffle=False,
        num_workers=int(cfg.data.num_workers),
    )

    model.set_adaptation_mode("kshot")     # LoRA-only
    model.eval()                            # BN frozen
    loss_fn = PhysicsGuidedLoss(
        lambda_kreg=cfg.train.get("lambda_kreg", 1.0),
        lambda_ktv=cfg.train.get("lambda_ktv", 0.01),
        k_max=int(cfg.model.decoder.get("k_max", 16)),
    )

    m_before = evaluate_dataset(model, query_loader, device, desc=f"K-shot pre [{split}]")
    optimizer = torch.optim.AdamW(list(model.lora_parameters()), lr=cfg.kshot.lr)
    bar = tqdm(range(int(cfg.kshot.steps)), desc=f"K-shot (K={K}) fit", leave=True, dynamic_ncols=True)
    nb = len(support_batches)
    for s in bar:
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for sb in support_batches:                 # grad-accumulate a full support pass
            terms = loss_fn(model(sb), sb, regime="pretrain")
            (terms["total"] / nb).backward()
            running += float(terms["total"].item())
        optimizer.step()
        bar.set_postfix(loss=f"{running / nb:.4f}")
    m_after = evaluate_dataset(model, query_loader, device, desc=f"K-shot post [{split}]")

    logger.log({f"kshot/before_{k}": v for k, v in m_before.items()}, 0)
    logger.log({f"kshot/after_{k}": v for k, v in m_after.items()}, 1)
    print(f"K-shot (K={K}) [{split}] | rmse {m_before.get('rmse', float('nan')):.4f} -> "
          f"{m_after.get('rmse', float('nan')):.4f}   "
          f"K_acc {m_before.get('k_accuracy', float('nan')):.4f} -> "
          f"{m_after.get('k_accuracy', float('nan')):.4f}")


def _run_kshot_edge_supervised(cfg: DictConfig, model: PhaseUnwrapper,
                               device: torch.device, logger: Logger) -> None:
    """Supervised LoRA cost-adaptation for the edge head (the MCF cost network).

    Fine-tunes ONLY the LoRA factors on K labelled support patches (per-edge CE),
    then SAVES the adapted checkpoint so it can be scored with the MCF solver
    (scripts/eval_learned_mcf.py). This is the 'adapt the costs' lever: the edge
    head's per-edge probabilities are the costs, so improving them on a few real
    patches should lower the learned-cost MCF RMSE on the rest of the domain.

    Clean-curve protocol: a FIXED held-out query (identical scenes for every K,
    fixed by ``seed``) is scored with MCF *before and after* adapting on K support
    patches drawn from a DISJOINT pool. Every point on the K-curve is therefore on
    the same scenes, and the K=0 (before) baseline is measured on that same query
    -- no leakage, no per-K subset drift. ``kshot.n_query`` sets the held-out size.
    """
    from torch.utils.data import DataLoader, Subset

    split = cfg.data.get("split", "test_real")
    full = dataset_module.build_dataset(cfg.data, split)
    K = int(cfg.kshot.k)
    n = len(full)
    n_query = int(cfg.kshot.get("n_query", 0)) or max(n // 2, 1)
    if n_query + K > n:
        raise ValueError(
            f"n_query({n_query}) + K({K}) > n({n}); lower kshot.n_query or kshot.k."
        )
    bs = int(cfg.data.batch_size)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(cfg.seed)).tolist()
    query_idx = perm[:n_query]                       # FIXED across K (same seed)
    support_idx = perm[n_query:n_query + K]          # disjoint support pool
    support_batches = [
        utils.move_batch_to_device(b, device)
        for b in DataLoader(Subset(full, support_idx), batch_size=bs, shuffle=False)
    ]
    query_loader = DataLoader(
        Subset(full, query_idx), batch_size=bs, shuffle=False,
        num_workers=int(cfg.data.num_workers),
    )

    model.set_adaptation_mode("kshot")     # LoRA-only
    model.eval()                            # BN frozen

    # K=0 baseline: zero-shot learned-cost MCF on the fixed query (before adapting).
    print(f"  zero-shot (K=0) on fixed held-out query (n={n_query})...")
    m_before = mcf_infer.evaluate_learned_cost(model, query_loader, device)

    s0 = support_batches[0]
    ce_weight = compute_class_weights(
        s0["k_x"], s0["k_y"], s0.get("valid_x"), s0.get("valid_y"),
        mode=cfg.train.get("ce_class_weights", "inverse_sqrt"),
    )
    optimizer = torch.optim.AdamW(list(model.lora_parameters()), lr=cfg.kshot.lr)
    nb = len(support_batches)
    bar = tqdm(range(int(cfg.kshot.steps)), desc=f"Edge K-shot (K={K}) fit", leave=True, dynamic_ncols=True)
    for _ in bar:
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for sb in support_batches:
            out = model(sb)
            loss = cross_entropy_loss(
                out["logits_h"], out["logits_v"], sb["k_x"], sb["k_y"],
                sb.get("valid_x"), sb.get("valid_y"), weight=ce_weight,
            )
            (loss / nb).backward()
            running += float(loss.item())
        optimizer.step()
        bar.set_postfix(ce=f"{running / nb:.4f}")

    # Score the SAME fixed query after adaptation.
    print(f"  scoring fixed held-out query after adaptation...")
    m_after = mcf_infer.evaluate_learned_cost(model, query_loader, device)

    out_path = os.path.join(cfg.checkpoint.dir, f"edge_kshot_k{K}.pt")
    save_checkpoint(out_path, model, optimizer, 0, cfg)
    logger.log({"kshot/learned_before": m_before["rmse_learned"],
                "kshot/learned_after": m_after["rmse_learned"], "kshot/k": K}, K)
    print(f"Edge K-shot (K={K}) [{split}] | fixed query n={n_query}, residue-free")
    print(f"  learned-cost MCF  {m_before['rmse_learned']:.3f} (K=0) -> "
          f"{m_after['rmse_learned']:.3f} (K={K})   "
          f"[unit-MCF {m_after['rmse_unit']:.3f}, floor {m_after['rmse_floor']:.3f}]")
    print(f"  adapted checkpoint: {out_path}")


def run_kshot(cfg: DictConfig, model: PhaseUnwrapper, device: torch.device, logger: Logger) -> None:
    """K-shot adaptation. Dispatches by head: supervised LoRA fine-tune for the
    pixel(_cls) head; prototypical refinement (below) for the edge head (§3.6).

    A few (``K in {1, 5}``) labelled support patches from the target domain are
    used to form per-direction class prototypes in the decoder's edge-feature
    space. Only the LoRA factors and a learnable temperature are refined so that:

    * support edges are correctly classified by proximity to prototypes
      (cross-entropy), and
    * the unlabelled query re-wraps consistently (congruence + sparsity).

    The remaining (query) patches are then evaluated with prototype-based
    classification and the full metric suite is reported.
    """
    if model.head_type in ("pixel", "pixel_cls"):
        return _run_kshot_pixel(cfg, model, device, logger)
    if model.head_type == "edge" and cfg.kshot.get("mode", "prototypical") == "supervised":
        return _run_kshot_edge_supervised(cfg, model, device, logger)

    split = cfg.data.get("split", "test_real")
    full = dataset_module.build_dataset(cfg.data, split)
    K = int(cfg.kshot.k)
    n = len(full)
    if n <= K:
        raise ValueError(f"Need more than K={K} samples in split {split!r} (have {n}).")

    # Deterministic support/query partition.
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(cfg.seed)).tolist()
    support_idx, query_idx = perm[:K], perm[K:]
    support_batch = utils.move_batch_to_device(
        next(iter(DataLoader(Subset(full, support_idx), batch_size=K, shuffle=False))), device
    )
    nw = int(cfg.data.num_workers)
    query_train_loader = DataLoader(
        Subset(full, query_idx), batch_size=int(cfg.data.batch_size), shuffle=True, num_workers=nw
    )
    query_eval_loader = DataLoader(
        Subset(full, query_idx), batch_size=int(cfg.data.batch_size), shuffle=False, num_workers=nw
    )

    model.set_adaptation_mode("kshot")
    model.eval()  # keep BatchNorm stats fixed; only LoRA + temperature adapt
    log_temp = torch.zeros((), device=device, requires_grad=True)  # temperature = exp(log_temp)
    optimizer = torch.optim.AdamW(list(model.lora_parameters()) + [log_temp], lr=cfg.kshot.lr)

    sk_x, sk_y = support_batch["k_x"], support_batch["k_y"]
    svx, svy = support_batch.get("valid_x"), support_batch.get("valid_y")
    ce_weight = compute_class_weights(
        sk_x, sk_y, svx, svy, mode=cfg.train.get("ce_class_weights", "inverse_sqrt")
    )
    query_iter = itertools.cycle(query_train_loader)

    bar = tqdm(range(int(cfg.kshot.steps)), desc=f"K-shot (K={K})", leave=True, dynamic_ncols=True)
    for s in bar:
        optimizer.zero_grad(set_to_none=True)
        temp = torch.exp(log_temp)

        # Prototypes from the support set (recomputed as LoRA features shift).
        _, ef_s = model.encode(support_batch)
        protos_h = compute_prototypes(ef_s, sk_x, svx)
        protos_v = compute_prototypes(ef_s, sk_y, svy)

        # Support prototypical cross-entropy.
        l_proto = cross_entropy_loss(
            prototypical_logits(ef_s, protos_h, temp),
            prototypical_logits(ef_s, protos_v, temp),
            sk_x, sk_y, svx, svy, weight=ce_weight,
        )

        # Unlabelled query physics consistency.
        qbatch = utils.move_batch_to_device(next(query_iter), device)
        out = model.forward_with_prototypes(
            qbatch, protos_h, protos_v, temperature=temp, straight_through=True
        )
        l_cong = congruence_loss(out["phi_hat"], qbatch["wrapped"])
        l_sparse = sparsity_loss(out["probs_h"], out["probs_v"], qbatch.get("valid_x"), qbatch.get("valid_y"))
        total = l_proto + cfg.train.lambda_cong * l_cong + cfg.train.lambda_sparse * l_sparse

        total.backward()
        optimizer.step()
        logger.log(
            {"kshot/proto_ce": l_proto.item(), "kshot/cong": l_cong.item(),
             "kshot/sparse": l_sparse.item(), "kshot/temperature": float(temp.item())},
            s,
        )
        bar.set_postfix(proto=f"{l_proto.item():.3f}", cong=f"{l_cong.item():.3f}")

    # Final prototypes and query evaluation.
    model.eval()
    with torch.no_grad():
        temp = torch.exp(log_temp)
        _, ef_s = model.encode(support_batch)
        protos_h = compute_prototypes(ef_s, sk_x, svx)
        protos_v = compute_prototypes(ef_s, sk_y, svy)

        sums: Dict[str, float] = {}
        total_n = 0
        bar = tqdm(query_eval_loader, desc=f"K-shot eval [{split}]", leave=True, dynamic_ncols=True)
        for batch in bar:
            batch = utils.move_batch_to_device(batch, device)
            bs = batch["wrapped"].shape[0]
            out = model.forward_with_prototypes(batch, protos_h, protos_v, temperature=temp, hard=True)
            for k, v in evaluate_outputs(out, batch).items():
                sums[k] = sums.get(k, 0.0) + v * bs
            total_n += bs
            if "rmse" in sums:
                bar.set_postfix(rmse=f"{sums['rmse'] / total_n:.4f}")

    metrics = {k: v / max(total_n, 1) for k, v in sums.items()}
    logger.log({f"kshot/{k}": v for k, v in metrics.items()}, 0)
    print(format_metrics_table(metrics, title=f"K-shot (K={K}) metrics on '{split}' query"))


# ===========================================================================
# Entry point
# ===========================================================================
_REGIME_DISPATCH = {
    "pretrain": run_pretrain,
    "zero_shot": run_zero_shot,
    "tta": run_tta,
    "kshot": run_kshot,
}


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    # Keep the terminal clean: silence INFO-level chatter from Hydra/libraries so
    # only the tqdm progress bars and end-of-epoch summaries are shown. Set
    # logging.verbose=true to restore full logs.
    if not cfg.logging.get("verbose", False):
        logging.getLogger().setLevel(logging.WARNING)

    utils.set_seed(cfg.seed, deterministic=cfg.deterministic)
    device = utils.get_device(cfg.device)
    print(f"Regime: {cfg.regime}  |  device: {device}  |  seed: {cfg.seed}")

    # When resuming, rebuild the architecture the checkpoint was trained with so
    # adaptation (tta/kshot) can't silently load into a mismatched model.
    if cfg.checkpoint.resume and os.path.exists(cfg.checkpoint.resume):
        _ck = torch.load(cfg.checkpoint.resume, map_location="cpu")
        _saved = _ck.get("config") if isinstance(_ck, dict) else None
        if _saved and "model" in _saved:
            cfg.model = OmegaConf.merge(cfg.model, OmegaConf.create(_saved["model"]))
            if "data" in _saved and "input_grad" in _saved["data"]:
                cfg.data.input_grad = _saved["data"]["input_grad"]

    # Keep the model's input channels in sync with the data encoding (2 for
    # (cos,sin), 4 when wrapped-gradient channels are appended).
    cfg.model.backbone.in_chans = dataset_module.input_channels(cfg.data)
    model = PhaseUnwrapper(model_cfg=cfg.model, img_size=cfg.data.get("img_size", 256)).to(device)

    if cfg.checkpoint.resume:
        load_checkpoint(cfg.checkpoint.resume, model, map_location=device)

    logger = Logger(cfg)
    try:
        regime = str(cfg.regime)
        if regime not in _REGIME_DISPATCH:
            raise ValueError(f"Unknown regime={regime!r}; choose from {list(_REGIME_DISPATCH)}.")
        _REGIME_DISPATCH[regime](cfg, model, device, logger)
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
