"""End-to-end differentiable cost fine-tuning (Stage-2 v2).

Loads a trained edge head and fine-tunes ONLY its LoRA factors so the per-edge
costs minimise the *final* unwrapping RMSE (through the MCF solver, via the
blackbox-solver gradient) -- rather than supervised edge classification. Requires
ground truth, so trains on DLPU 'train'. Each step solves MCF twice per image, so
keep batches/steps modest; this is a refinement of the supervised edge head.

    CUDA_VISIBLE_DEVICES=1 python -m scripts.train_diff \\
      checkpoint.resume=checkpoints/pretrain_last.pt data.source=dlpu \\
      train.lr=1e-4 train.steps=300 logging.backend=none
"""

from __future__ import annotations

import os

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src import dataset as dataset_module
from src import mcf_infer
from src import utils
from src.models.unwrapper import PhaseUnwrapper


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not cfg.checkpoint.resume or not os.path.exists(cfg.checkpoint.resume):
        raise SystemExit(f"Need a trained edge-head checkpoint (got {cfg.checkpoint.resume!r}).")
    ckpt = torch.load(cfg.checkpoint.resume, map_location=device)
    saved = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    if saved and "model" in saved:
        cfg.model = OmegaConf.merge(cfg.model, OmegaConf.create(saved["model"]))
        if "data" in saved and "input_grad" in saved["data"]:
            cfg.data.input_grad = saved["data"]["input_grad"]
    cfg.model.backbone.in_chans = dataset_module.input_channels(cfg.data)
    model = PhaseUnwrapper(model_cfg=cfg.model, img_size=cfg.data.get("img_size", 256)).to(device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)
    if model.head_type != "edge":
        raise SystemExit(f"Differentiable cost training needs head_type=edge; got {model.head_type!r}.")

    model.set_adaptation_mode("kshot")     # LoRA-only (PEFT)
    model.eval()                            # BN frozen
    loader = dataset_module.build_dataloader(cfg.data, split="train", shuffle=True, device=device)
    lam = float(cfg.train.get("diff_lam", 20.0))
    steps = int(cfg.train.get("steps", 300))
    opt = torch.optim.AdamW(list(model.lora_parameters()), lr=float(cfg.train.lr))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)  # stabilise the tail
    from src.train import save_checkpoint
    best_path = os.path.join(cfg.checkpoint.dir, "edge_diff_best.pt")

    it = iter(loader)
    bar = tqdm(range(steps), desc="diff-cost fine-tune", dynamic_ncols=True)
    hist, best = [], float("inf")
    for s in bar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        batch = utils.move_batch_to_device(batch, device)
        opt.zero_grad(set_to_none=True)
        loss = mcf_diff_loss(model, batch, lam)
        loss.backward()
        opt.step(); sched.step()
        hist.append(float(loss.item()))
        rm = float(np.mean(hist[-25:]))
        bar.set_postfix(rmse=f"{rm:.3f}")
        if (s + 1) >= 25 and rm < best:             # save the BEST running-mean checkpoint
            best = rm; save_checkpoint(best_path, model, opt, s, cfg)
        if (s + 1) % 25 == 0:                        # proper lines survive in scrollback
            print(f"  step {s + 1:4d}/{steps}: rmse(last25)={rm:.4f}  best={best:.4f}", flush=True)

    np.savetxt(os.path.join(cfg.checkpoint.dir, "train_diff_loss.txt"), np.array(hist))
    n0 = min(25, len(hist))
    print(f"loss trend: first {n0} = {np.mean(hist[:n0]):.4f}  ->  last {n0} = {np.mean(hist[-n0:]):.4f}  "
          f"| BEST running-mean = {best:.4f}  (saved {best_path})")
    out = os.path.join(cfg.checkpoint.dir, "edge_diff_last.pt")
    save_checkpoint(out, model, opt, 0, cfg)
    print(f"saved: {out} (last) and {best_path} (best -- use this one)")
    print(f"  score it: python -m scripts.eval_learned_mcf checkpoint.resume={out} "
          f"data.source=dlpu eval.split=test_real logging.backend=none")


def mcf_diff_loss(model, batch, lam):
    from src import mcf_diff
    return mcf_diff.differentiable_unwrap_loss(model, batch, lam=lam)


if __name__ == "__main__":
    main()
