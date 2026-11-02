"""Train one ext_dl baseline with the repository's unchanged Hydra protocol.

Checkpoint naming remains ``<checkpoint.dir>/{baseline}_last.pt``.  For the
two-stage MoDL-PU adaptation, train stage 1 first, then resume that checkpoint
and select ``ext.modl_stage=2``; the rolling filename remains unchanged.
"""

from __future__ import annotations

import math
import os

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src import dataset as dataset_module
from src import utils
from .registry import BUILDERS, compute_loss, _kcls  # _kcls re-exported for compatibility


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    name = str(cfg.get("baseline", "dlpu_cnn"))
    if name not in BUILDERS:
        raise SystemExit(f"baseline must be one of {list(BUILDERS)}, got {name!r}")
    utils.set_seed(cfg.seed, deterministic=cfg.deterministic)
    device = utils.get_device(cfg.device)
    in_chans = dataset_module.input_channels(cfg.data)
    model = BUILDERS[name](cfg, in_chans).to(device)

    resume = cfg.checkpoint.get("resume", None)
    if resume:
        if not os.path.exists(resume):
            raise SystemExit(f"checkpoint.resume does not exist: {resume}")
        state = torch.load(resume, map_location=device)
        model.load_state_dict(state.get("model", state), strict=True)
        print(f"Resumed {name} from {resume}")
    print(f"Baseline={name} params={model.count_parameters()['total']:,} device={device}")

    loader = dataset_module.build_dataloader(cfg.data, split="train", device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    epochs = int(cfg.train.epochs); steps_per_epoch = max(1, len(loader))
    warmup = int(cfg.train.get("warmup_epochs", 1) * steps_per_epoch); total = epochs * steps_per_epoch

    def lr_factor(step):
        if step < warmup: return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return .5 * (1. + math.cos(math.pi * min(1., progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    os.makedirs(cfg.checkpoint.dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint.dir, f"{name}_last.pt")
    for epoch in range(epochs):
        model.train(); running = 0.0
        bar = tqdm(loader, desc=f"[{name}] epoch {epoch+1}/{epochs}", dynamic_ncols=True)
        for batch in bar:
            batch = utils.move_batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            terms = compute_loss(name, cfg, model(batch), batch)
            terms["total"].backward()
            if cfg.train.get("grad_clip", 0):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step(); scheduler.step(); running += float(terms["total"].item())
            bar.set_postfix(loss=f"{terms['total'].item():.4f}")
        # Preserve the original ext_dl checkpoint payload as well as its name.
        torch.save(model.state_dict(), path)
        print(f"epoch {epoch+1}: mean loss {running/max(1,len(loader)):.4f}")
    print(f"Training complete. Checkpoint: {path}")


if __name__ == "__main__":
    main()
