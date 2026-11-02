"""Lazy model/loss registry preserving the original ext_dl public interface."""

from __future__ import annotations

import importlib
from typing import Dict


MODEL_MODULES: Dict[str, str] = {
    "dlpu_cnn": "ext_dl.dlpu_cnn",
    "phasenet2": "ext_dl.phasenet2",
    "gradient_net": "ext_dl.gradient_net",
    "attention_unet": "ext_dl.attention_unet",
    "deeplabv3plus": "ext_dl.deeplabv3plus",
    "unetpp": "ext_dl.unetpp",
    "denet": "ext_dl.denet",
    "modl_pu": "ext_dl.modl_pu",
    "unwrap_net": "ext_dl.unwrap_net",
    "dmp_punet": "ext_dl.dmp_punet",
    "waveunet": "ext_dl.waveunet",
}


def _kcls(cfg):
    return int(cfg.model.decoder.get("k_max", 16))


def _ext(cfg, key, default):
    return cfg.get("ext", {}).get(key, default)


def build_model(name, cfg, in_chans):
    module = importlib.import_module(MODEL_MODULES[name])
    if name == "dlpu_cnn": return module.DLPUCNNBaseline(in_chans=in_chans)
    if name == "phasenet2": return module.PhaseNet2Baseline(in_chans=in_chans, k_max=_kcls(cfg))
    if name == "gradient_net": return module.GradientNetBaseline(in_chans=in_chans)
    if name == "attention_unet": return module.AttentionUNetBaseline(in_chans=in_chans)
    if name == "deeplabv3plus": return module.DeepLabV3PlusBaseline(in_chans=in_chans, k_max=_kcls(cfg))
    if name == "unetpp": return module.UNetPPBaseline(in_chans=in_chans)
    if name == "denet": return module.DENetBaseline()
    if name == "modl_pu": return module.MoDLPUBaseline(in_chans=in_chans)
    if name == "unwrap_net": return module.UnwrapNetBaseline(in_chans=in_chans)
    if name == "dmp_punet": return module.DMPPUNetBaseline(in_chans=in_chans)
    if name == "waveunet": return module.WaveUNetBaseline(in_chans=in_chans)
    raise ValueError(f"Unknown baseline {name!r}; choose from {list(MODEL_MODULES)}")


def compute_loss(name, cfg, outputs, batch):
    module = importlib.import_module(MODEL_MODULES[name])
    if name == "dlpu_cnn": return module.dlpu_loss(outputs, batch, lambda_tv=float(cfg.train.get("lambda_ktv", .01)))
    if name == "phasenet2": return module.phasenet2_loss(outputs, batch, k_max=_kcls(cfg))
    if name == "gradient_net": return module.loss_from_k_max(outputs, batch, k_max=int(cfg.data.get("k_max", 1)), lambda_cong=float(cfg.train.get("lambda_cong_gn", .1)))
    if name == "attention_unet": return module.attention_unet_loss(outputs, batch, lambda_tv=float(cfg.train.get("lambda_ktv", .01)))
    if name == "deeplabv3plus": return module.deeplabv3plus_loss(outputs, batch, k_max=_kcls(cfg))
    if name == "unetpp": return module.unetpp_loss(outputs, batch, lambda_tv=float(cfg.train.get("lambda_ktv", .01)))
    if name == "denet": return module.denet_loss(outputs, batch, k_max=int(cfg.data.get("k_max", 1)))
    if name == "modl_pu": return module.modl_pu_loss(outputs, batch, stage=int(_ext(cfg, "modl_stage", 2)), lambda_grad=float(_ext(cfg, "lambda_grad", .3)), lambda_congruence=float(_ext(cfg, "lambda_congruence", .1)))
    if name == "unwrap_net": return module.unwrap_net_loss(outputs, batch, float(_ext(cfg, "lambda_grad", .2)), float(_ext(cfg, "lambda_congruence", .05)))
    if name == "dmp_punet": return module.dmp_punet_loss(outputs, batch, float(_ext(cfg, "lambda_grad", .2)), float(_ext(cfg, "lambda_congruence", .05)))
    if name == "waveunet": return module.waveunet_loss(outputs, batch)
    raise ValueError(name)


class _Builder:
    def __init__(self, name): self.name = name
    def __call__(self, cfg, in_chans): return build_model(self.name, cfg, in_chans)


BUILDERS = {name: _Builder(name) for name in MODEL_MODULES}
