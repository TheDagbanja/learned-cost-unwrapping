"""The end-to-end phase unwrapper (proposal Fig. 1).

Wires the three components together:

    (cos psi, sin psi)  --Swin-T+LoRA-->  multi-scale features
                        --U-Net decoder->  per-edge ambiguity logits (B,6,H,W)
                        --integrator---->  unwrapped phase phi_hat

The per-edge logits are split into horizontal (channels 0:3) and vertical
(channels 3:6) ambiguity distributions over ``{-1, 0, +1}``. Corrected gradients
are formed as ``g_e = W(Delta psi)_e + 2*pi*E[k_e]`` during training (soft,
differentiable) and with ``argmax`` ambiguity at inference (proposal §3.4), then
integrated with coherence-derived weights.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import utils
from ..integrator import WeightedLeastSquaresIntegrator
from .backbone import SwinLoRABackbone, iter_lora_parameters
from .decoder import UNetDecoder

_ADAPTATION_MODES = ("pretrain", "tta", "kshot", "frozen")


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# ---------------------------------------------------------------------------
# Prototypical few-shot helpers (proposal §3.6)
# ---------------------------------------------------------------------------
def compute_prototypes(
    edge_features: torch.Tensor,
    k_class: torch.Tensor,
    valid: torch.Tensor | None = None,
    num_classes: int = 3,
) -> torch.Tensor:
    r"""Class prototypes ``c_l = mean_{e in S_l} h_e`` in the edge-feature space.

    Parameters
    ----------
    edge_features:
        Decoder edge features ``(B, C, H, W)``.
    k_class:
        Ambiguity class indices ``(B, H, W)`` in ``{0, 1, 2}`` for one direction.
    valid:
        Optional boolean mask ``(B, H, W)`` excluding padding edges.
    num_classes:
        Number of ambiguity classes (3).

    Returns
    -------
    torch.Tensor
        Prototypes ``(num_classes, C)``. Classes absent from the support set get
        a zero prototype (their distance term is then uninformative).
    """
    b, c, h, w = edge_features.shape
    feats = edge_features.permute(0, 2, 3, 1).reshape(-1, c)   # (N, C)
    labels = k_class.reshape(-1)
    mask = (valid.reshape(-1) if valid is not None else torch.ones_like(labels, dtype=torch.bool))
    protos = []
    for cls in range(num_classes):
        sel = mask & (labels == cls)
        if sel.any():
            protos.append(feats[sel].mean(dim=0))
        else:
            protos.append(feats.new_zeros(c))
    return torch.stack(protos, dim=0)


def prototypical_logits(
    edge_features: torch.Tensor, prototypes: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    r"""Per-edge logits as ``-||h_e - c_l||^2 / temperature``.

    ``edge_features`` is ``(B, C, H, W)`` and ``prototypes`` is ``(num_classes, C)``;
    returns ``(B, num_classes, H, W)``. Computed via the expansion
    ``||h - c||^2 = ||h||^2 - 2 h.c + ||c||^2`` (differentiable in both).
    """
    f = edge_features.permute(0, 2, 3, 1)                       # (B, H, W, C)
    h2 = (f ** 2).sum(dim=-1, keepdim=True)                     # (B, H, W, 1)
    c2 = (prototypes ** 2).sum(dim=-1)                          # (num_classes,)
    hc = torch.einsum("bhwc,lc->bhwl", f, prototypes)           # (B, H, W, L)
    d2 = h2 - 2.0 * hc + c2                                     # (B, H, W, L)
    # temperature may be a python float or a (learnable) scalar tensor.
    logits = -d2 / temperature
    return logits.permute(0, 3, 1, 2).contiguous()             # (B, L, H, W)


class PhaseUnwrapper(nn.Module):
    """Backbone + decoder + differentiable integrator.

    Parameters
    ----------
    model_cfg:
        The ``model`` config group (Hydra/OmegaConf node or plain dict) with
        ``backbone``, ``lora``, ``decoder`` and ``integrator`` subsections, as in
        ``configs/model.yaml``.
    img_size:
        Spatial input size the Swin encoder is built for (default 256).
    """

    def __init__(self, model_cfg: Any = None, img_size: int = 256) -> None:
        super().__init__()
        bk = _cfg_get(model_cfg, "backbone", {}) or {}
        lo = _cfg_get(model_cfg, "lora", {}) or {}
        de = _cfg_get(model_cfg, "decoder", {}) or {}
        it = _cfg_get(model_cfg, "integrator", {}) or {}

        lora_cfg = {
            "rank": int(_cfg_get(lo, "rank", 8)),
            "alpha": float(_cfg_get(lo, "alpha", 16.0)),
            "dropout": float(_cfg_get(lo, "dropout", 0.0)),
            "targets": tuple(_cfg_get(lo, "targets", ("q", "v"))),
        }

        self.backbone = SwinLoRABackbone(
            name=_cfg_get(bk, "name", "swin_tiny_patch4_window7_224"),
            in_chans=int(_cfg_get(bk, "in_chans", 2)),
            img_size=img_size,
            pretrained_imagenet=bool(_cfg_get(bk, "pretrained_imagenet", False)),
            lora=lora_cfg,
            freeze_backbone=bool(_cfg_get(bk, "freeze", True)),
        )

        # Prediction head:
        #   "edge"      -> per-edge ambiguity + weighted-LS integration (proposal core)
        #   "pixel"     -> per-pixel wrap count K via regression (residue-free)
        #   "pixel_cls" -> per-pixel wrap count K via classification over the
        #                  integer range [-k_max, k_max] (sharper at fringe
        #                  boundaries; PhaseNet-2.0 style). Residue-free.
        self.head_type = str(_cfg_get(de, "head_type", "edge"))
        self.k_max = int(_cfg_get(de, "k_max", 16))
        if self.head_type == "pixel":
            out_channels = 1
        elif self.head_type == "pixel_cls":
            out_channels = 2 * self.k_max + 1
        else:
            out_channels = int(_cfg_get(de, "out_channels", 6))

        self.decoder = UNetDecoder(
            encoder_channels=self.backbone.feature_channels,
            decoder_channels=tuple(_cfg_get(de, "channels", (256, 128, 96, 64, 32))),
            in_chans=int(_cfg_get(bk, "in_chans", 2)),
            stem_channels=tuple(_cfg_get(de, "stem_channels", (48, 32))),
            out_channels=out_channels,
        )

        self.integrator = WeightedLeastSquaresIntegrator(
            solver=_cfg_get(it, "solver", "auto"),
            cg_iters=int(_cfg_get(it, "cg_iters", 100)),
            cg_tol=float(_cfg_get(it, "cg_tol", 1e-6)),
        )

        # Synthetic-pretrained backbone weights, if provided.
        syn = _cfg_get(bk, "pretrained_synthetic", None)
        if syn:
            self.backbone.load_synthetic_weights(syn)

        self.adaptation_mode = "pretrain"

    # ------------------------------------------------------------------ encode
    def encode(self, batch: Dict[str, torch.Tensor]):
        """Run backbone + decoder, returning ``(logits, edge_features)``.

        ``logits`` is ``(B, 6, H, W)`` and ``edge_features`` ``(B, C, H, W)`` at
        the input resolution (with a guard against Swin window-padding drift).
        """
        x = batch["input"]
        h, w = x.shape[-2], x.shape[-1]
        features = self.backbone(x)
        logits, edge_features = self.decoder(features, x)
        if logits.shape[-2:] != (h, w):
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            edge_features = F.interpolate(
                edge_features, size=(h, w), mode="bilinear", align_corners=False
            )
        return logits, edge_features

    def _assemble(
        self,
        logits: torch.Tensor,
        edge_features: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        hard: bool,
        straight_through: bool = False,
        cut_threshold: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Turn ``(B, 6, H, W)`` per-edge logits into the full pipeline output.

        Ambiguity used for integration:

        * ``hard=True``           -> argmax integer (inference).
        * ``straight_through``    -> argmax value in the forward pass, soft
          ``E[k]`` gradient in the backward pass.
        * otherwise               -> soft ``E[k]`` (fully differentiable).

        ``cut_threshold`` (inference robustness): a predicted non-zero ambiguity
        is only applied if its class probability exceeds the threshold; otherwise
        the edge is treated as ``k=0``. Because a single wrong cut injects a 2*pi
        error that the integrator spreads across the field, gating low-confidence
        cuts trades recall for the precision that actually matters here.
        """
        b, _, h, w = logits.shape
        psi = batch["wrapped"]
        if psi.dim() == 4:                                # (B,1,H,W) -> (B,H,W)
            psi = psi.squeeze(1)

        # Split into (direction, class): (B, 2, 3, H, W).
        logits_dc = logits.view(b, 2, 3, h, w)
        probs = torch.softmax(logits_dc, dim=2)
        probs_h, probs_v = probs[:, 0], probs[:, 1]       # (B,3,H,W) each

        ek_h = utils.expected_ambiguity(probs_h)          # (B,H,W)
        ek_v = utils.expected_ambiguity(probs_v)
        conf_h, idx_h = probs_h.max(dim=1)
        conf_v, idx_v = probs_v.max(dim=1)
        k_h = idx_h.float() - 1.0                          # class -> k (hard)
        k_v = idx_v.float() - 1.0
        if cut_threshold > 0.0:
            k_h = k_h.masked_fill((k_h != 0) & (conf_h <= cut_threshold), 0.0)
            k_v = k_v.masked_fill((k_v != 0) & (conf_v <= cut_threshold), 0.0)

        if hard:
            amb_h, amb_v = k_h, k_v
        elif straight_through:
            amb_h = ek_h + (k_h - ek_h).detach()          # value=hard, grad=soft
            amb_v = ek_v + (k_v - ek_v).detach()
        else:
            amb_h, amb_v = ek_h, ek_v

        # Corrected gradients g_e = W(Delta psi)_e + 2*pi*amb_e (full-res padded).
        wgx, wgy = utils.wrapped_gradients(psi, pad=True)
        g_x = wgx + utils.TWO_PI * amb_h
        g_y = wgy + utils.TWO_PI * amb_v

        phi_hat = self.integrator(g_x, g_y, batch.get("weights_x"), batch.get("weights_y"))

        return {
            "logits": logits,
            "logits_h": logits_dc[:, 0],
            "logits_v": logits_dc[:, 1],
            "probs_h": probs_h,
            "probs_v": probs_v,
            "ek_h": ek_h,
            "ek_v": ek_v,
            "g_x": g_x,
            "g_y": g_y,
            "phi_hat": phi_hat.unsqueeze(1),               # (B,1,H,W)
            "edge_features": edge_features,
        }

    def _assemble_pixel(
        self, k_reg: torch.Tensor, edge_features: torch.Tensor,
        batch: Dict[str, torch.Tensor], hard: bool,
    ) -> Dict[str, torch.Tensor]:
        """Per-pixel wrap-count head: ``phi_hat = psi + 2*pi*K``.

        The network regresses a continuous wrap count ``K``; ``phi_hat`` is a
        scalar field, so it is residue-free by construction. At inference ``K`` is
        rounded to integers (errors stay local: one wrong pixel = a 2*pi error at
        that pixel only). ``g_x``/``g_y`` are the actual gradients of ``phi_hat``
        (so the residue metric reads ~0).
        """
        psi = batch["wrapped"]
        if psi.dim() == 4:
            psi = psi.squeeze(1)
        k = k_reg.squeeze(1)                              # (B, H, W) continuous
        k_used = torch.round(k) if hard else k
        phi_hat = psi + utils.TWO_PI * k_used
        gx, gy = utils.edge_diff_x(phi_hat), utils.edge_diff_y(phi_hat)
        g_x = utils.pad_edge_x(gx, 0.0)
        g_y = utils.pad_edge_y(gy, 0.0)
        return {
            "k_reg": k_reg,                              # (B,1,H,W)
            "k_pred": torch.round(k).unsqueeze(1),
            "phi_hat": phi_hat.unsqueeze(1),
            "g_x": g_x,
            "g_y": g_y,
            "edge_features": edge_features,
        }

    def _assemble_pixel_cls(
        self, logits: torch.Tensor, edge_features: torch.Tensor,
        batch: Dict[str, torch.Tensor], hard: bool,
    ) -> Dict[str, torch.Tensor]:
        """Per-pixel wrap-count *classification* head over ``[-k_max, k_max]``.

        Gives a crisp per-pixel integer decision (sharper than regression at the
        fringe boundaries where K steps by 1). ``phi_hat = psi + 2*pi*K`` is still
        residue-free. The class index ``c`` maps to ``K = c - k_max`` (centred;
        the unidentifiable global offset is removed by the zero-mean metric).
        """
        psi = batch["wrapped"]
        if psi.dim() == 4:
            psi = psi.squeeze(1)
        probs = torch.softmax(logits, dim=1)                  # (B, 2k_max+1, H, W)
        kvals = torch.arange(
            -self.k_max, self.k_max + 1, device=logits.device, dtype=logits.dtype
        ).view(1, -1, 1, 1)
        ek = (probs * kvals).sum(dim=1)                       # (B,H,W) soft expected K
        # Round the *expected* K (ordinal-hybrid): errors stay bounded to +-1,
        # unlike argmax which can jump several classes on a misclassification.
        k_round = torch.round(ek)
        k_used = k_round if hard else ek
        phi_hat = psi + utils.TWO_PI * k_used
        g_x = utils.pad_edge_x(utils.edge_diff_x(phi_hat), 0.0)
        g_y = utils.pad_edge_y(utils.edge_diff_y(phi_hat), 0.0)
        return {
            "k_logits": logits,
            "k_reg": ek.unsqueeze(1),                         # soft K (for TV + metrics)
            "k_pred": k_round.unsqueeze(1),
            "phi_hat": phi_hat.unsqueeze(1),
            "g_x": g_x,
            "g_y": g_y,
            "edge_features": edge_features,
        }

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        hard: bool = False,
        straight_through: bool = False,
        cut_threshold: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Run the full pipeline.

        Edge head: see :meth:`_assemble`. Pixel heads: see :meth:`_assemble_pixel`
        / :meth:`_assemble_pixel_cls`. ``hard`` selects integer (inference) vs
        continuous (training) ambiguity.
        """
        out, edge_features = self.encode(batch)
        if self.head_type == "pixel":
            return self._assemble_pixel(out, edge_features, batch, hard)
        if self.head_type == "pixel_cls":
            return self._assemble_pixel_cls(out, edge_features, batch, hard)
        return self._assemble(out, edge_features, batch, hard, straight_through, cut_threshold)

    def forward_with_prototypes(
        self,
        batch: Dict[str, torch.Tensor],
        prototypes_h: torch.Tensor,
        prototypes_v: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = False,
        straight_through: bool = False,
        cut_threshold: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Classify edges by proximity to prototypes instead of the decoder head.

        Used for K-shot prototypical refinement (proposal §3.6): per-edge logits
        are the negative squared distance from the decoder edge-features to the
        per-direction class prototypes. The downstream integration is unchanged.
        """
        _, edge_features = self.encode(batch)
        logits_h = prototypical_logits(edge_features, prototypes_h, temperature)
        logits_v = prototypical_logits(edge_features, prototypes_v, temperature)
        logits = torch.cat((logits_h, logits_v), dim=1)   # (B,6,H,W)
        return self._assemble(logits, edge_features, batch, hard, straight_through, cut_threshold)

    # -------------------------------------------------------------- adaptation
    def set_adaptation_mode(self, mode: str) -> None:
        """Set which parameters are trainable for a given regime.

        * ``pretrain`` -- **full fine-tuning** of the backbone + decoder on the
          (synthetic / source) data; the LoRA factors stay at their zero init.
          This is the stage that *creates* the prior — the backbone must learn.
        * ``tta`` -- adaptation: backbone + decoder frozen, only LoRA factors train.
        * ``kshot`` -- adaptation: only LoRA factors train; prototypes replace the
          head (§3.6).
        * ``frozen`` -- nothing trains (zero-shot inference).
        """
        if mode not in _ADAPTATION_MODES:
            raise ValueError(f"mode must be one of {_ADAPTATION_MODES}, got {mode!r}.")
        self.adaptation_mode = mode

        if mode == "pretrain":
            # Train the full backbone (and decoder); keep LoRA as a zero-init no-op
            # so adaptation can later start cleanly from the pretrained weights.
            for p in self.backbone.parameters():
                p.requires_grad_(True)
            for p in self.backbone.lora_parameters():
                p.requires_grad_(False)
            decoder_trainable = True
        else:
            # Adaptation / inference: freeze the backbone base, train only LoRA.
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            lora_trainable = mode in ("tta", "kshot")
            for p in self.backbone.lora_parameters():
                p.requires_grad_(lora_trainable)
            decoder_trainable = False

        for p in self.decoder.parameters():
            p.requires_grad_(decoder_trainable)

    def lora_parameters(self) -> Iterator[nn.Parameter]:
        return iter_lora_parameters(self.backbone.encoder)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)

    def count_parameters(self) -> Dict[str, int]:
        """Return total / trainable / LoRA parameter counts (for logging)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora = sum(p.numel() for p in self.lora_parameters())
        return {"total": total, "trainable": trainable, "lora": lora}


def _smoke_test() -> None:
    """Build the model and run one forward/backward pass on random input.

    Run with ``python -m src.models.unwrapper`` on a machine with torch + timm.
    Validates the Swin-T window padding at 256x256, the decoder shapes, the
    differentiable integrator, and the LoRA-only trainable parameter accounting.
    """
    torch.manual_seed(0)
    b, h, w = 2, 256, 256
    psi = (torch.rand(b, h, w) * 2 - 1) * utils.PI
    batch = {
        "input": utils.encode_input(psi, include_grad=True),   # (B, 4, H, W)
        "wrapped": psi.unsqueeze(1),
    }
    for head in ("pixel", "pixel_cls", "edge"):
        model = PhaseUnwrapper(
            model_cfg={"backbone": {"in_chans": 4}, "decoder": {"head_type": head}}, img_size=256
        )
        model.set_adaptation_mode("pretrain")
        counts = model.count_parameters()
        out = model(batch, hard=False)
        assert out["phi_hat"].shape == (b, 1, h, w)
        extra = (f"k_reg {tuple(out['k_reg'].shape)}" if head == "pixel"
                 else f"k_logits {tuple(out['k_logits'].shape)}" if head == "pixel_cls"
                 else f"logits {tuple(out['logits'].shape)}")
        print(f"[smoke] head={head:5s} total={counts['total']:,} "
              f"trainable={counts['trainable']:,} ({100 * counts['lora'] / counts['total']:.3f}% LoRA) "
              f"phi_hat {tuple(out['phi_hat'].shape)}  {extra}")
        out["phi_hat"].pow(2).mean().backward()
        assert any(p.grad is not None for p in model.decoder.parameters())
    print("[smoke] PASSED (both heads)")


if __name__ == "__main__":
    _smoke_test()
