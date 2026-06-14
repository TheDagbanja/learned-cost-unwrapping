"""Swin-Transformer-Tiny backbone with targeted LoRA injection.

The backbone follows proposal §3.3:

* Input is the 2-channel ``(cos psi, sin psi)`` encoding (``in_chans=2``).
* A Swin-T encoder is used in ``features_only`` mode to expose the four
  hierarchical feature maps (strides 4/8/16/32) that feed the U-Net decoder.
* Low-Rank Adaptation (Eq. 5) is injected **only** into the query (``Wq``) and
  value (``Wv``) projections of every attention block, with rank ``r=8`` and
  scaling ``alpha=16``. All other backbone weights are frozen.

timm's Swin packs ``Wq, Wk, Wv`` into a single fused ``qkv`` linear, so the
adapter writes its low-rank update into the ``q`` and ``v`` output blocks only,
leaving the ``k`` block untouched. This is mathematically identical to adapting
separate ``Wq`` and ``Wv`` matrices.

Requires ``timm>=0.9.12`` (dynamic ``img_size`` + ``features_only`` for Swin).
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Sequence, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# LoRA primitives
# ---------------------------------------------------------------------------
class LoRAAdapter(nn.Module):
    r"""Low-rank update ``Delta = (alpha / r) * B @ A`` (proposal Eq. 5).

    ``A in R^{r x in}`` is initialised with Kaiming-uniform and ``B in R^{out x r}``
    with zeros, so the adapter starts as a no-op (``Delta = 0``) and training
    departs smoothly from the frozen backbone.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.rank = rank
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features) -> (..., out_features)
        return (self.dropout(x) @ self.A.t()) @ self.B.t() * self.scaling


class LoRAFusedQKV(nn.Module):
    """Wrap a frozen fused ``qkv`` ``nn.Linear`` and add LoRA to chosen blocks.

    The base linear is kept frozen; only the adapter factors are trainable. The
    output shape is identical to the original ``qkv`` linear, so this module is a
    drop-in replacement inside timm's ``WindowAttention``.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        targets: Sequence[str] = ("q", "v"),
    ) -> None:
        super().__init__()
        if base.out_features % 3 != 0:
            raise ValueError(
                "Fused qkv linear must have out_features divisible by 3; "
                f"got {base.out_features}."
            )
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.dim = base.out_features // 3
        self.targets = tuple(targets)
        in_f = base.in_features
        self.adapters = nn.ModuleDict()
        for name in ("q", "k", "v"):
            if name in self.targets:
                self.adapters[name] = LoRAAdapter(in_f, self.dim, rank, alpha, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)  # (..., 3*dim)
        zeros_shape = x.shape[:-1] + (self.dim,)
        dq = self.adapters["q"](x) if "q" in self.adapters else x.new_zeros(zeros_shape)
        dk = self.adapters["k"](x) if "k" in self.adapters else x.new_zeros(zeros_shape)
        dv = self.adapters["v"](x) if "v" in self.adapters else x.new_zeros(zeros_shape)
        return out + torch.cat((dq, dk, dv), dim=-1)


# ---------------------------------------------------------------------------
# Injection utilities
# ---------------------------------------------------------------------------
def _replace_submodule(root: nn.Module, qualified_name: str, new: nn.Module) -> None:
    """Replace ``root.<qualified_name>`` with ``new`` (supports numeric indices)."""
    parts = qualified_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new
    else:
        setattr(parent, last, new)


def inject_lora_qkv(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    targets: Sequence[str] = ("q", "v"),
) -> List[str]:
    """Inject LoRA into every fused ``qkv`` attention linear of ``model``.

    Returns the list of replaced module names. Detection is structural: any
    submodule named ``qkv`` that is an ``nn.Linear`` with ``out_features ==
    3 * in_features`` is treated as a Swin attention projection.
    """
    to_replace: List[Tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if (
            name.endswith("qkv")
            and isinstance(module, nn.Linear)
            and module.out_features == 3 * module.in_features
        ):
            to_replace.append((name, module))

    replaced: List[str] = []
    for name, base in to_replace:
        wrapped = LoRAFusedQKV(base, rank=rank, alpha=alpha, dropout=dropout, targets=targets)
        _replace_submodule(model, name, wrapped)
        replaced.append(name)
    if not replaced:
        raise RuntimeError(
            "No fused 'qkv' attention linears found for LoRA injection. "
            "Check the backbone name / timm version."
        )
    return replaced


def iter_lora_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    """Yield only the LoRA adapter parameters of ``model``."""
    for module in model.modules():
        if isinstance(module, LoRAAdapter):
            yield from module.parameters()


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    """Freeze everything in ``model`` except LoRA adapter parameters."""
    for p in model.parameters():
        p.requires_grad_(False)
    for p in iter_lora_parameters(model):
        p.requires_grad_(True)


# ---------------------------------------------------------------------------
# Backbone wrapper
# ---------------------------------------------------------------------------
class SwinLoRABackbone(nn.Module):
    """Synthetic-pretrained Swin-T feature extractor with targeted LoRA.

    Parameters
    ----------
    name:
        timm model id, e.g. ``"swin_tiny_patch4_window7_224"``.
    in_chans:
        Number of input channels (2 for the ``(cos, sin)`` encoding).
    img_size:
        Spatial size the encoder is built for (256 for InSAR-DLPU).
    pretrained_imagenet:
        Load timm's ImageNet weights (development convenience only; the proposal
        uses synthetic-interferogram pretraining instead).
    lora:
        Dict with keys ``rank``, ``alpha``, ``dropout``, ``targets``.
    freeze_backbone:
        If ``True``, freeze all non-LoRA backbone weights.
    out_indices:
        Which hierarchical stages to return (default all four).
    """

    def __init__(
        self,
        name: str = "swin_tiny_patch4_window7_224",
        in_chans: int = 2,
        img_size: int = 256,
        pretrained_imagenet: bool = False,
        lora: Dict | None = None,
        freeze_backbone: bool = True,
        out_indices: Sequence[int] = (0, 1, 2, 3),
    ) -> None:
        super().__init__()
        import timm  # heavy; imported lazily

        lora = lora or {"rank": 8, "alpha": 16.0, "dropout": 0.0, "targets": ("q", "v")}
        self.encoder = timm.create_model(
            name,
            pretrained=pretrained_imagenet,
            features_only=True,
            out_indices=tuple(out_indices),
            in_chans=in_chans,
            img_size=img_size,
        )
        self.feature_channels: List[int] = list(self.encoder.feature_info.channels())
        self.reductions: List[int] = list(self.encoder.feature_info.reduction())

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        self.lora_layers = inject_lora_qkv(self.encoder, **lora)

    # -- feature-layout helper -------------------------------------------------
    @staticmethod
    def _to_nchw(feat: torch.Tensor, channels: int) -> torch.Tensor:
        """Normalise a Swin feature map to NCHW regardless of timm's layout."""
        if feat.dim() != 4:
            raise ValueError(f"Expected 4-D feature, got shape {tuple(feat.shape)}.")
        if feat.shape[1] == channels:
            return feat                                   # already NCHW
        if feat.shape[-1] == channels:
            return feat.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
        return feat  # ambiguous (e.g. square stage); assume NCHW

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = self.encoder(x)
        return [self._to_nchw(f, c) for f, c in zip(feats, self.feature_channels)]

    # -- convenience -----------------------------------------------------------
    def lora_parameters(self) -> Iterator[nn.Parameter]:
        return iter_lora_parameters(self.encoder)

    def load_synthetic_weights(self, path: str, strict: bool = False) -> None:
        """Load synthetic-interferogram-pretrained encoder weights.

        Non-strict by default so a checkpoint saved before LoRA injection (i.e.
        without adapter keys) still loads cleanly.
        """
        state = torch.load(path, map_location="cpu")
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        self.encoder.load_state_dict(state, strict=strict)
