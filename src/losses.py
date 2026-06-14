r"""Physics-guided losses (proposal §3.5).

Four terms, combined differently per training/adaptation regime:

* ``L_CE``     -- per-edge cross-entropy on the ambiguity labels (Eq. 6).
* ``L_cong``   -- congruence: ``mean(1 - cos(phi_hat - psi))`` (Eq. 7), zero when
                  re-wrapping the prediction reproduces the observation.
* ``L_sparse`` -- relaxed cut count ``mean(1 - p_{e,0})`` (Eq. 8), encoding that
                  non-zero ambiguities are sparse.
* ``L_ent``    -- mean per-edge prediction entropy, a confidence-sharpening term
                  used only in unsupervised test-time adaptation (Eq. 10).

Objectives::

    pretrain (Eq. 9): L = L_CE + lambda1 * L_cong + lambda2 * L_sparse
    TTA      (Eq. 10): L =        L_cong + lambda2 * L_sparse + lambda3 * L_ent

All per-edge terms respect the validity masks so the padding column/row (the
non-existent border edges) never contribute.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean of ``x`` over elements where ``mask`` is truthy (per whole batch)."""
    if mask is None:
        return x.mean()
    m = mask.to(x.dtype)
    return (x * m).sum() / m.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# Individual loss terms
# ---------------------------------------------------------------------------
def compute_class_weights(
    k_h: torch.Tensor,
    k_v: torch.Tensor,
    valid_h: Optional[torch.Tensor],
    valid_v: Optional[torch.Tensor],
    mode: str = "inverse_sqrt",
    num_classes: int = 3,
) -> Optional[torch.Tensor]:
    r"""Per-class cross-entropy weights from label frequencies in a batch.

    The ambiguity classes are extremely imbalanced (k=0 is ~98% of edges), so
    plain cross-entropy collapses to predicting "k=0" everywhere. These weights
    counteract that. ``mode``:

    * ``"none"`` -> ``None`` (unweighted).
    * ``"inverse"`` / ``"median"`` -> inverse-frequency / median-frequency
      balancing (strong; k=0 weight ~0.02).
    * ``"inverse_sqrt"`` -> inverse square-root frequency (moderate; default).

    The returned weight vector ``(num_classes,)`` is normalised to mean 1.
    """
    if mode in (None, "none", False):
        return None
    device = k_h.device
    counts = torch.zeros(num_classes, device=device)
    for k, valid in ((k_h, valid_h), (k_v, valid_v)):
        sel = k[valid.bool()] if valid is not None else k.reshape(-1)
        counts += torch.bincount(sel.reshape(-1), minlength=num_classes).float()
    freq = counts.clamp_min(1.0)
    freq = freq / freq.sum()
    if mode == "inverse":
        w = 1.0 / freq
    elif mode == "inverse_sqrt":
        w = 1.0 / torch.sqrt(freq)
    elif mode == "median":
        w = freq.median() / freq
    else:
        raise ValueError(f"Unknown class-weight mode {mode!r}.")
    return w * num_classes / w.sum()


def cross_entropy_loss(
    logits_h: torch.Tensor,
    logits_v: torch.Tensor,
    k_h: torch.Tensor,
    k_v: torch.Tensor,
    valid_h: Optional[torch.Tensor] = None,
    valid_v: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-edge cross-entropy averaged over horizontal and vertical edges (Eq. 6).

    ``logits_*`` are ``(B, 3, H, W)``; ``k_*`` are ``(B, H, W)`` long class maps.
    ``weight`` is an optional ``(3,)`` per-class weight (see
    :func:`compute_class_weights`) to counteract the severe class imbalance.
    """
    ce_h = F.cross_entropy(logits_h, k_h, weight=weight, reduction="none")  # (B, H, W)
    ce_v = F.cross_entropy(logits_v, k_v, weight=weight, reduction="none")
    return 0.5 * (_masked_mean(ce_h, valid_h) + _masked_mean(ce_v, valid_v))


def _focal_one(
    logits: torch.Tensor, target: torch.Tensor, gamma: float, weight: Optional[torch.Tensor]
) -> torch.Tensor:
    """Per-edge focal loss ``(1 - p_t)^gamma * (-log p_t)`` for one direction."""
    logp = F.log_softmax(logits, dim=1)                       # (B, 3, H, W)
    logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)   # (B, H, W)
    pt = logp_t.exp()
    loss = -((1.0 - pt) ** gamma) * logp_t
    if weight is not None:
        loss = loss * weight[target]                          # optional class balancing
    return loss


def focal_loss(
    logits_h: torch.Tensor,
    logits_v: torch.Tensor,
    k_h: torch.Tensor,
    k_v: torch.Tensor,
    valid_h: Optional[torch.Tensor] = None,
    valid_v: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Multi-class focal loss over horizontal and vertical edges.

    Focal loss down-weights well-classified (easy) edges by ``(1 - p_t)^gamma``,
    concentrating learning on the genuinely hard edges (true cuts and confusable
    boundaries). Unlike static inverse-frequency weighting it does not blanket-
    reward predicting the rare classes, so it is far less prone to the
    over-prediction (false-cut) failure that wrecks unwrapping RMSE.
    ``weight`` is an optional ``(3,)`` class-balancing factor (focal ``alpha``).
    """
    fl_h = _focal_one(logits_h, k_h, gamma, weight)
    fl_v = _focal_one(logits_v, k_v, gamma, weight)
    return 0.5 * (_masked_mean(fl_h, valid_h) + _masked_mean(fl_v, valid_v))


def congruence_loss(
    phi_hat: torch.Tensor, psi: torch.Tensor, align: bool = True
) -> torch.Tensor:
    r"""Congruence loss ``mean(1 - cos(phi_hat - psi))`` (Eq. 7).

    ``phi_hat`` and ``psi`` are ``(B, H, W)`` (or ``(B, 1, H, W)``). When ``align``
    is true (default) the prediction is first reference-aligned to the
    observation (see :func:`utils.align_reference`), so a perfect prediction
    yields exactly zero despite the integrator's arbitrary additive constant.
    """
    if phi_hat.dim() == 4:
        phi_hat = phi_hat.squeeze(1)
    if psi.dim() == 4:
        psi = psi.squeeze(1)
    if align:
        phi_hat = utils.align_reference(phi_hat, psi)
    return (1.0 - torch.cos(phi_hat - psi)).mean()


def curl_loss(g_x: torch.Tensor, g_y: torch.Tensor) -> torch.Tensor:
    r"""Residue (loop-curl) consistency loss ``mean(curl(g)^2) / (2 pi)^2``.

    The loop curl of the corrected gradient field,
    ``curl[i,j] = g_x[i,j] + g_y[i,j+1] - g_x[i+1,j] - g_y[i,j]``, is zero exactly
    when the field is integrable (no residues). A *wrong* cut creates a residue,
    so penalising ``curl^2`` directly punishes inconsistent cuts and drives the
    predicted ambiguities to cancel the observation's residues.

    Unlike the congruence loss this is **local** (a 2x2 loop, no Poisson solve),
    so its gradient is well-conditioned and stable. ``g_x``/``g_y`` are the
    full-resolution padded corrected gradients ``(B, H, W)``; the slicing uses
    only valid edges. Normalised by ``(2 pi)^2`` so a single residue contributes ~1.
    """
    curl = (
        g_x[..., :-1, :-1] + g_y[..., :-1, 1:]
        - g_x[..., 1:, :-1] - g_y[..., :-1, :-1]
    )
    return (curl ** 2).mean() / (utils.TWO_PI ** 2)


def sparsity_loss(
    probs_h: torch.Tensor,
    probs_v: torch.Tensor,
    valid_h: Optional[torch.Tensor] = None,
    valid_v: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Relaxed cut count ``mean(1 - p_{e,0})`` (Eq. 8).

    ``p_{e,0}`` is the probability of the zero-ambiguity class
    (``utils.ZERO_AMBIGUITY_CLASS``). ``probs_*`` are softmax maps ``(B, 3, H, W)``.
    """
    z = utils.ZERO_AMBIGUITY_CLASS
    s_h = 1.0 - probs_h[:, z]                                 # (B, H, W)
    s_v = 1.0 - probs_v[:, z]
    return 0.5 * (_masked_mean(s_h, valid_h) + _masked_mean(s_v, valid_v))


def entropy_loss(
    probs_h: torch.Tensor,
    probs_v: torch.Tensor,
    valid_h: Optional[torch.Tensor] = None,
    valid_v: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean per-edge prediction entropy (TTA confidence-sharpening term).

    Lower entropy = more confident edges; minimised during test-time adaptation.
    """
    e_h = -(probs_h * torch.log(probs_h.clamp_min(eps))).sum(dim=1)   # (B, H, W)
    e_v = -(probs_v * torch.log(probs_v.clamp_min(eps))).sum(dim=1)
    return 0.5 * (_masked_mean(e_h, valid_h) + _masked_mean(e_v, valid_v))


# ---------------------------------------------------------------------------
# Combined objective
# ---------------------------------------------------------------------------
class PhysicsGuidedLoss(nn.Module):
    """Assemble the regime-specific objective from the four terms.

    Parameters
    ----------
    lambda_cong, lambda_sparse, lambda_ent:
        Weights ``lambda1``, ``lambda2``, ``lambda3`` from Eqs. 9-10.
    align_congruence:
        Reference-align ``phi_hat`` to ``psi`` inside the congruence loss.
    """

    def __init__(
        self,
        lambda_cong: float = 1.0,
        lambda_sparse: float = 0.05,
        lambda_ent: float = 1.0,
        lambda_curl: float = 0.0,
        lambda_ktv: float = 0.01,
        lambda_kreg: float = 1.0,
        k_max: int = 16,
        align_congruence: bool = True,
        ce_class_weights: str = "inverse_sqrt",
        cut_loss: str = "ce",
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.lambda_cong = lambda_cong
        self.lambda_sparse = lambda_sparse
        self.lambda_ent = lambda_ent
        self.lambda_curl = lambda_curl
        self.lambda_ktv = lambda_ktv
        self.lambda_kreg = lambda_kreg
        self.k_max = k_max
        self.align_congruence = align_congruence
        self.ce_class_weights = ce_class_weights
        self.cut_loss = cut_loss
        self.focal_gamma = focal_gamma

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
        regime: str = "pretrain",
    ) -> Dict[str, torch.Tensor]:
        """Return a dict of the individual terms plus the regime ``total``.

        ``regime`` selects which terms enter ``total``:

        * ``pretrain`` / ``kshot`` -> ``L_CE + lambda1 L_cong + lambda2 L_sparse``
        * ``tta``                  -> ``L_cong + lambda2 L_sparse + lambda3 L_ent``
        * ``zero_shot``            -> ``L_cong`` (diagnostic only; no parameter update)
        """
        # --- Per-pixel wrap-count classification head (ordinal-hybrid) -----
        if "k_logits" in outputs:
            terms: Dict[str, torch.Tensor] = {}
            logits = outputs["k_logits"]
            ek = outputs["k_reg"]                              # soft expected K (centred)
            tv = (ek[..., :, 1:] - ek[..., :, :-1]).abs().mean() + (ek[..., 1:, :] - ek[..., :-1, :]).abs().mean()
            terms["ktv"] = tv
            # Prediction entropy (the unsupervised TTA signal; always computed).
            logp = F.log_softmax(logits, dim=1)
            terms["ent"] = -(logp.exp() * logp).sum(dim=1).mean()
            if regime == "tta":
                # Unsupervised: minimise entropy (TENT-style) + smoothness prior.
                terms["total"] = terms["ent"] + self.lambda_ktv * terms["ktv"]
                return terms
            if "k_pixel" not in batch:
                raise ValueError("supervised pixel_cls head requires 'k_pixel' labels.")
            kp = batch["k_pixel"]                              # (B,1,H,W) raw integer K
            center = torch.round(kp.mean(dim=(-2, -1), keepdim=True))
            cls = (kp - center + self.k_max).clamp(0, 2 * self.k_max).long().squeeze(1)
            terms["kce"] = F.cross_entropy(logits, cls)
            # Ordinal term: pull the *expected* K to the (centred) integer target,
            # so errors stay bounded to +-1 (fixes pure-CE's large mis-class errors).
            terms["kreg"] = F.smooth_l1_loss(ek, kp - center)
            terms["total"] = terms["kce"] + self.lambda_kreg * terms["kreg"] + self.lambda_ktv * terms["ktv"]
            return terms

        # --- Per-pixel wrap-count regression head -------------------------
        if "k_reg" in outputs:
            terms: Dict[str, torch.Tensor] = {}
            if regime == "tta":
                raise ValueError(
                    "Unsupervised TTA needs a prediction distribution; the 'pixel' "
                    "regression head has none. Use head_type=pixel_cls (entropy) or edge."
                )
            if "k_pixel" not in batch:
                raise ValueError("pixel head requires 'k_pixel' labels.")
            terms["kreg"] = F.smooth_l1_loss(outputs["k_reg"], batch["k_pixel"])
            # Total-variation prior: wrap count is piecewise-constant (jumps by 1
            # at fringe boundaries), so penalise spurious high-frequency wiggle.
            k = outputs["k_reg"]
            tv = (k[..., :, 1:] - k[..., :, :-1]).abs().mean() + (k[..., 1:, :] - k[..., :-1, :]).abs().mean()
            terms["ktv"] = tv
            terms["total"] = terms["kreg"] + self.lambda_ktv * terms["ktv"]
            return terms

        valid_h = batch.get("valid_x")
        valid_v = batch.get("valid_y")
        probs_h, probs_v = outputs["probs_h"], outputs["probs_v"]

        terms: Dict[str, torch.Tensor] = {}
        terms["cong"] = congruence_loss(
            outputs["phi_hat"], batch["wrapped"], align=self.align_congruence
        )
        terms["sparse"] = sparsity_loss(probs_h, probs_v, valid_h, valid_v)
        terms["ent"] = entropy_loss(probs_h, probs_v, valid_h, valid_v)
        terms["curl"] = curl_loss(outputs["g_x"], outputs["g_y"])

        has_labels = "k_x" in batch and "k_y" in batch
        if has_labels:
            weight = compute_class_weights(
                batch["k_x"], batch["k_y"], valid_h, valid_v, mode=self.ce_class_weights
            )
            if self.cut_loss == "focal":
                terms["ce"] = focal_loss(
                    outputs["logits_h"], outputs["logits_v"],
                    batch["k_x"], batch["k_y"], valid_h, valid_v,
                    gamma=self.focal_gamma, weight=weight,
                )
            else:
                terms["ce"] = cross_entropy_loss(
                    outputs["logits_h"], outputs["logits_v"],
                    batch["k_x"], batch["k_y"], valid_h, valid_v, weight=weight,
                )

        if regime in ("pretrain", "kshot"):
            if not has_labels:
                raise ValueError(f"regime={regime!r} requires ambiguity labels (k_x/k_y).")
            total = (
                terms["ce"]
                + self.lambda_cong * terms["cong"]
                + self.lambda_curl * terms["curl"]
                + self.lambda_sparse * terms["sparse"]
            )
        elif regime == "tta":
            total = terms["cong"] + self.lambda_sparse * terms["sparse"] + self.lambda_ent * terms["ent"]
        elif regime == "zero_shot":
            total = terms["cong"]
        else:
            raise ValueError(f"Unknown regime={regime!r}.")

        terms["total"] = total
        return terms
