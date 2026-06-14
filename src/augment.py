"""Augmentation for sim->real robustness.

This module addresses the *distribution shift* that caps zero-shot accuracy on
the real TanDEM-X subset (``test_real``): a prior trained on InSAR-DLPU's
simulated noise is *confidently wrong* on real decorrelation statistics, which
both lowers the OOD floor and starves entropy-based TTA of gradient.

Two roles:

1. **Train-time raw-array augmentation** (:class:`PhaseAugmentor`). Applied to
   the ``(wrapped, absolute)`` pair *before* :func:`dataset.build_sample`, so all
   labels (``k_pixel``, ``k_x``, ``k_y``) are recomputed exactly from the
   augmented pair. It (a) preserves the original wrapped noise via the residual
   ``r = W(psi - phi)``, (b) jitters fringe density by scaling ``phi`` about its
   mean, (c) injects extra, spatially *patchy* decorrelation noise, and (d)
   applies a D4 symmetry. This widens the training noise distribution toward the
   real regime -> raises the OOD floor and de-confidences the prior.

2. **TTA-time augmentation-consistency** (:func:`augment_batch_view` + the D4
   tensor helpers). The unwrapped reconstruction ``phi_hat`` is *physically
   invariant* to a D4 symmetry of the input and to added wrapped-phase noise
   (both still reconstruct the same absolute field up to a global constant).
   Enforcing that invariance is a genuine unsupervised signal that has gradient
   even where the prediction entropy is flat (confidently wrong) -- exactly the
   failure mode of pure TENT under large shift.

Note: the TTA D4 set is restricted to *involutions* (identity, rot180, fliplr,
flipud, transpose) so the inverse equals the forward op -- no inverse-mapping
bugs. Transpose is only used on square tiles.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from . import utils


# ===========================================================================
# Train-time raw-array augmentation
# ===========================================================================
def _wrap_np(x: np.ndarray) -> np.ndarray:
    """Wrap to (-pi, pi]; numpy mirror of :func:`utils.wrap_to_pi`."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _apply_d4_np(arr: np.ndarray, op: int) -> np.ndarray:
    """Apply one of 8 D4 symmetries to a 2-D array (same op used for every field)."""
    if op == 0:
        return arr
    if op == 1:
        return np.rot90(arr, 1)
    if op == 2:
        return np.rot90(arr, 2)
    if op == 3:
        return np.rot90(arr, 3)
    if op == 4:
        return np.fliplr(arr)
    if op == 5:
        return np.flipud(arr)
    if op == 6:
        return arr.T
    return np.fliplr(np.rot90(arr, 1))  # op == 7


class PhaseAugmentor:
    """Physically-grounded augmentation of a ``(wrapped, absolute)`` pair.

    Parameters
    ----------
    prob:
        Probability of augmenting a given sample (else returned unchanged).
    d4:
        Apply a random D4 symmetry (geometry only; phase values unchanged).
    scale:
        ``(lo, hi)`` range for fringe-density scaling of the absolute phase about
        its mean. ``None`` disables. Keep modest so the wrap count stays within
        ``k_max`` (with DLPU's ~22 max and hi=1.2 the peak is ~27 < 32).
    noise_std:
        ``(lo, hi)`` range for the std (radians) of the *extra* decorrelation
        noise added on top of the preserved original noise.
    patchy:
        Modulate the extra noise by a smooth low-frequency field in ``[0, 1]`` so
        it forms spatially-correlated low-coherence patches (more realistic than
        i.i.d. noise).
    patch_scale:
        Gaussian-blur sigma (pixels) for the patchy field.
    """

    def __init__(
        self,
        prob: float = 0.9,
        d4: bool = True,
        scale: Optional[Tuple[float, float]] = (0.85, 1.20),
        noise_std: Tuple[float, float] = (0.0, 0.6),
        patchy: bool = True,
        patch_scale: float = 32.0,
        seed: Optional[int] = None,
    ) -> None:
        self.prob = float(prob)
        self.d4 = bool(d4)
        self.scale = tuple(scale) if scale is not None else None
        self.noise_std = tuple(noise_std)
        self.patchy = bool(patchy)
        self.patch_scale = float(patch_scale)
        # Per-worker RNG; seeded lazily so DataLoader workers diverge.
        self._seed = seed
        self._rng: Optional[np.random.Generator] = None

    def _get_rng(self) -> np.random.Generator:
        if self._rng is None:
            # Mix in the worker id so forked workers don't share a stream.
            info = torch.utils.data.get_worker_info()
            wid = 0 if info is None else int(info.id)
            base = 0 if self._seed is None else int(self._seed)
            self._rng = np.random.default_rng(base + 1009 * (wid + 1))
        return self._rng

    def _smooth_field(self, shape: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
        f = rng.standard_normal(shape).astype(np.float32)
        try:
            from scipy.ndimage import gaussian_filter

            f = gaussian_filter(f, sigma=self.patch_scale)
        except Exception:
            pass  # fall back to white noise if scipy is unavailable
        f = f - float(f.min())
        m = float(f.max())
        return f / (m + 1e-6)

    def __call__(
        self,
        wrapped: np.ndarray,
        absolute: Optional[np.ndarray],
        coherence: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        # No labels -> we cannot regenerate a consistent wrap; only geometry is safe.
        rng = self._get_rng()
        if absolute is None:
            if self.d4 and rng.random() <= self.prob:
                op = int(rng.integers(0, 8))
                wrapped = np.ascontiguousarray(_apply_d4_np(wrapped, op))
                if coherence is not None:
                    coherence = np.ascontiguousarray(_apply_d4_np(coherence, op))
            return wrapped, absolute, coherence

        if rng.random() > self.prob:
            return wrapped, absolute, coherence

        psi = wrapped.astype(np.float32, copy=False)
        phi = absolute.astype(np.float32, copy=False)
        # Preserve the original wrapped noise as a residual we can re-wrap with.
        r = _wrap_np(psi - phi)

        # (b) fringe-density jitter: scale the absolute field about its mean.
        if self.scale is not None:
            s = float(rng.uniform(*self.scale))
            m = float(phi.mean())
            phi = (phi - m) * s + m

        # (c) extra (patchy) decorrelation noise on top of the preserved noise.
        smax = float(rng.uniform(*self.noise_std))
        if smax > 1e-4:
            noise = rng.standard_normal(phi.shape).astype(np.float32)
            if self.patchy:
                noise = noise * self._smooth_field(phi.shape, rng)
            r = r + smax * noise

        psi = _wrap_np(phi + r)

        # (d) D4 symmetry applied identically to every field.
        if self.d4:
            op = int(rng.integers(0, 8))
            psi = _apply_d4_np(psi, op)
            phi = _apply_d4_np(phi, op)
            if coherence is not None:
                coherence = np.ascontiguousarray(_apply_d4_np(coherence, op))

        return np.ascontiguousarray(psi), np.ascontiguousarray(phi), coherence


def build_augmentor(cfg: Any, seed: Optional[int] = None) -> Optional[PhaseAugmentor]:
    """Construct a :class:`PhaseAugmentor` from a ``data.augment`` config block.

    Returns ``None`` when augmentation is absent or disabled. Recognised keys:
    ``enabled, prob, d4, scale, noise_std, patchy, patch_scale``.
    """
    if cfg is None:
        return None
    get = (lambda k, d=None: cfg.get(k, d)) if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    if not bool(get("enabled", False)):
        return None
    scale = get("scale", [0.85, 1.20])
    noise = get("noise_std", [0.0, 0.6])
    return PhaseAugmentor(
        prob=float(get("prob", 0.9)),
        d4=bool(get("d4", True)),
        scale=(tuple(scale) if scale is not None else None),
        noise_std=tuple(noise),
        patchy=bool(get("patchy", True)),
        patch_scale=float(get("patch_scale", 32.0)),
        seed=seed,
    )


# ===========================================================================
# TTA-time augmentation-consistency (tensor helpers)
# ===========================================================================
#: D4 involutions: each is its own inverse, so invert == apply. Transpose (6)
#: is only valid on square tiles and is filtered out otherwise.
_TTA_INVOLUTIONS: Tuple[int, ...] = (2, 4, 5, 6)


def apply_d4_torch(x: torch.Tensor, op: int) -> torch.Tensor:
    """Apply a D4 involution to a ``(B, C, H, W)`` tensor (op in :data:`_TTA_INVOLUTIONS`)."""
    if op == 0:
        return x
    if op == 2:
        return torch.rot90(x, 2, dims=(-2, -1))
    if op == 4:
        return torch.flip(x, dims=(-1,))
    if op == 5:
        return torch.flip(x, dims=(-2,))
    if op == 6:
        return x.transpose(-2, -1)
    raise ValueError(f"op {op} is not a supported TTA involution")


def sample_tta_ops(shape: Tuple[int, ...], rng: np.random.Generator, n: int = 1) -> List[int]:
    """Sample ``n`` non-identity D4 involutions valid for a tensor of ``shape``."""
    h, w = int(shape[-2]), int(shape[-1])
    ops = [o for o in _TTA_INVOLUTIONS if o != 6 or h == w]
    return [int(rng.choice(ops)) for _ in range(n)]


def augment_batch_view(
    batch: Dict[str, torch.Tensor],
    op: int,
    noise_std: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Build an augmented view of ``batch`` for consistency TTA.

    Adds optional wrapped-phase noise, applies the D4 involution ``op`` to the
    wrapped field, and regenerates ``input`` from it (preserving the 2- or
    4-channel encoding). Other tensors are passed through unchanged -- for the
    pixel head ``phi_hat`` depends only on ``wrapped`` + the predicted ``K``, and
    on DLPU the integrator weights are uniform (D4-invariant), so this view is
    physically consistent for the heads in use.
    """
    aug = dict(batch)
    psi = batch["wrapped"]  # (B, 1, H, W)
    if noise_std and noise_std > 0:
        noise = torch.randn(psi.shape, device=psi.device, dtype=psi.dtype, generator=generator)
        psi = utils.wrap_to_pi(psi + noise_std * noise)
    psi = apply_d4_torch(psi, op)
    aug["wrapped"] = psi
    include_grad = batch["input"].shape[1] == 4
    aug["input"] = utils.encode_input(psi.squeeze(1), include_grad=include_grad)
    return aug


def consistency_loss(
    phi_student: torch.Tensor,
    phi_teacher: torch.Tensor,
    op: int,
) -> torch.Tensor:
    """Augmentation-consistency between two ``phi_hat`` maps.

    The student was predicted on the D4-``op`` (and possibly noised) view; map it
    back with the same involution, remove the global-constant ambiguity from both
    fields, and compare. Smooth-L1 is robust to the occasional outlier pixel.
    """
    phi_s = apply_d4_torch(phi_student, op)        # involution: inverse == forward
    phi_s = utils.zero_mean(phi_s)
    phi_t = utils.zero_mean(phi_teacher).detach()
    return torch.nn.functional.smooth_l1_loss(phi_s, phi_t)
