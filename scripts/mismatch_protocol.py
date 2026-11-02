"""Paired synthetic observations for the cost-mismatch experiment.

This module is deliberately NumPy-only so the observation protocol can be
checked without loading the network, CUDA, or the phase-unwrapping solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


PROTOCOL_VERSION = 3
REGIMES = ("homogeneous", "heterogeneous", "decorrelation_patches", "mixed")


def snaphu_default_coherence_threshold(nlooks: float) -> float:
    """Default SMOOTH/DEFO rho cutoff in SNAPHU 2.0.7.

    rho0 = 1.3 / NCORRLOOKS + 0.14; correlations below 1.2 * rho0
    are replaced by zero. This diagnostic assumes unmodified SNAPHU defaults.
    """
    if float(nlooks) < 1.0:
        raise ValueError("SNAPHU equivalent coherence looks must be >= 1")
    return 1.2 * (1.3 / float(nlooks) + 0.14)


@dataclass(frozen=True)
class SceneLatents:
    real_noise: np.ndarray
    imag_noise: np.ndarray
    patch_centers: np.ndarray
    patch_radii: np.ndarray


def make_scene_latents(shape: Tuple[int, int], seed: int, scene_index: int) -> SceneLatents:
    """One deterministic latent draw, shared by every regime and severity."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(scene_index)]))
    h, w = shape
    return SceneLatents(
        real_noise=rng.standard_normal(shape).astype(np.float32),
        imag_noise=rng.standard_normal(shape).astype(np.float32),
        patch_centers=rng.uniform(0.0, 1.0, size=(4, 2)),
        patch_radii=rng.uniform(0.08, 0.18, size=4),
    )


def coherence_field(shape: Tuple[int, int], regime: str, severity: float,
                    latents: SceneLatents) -> np.ndarray:
    """Known coherence field; all regimes coincide at zero severity.

    Heterogeneity and localized decorrelation increase continuously with
    severity. Patch locations/radii are fixed for each scene, allowing paired
    comparisons across severities rather than redrawing the spatial pattern.
    """
    if regime not in REGIMES:
        raise ValueError(f"Unknown mismatch regime: {regime!r}")
    s = float(severity)
    if not 0.0 <= s <= 1.0:
        raise ValueError("severity must be in [0, 1]")
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    base = 0.92 - 0.55 * s
    gamma = np.full(shape, base, dtype=np.float64)
    if regime in ("heterogeneous", "mixed"):
        gamma += 0.25 * s * np.sin(2 * np.pi * xx / w) * np.cos(2 * np.pi * yy / h)
    if regime in ("decorrelation_patches", "mixed"):
        for (cy, cx), radius_fraction in zip(latents.patch_centers, latents.patch_radii):
            radius = radius_fraction * min(h, w)
            mask = (yy - cy * (h - 1)) ** 2 + (xx - cx * (w - 1)) ** 2 <= radius ** 2
            gamma[mask] -= s * max(base - 0.03, 0.0)
    if regime == "mixed":
        gamma -= 0.10 * s * (1 + np.sin(2 * np.pi * (3 * xx / w + 2 * yy / h)))
    return np.clip(gamma, 0.02, 0.98).astype(np.float32)


def noisy_wrapped(phi: np.ndarray, gamma: np.ndarray,
                  latents: SceneLatents, looks: int = 1) -> np.ndarray:
    """Circular complex-Gaussian observation with *shared* standardized noise.

    This is a controlled stress-test model, not a claim to reproduce all SAR
    interferometric statistics. `looks` is fixed across compared methods.
    """
    if int(looks) < 1:
        raise ValueError("looks must be a positive integer")
    signal = gamma * np.exp(1j * phi)
    sigma = np.sqrt(np.maximum(0.0, 1.0 - gamma.astype(np.float64) ** 2) / (2 * int(looks)))
    noise = sigma * (latents.real_noise + 1j * latents.imag_noise)
    return np.angle(signal + noise).astype(np.float32)
