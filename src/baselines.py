"""Classical unwrapping baselines for comparison.

The one that matters for a Zebker-lab-aligned paper is **SNAPHU** (Chen & Zebker
2001) -- the standard statistical-cost network-flow unwrapper. We call the real
SNAPHU via the ``snaphu`` Python wrapper (``pip install snaphu``), feeding it the
wrapped phase as a unit-amplitude interferogram plus a correlation map. DLPU has
no coherence band, so we pass a phase-derivative-variance pseudo-coherence (the
same information our stat-cost MCF uses) for a fair comparison.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import mcf


def quality_guided_unwrap(psi: np.ndarray) -> np.ndarray:
    """Quality-guided / path-following unwrap (Herraez et al. 2002), via scikit-image.
    A standard non-network-flow baseline; NOT residue-free (errors propagate along
    the path). psi: wrapped phase (H, W). Returns unwrapped phase (H, W)."""
    from skimage.restoration import unwrap_phase
    return np.asarray(unwrap_phase(np.asarray(psi, dtype=np.float64)), dtype=np.float64)


def ls_unwrap(psi: np.ndarray) -> np.ndarray:
    """Least-squares (Ghiglia-Romero) unwrap: DCT-integrate the wrapped gradients.
    The classic L2 relaxation; smears 2pi errors across the scene."""
    import torch
    from .integrator import poisson_dct_solve
    psi = np.asarray(psi, dtype=np.float64)
    gx = mcf.wrap(psi[:, 1:] - psi[:, :-1])
    gy = mcf.wrap(psi[1:, :] - psi[:-1, :])
    phi = poisson_dct_solve(torch.from_numpy(gx)[None], torch.from_numpy(gy)[None])
    return phi[0].numpy()


def goldstein_unwrap(psi: np.ndarray, max_box: int = 40) -> np.ndarray:
    """Goldstein-Zebker-Werner (1988) residue branch-cut unwrapping.

    Greedy box-growing connects opposite-charge residues (and residues to the
    border) with branch cuts; the phase is then path-followed (flood-filled)
    without crossing cuts. The classic residue method; Costantini's MCF (our
    unit-cost variant) is its globally-optimal successor. Pure NumPy/Python, so
    practical for benchmark-sized scenes (<=~512); large scenes are slow.
    """
    from collections import deque
    psi = np.asarray(psi, dtype=np.float64)
    M, N = psi.shape
    R = mcf.residues(psi).astype(int)             # (M-1, N-1) plaquette charges
    cut = np.zeros((M, N), dtype=bool)
    bal = R == 0

    def mark(r0, c0, r1, c1):
        k = max(abs(r1 - r0), abs(c1 - c0)) + 1
        for r, c in zip(np.round(np.linspace(r0, r1, k)).astype(int),
                        np.round(np.linspace(c0, c1, k)).astype(int)):
            cut[r, c] = True
            if r + 1 < M:
                cut[r + 1, c] = True
            if c + 1 < N:
                cut[r, c + 1] = True

    for ri, ci in zip(*np.where(R != 0)):
        if bal[ri, ci]:
            continue
        charge = int(R[ri, ci]); bal[ri, ci] = True
        for bs in range(1, max_box + 1):
            if charge == 0:
                break
            if ri - bs <= 0 or ci - bs <= 0 or ri + bs >= M - 2 or ci + bs >= N - 2:
                d = [ri, (M - 2) - ri, ci, (N - 2) - ci]; k = int(np.argmin(d))
                tgt = [(0, ci), (M - 2, ci), (ri, 0), (ri, N - 2)][k]
                mark(ri, ci, tgt[0], tgt[1]); charge = 0; break
            rlo, rhi, clo, chi = ri - bs, ri + bs, ci - bs, ci + bs
            sub = R[rlo:rhi + 1, clo:chi + 1]
            for y, x in zip(*np.where(sub != 0)):
                rr, cc = rlo + y, clo + x
                if (rr, cc) == (ri, ci) or bal[rr, cc]:
                    continue
                mark(ri, ci, rr, cc); charge += int(R[rr, cc]); bal[rr, cc] = True
                if charge == 0:
                    break

    phi = np.zeros((M, N)); seen = np.zeros((M, N), bool)
    for sy, sx in np.argwhere(~cut):
        if seen[sy, sx]:
            continue
        phi[sy, sx] = psi[sy, sx]; seen[sy, sx] = True
        dq = deque([(sy, sx)])
        while dq:
            y, x = dq.popleft()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < M and 0 <= nx < N and not seen[ny, nx] and not cut[ny, nx]:
                    phi[ny, nx] = phi[y, x] + mcf.wrap(psi[ny, nx] - psi[y, x])
                    seen[ny, nx] = True; dq.append((ny, nx))
    for y, x in np.argwhere(cut & ~seen):           # cut pixels: best-effort from a neighbor
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < M and 0 <= nx < N and seen[ny, nx]:
                phi[y, x] = phi[ny, nx] + mcf.wrap(psi[y, x] - psi[ny, nx]); seen[y, x] = True
                break
    return phi


def snaphu_available() -> bool:
    try:
        import snaphu  # noqa: F401
        return True
    except Exception:
        return False


def snaphu_unwrap(psi: np.ndarray,
                  coherence: Optional[np.ndarray] = None,
                  nlooks: float = 1.0,
                  cost: str = "smooth") -> np.ndarray:
    """Unwrap with the real SNAPHU (Chen & Zebker).

    psi: wrapped phase (H, W). coherence: optional [0,1] correlation; if None a
    phase-derivative-variance pseudo-coherence is used. Returns unwrapped phase
    (H, W) float64, mean-aligned externally for RMSE.
    """
    try:
        import snaphu
    except ImportError as e:
        raise ImportError(
            "SNAPHU baseline needs the 'snaphu' package: pip install snaphu"
        ) from e

    psi = np.asarray(psi, dtype=np.float64)
    igram = np.exp(1j * psi).astype(np.complex64)
    if coherence is None:
        corr = mcf.quality_map(psi).astype(np.float32)
    else:
        corr = np.clip(np.asarray(coherence, dtype=np.float32), 0.0, 1.0)
    corr = np.clip(corr, 1e-3, 1.0)

    unw, _ = snaphu.unwrap(igram, corr, nlooks=float(nlooks), cost=cost, init="mcf")
    return np.asarray(unw, dtype=np.float64)
