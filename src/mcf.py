"""Minimum-cost-flow (MCF) phase unwrapping -- the integer solver under SNAPHU.

This is the rigorous, *residue-free-by-construction* core of the planned method
(Costantini 1998; Chen & Zebker 2001). It solves for the integer ambiguity field
``k`` on the grid edges as a minimum-cost flow on the dual graph, where the
loop *residues* of the wrapped phase are the flow sources/sinks and the per-edge
*costs* are the prior (uniform, coherence-derived, or -- the eventual
contribution -- learned and end-to-end differentiable).

Convention (matches src/utils edge layout):
  psi               : (M, N) wrapped phase
  dx[i,j] = W(psi[i,j+1]-psi[i,j])   horizontal edge gradient, shape (M, N-1)
  dy[i,j] = W(psi[i+1,j]-psi[i,j])   vertical   edge gradient, shape (M-1, N)
  residue R[i,j] (plaquette i,j, shape (M-1,N-1)) =
      round( (dx[i,j] + dy[i,j+1] - dx[i+1,j] - dy[i,j]) / 2pi )

The unwrapped gradients gx = dx + 2pi*kx, gy = dy + 2pi*ky are made curl-free by
choosing integer kx,ky that satisfy, at every plaquette,
      kx[i,j] + ky[i,j+1] - kx[i+1,j] - ky[i,j] = -R[i,j].
We minimise sum(wx*|kx|) + sum(wy*|ky|) -> a min-cost flow. The result is then
integrated (curl-free, so path-independent) to recover the unwrapped phase.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

TWO_PI = 2.0 * np.pi
GROUND = ("ground",)  # sentinel node id for the boundary "earth"


def wrap(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % TWO_PI - np.pi


def wrapped_gradients(psi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dx = wrap(psi[:, 1:] - psi[:, :-1])   # (M, N-1)
    dy = wrap(psi[1:, :] - psi[:-1, :])   # (M-1, N)
    return dx, dy


def residues(psi: np.ndarray) -> np.ndarray:
    """Loop-curl residues at each 2x2 plaquette, shape (M-1, N-1), in {-1,0,+1}."""
    dx, dy = wrapped_gradients(psi)
    # dx[i,j] - dx[i+1,j] : top minus bottom horizontal edges  (use rows 0..M-2)
    curl = dx[:-1, :] + dy[:, 1:] - dx[1:, :] - dy[:, :-1]
    return np.round(curl / TWO_PI).astype(np.int64)


def aligned_gtk_floor(psi: np.ndarray, phi: np.ndarray, n_offsets: int = 128) -> float:
    """Lower bound on mean-aligned RMSE over every congruent unwrapping of ``psi``.

    Any congruent field is ``phi_hat = psi + 2*pi*K``. The evaluation metric
    subtracts the mean residual, i.e. it grants a free global offset ``c``, so the
    optimal integer field must be chosen jointly with that offset::

        floor = min_c  std( wrap( (phi - psi) - c ) )

    which is the closed form of
    ``min_c aligned_rmse(psi + 2*pi*round((phi - psi - c)/2*pi), phi)``.

    The earlier definition fixed ``c = 0`` (per-pixel nearest integer). That
    minimises the *unaligned* residual and is therefore NOT a bound for the
    mean-aligned metric -- methods beat it on ~1 scene in 4 (see
    ``audit/DIAGNOSIS.md`` sec.3-I1). Near ``+/-pi`` the per-pixel rounding splits
    neighbouring pixels between ``+pi`` and ``-pi``, giving a bimodal, high-variance
    residual; committing one integer across a region is lower-variance once the
    mean is removed.
    """
    e = np.asarray(phi, dtype=np.float64) - np.asarray(psi, dtype=np.float64)
    return float(min(wrap(e - c).std()
                     for c in np.linspace(0.0, TWO_PI, int(n_offsets), endpoint=False)))


def quality_map(psi: np.ndarray, win: int = 5) -> np.ndarray:
    """Phase-derivative-variance quality -> pseudo-coherence in [0,1].

    Classical no-coherence quality estimate (Ghiglia & Pritt): the local variance
    of the wrapped gradients is high in noisy/low-coherence regions. We return
    ``1 - normalised variance`` so it behaves like a coherence map and plugs into
    the MCF cost (high quality -> high cost -> don't cut there). This is the
    hand-designed statistical cost that the learned network will replace.
    """
    from scipy.ndimage import uniform_filter

    M, N = psi.shape
    dx, dy = wrapped_gradients(psi)
    gx = np.zeros((M, N)); gx[:, :-1] = dx
    gy = np.zeros((M, N)); gy[:-1, :] = dy

    def local_var(g):
        m = uniform_filter(g, win)
        m2 = uniform_filter(g * g, win)
        return np.maximum(m2 - m * m, 0.0)

    q = local_var(gx) + local_var(gy)
    q = q / (q.max() + 1e-6)
    return 1.0 - q


def integrate_gradients(dx: np.ndarray, dy: np.ndarray, ref: float = 0.0) -> np.ndarray:
    """Reconstruct a field from curl-free gradients by cumulative summation.

    phi[0,0]=ref; first column from dy[:,0], each row from dx[i,:]. Path-independent
    only when curl-free (which holds after MCF), so any spanning path is valid.
    """
    M = dx.shape[0]
    N = dx.shape[1] + 1
    phi = np.zeros((M, N), dtype=np.float64)
    phi[0, 1:] = np.cumsum(dx[0, :])           # top row
    phi[1:, 0] = np.cumsum(dy[:, 0])           # first column
    # fill remaining by integrating dx along each row from the first column
    for i in range(1, M):
        phi[i, 1:] = phi[i, 0] + np.cumsum(dx[i, :])
    return phi + ref


def _build_dual(M: int, N: int, R: np.ndarray,
                cpx: np.ndarray, cnx: np.ndarray,
                cpy: np.ndarray, cny: np.ndarray, cost_scale: int):
    """Build the dual min-cost-flow problem (solver-agnostic), asymmetric costs.

    Each primal edge ``e`` has coefficient +1 in one plaquette ("plus") and -1 in
    the adjacent one ("minus"); boundary edges use a single GROUND node. The
    constraint  sum(coef * k) = -R  is flow conservation when an arc minus->plus
    carries f_e = k_e, i.e. (inflow - outflow) = -R at every node.

    Costs are *directional*: positive flow (k_e=+1, the minus->plus arc) costs
    ``cpx/cpy``; negative flow (k_e=-1, the plus->minus arc) costs ``cnx/cny``.
    This lets a network's P(k=+1) and P(k=-1) set the two directions separately.
    Each stored arc is (minus_id, plus_id, cost_pos_int, cost_neg_int, key).

    Returns: n_nodes, supply (list, supply = outflow-inflow = R), arcs.
    """
    Pr, Pc = M - 1, N - 1
    ground = Pr * Pc
    n_nodes = Pr * Pc + 1

    def nid(i, j):
        return i * Pc + j

    supply = [0] * n_nodes
    total = 0
    for i in range(Pr):
        for j in range(Pc):
            supply[nid(i, j)] = int(R[i, j])
            total += int(R[i, j])
    supply[ground] = -total

    def pm_x(r, c):
        plus = nid(r, c) if r <= Pr - 1 else ground       # top edge of plaquette (r,c)
        minus = nid(r - 1, c) if r - 1 >= 0 else ground   # bottom edge of (r-1,c)
        return plus, minus

    def pm_y(r, c):
        plus = nid(r, c - 1) if c - 1 >= 0 else ground    # right edge of (r,c-1)
        minus = nid(r, c) if c <= Pc - 1 else ground      # left edge of (r,c)
        return plus, minus

    def ci(w):
        return max(int(round(float(w) * cost_scale)), 1)

    arcs = []  # (minus_id, plus_id, cost_pos_int, cost_neg_int, key)
    for r in range(M):
        for c in range(N - 1):
            plus, minus = pm_x(r, c)
            if plus != minus:
                arcs.append((minus, plus, ci(cpx[r, c]), ci(cnx[r, c]), ("x", r, c)))
    for r in range(M - 1):
        for c in range(N):
            plus, minus = pm_y(r, c)
            if plus != minus:
                arcs.append((minus, plus, ci(cpy[r, c]), ci(cny[r, c]), ("y", r, c)))
    return n_nodes, supply, arcs


def _flows_ortools(n_nodes, supply, arcs, big):
    """Solve with OR-Tools SimpleMinCostFlow (exact; handles parallel arcs)."""
    from ortools.graph.python import min_cost_flow as ortools_mcf

    smcf = ortools_mcf.SimpleMinCostFlow()
    fwd, bwd = [], []
    for minus, plus, cpos, cneg, _ in arcs:
        fwd.append(smcf.add_arc_with_capacity_and_unit_cost(minus, plus, big, cpos))
        bwd.append(smcf.add_arc_with_capacity_and_unit_cost(plus, minus, big, cneg))
    for node in range(n_nodes):
        smcf.set_node_supply(node, supply[node])
    status = smcf.solve()
    if status != smcf.OPTIMAL:
        raise RuntimeError(f"OR-Tools min-cost-flow status={status}")
    return [smcf.flow(fwd[i]) - smcf.flow(bwd[i]) for i in range(len(arcs))]


def _flows_networkx(n_nodes, supply, arcs, big):
    """Solve with networkx (validated reference fallback).

    networkx ``DiGraph`` cannot hold parallel arcs, which only collide at the few
    boundary/corner edges sharing the GROUND node; we keep the first (a negligible
    boundary approximation). OR-Tools is exact -- prefer it for production.
    """
    import networkx as nx

    G = nx.DiGraph()
    for node in range(n_nodes):
        G.add_node(node, demand=-supply[node])             # inflow-outflow = -R
    for minus, plus, cpos, cneg, _ in arcs:
        if not G.has_edge(minus, plus):
            G.add_edge(minus, plus, weight=cpos, capacity=big)
        if not G.has_edge(plus, minus):
            G.add_edge(plus, minus, weight=cneg, capacity=big)
    flow = nx.min_cost_flow(G)
    return [flow.get(mi, {}).get(pl, 0) - flow.get(pl, {}).get(mi, 0)
            for mi, pl, _, _, _ in arcs]


def _solve_min_cost_flow(M: int, N: int, R: np.ndarray,
                         cpx: np.ndarray, cnx: np.ndarray,
                         cpy: np.ndarray, cny: np.ndarray,
                         cost_scale: int = 1000,
                         backend: str = "auto") -> Tuple[np.ndarray, np.ndarray]:
    """Solve the dual MCF; return integer ambiguity fields (kx, ky).

    backend: "auto" (OR-Tools if importable, else networkx), "ortools", "networkx".
    """
    n_nodes, supply, arcs = _build_dual(M, N, R, cpx, cnx, cpy, cny, cost_scale)
    big = M * N + 10

    if backend in ("auto", "ortools"):
        try:
            flows = _flows_ortools(n_nodes, supply, arcs, big)
        except ImportError:
            if backend == "ortools":
                raise
            flows = _flows_networkx(n_nodes, supply, arcs, big)
    else:
        flows = _flows_networkx(n_nodes, supply, arcs, big)

    kx = np.zeros((M, N - 1), dtype=np.int64)
    ky = np.zeros((M - 1, N), dtype=np.int64)
    for (minus, plus, cpos, cneg, key), f in zip(arcs, flows):
        kind, r, c = key
        if kind == "x":
            kx[r, c] = f
        else:
            ky[r, c] = f
    return kx, ky


# ---------------------------------------------------------------------------
# Cost construction (the hook where learning plugs in)
# ---------------------------------------------------------------------------
def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-12)


def edge_costs_from_logits(logits6: np.ndarray, eps: float = 1e-3,
                           clip: float = 8.0):
    """Per-edge MCF costs from a network's per-edge ambiguity logits.

    ``logits6`` is ``(6, H, W)`` (single image): channels 0:3 are the horizontal
    edge classes ``(k=-1, 0, +1)``, 3:6 the vertical (matching utils edge layout).
    Cost of a +1 cut = ``-log P(k=+1)``, of a -1 cut = ``-log P(k=-1)`` -> cheap to
    cut where the network expects a cut. Returns ``(cpx, cnx, cpy, cny)`` with
    shapes ``(H, W-1)`` and ``(H-1, W)`` (valid edges; last col/row dropped).
    """
    logits6 = np.asarray(logits6, dtype=np.float64)
    if logits6.shape[0] != 6:
        raise ValueError(f"expected (6,H,W) edge logits, got {logits6.shape}")
    ph = _softmax(logits6[0:3], axis=0)   # (3, H, W) horizontal
    pv = _softmax(logits6[3:6], axis=0)   # (3, H, W) vertical

    def cost(p):
        return np.clip(-np.log(p + eps), 0.0, clip)

    cpx = cost(ph[2, :, :-1])             # P(k=+1) on horizontal edges (H, W-1)
    cnx = cost(ph[0, :, :-1])             # P(k=-1)
    cpy = cost(pv[2, :-1, :])             # vertical edges (H-1, W)
    cny = cost(pv[0, :-1, :])
    return cpx, cnx, cpy, cny


def uncertainty_from_logits(logits6: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Per-pixel unwrapping reliability from the edge head's ambiguity distributions.

    The Shannon entropy of each per-edge ``P(k_e)`` distribution measures how
    unsure the network is whether/where to cut; averaging the incident horizontal
    and vertical edge entropies gives a per-pixel uncertainty map ``(H, W)`` (nats,
    in [0, ln 3]). High values flag where the unwrapping is least trustworthy --
    a statistically-grounded confidence product alongside the unwrapped phase.
    """
    logits6 = np.asarray(logits6, dtype=np.float64)
    ph = _softmax(logits6[0:3], axis=0)   # (3, H, W) horizontal
    pv = _softmax(logits6[3:6], axis=0)   # (3, H, W) vertical
    ent_h = -(ph * np.log(ph + eps)).sum(axis=0)   # (H, W)
    ent_v = -(pv * np.log(pv + eps)).sum(axis=0)
    return 0.5 * (ent_h + ent_v)


# ---------------------------------------------------------------------------
# Public unwrap entry point
# ---------------------------------------------------------------------------
def unwrap_mcf(psi: np.ndarray,
               wx: Optional[np.ndarray] = None,
               wy: Optional[np.ndarray] = None,
               coherence: Optional[np.ndarray] = None,
               edge_costs: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
               coh_weight: float = 0.0,
               backend: str = "auto") -> np.ndarray:
    """Unwrap ``psi`` via minimum-cost flow. Residue-free by construction.

    Cost precedence (lower cost = cheaper to place a cut there):
      * ``edge_costs=(cpx, cnx, cpy, cny)`` -- directional learned (negative
        log-likelihood) per-edge costs from :func:`edge_costs_from_logits`.
        If ``coherence`` is also given with ``coh_weight>0``, the costs become a
        COMBINED maximum-likelihood cost ``learned_NLL + coh_weight * coh_cost``
        (learned where coherence is uninformative, physical where it is) -- a
        learned+statistical cost for the network-flow solver.
      * ``wx``/``wy`` -- symmetric per-edge costs supplied directly.
      * ``coherence`` (alone) -- symmetric cost = mean endpoint coherence.
      * none -- unit costs (Costantini L1).
    """
    psi = np.asarray(psi, dtype=np.float64)
    M, N = psi.shape
    dx, dy = wrapped_gradients(psi)
    R = residues(psi)

    if edge_costs is not None:
        cpx, cnx, cpy, cny = (np.asarray(a, dtype=np.float64).copy() for a in edge_costs)
        if coherence is not None and coh_weight > 0:
            coh = np.asarray(coherence, dtype=np.float64)
            cohx = 0.5 * (coh[:, 1:] + coh[:, :-1])       # (M, N-1)
            cohy = 0.5 * (coh[1:, :] + coh[:-1, :])       # (M-1, N)
            cpx += coh_weight * cohx; cnx += coh_weight * cohx
            cpy += coh_weight * cohy; cny += coh_weight * cohy
    else:
        if wx is None or wy is None:
            if coherence is not None:
                coh = np.asarray(coherence, dtype=np.float64)
                wx = 0.5 * (coh[:, 1:] + coh[:, :-1]) + 1e-3   # (M, N-1)
                wy = 0.5 * (coh[1:, :] + coh[:-1, :]) + 1e-3   # (M-1, N)
            else:
                wx = np.ones((M, N - 1))
                wy = np.ones((M - 1, N))
        cpx = cnx = wx   # symmetric
        cpy = cny = wy

    if R.any():
        kx, ky = _solve_min_cost_flow(M, N, R, cpx, cnx, cpy, cny, backend=backend)
    else:
        kx = np.zeros((M, N - 1), dtype=np.int64)
        ky = np.zeros((M - 1, N), dtype=np.int64)

    gx = dx + TWO_PI * kx
    gy = dy + TWO_PI * ky
    return integrate_gradients(gx, gy, ref=psi[0, 0])
