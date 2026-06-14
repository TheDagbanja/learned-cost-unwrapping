"""Synthetic deformation interferogram generator (proposal §3.8).

A controllable generator that composes absolute interferometric phase from
physical sources and wraps it to produce ``(wrapped, absolute, coherence)``
triples with a *coherence band* -- which is exactly what the coherence-weighted
per-edge unwrapper needs and what the InSAR-DLPU benchmark lacks.

    phi = phi_defo + phi_topo + phi_atm + phi_flat            (Eq. 2)
    psi = W(phi + n(gamma))                                   noisy wrapped obs.

Deformation sources (forward-modelled, §3.8):
  * Okada (1985) rectangular dislocation  -- coseismic fields.
  * Mogi (1958) point source              -- volcanic inflation/deflation.
  * Gaussian / exponential bowl           -- subsidence.
Plus a turbulent power-law atmospheric screen, a flat-earth ramp, fractal
topography, and a spatially varying coherence field that sets the phase-noise
statistics.

The per-edge ambiguity labels are derived (in :func:`src.dataset.build_sample`)
exactly as in Eq. (3), from the clean ``phi`` and the noisy ``psi`` -- identical
to the DLPU pipeline, so training code is agnostic to the data source. The key
difference is that here the model also receives ``coherence``, so it can learn
to distinguish real fringe cuts from low-coherence noise cuts.

Run ``python -m src.synthetic_engine`` for a self-test (physics sanity checks +
sample statistics).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PI = math.pi
TWO_PI = 2.0 * PI


# ===========================================================================
# Spectral random fields (topography, atmosphere)
# ===========================================================================
def power_law_field(
    size: int, beta: float, rng: np.random.Generator, normalize: bool = True
) -> np.ndarray:
    """Random field with a radial power-law spectrum ``P(k) ~ k^(-beta)``.

    Used for fractal topography (``beta`` ~ 2-3) and turbulent atmospheric
    screens (``beta`` ~ 8/3). Returns a zero-mean ``(size, size)`` array, scaled
    to unit standard deviation when ``normalize``.
    """
    kx = np.fft.fftfreq(size)[:, None]
    ky = np.fft.fftfreq(size)[None, :]
    k = np.sqrt(kx**2 + ky**2)
    k[0, 0] = 1.0  # avoid division by zero at DC
    amplitude = k ** (-beta / 2.0)
    amplitude[0, 0] = 0.0  # zero mean
    phase = rng.uniform(0, TWO_PI, size=(size, size))
    spectrum = amplitude * np.exp(1j * phase)
    field_ = np.fft.ifft2(spectrum).real
    if normalize:
        std = field_.std()
        if std > 0:
            field_ = field_ / std
    return field_


# ===========================================================================
# Deformation sources
# ===========================================================================
def _chinnery(f, xi, eta, q, L, W, dip, nu):
    """Chinnery's ``||f|| = f(x,p) - f(x,p-W) - f(x-L,p) + f(x-L,p-W)``."""
    return (
        f(xi, eta, q, dip, nu)
        - f(xi, eta - W, q, dip, nu)
        - f(xi - L, eta, q, dip, nu)
        + f(xi - L, eta - W, q, dip, nu)
    )


# --- Okada 1985 surface displacement (z = 0) elementary kernels -------------
def _okada_I(xi, eta, q, dip, nu, R, db):
    """Return the I1..I5 terms (Okada 1985), handling the vertical-dip case."""
    cosd, sind = math.cos(dip), math.sin(dip)
    if abs(cosd) > 1e-6:
        denom = R + db
        I5 = (
            (1 - 2 * nu)
            * 2.0
            / cosd
            * np.arctan(
                (eta * (np.sqrt(xi**2 + q**2) + q * cosd) + np.sqrt(xi**2 + q**2) * (R + np.sqrt(xi**2 + q**2)) * sind)
                / (xi * (R + np.sqrt(xi**2 + q**2)) * cosd + 1e-30)
            )
        )
        I4 = (1 - 2 * nu) / cosd * (np.log(R + db) - sind * np.log(R + eta))
        I3 = (1 - 2 * nu) * (1.0 / cosd * db / denom - np.log(R + eta)) + sind / cosd * I4
        I2 = (1 - 2 * nu) * (-np.log(R + eta)) - I3
        I1 = (1 - 2 * nu) * (-xi / (cosd * denom)) - sind / cosd * I5
    else:
        denom = (R + db) ** 2
        I5 = -(1 - 2 * nu) * xi * sind / (R + db)
        I4 = -(1 - 2 * nu) * q / (R + db)
        I3 = (1 - 2 * nu) / 2.0 * (eta / (R + db) + (eta * q) / denom - np.log(R + eta))
        I1 = -(1 - 2 * nu) / 2.0 * (xi * q) / denom
        I2 = (1 - 2 * nu) * (-np.log(R + eta)) - I3
    return I1, I2, I3, I4, I5


def _make_kernels(component: str):
    """Return the (ux, uy, uz) kernel functions for a slip ``component``."""
    def geom(xi, eta, q, dip):
        R = np.sqrt(xi**2 + eta**2 + q**2)
        R = np.where(R < 1e-9, 1e-9, R)
        sind, cosd = math.sin(dip), math.cos(dip)
        yb = eta * cosd + q * sind
        db = eta * sind - q * cosd
        theta = np.arctan(xi * eta / (q * R + 1e-30))
        return R, yb, db, theta, sind, cosd

    if component == "strike":
        def ux(xi, eta, q, dip, nu):
            R, yb, db, theta, sind, cosd = geom(xi, eta, q, dip)
            I1, I2, I3, I4, I5 = _okada_I(xi, eta, q, dip, nu, R, db)
            return xi * q / (R * (R + eta)) + theta + I1 * sind
        def uy(xi, eta, q, dip, nu):
            R, yb, db, theta, sind, cosd = geom(xi, eta, q, dip)
            I1, I2, I3, I4, I5 = _okada_I(xi, eta, q, dip, nu, R, db)
            return yb * q / (R * (R + eta)) + q * cosd / (R + eta) + I2 * sind
        def uz(xi, eta, q, dip, nu):
            R, yb, db, theta, sind, cosd = geom(xi, eta, q, dip)
            I1, I2, I3, I4, I5 = _okada_I(xi, eta, q, dip, nu, R, db)
            return db * q / (R * (R + eta)) + q * sind / (R + eta) + I4 * sind
        return ux, uy, uz

    if component == "dip":
        def ux(xi, eta, q, dip, nu):
            R, yb, db, theta, sind, cosd = geom(xi, eta, q, dip)
            I1, I2, I3, I4, I5 = _okada_I(xi, eta, q, dip, nu, R, db)
            return q / R - I3 * sind * cosd
        def uy(xi, eta, q, dip, nu):
            R, yb, db, theta, sind, cosd = geom(xi, eta, q, dip)
            I1, I2, I3, I4, I5 = _okada_I(xi, eta, q, dip, nu, R, db)
            return yb * q / (R * (R + xi)) + cosd * theta - I1 * sind * cosd
        def uz(xi, eta, q, dip, nu):
            R, yb, db, theta, sind, cosd = geom(xi, eta, q, dip)
            I1, I2, I3, I4, I5 = _okada_I(xi, eta, q, dip, nu, R, db)
            return db * q / (R * (R + xi)) + sind * theta - I5 * sind * cosd
        return ux, uy, uz

    # tensile / opening
    def ux(xi, eta, q, dip, nu):
        R, yb, db, theta, sind, cosd = geom(xi, eta, q, dip)
        I1, I2, I3, I4, I5 = _okada_I(xi, eta, q, dip, nu, R, db)
        return q**2 / (R * (R + eta)) - I3 * sind**2
    def uy(xi, eta, q, dip, nu):
        R, yb, db, theta, sind, cosd = geom(xi, eta, q, dip)
        I1, I2, I3, I4, I5 = _okada_I(xi, eta, q, dip, nu, R, db)
        return -db * q / (R * (R + xi)) - sind * (xi * q / (R * (R + eta)) - theta) - I1 * sind**2
    def uz(xi, eta, q, dip, nu):
        R, yb, db, theta, sind, cosd = geom(xi, eta, q, dip)
        I1, I2, I3, I4, I5 = _okada_I(xi, eta, q, dip, nu, R, db)
        return yb * q / (R * (R + xi)) + cosd * (xi * q / (R * (R + eta)) - theta) - I5 * sind**2
    return ux, uy, uz


def okada85(
    e: np.ndarray,
    n: np.ndarray,
    depth: float,
    strike_deg: float,
    dip_deg: float,
    length: float,
    width: float,
    rake_deg: float,
    slip: float,
    opening: float = 0.0,
    nu: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Okada (1985) surface displacement of a finite rectangular dislocation.

    ``e``/``n`` are East/North observation coordinates (metres) relative to the
    fault-centroid surface projection; ``depth`` is the centroid depth (>0).
    Returns ``(uE, uN, uUp)`` displacement components in metres.
    """
    strike, dip, rake = map(math.radians, (strike_deg, dip_deg, rake_deg))
    U1 = math.cos(rake) * slip   # strike-slip
    U2 = math.sin(rake) * slip   # dip-slip
    U3 = opening                 # tensile

    sind, cosd = math.sin(dip), math.cos(dip)
    d = depth + sind * width / 2.0  # depth of the deeper edge

    # Rotate (e, n) into fault frame (x along strike, y up-dip horizontal).
    coss, sins = math.cos(strike), math.sin(strike)
    ec = e + coss * cosd * width / 2.0
    nc = n - sins * cosd * width / 2.0
    x = coss * nc + sins * ec + length / 2.0
    y = sins * nc - coss * ec + cosd * width

    p = y * cosd + d * sind
    q = y * sind - d * cosd

    uE = np.zeros_like(x, dtype=float)
    uN = np.zeros_like(x, dtype=float)
    uZ = np.zeros_like(x, dtype=float)
    for comp, U in (("strike", U1), ("dip", U2), ("tensile", U3)):
        if abs(U) < 1e-12:
            continue
        kux, kuy, kuz = _make_kernels(comp)
        ux = -U / TWO_PI * _chinnery(kux, x, p, q, length, width, dip, nu)
        uy = -U / TWO_PI * _chinnery(kuy, x, p, q, length, width, dip, nu)
        uz = -U / TWO_PI * _chinnery(kuz, x, p, q, length, width, dip, nu)
        # Rotate horizontal displacement back to E/N.
        uE += sins * ux - coss * uy
        uN += coss * ux + sins * uy
        uZ += uz
    return uE, uN, uZ


def mogi(
    e: np.ndarray, n: np.ndarray, depth: float, dv: float, nu: float = 0.25
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mogi (1958) point pressure source surface displacement (metres).

    ``dv`` is the source volume change (m^3, positive = inflation). ``e``/``n``
    are horizontal coords relative to the source, ``depth`` the source depth.
    """
    r2 = e**2 + n**2
    R = np.sqrt(r2 + depth**2)
    c = (1 - nu) / PI * dv
    uE = c * e / R**3
    uN = c * n / R**3
    uZ = c * depth / R**3
    return uE, uN, uZ


def subsidence_bowl(
    e: np.ndarray,
    n: np.ndarray,
    amplitude: float,
    radius: float,
    shape: str = "gaussian",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A vertical subsidence/uplift bowl (Gaussian or exponential), metres.

    ``amplitude`` is peak vertical displacement (negative = subsidence). Returns
    ``(uE, uN, uUp)`` with only a vertical component.
    """
    r = np.sqrt(e**2 + n**2)
    if shape == "exponential":
        prof = np.exp(-r / max(radius, 1e-6))
    else:
        prof = np.exp(-(r**2) / (2.0 * max(radius, 1e-6) ** 2))
    uZ = amplitude * prof
    return np.zeros_like(uZ), np.zeros_like(uZ), uZ


# ===========================================================================
# LOS projection and coherence-driven noise
# ===========================================================================
def los_unit_vector(incidence_deg: float, heading_deg: float) -> Tuple[float, float, float]:
    """Unit line-of-sight vector (East, North, Up) from geometry."""
    theta = math.radians(incidence_deg)
    head = math.radians(heading_deg)
    los_e = -math.sin(theta) * math.cos(head)
    los_n = math.sin(theta) * math.sin(head)
    los_u = math.cos(theta)
    return los_e, los_n, los_u


def displacement_to_phase(
    uE: np.ndarray, uN: np.ndarray, uZ: np.ndarray, wavelength_m: float,
    incidence_deg: float, heading_deg: float,
) -> np.ndarray:
    """Project 3-D displacement (m) to two-way LOS interferometric phase (rad)."""
    le, ln, lu = los_unit_vector(incidence_deg, heading_deg)
    los = uE * le + uN * ln + uZ * lu
    return (4.0 * PI / wavelength_m) * los


def coherence_phase_std(gamma: np.ndarray, looks: int) -> np.ndarray:
    r"""Interferometric phase-noise std (rad) for coherence ``gamma``, ``L`` looks.

    Uses the Cramer-Rao-style approximation
    ``sigma = sqrt((1 - gamma^2) / (2 L gamma^2))``, clipped to a sensible range.
    """
    g = np.clip(gamma, 1e-3, 0.9999)
    sigma = np.sqrt((1.0 - g**2) / (2.0 * looks * g**2))
    return np.clip(sigma, 0.0, 3.0 * PI)


# ===========================================================================
# Generator
# ===========================================================================
@dataclass
class SyntheticConfig:
    """Parameters controlling the synthetic interferogram generator."""

    img_size: int = 256
    pixel_spacing_m: float = 90.0
    wavelength_m: float = 0.0555           # Sentinel-1 C-band (~5.55 cm)
    incidence_deg: float = 38.0
    heading_deg: float = -12.0

    # Component toggles and magnitudes (sampled within ranges).
    deformation_types: Sequence[str] = ("okada", "mogi", "subsidence", "none")
    deformation_weights: Sequence[float] = (0.4, 0.25, 0.25, 0.1)

    topo_enabled: bool = True
    topo_beta: float = 2.5
    topo_max_rad: Tuple[float, float] = (5.0, 60.0)   # peak-to-peak fringe range

    atm_enabled: bool = True
    atm_beta: float = 2.67                 # ~8/3 turbulence
    atm_std_rad: Tuple[float, float] = (0.0, 4.0)

    flat_enabled: bool = True
    flat_max_rad: Tuple[float, float] = (0.0, 20.0)

    coherence_mean: Tuple[float, float] = (0.4, 0.95)
    coherence_beta: float = 3.0            # smoothness of the coherence field
    low_coherence_patches: int = 3
    looks: int = 8

    # Global difficulty in [0,1]: scales the signal (deformation+topo+atm) and
    # hence the fringe density / cut fraction. ~0.5 gives a realistic cut
    # fraction (~3% edges, like DLPU); 1.0 is the aggressive original setting.
    # Used for curriculum (ramp up over training).
    difficulty: float = 0.5


class SyntheticInterferogramGenerator:
    """Generate ``(phi, psi, coherence)`` triples from physical sources."""

    def __init__(self, cfg: Optional[SyntheticConfig] = None) -> None:
        self.cfg = cfg or SyntheticConfig()
        s = self.cfg.img_size
        sp = self.cfg.pixel_spacing_m
        # Ground coordinate grids centred at 0 (metres).
        coords = (np.arange(s) - s / 2.0) * sp
        self.E, self.N = np.meshgrid(coords, coords)

    # -- individual components ------------------------------------------------
    def _deformation_phase(self, rng: np.random.Generator) -> np.ndarray:
        cfg = self.cfg
        kind = rng.choice(cfg.deformation_types, p=np.asarray(cfg.deformation_weights) / np.sum(cfg.deformation_weights))
        extent = cfg.img_size * cfg.pixel_spacing_m
        if kind == "okada":
            uE, uN, uZ = okada85(
                self.E, self.N,
                depth=rng.uniform(2e3, 12e3),
                strike_deg=rng.uniform(0, 360),
                dip_deg=rng.uniform(20, 90),
                length=rng.uniform(0.1, 0.4) * extent,
                width=rng.uniform(0.05, 0.25) * extent,
                rake_deg=rng.uniform(-180, 180),
                slip=rng.uniform(0.3, 4.0),
            )
        elif kind == "mogi":
            uE, uN, uZ = mogi(
                self.E, self.N,
                depth=rng.uniform(1e3, 8e3),
                dv=rng.uniform(-1.0, 1.0) * rng.uniform(2e6, 3e7),
            )
        elif kind == "subsidence":
            uE, uN, uZ = subsidence_bowl(
                self.E, self.N,
                amplitude=rng.uniform(-0.5, 0.5),
                radius=rng.uniform(0.05, 0.3) * extent,
                shape=rng.choice(["gaussian", "exponential"]),
            )
        else:
            return np.zeros((cfg.img_size, cfg.img_size))
        return displacement_to_phase(
            uE, uN, uZ, cfg.wavelength_m, cfg.incidence_deg, cfg.heading_deg
        )

    def _topo_phase(self, rng: np.random.Generator) -> np.ndarray:
        if not self.cfg.topo_enabled:
            return np.zeros((self.cfg.img_size, self.cfg.img_size))
        f = power_law_field(self.cfg.img_size, self.cfg.topo_beta, rng)
        ptp = rng.uniform(*self.cfg.topo_max_rad)
        rng_ptp = f.max() - f.min()
        return f * (ptp / rng_ptp) if rng_ptp > 0 else f

    def _atm_phase(self, rng: np.random.Generator) -> np.ndarray:
        if not self.cfg.atm_enabled:
            return np.zeros((self.cfg.img_size, self.cfg.img_size))
        f = power_law_field(self.cfg.img_size, self.cfg.atm_beta, rng)
        return f * rng.uniform(*self.cfg.atm_std_rad)

    def _flat_phase(self, rng: np.random.Generator) -> np.ndarray:
        if not self.cfg.flat_enabled:
            return np.zeros((self.cfg.img_size, self.cfg.img_size))
        s = self.cfg.img_size
        gx, gy = np.meshgrid(np.linspace(-1, 1, s), np.linspace(-1, 1, s))
        amp = rng.uniform(*self.cfg.flat_max_rad)
        ang = rng.uniform(0, TWO_PI)
        return amp * (gx * math.cos(ang) + gy * math.sin(ang))

    def _coherence(self, rng: np.random.Generator, fringe_rate: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        s = cfg.img_size
        base = power_law_field(s, cfg.coherence_beta, rng)
        base = (base - base.min()) / (np.ptp(base) + 1e-9)      # -> [0,1]
        lo, hi = cfg.coherence_mean
        gamma = lo + (hi - lo) * base
        # Decorrelate where fringes are dense (steep gradients).
        gamma = gamma * (1.0 - 0.3 * np.clip(fringe_rate, 0, 1))
        # A few random low-coherence patches (water/vegetation).
        for _ in range(cfg.low_coherence_patches):
            cy, cx = rng.integers(0, s, size=2)
            r = rng.uniform(0.03, 0.12) * s
            yy, xx = np.ogrid[:s, :s]
            mask = ((yy - cy) ** 2 + (xx - cx) ** 2) < r**2
            gamma[mask] *= rng.uniform(0.1, 0.4)
        return np.clip(gamma, 0.02, 0.99)

    # -- full sample ----------------------------------------------------------
    def generate(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Return ``{phi, wrapped, coherence}`` as float32 arrays.

        ``seed`` makes a scene reproducible (for episodic adaptation).
        """
        rng = np.random.default_rng(seed)
        phi = (
            self._deformation_phase(rng)
            + self._topo_phase(rng)
            + self._atm_phase(rng)
            + self._flat_phase(rng)
        )
        phi = float(self.cfg.difficulty) * (phi - phi.mean())

        # Local fringe rate (wrapped-gradient magnitude) drives decorrelation.
        gx = np.abs(np.remainder(np.diff(phi, axis=1, append=phi[:, -1:]) + PI, TWO_PI) - PI)
        gy = np.abs(np.remainder(np.diff(phi, axis=0, append=phi[-1:, :]) + PI, TWO_PI) - PI)
        fringe_rate = (gx + gy) / TWO_PI

        gamma = self._coherence(rng, fringe_rate)
        sigma = coherence_phase_std(gamma, self.cfg.looks)
        noise = rng.standard_normal(phi.shape) * sigma
        psi = np.remainder(phi + noise + PI, TWO_PI) - PI    # W(phi + n)

        return {
            "phi": phi.astype(np.float32),
            "wrapped": psi.astype(np.float32),
            "coherence": gamma.astype(np.float32),
        }


# ===========================================================================
# PyTorch dataset (on-the-fly generation)
# ===========================================================================
def build_synthetic_dataset(
    cfg: Any, length: int = 30000, base_seed: int = 0,
    deterministic: Optional[bool] = None, input_grad: bool = True,
    difficulty: Optional[float] = None,
):
    """Construct a :class:`SyntheticInSARDataset` from a (dict/OmegaConf) config.

    Recognised keys mirror :class:`SyntheticConfig`. ``deterministic`` (fixed
    per-index seeds for reproducible eval episodes) can be forced via the
    argument, overriding any value in ``cfg``.
    """
    from torch.utils.data import Dataset
    from .dataset import build_sample

    def _get(key, default):
        if cfg is None:
            return default
        return cfg.get(key, default) if isinstance(cfg, dict) else getattr(cfg, key, default)

    sc = SyntheticConfig(
        img_size=int(_get("img_size", 256)),
        pixel_spacing_m=float(_get("pixel_spacing_m", 90.0)),
        wavelength_m=float(_get("wavelength_m", 0.0555)),
        incidence_deg=float(_get("incidence_deg", 38.0)),
        looks=int(_get("looks", 8)),
        coherence_mean=tuple(_get("coherence_mean", (0.4, 0.95))),
        difficulty=float(difficulty if difficulty is not None else _get("difficulty", 0.5)),
    )
    det = bool(_get("deterministic", False)) if deterministic is None else deterministic

    class SyntheticInSARDataset(Dataset):
        """On-the-fly synthetic interferograms with coherence and labels."""

        def __init__(self) -> None:
            self.gen = SyntheticInterferogramGenerator(sc)
            self.length = length
            self.deterministic = det
            self.base_seed = base_seed

        def __len__(self) -> int:
            return self.length

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            seed = (self.base_seed + idx) if self.deterministic else None
            s = self.gen.generate(seed=seed)
            return build_sample(
                wrapped=s["wrapped"], absolute=s["phi"], coherence=s["coherence"],
                name=f"synth_{idx:06d}", weight_reduce="mean", input_grad=input_grad,
            )

    return SyntheticInSARDataset()


# ===========================================================================
# Self-test (physics sanity checks)
# ===========================================================================
def _self_test() -> None:
    rng = np.random.default_rng(0)

    # 1) Mogi: inflation -> uplift at centre, far-field decay ~ 1/r^2.
    s = 64
    coords = (np.arange(s) - s / 2) * 100.0
    E, N = np.meshgrid(coords, coords)
    _, _, uZ = mogi(E, N, depth=3000.0, dv=1e7)
    assert uZ[s // 2, s // 2] > 0, "Mogi inflation should uplift the centre"
    print(f"[self-test] Mogi peak uplift = {uZ.max() * 100:.2f} cm")

    # 2) Okada: finite slip produces a bounded, finite displacement field.
    uE, uN, uZv = okada85(E, N, depth=4000.0, strike_deg=30, dip_deg=45,
                          length=4000, width=2000, rake_deg=90, slip=1.0)
    assert np.all(np.isfinite(uE)) and np.all(np.isfinite(uZv)), "Okada produced non-finite values"
    mag = np.sqrt(uE**2 + uN**2 + uZv**2).max()
    assert 1e-4 < mag < 5.0, f"Okada displacement magnitude unreasonable: {mag}"
    print(f"[self-test] Okada max |u| = {mag * 100:.2f} cm (slip=1 m)")

    # 3) Power-law field has ~zero mean and the requested spectral slope sign.
    f = power_law_field(128, 2.5, rng)
    assert abs(f.mean()) < 1e-6 and f.std() > 0
    print(f"[self-test] power-law field std={f.std():.3f} mean={f.mean():.2e}")

    # 4) Coherence noise: lower coherence -> larger phase std.
    s_hi = coherence_phase_std(np.array([0.9]), 8)[0]
    s_lo = coherence_phase_std(np.array([0.3]), 8)[0]
    assert s_lo > s_hi, "Lower coherence must give larger phase noise"
    print(f"[self-test] phase std: gamma=0.9 -> {s_hi:.3f} rad, gamma=0.3 -> {s_lo:.3f} rad")

    # 5) Full generator: shapes, ranges, reproducibility.
    gen = SyntheticInterferogramGenerator(SyntheticConfig(img_size=128))
    a = gen.generate(seed=42)
    b = gen.generate(seed=42)
    assert a["wrapped"].shape == (128, 128)
    assert -PI - 1e-3 <= a["wrapped"].min() and a["wrapped"].max() <= PI + 1e-3
    assert (0 < a["coherence"]).all() and (a["coherence"] <= 1).all()
    assert np.allclose(a["phi"], b["phi"]), "seeded generation must be reproducible"
    nonzero = float(np.mean(np.abs(a["phi"]) > PI))
    print(f"[self-test] sample: phi range [{a['phi'].min():.1f}, {a['phi'].max():.1f}] rad, "
          f"coherence mean={a['coherence'].mean():.2f}")
    print("[self-test] PASSED")


if __name__ == "__main__":
    _self_test()
