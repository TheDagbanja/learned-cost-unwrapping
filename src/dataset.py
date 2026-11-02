"""PyTorch datasets and dataloaders for InSAR phase unwrapping.

Two dataset classes are provided:

* :class:`InSARDLPUDataset` -- the public InSAR-DLPU benchmark, stored as
  ``.mat`` pairs (wrapped ``input`` / absolute ``output``, 256x256 float32).
* :class:`GeoTIFFInSARDataset` -- real Sentinel-1 (Stream C) multi-band
  GeoTIFF / ENVI files, where band 1 is wrapped phase and band 2 is coherence.
  These have no trustworthy unwrapped ground truth, so labels are omitted and
  the coherence band is turned into the edge weights ``w_e`` consumed by the
  weighted least-squares integrator (Eq. 4).

Every sample is a ``dict`` of tensors with a shared schema (see
:func:`build_sample`), so the training and adaptation code is agnostic to which
source a batch came from.

Note on labels: in InSAR-DLPU the wrapped field carries decorrelation noise
(``psi = W(phi + n)``) while ``output`` is the clean absolute phase. Per-edge
ambiguity labels are therefore derived exactly as in Eq. (3),
``k_e = round((Delta phi - W(Delta psi)) / 2pi)`` and clamped to ``{-1,0,+1}``;
the noise is what produces the sparse non-zero cuts.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import utils

# ---------------------------------------------------------------------------
# Sample schema
# ---------------------------------------------------------------------------
# Each sample dict contains (shapes for a single item, batched by DataLoader):
#   "input"      : (2, H, W) float  -- (cos psi, sin psi) network input
#   "wrapped"    : (1, H, W) float  -- raw wrapped phase psi (losses / eval)
#   "coherence"  : (1, H, W) float  -- pixel coherence in [0, 1]
#   "weights_x"  : (H, W)    float  -- horizontal edge weights w_e
#   "weights_y"  : (H, W)    float  -- vertical   edge weights w_e
#   "valid_x"    : (H, W)    bool   -- horizontal edge validity (last col False)
#   "valid_y"    : (H, W)    bool   -- vertical   edge validity (last row False)
#   "name"       : str              -- source filename (no extension)
#   "absolute"   : (1, H, W) float  -- clean absolute phase   (if available)
#   "k_x"        : (H, W)    long   -- horizontal ambiguity class (if available)
#   "k_y"        : (H, W)    long   -- vertical   ambiguity class (if available)
# ---------------------------------------------------------------------------

_DLPU_SPLIT_DIRS = {
    "train": ("train_wrapped", "train_absolute"),
    # Unaugmented view of the training files for classical-baseline calibration.
    # Keeping a distinct split name prevents build_dataset() from enabling the
    # train-time phase augmentation used by learned models.
    "train_eval": ("train_wrapped", "train_absolute"),
    "test": ("test_wrapped", "test_absolute"),
    "test_real": ("test_wrapped_real", "test_absolute_real"),
}


def _select_files(
    files: list[str], limit: Optional[int], strategy: str = "head", seed: int = 0
) -> list[str]:
    """Return a deterministic subset without changing the default head semantics."""
    if limit is None or int(limit) >= len(files):
        return files
    count = int(limit)
    if count <= 0:
        raise ValueError(f"limit must be positive or null, got {limit!r}")
    if strategy == "head":
        return files[:count]
    if strategy == "even":
        indices = np.linspace(0, len(files) - 1, count, dtype=np.int64)
    elif strategy == "random":
        indices = np.sort(np.random.default_rng(int(seed)).choice(len(files), count, replace=False))
    else:
        raise ValueError("sample_strategy must be one of: head, even, random")
    return [files[int(index)] for index in indices]


# ---------------------------------------------------------------------------
# .mat loading
# ---------------------------------------------------------------------------
def _load_mat_array(path: str, key: str) -> np.ndarray:
    """Load a single 2-D array from a MATLAB ``.mat`` file.

    Handles both classic (v5, ``scipy.io.loadmat``) and v7.3/HDF5 (``h5py``)
    files transparently. ``key`` is the variable name (``input`` or ``output``
    in InSAR-DLPU).
    """
    try:
        import scipy.io as sio

        mat = sio.loadmat(path)
        return np.asarray(mat[key], dtype=np.float32)
    except (NotImplementedError, ValueError):
        # v7.3 files are HDF5 and raise here; fall back to h5py.
        import h5py

        with h5py.File(path, "r") as f:
            arr = np.asarray(f[key], dtype=np.float32)
        # h5py returns Fortran-order (transposed) relative to MATLAB.
        return arr.T if arr.ndim == 2 else arr


# ---------------------------------------------------------------------------
# Shared sample builder
# ---------------------------------------------------------------------------
def build_sample(
    wrapped: np.ndarray,
    absolute: Optional[np.ndarray] = None,
    coherence: Optional[np.ndarray] = None,
    *,
    name: str = "",
    weight_reduce: str = "mean",
    clamp_k: bool = True,
    input_grad: bool = True,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    """Assemble the canonical sample dict from raw numpy arrays.

    Parameters
    ----------
    wrapped:
        Wrapped phase ``(H, W)`` in ``(-pi, pi]``.
    absolute:
        Optional clean absolute phase ``(H, W)``. When provided, per-edge
        ambiguity labels are computed.
    coherence:
        Optional pixel coherence ``(H, W)`` in ``[0, 1]``. When provided, edge
        weights are derived from it; otherwise uniform weights are used.
    name:
        Identifier carried through for logging / visualisation.
    weight_reduce:
        Endpoint combiner for coherence -> edge weights (``mean|min|geo``).
    clamp_k:
        Clamp ambiguity to ``{-1, 0, +1}`` (proposal default).
    input_grad:
        Append wrapped-gradient channels to the network input (4 channels total),
        giving the classifier the local cut cue directly (see
        :func:`utils.encode_input`).
    """
    psi = torch.from_numpy(np.ascontiguousarray(wrapped)).to(dtype)
    h, w = psi.shape[-2], psi.shape[-1]

    sample: Dict[str, Any] = {
        "input": utils.encode_input(psi, include_grad=input_grad),  # (2 or 4, H, W)
        "wrapped": psi.unsqueeze(0),                  # (1, H, W)
        "name": name,
        # A tensor flag survives DataLoader collation.  It prevents benchmark
        # code from mistaking the all-one placeholder used when coherence is
        # absent for a measured or simulated coherence band.
        "coherence_available": torch.tensor(coherence is not None, dtype=torch.bool),
    }

    # --- coherence + edge weights ---
    if coherence is not None:
        coh = torch.from_numpy(np.ascontiguousarray(coherence)).to(dtype)
        coh = coh.clamp(0.0, 1.0)
        wx, wy = utils.coherence_to_edge_weights(coh, reduce=weight_reduce, pad=True)
    else:
        coh = torch.ones((h, w), dtype=dtype)
        wx, wy = utils.default_edge_weights((h, w), dtype=dtype)
    sample["coherence"] = coh.unsqueeze(0)            # (1, H, W)
    sample["weights_x"] = wx
    sample["weights_y"] = wy

    # --- validity masks ---
    mask_x, mask_y = utils.edge_valid_masks(h, w)
    sample["valid_x"] = mask_x
    sample["valid_y"] = mask_y

    # --- labels (only when ground-truth absolute phase is available) ---
    if absolute is not None:
        phi = torch.from_numpy(np.ascontiguousarray(absolute)).to(dtype)
        sample["absolute"] = phi.unsqueeze(0)         # (1, H, W)
        # Per-edge ambiguity labels (edge head).
        cls_x, cls_y = utils.compute_ambiguity_labels(psi, phi, pad=True, clamp=clamp_k)
        sample["k_x"] = cls_x                          # (H, W) long
        sample["k_y"] = cls_y                          # (H, W) long
        # Per-pixel wrap count (pixel head): phi = psi + 2*pi*K, integer K.
        sample["k_pixel"] = torch.round((phi - psi) / utils.TWO_PI).unsqueeze(0)  # (1,H,W) float

    return sample


# ---------------------------------------------------------------------------
# InSAR-DLPU dataset
# ---------------------------------------------------------------------------
def resolve_dlpu_root(root: str) -> str:
    """Resolve the directory that actually holds the ``*_wrapped`` split folders.

    The released archive nests the data several levels deep
    (``.../InSAR-DLPU/InSAR-DLPU/InSAR-DLPU/``). This walks down through any
    chain of single ``InSAR-DLPU`` subfolders until it finds one containing a
    recognised split directory, so configs can simply point at ``data/``.
    """
    candidates = [root]
    # Breadth-first descent, bounded depth, following InSAR-DLPU-named folders.
    for _ in range(6):
        new_candidates = []
        for c in candidates:
            if not os.path.isdir(c):
                continue
            if any(os.path.isdir(os.path.join(c, d[0])) for d in _DLPU_SPLIT_DIRS.values()):
                return c
            for entry in os.listdir(c):
                p = os.path.join(c, entry)
                if os.path.isdir(p) and "insar" in entry.lower():
                    new_candidates.append(p)
        candidates = new_candidates
        if not candidates:
            break
    raise FileNotFoundError(
        f"Could not locate InSAR-DLPU split folders under {root!r}. "
        f"Expected a directory containing e.g. 'train_wrapped'."
    )


class InSARDLPUDataset(Dataset):
    """InSAR-DLPU benchmark of wrapped/absolute ``.mat`` pairs.

    Parameters
    ----------
    root:
        Path to the dataset; may point at any ancestor of the split folders
        (see :func:`resolve_dlpu_root`).
    split:
        One of ``"train"``, ``"test"`` (simulated) or ``"test_real"``
        (real TanDEM-X, the primary out-of-distribution target).
    return_labels:
        If ``True`` (default) compute per-edge ambiguity labels from the
        absolute phase. Set ``False`` to emulate the label-free TTA setting.
    limit:
        Optional cap on the number of samples (handy for quick local smoke
        tests on CPU).
    transform:
        Optional callable applied to the assembled sample dict (augmentation).
    augment:
        Optional callable ``(wrapped, absolute, coherence) -> (wrapped, absolute,
        coherence)`` applied to the *raw arrays before* :func:`build_sample`, so
        labels are recomputed exactly from the augmented pair (see
        :class:`augment.PhaseAugmentor`). Use only on the train split.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        return_labels: bool = True,
        clamp_k: bool = True,
        limit: Optional[int] = None,
        sample_strategy: str = "head",
        sample_seed: int = 0,
        input_grad: bool = True,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        augment: Optional[Callable[..., Any]] = None,
    ) -> None:
        if split not in _DLPU_SPLIT_DIRS:
            raise ValueError(
                f"split must be one of {list(_DLPU_SPLIT_DIRS)}, got {split!r}."
            )
        self.root = resolve_dlpu_root(root)
        self.split = split
        self.return_labels = return_labels
        self.clamp_k = clamp_k
        self.input_grad = input_grad
        self.transform = transform
        self.augment = augment

        wrapped_dir, absolute_dir = _DLPU_SPLIT_DIRS[split]
        self.wrapped_dir = os.path.join(self.root, wrapped_dir)
        self.absolute_dir = os.path.join(self.root, absolute_dir)

        files = sorted(glob.glob(os.path.join(self.wrapped_dir, "*.mat")))
        if not files:
            raise FileNotFoundError(f"No .mat files in {self.wrapped_dir!r}.")
        self.wrapped_files = _select_files(files, limit, sample_strategy, sample_seed)

    def __len__(self) -> int:
        return len(self.wrapped_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        wpath = self.wrapped_files[idx]
        base = os.path.basename(wpath)
        name = os.path.splitext(base)[0]

        wrapped = _load_mat_array(wpath, "input")

        absolute = None
        if self.return_labels:
            apath = os.path.join(self.absolute_dir, base)
            if os.path.exists(apath):
                absolute = _load_mat_array(apath, "output")

        if self.augment is not None:
            wrapped, absolute, _ = self.augment(wrapped, absolute, None)

        sample = build_sample(
            wrapped=wrapped,
            absolute=absolute,
            coherence=None,           # InSAR-DLPU has no coherence band
            name=name,
            clamp_k=self.clamp_k,
            input_grad=self.input_grad,
        )
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


# ---------------------------------------------------------------------------
# Real Sentinel-1 / GeoTIFF dataset (Stream C)
# ---------------------------------------------------------------------------
class GeoTIFFInSARDataset(Dataset):
    """Multi-band GeoTIFF / ENVI interferograms (real Sentinel-1, Stream C).

    Each file must contain at least two bands: wrapped phase and coherence.
    There is no reliable unwrapped ground truth, so no ambiguity labels are
    produced; the coherence band becomes the edge weights ``w_e`` used by the
    integrator, mirroring SNAPHU's physical prior.

    Parameters
    ----------
    files:
        Either a list of file paths, a directory, or a glob pattern.
    wrapped_band, coherence_band:
        1-indexed band numbers (rasterio convention) for the wrapped phase and
        coherence respectively.
    absolute_band:
        Optional 1-indexed band for an unwrapped reference (e.g. a SNAPHU
        result) used only for evaluation; ``None`` if unavailable.
    weight_reduce:
        Endpoint combiner for coherence -> edge weights.
    transform:
        Optional callable applied to the sample dict.
    """

    def __init__(
        self,
        files: Sequence[str] | str,
        wrapped_band: int = 1,
        coherence_band: int = 2,
        absolute_band: Optional[int] = None,
        weight_reduce: str = "mean",
        input_grad: bool = True,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        self.files = self._resolve_files(files)
        if not self.files:
            raise FileNotFoundError(f"No GeoTIFF/ENVI files found for {files!r}.")
        self.wrapped_band = wrapped_band
        self.coherence_band = coherence_band
        self.absolute_band = absolute_band
        self.weight_reduce = weight_reduce
        self.input_grad = input_grad
        self.transform = transform

    @staticmethod
    def _resolve_files(files: Sequence[str] | str) -> List[str]:
        if isinstance(files, str):
            if os.path.isdir(files):
                patterns = ("*.tif", "*.tiff", "*.img", "*.dat")
                out: List[str] = []
                for pat in patterns:
                    out.extend(glob.glob(os.path.join(files, pat)))
                return sorted(out)
            return sorted(glob.glob(files))
        return list(files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import rasterio  # lazy import: only needed for Stream C

        path = self.files[idx]
        name = os.path.splitext(os.path.basename(path))[0]
        with rasterio.open(path) as src:
            wrapped = src.read(self.wrapped_band).astype(np.float32)
            coherence = src.read(self.coherence_band).astype(np.float32)
            absolute = (
                src.read(self.absolute_band).astype(np.float32)
                if self.absolute_band is not None
                else None
            )

        sample = build_sample(
            wrapped=wrapped,
            absolute=absolute,           # usually None for real data
            coherence=coherence,
            name=name,
            weight_reduce=self.weight_reduce,
            input_grad=self.input_grad,
        )
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


# ---------------------------------------------------------------------------
# Config-driven factory helpers
# ---------------------------------------------------------------------------
def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from either an attribute-style (OmegaConf) or dict config."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def build_dataset(cfg: Any, split: str, difficulty: Optional[float] = None) -> Dataset:
    """Construct a dataset for ``split`` from a (Hydra/OmegaConf or dict) config.

    Recognised ``cfg`` (the ``data`` group) keys::

        source        : "dlpu" (default) | "geotiff" | "synthetic"
        root          : dataset root path
        return_labels : bool
        clamp_k       : bool
        limit         : Optional[int]            # per-split sample cap
        # geotiff-only:
        files         : list/dir/glob
        wrapped_band, coherence_band, absolute_band, weight_reduce
        # synthetic-only (see SyntheticConfig):
        synthetic     : dict of generator params
        epoch_size, val_size : per-split sample counts
    """
    source = _cfg_get(cfg, "source", "dlpu")
    input_grad = bool(_cfg_get(cfg, "input_grad", True))
    # Train-time augmentation (sim->real robustness) is applied to the train
    # split only; eval/val splits are always seen clean.
    augment = None
    if split == "train":
        from .augment import build_augmentor

        augment = build_augmentor(_cfg_get(cfg, "augment", None), seed=_cfg_get(cfg, "seed", None))
    if source == "dlpu":
        return InSARDLPUDataset(
            root=_cfg_get(cfg, "root", "./data"),
            split=split,
            return_labels=bool(_cfg_get(cfg, "return_labels", True)),
            clamp_k=bool(_cfg_get(cfg, "clamp_k", True)),
            limit=_cfg_get(cfg, "limit", None),
            sample_strategy=str(_cfg_get(cfg, "sample_strategy", "head")),
            sample_seed=int(_cfg_get(cfg, "sample_seed", 0)),
            input_grad=input_grad,
            augment=augment,
        )
    if source == "geotiff":
        return GeoTIFFInSARDataset(
            files=_cfg_get(cfg, "files", _cfg_get(cfg, "root", "./data")),
            wrapped_band=int(_cfg_get(cfg, "wrapped_band", 1)),
            coherence_band=int(_cfg_get(cfg, "coherence_band", 2)),
            absolute_band=_cfg_get(cfg, "absolute_band", None),
            weight_reduce=_cfg_get(cfg, "weight_reduce", "mean"),
            input_grad=input_grad,
        )
    if source == "synthetic":
        from .synthetic_engine import build_synthetic_dataset

        syn_cfg = _cfg_get(cfg, "synthetic", {})
        # The non-train splits use deterministic seeds (reproducible eval) and a
        # smaller size, offset so train/val episodes never coincide.
        is_train = split == "train"
        length = int(_cfg_get(cfg, "epoch_size", 30000)) if is_train else int(_cfg_get(cfg, "val_size", 1000))
        base_seed = 0 if is_train else 10_000_000
        return build_synthetic_dataset(
            syn_cfg, length=length, base_seed=base_seed,
            deterministic=(None if is_train else True), input_grad=input_grad,
            difficulty=difficulty,
        )
    raise ValueError(f"Unknown data.source={source!r}; use 'dlpu', 'geotiff' or 'synthetic'.")


def input_channels(cfg: Any) -> int:
    """Number of network input channels implied by the data config.

    2 for ``(cos, sin)``; 4 when wrapped-gradient channels are appended
    (``data.input_grad``, default true). Used to keep the model's ``in_chans``
    in sync with the data without hardcoding.
    """
    return 4 if bool(_cfg_get(cfg, "input_grad", True)) else 2


def build_dataloader(
    cfg: Any,
    split: str,
    shuffle: Optional[bool] = None,
    device: Optional[torch.device] = None,
    difficulty: Optional[float] = None,
) -> DataLoader:
    """Build a ``DataLoader`` for ``split`` from a config.

    ``cfg`` is the ``data`` config group. Relevant keys: ``batch_size``,
    ``num_workers``, ``pin_memory``, ``drop_last``. ``pin_memory`` defaults to
    ``True`` only when training on CUDA. ``shuffle`` defaults to ``True`` for
    the ``train`` split and ``False`` otherwise.
    """
    dataset = build_dataset(cfg, split, difficulty=difficulty)
    if shuffle is None:
        shuffle = split == "train"

    use_cuda = device is not None and device.type == "cuda"
    pin_memory = bool(_cfg_get(cfg, "pin_memory", use_cuda))
    num_workers = int(_cfg_get(cfg, "num_workers", 4))

    return DataLoader(
        dataset,
        batch_size=int(_cfg_get(cfg, "batch_size", 8)),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=bool(_cfg_get(cfg, "drop_last", split == "train")),
        persistent_workers=num_workers > 0,
    )
