"""Diagnose the DLPU sim->real gap at the *data* level.

Three failed label-free attempts (entropy TTA, consistency TTA, train-time
augmentation) to lift the real-subset floor warrant ruling out a representation
mismatch before blaming the domain gap. This compares the distributions of the
simulated 'test' split and the real 'test_real' split for the quantities the
model actually consumes / predicts:

  * wrapped phase range        -> must be ~(-pi, pi] for the (cos,sin) encoding
  * absolute phase range/std   -> overall scale / dynamic range
  * per-pixel wrap count K      -> centered range and the fraction CLIPPED at
                                   +/-k_max (if real needs |K|>k_max the head
                                   literally cannot represent the answer)
  * GT residual |phi - (psi+2pi*round((phi-psi)/2pi))|  -> the achievable
                                   reconstruction floor (oracle), i.e. how good
                                   any integer-K method can be on this data
  * fringe density |grad phi|   -> signal structure / steepness

Run on the remote box:
    python -m scripts.diagnose_real_gap
    python -m scripts.diagnose_real_gap --k_max 32 --n 100 --root ./data
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from src import dataset as dataset_module
from src import utils


def _stats(x: np.ndarray) -> str:
    return (f"min={x.min():+.3f} max={x.max():+.3f} mean={x.mean():+.3f} "
            f"std={x.std():.3f}")


def summarize(split: str, root: str, n: int, k_max: int) -> dict:
    cfg = {"source": "dlpu", "root": root, "input_grad": True,
           "return_labels": True, "clamp_k": True}
    ds = dataset_module.build_dataset(cfg, split)
    n = min(n, len(ds))
    print(f"\n=== split={split!r}  ({len(ds)} samples, inspecting {n}) ===")

    wr, ab, kc, kabs, clip, resid, grad, kspan = [], [], [], [], [], [], [], []
    for i in range(n):
        s = ds[i]
        psi = s["wrapped"][0].numpy()
        wr.append(psi)
        if "absolute" not in s:
            continue
        phi = s["absolute"][0].numpy()
        ab.append(phi)
        kp = s["k_pixel"][0].numpy()
        kcent = kp - np.round(kp.mean())              # centered (offset-invariant)
        kc.append(kcent)
        kabs.append(np.abs(kcent).max())
        kspan.append(kp.max() - kp.min())
        clip.append(float(np.mean(np.abs(kcent) > k_max)))
        # achievable integer-K floor (oracle): residual after best integer round
        phihat = psi + utils.TWO_PI * np.round((phi - psi) / utils.TWO_PI)
        r = phihat - phi
        r = r - r.mean()
        resid.append(np.sqrt(np.mean(r ** 2)))
        gx = np.abs(np.diff(phi, axis=1)); gy = np.abs(np.diff(phi, axis=0))
        grad.append(0.5 * (gx.mean() + gy.mean()))

    wr = np.concatenate([w.ravel() for w in wr])
    print(f"  wrapped psi      : {_stats(wr)}   (expect ~[-pi,pi]={np.pi:.3f})")
    if ab:
        ab_all = np.concatenate([a.ravel() for a in ab])
        print(f"  absolute phi     : {_stats(ab_all)}")
        print(f"  centered K range : max|K|={max(kabs):.0f}  "
              f"mean span(max-min)={np.mean(kspan):.1f}  "
              f"frac |K|>{k_max}={np.mean(clip)*100:.2f}%")
        print(f"  oracle K floor   : rmse(GT-K reconstruction) "
              f"mean={np.mean(resid):.4f}  p90={np.percentile(resid,90):.4f}")
        print(f"  fringe |grad phi|: mean={np.mean(grad):.4f} rad/px")
        return {"wr": wr, "kabs": float(max(kabs)), "clip": float(np.mean(clip)),
                "oracle": float(np.mean(resid)), "grad": float(np.mean(grad))}
    return {"wr": wr}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k_max", type=int, default=32)
    args = ap.parse_args()

    torch.manual_seed(0)
    sim = summarize("test", args.root, args.n, args.k_max)
    real = summarize("test_real", args.root, args.n, args.k_max)

    print("\n=== sim -> real deltas ===")
    if "oracle" in sim and "oracle" in real:
        print(f"  oracle floor   : sim={sim['oracle']:.3f}  real={real['oracle']:.3f}  "
              f"(if real >> sim, the integer-K target itself is harder)")
        print(f"  max|K| clipped : sim={sim['clip']*100:.2f}%  real={real['clip']*100:.2f}%  "
              f"(if real >0, raise k_max)")
        print(f"  fringe density : sim={sim['grad']:.3f}  real={real['grad']:.3f}  rad/px")
    print(f"  wrapped range  : sim[{sim['wr'].min():+.2f},{sim['wr'].max():+.2f}]  "
          f"real[{real['wr'].min():+.2f},{real['wr'].max():+.2f}]  "
          f"(mismatch => preprocessing bug)")


if __name__ == "__main__":
    main()
