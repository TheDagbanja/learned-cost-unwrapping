"""Publication figures for the TGRS method-extension manuscript.

Generates the five data-driven figures the paper needs, styled for IEEE
double-column (vector PDF + PNG preview). Three figures (slip decomposition,
GPS, ablation-volume) run immediately from the recorded leaderboard numbers
embedded below -- these need no server data and match Tables I/II of the
manuscript exactly. Two figures (reliability ROC, spatial cycle-slip map) need
per-pixel arrays produced by the eval harness and are documented inline.

    python -m scripts.paper_figures --figure all --out manuscript/Extension/figs

Individual figures:
    --figure slip      Fig. 3  RMSE vs slip-free error, per method (sim + real)
    --figure gps       Fig. 4  GPS line-of-sight residual per method
    --figure ablation  Fig. 5  RMSE vs training volume (with SNAPHU reference)
    --figure roc       Fig. 6  reliability ROC + coverage-risk   [needs --calib .npz]
    --figure slipmap   Fig. 7  spatial cycle-slip maps ours vs DL [needs --scene .npz]

The embedded numbers can be overridden with --leaderboard <CSV> (columns
split,name,rmse,slip,eps[,residues],family; split in {sim,test}->sim else real).
The DL leaderboards written by ``ext_dl/eval_dl.py`` (dl_leaderboard_test.csv,
dl_leaderboard_test_real.csv) use exactly this schema, so they can be merged
with the classical/ours rows and passed here.
"""

from __future__ import annotations

import argparse
import os

import numpy as np


# ---------------------------------------------------------------------------
# Recorded leaderboard numbers (fallback defaults; override with --leaderboard).
# Each row: name, RMSE, slip% , slip-free eps, family.
# family in {classical, snaphu, dl, hybrid, ours, ref, floor}.
# DL rows are the genuine InSAR-DLPU baselines trained by ext_dl (see Tables I/II).
# ---------------------------------------------------------------------------
SIM = [
    ("Goldstein",       8.368, 30.9, 1.161, "classical"),
    ("Least squares",   4.327, 23.7, 1.318, "classical"),
    ("Quality-guided",  4.079, 25.3, 1.095, "classical"),
    ("Unit-cost MCF",   2.315,  9.1, 0.877, "classical"),
    ("SNAPHU",          1.475,  4.4, 0.814, "snaphu"),
    ("PhaseNet-2",      2.642, 19.5, 1.210, "dl"),
    ("Gradient-net",    2.829, 14.1, 1.136, "dl"),
    ("DeepLabV3+",      3.099, 24.1, 1.240, "dl"),
    ("Attention U-Net", 4.542, 29.8, 1.259, "dl"),
    ("DLPU-CNN",        4.933, 33.2, 1.313, "dl"),
    ("UNet++",          6.617, 42.3, 1.428, "dl"),
    ("DENet (hybrid)",  0.931,  0.9, 0.759, "hybrid"),
    ("Learned (ours)",  0.895,  1.0, 0.759, "ours"),
    ("GT-grad (ref.)",  0.839,  0.8, 0.757, "ref"),
]
SIM_FLOOR = 0.756
REAL = [
    ("Least squares",   7.421, 59.3, 1.773, "classical"),
    ("Goldstein",       7.731, 50.4, 1.660, "classical"),
    ("Unit-cost MCF",   5.958, 45.3, 1.601, "classical"),
    ("SNAPHU",          5.304, 41.5, 1.580, "snaphu"),
    ("PhaseNet-2",      6.284, 54.8, 1.732, "dl"),
    ("DLPU-CNN",        6.464, 57.8, 1.801, "dl"),
    ("DeepLabV3+",      6.570, 56.1, 1.737, "dl"),
    ("Attention U-Net", 7.102, 61.9, 1.788, "dl"),
    ("Gradient-net",    7.501, 60.5, 1.769, "dl"),
    ("UNet++",          8.057, 67.2, 1.796, "dl"),
    ("DENet (hybrid)",  3.150, 24.6, 1.458, "hybrid"),
    ("GT-grad (ref.)",  3.709, 31.1, 1.519, "ref"),
    ("Learned (ours)",  3.024, 22.8, 1.463, "ours"),
    ("Quality-guided",  2.383, 12.5, 1.357, "classical"),
]
REAL_FLOOR = 1.270

# GPS line-of-sight residual (cm) at the 18 shared Ridgecrest stations.
# ours/SNAPHU/DENet are numerically identical at every station (all high-coh,
# far-field -- no genuine ambiguity for any solver-based method to resolve).
# Ridgecrest, coherence-gated (>=0.5), 20 stations. ours/SNAPHU/DENet identical.
GPS = [
    ("Learned (ours)",  1.8,  "ours"),
    ("SNAPHU",          1.8,  "snaphu"),
    ("DENet (hybrid)",  1.8,  "hybrid"),
    ("Gradient-net",    5.3,  "dl"),
    ("PhaseNet-2",      6.8,  "dl"),
    ("Attention U-Net", 7.0,  "dl"),
    ("UNet++",          7.6,  "dl"),
    ("DeepLabV3+",      8.2,  "dl"),
    ("DLPU-CNN",       12.7,  "dl"),
]

# Training-volume ablation (pairs -> RMSE) and the SNAPHU reference.
VOLUME = [(1000, 1.307), (5000, 0.968), (10000, 0.915), (20000, 0.882), (30000, 0.874)]
VOLUME_SNAPHU = 1.475

# Wong colorblind-safe palette. hybrid = sky blue (solver-based, near ours).
C = {
    "classical": "#999999", "snaphu": "#E69F00", "dl": "#D55E00",
    "hybrid": "#56B4E9", "ours": "#0072B2", "ref": "#009E73", "floor": "#000000",
}
# Legend labels for the family groups drawn on the slip figure.
FAMILY_LABELS = [
    ("classical", "classical"), ("snaphu", "SNAPHU"), ("dl", "pure deep net"),
    ("hybrid", "hybrid (DENet)"), ("ours", "learned cost (ours)"), ("ref", "GT-grad ref."),
]


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "figure.dpi": 150, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(out, f"{name}.{ext}")
        fig.savefig(p, dpi=300)
    print(f"  wrote {os.path.join(out, name)}.pdf (+ .png)")


def _load_leaderboard(path):
    """Optional CSV override. Columns: split,name,rmse,slip,eps[,residues],family.
    split in {sim,test} -> simulated table, anything else -> real table."""
    import csv
    sim, real = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            row = (r["name"], float(r["rmse"]), float(r["slip"]),
                   float(r["eps"]), r.get("family", "classical"))
            (sim if r["split"] in ("sim", "test") else real).append(row)
    return sim or SIM, real or REAL


# ---------------------------------------------------------------------------
# Fig. 3 — slip decomposition
# ---------------------------------------------------------------------------
def fig_slip(out, sim=SIM, real=REAL):
    """RMSE (full bar) vs slip-free eps (overlaid), per method, sorted by RMSE,
    with slip% annotated and the floor as a vertical line. The point of the
    figure: RMSE spans ~10x while eps stays near the floor, so the spread is the
    cycle-slip rate. Bar colour encodes the method family (classical / pure deep
    / hybrid / ours), making the solver-based-vs-solver-free split visible."""
    import matplotlib.patches as mpatches
    plt = _style()
    nrows = max(len(sim), len(real))
    height = max(2.9, 0.30 * nrows)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, height))
    for ax, data, floor, title in ((axes[0], sim, SIM_FLOOR, "(a) Simulated (1000 scenes)"),
                                   (axes[1], real, REAL_FLOOR, "(b) Real TanDEM-X (100 scenes)")):
        rows = sorted(data, key=lambda r: r[1], reverse=True)
        y = np.arange(len(rows))
        rmse = [r[1] for r in rows]
        eps = [r[3] for r in rows]
        cols = [C.get(r[4], C["classical"]) for r in rows]
        ax.barh(y, rmse, color=cols, alpha=0.35, height=0.7)
        ax.barh(y, eps, color=cols, alpha=0.95, height=0.35)
        ax.axvline(floor, color=C["floor"], ls="--", lw=0.8)
        ax.text(floor, len(rows) - 0.4, " floor", fontsize=6, va="top", color=C["floor"])
        for yi, r in zip(y, rows):
            ax.text(r[1] + 0.02 * max(rmse), yi, f"{r[2]:.0f}%", va="center", fontsize=6)
        ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
        ax.set_xlabel("aligned error (rad)"); ax.set_title(title)
        ax.set_xlim(0, max(rmse) * 1.18)
        ax.invert_yaxis()
    # family legend (shared) + a note that the thin bar is slip-free eps
    handles = [mpatches.Patch(color=C[k], label=lab) for k, lab in FAMILY_LABELS]
    axes[0].legend(handles=handles, loc="lower right", frameon=False, fontsize=6)
    fig.tight_layout()
    _save(fig, out, "fig_slip_decomposition")


# ---------------------------------------------------------------------------
# Fig. 4 — GPS validation
# ---------------------------------------------------------------------------
def fig_gps(out, gps=GPS):
    plt = _style()
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    rows = sorted(gps, key=lambda r: r[1])
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[C.get(r[2], C["dl"]) for r in rows], height=0.7)
    for yi, r in zip(y, rows):
        ax.text(r[1] + 0.3, yi, f"{r[1]:.1f}", va="center", fontsize=6)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("LOS residual vs GPS (cm)")
    ax.axvline(1.8, color=C["ours"], ls=":", lw=0.8)
    ax.set_xlim(0, max(r[1] for r in rows) * 1.15)
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, out, "fig_gps")


# ---------------------------------------------------------------------------
# Fig. 5 — training-volume ablation
# ---------------------------------------------------------------------------
def fig_ablation(out, vol=VOLUME, snaphu=VOLUME_SNAPHU):
    plt = _style()
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    xs = [v[0] for v in vol]; ys = [v[1] for v in vol]
    ax.plot(xs, ys, "o-", color=C["ours"], lw=1.4, ms=4, label="learned cost (ours)")
    ax.axhline(snaphu, color=C["snaphu"], ls="--", lw=1.0, label="SNAPHU")
    ax.fill_between([xs[0], xs[-1]], snaphu, max(ys) * 1.02, color=C["snaphu"], alpha=0.06)
    ax.set_xscale("log")
    ax.set_xlabel("training pairs"); ax.set_ylabel("aligned RMSE (rad)")
    ax.legend(frameon=False)
    ax.annotate("5k pairs already\nbeats SNAPHU", xy=(5000, 0.968), xytext=(7000, 1.15),
                fontsize=6, arrowprops=dict(arrowstyle="->", lw=0.6))
    fig.tight_layout()
    _save(fig, out, "fig_ablation_volume")


# ---------------------------------------------------------------------------
# Fig. 6 — reliability ROC + coverage-risk
# ---------------------------------------------------------------------------
def fig_roc(out, calib_npz):
    """Needs a .npz with per-pixel arrays over the 500-scene calibration set:
       entropy   : float, the reliability map value U at each scored pixel
       is_error  : bool,  whether that pixel carries a cycle error |round(d/2pi)|>0
       grad_proxy: float (optional), a wrapped-gradient quality proxy for comparison
       err_value : float (optional), signed sub-cycle error for coverage-risk
    Produce it from the eval harness, e.g. accumulating U and the cycle-error
    mask over the 500 test scenes into arrays and np.savez."""
    plt = _style()
    d = np.load(calib_npz)
    ent, err = d["entropy"].ravel(), d["is_error"].ravel().astype(bool)
    proxy = d["grad_proxy"].ravel() if "grad_proxy" in d else None

    def roc(score):
        thr = np.quantile(score, np.linspace(0, 1, 200))
        tpr = [(score[err] >= t).mean() for t in thr]
        fpr = [(score[~err] >= t).mean() for t in thr]
        fpr, tpr = np.array(fpr), np.array(tpr)
        order = np.argsort(fpr)
        trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
        auc = trapz(tpr[order], fpr[order])
        return fpr, tpr, abs(auc)

    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.7))
    fpr, tpr, auc = roc(ent)
    ax[0].plot(fpr, tpr, color=C["ours"], lw=1.5, label=f"entropy (AUC {auc:.3f})")
    if proxy is not None:
        f2, t2, a2 = roc(proxy)
        ax[0].plot(f2, t2, color=C["classical"], lw=1.2, ls="--", label=f"grad. proxy (AUC {a2:.3f})")
    ax[0].plot([0, 1], [0, 1], "k:", lw=0.6)
    ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate")
    ax[0].set_title("(a) cycle-error detection"); ax[0].legend(frameon=False, loc="lower right")

    # coverage-risk: sort by confidence (low entropy first), cumulative slip-free error.
    err_val = d["err_value"].ravel() if "err_value" in d else err.astype(float)
    idx = np.argsort(ent)
    frac = np.linspace(0.02, 1.0, 50)
    risk = [np.sqrt((err_val[idx[:int(f * len(idx))]] ** 2).mean()) for f in frac]
    ax[1].plot(frac * 100, risk, color=C["ours"], lw=1.5)
    ax[1].set_xlabel("pixels retained (%, most confident first)")
    ax[1].set_ylabel("RMSE of retained (rad)"); ax[1].set_title("(b) coverage-risk")
    fig.tight_layout()
    _save(fig, out, "fig_reliability")


# ---------------------------------------------------------------------------
# Fig. 7 — spatial cycle-slip maps
# ---------------------------------------------------------------------------
def fig_slipmap(out, scene_npz):
    """Needs a .npz for ONE representative simulated scene with:
       psi  : wrapped phase (H,W)
       gt   : ground-truth absolute phase (H,W)
       ours : our unwrapped result (H,W)
       dl   : a deep baseline's unwrapped result (H,W)  [e.g. phasenet2]
    Dramatizes the residue-count critique: the DL output reports zero residues
    yet slips on whole-cycle patches; ours is congruent and slip-free."""
    plt = _style()
    TWO_PI = 2 * np.pi
    d = np.load(scene_npz)
    psi, gt = d["psi"], d["gt"]

    def slipmask(u):
        e = u - gt
        e = e - e.mean()
        return np.round(e / TWO_PI)

    fig, ax = plt.subplots(2, 3, figsize=(7.16, 4.6))
    panels = [
        ("Wrapped $\\psi$", psi, "twilight", None),
        ("Ours", d["ours"], "viridis", None),
        ("DL baseline", d["dl"], "viridis", None),
        ("Ground truth", gt, "viridis", None),
        ("Ours: cycle slips", slipmask(d["ours"]), "RdBu", (-2, 2)),
        ("DL: cycle slips", slipmask(d["dl"]), "RdBu", (-2, 2)),
    ]
    for a, (t, img, cmap, lim) in zip(ax.ravel(), panels):
        kw = dict(cmap=cmap)
        if lim: kw.update(vmin=lim[0], vmax=lim[1])
        im = a.imshow(img, **kw); a.set_title(t); a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
    ax.ravel()[4].set_title(f"Ours: {(slipmask(d['ours'])!=0).mean()*100:.1f}% slipped")
    ax.ravel()[5].set_title(f"DL: {(slipmask(d['dl'])!=0).mean()*100:.1f}% slipped")
    fig.tight_layout()
    _save(fig, out, "fig_slipmap")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", default="all",
                    choices=["all", "slip", "gps", "ablation", "roc", "slipmap"])
    ap.add_argument("--out", default="manuscript/Extension/figs")
    ap.add_argument("--leaderboard", default=None, help="CSV to override slip-decomp numbers")
    ap.add_argument("--calib", default=None, help=".npz for the ROC figure")
    ap.add_argument("--scene", default=None, help=".npz for the slip-map figure")
    args = ap.parse_args()

    sim, real = (_load_leaderboard(args.leaderboard) if args.leaderboard else (SIM, REAL))
    want = args.figure
    if want in ("all", "slip"):
        fig_slip(args.out, sim, real)
    if want in ("all", "gps"):
        fig_gps(args.out)
    if want in ("all", "ablation"):
        fig_ablation(args.out)
    if want in ("all", "roc"):
        if args.calib:
            fig_roc(args.out, args.calib)
        elif want == "roc":
            raise SystemExit("--figure roc needs --calib <calibration.npz> (see fig_roc docstring).")
        else:
            print("  [roc skipped: pass --calib <calibration.npz> to generate it]")
    if want in ("all", "slipmap"):
        if args.scene:
            fig_slipmap(args.out, args.scene)
        elif want == "slipmap":
            raise SystemExit("--figure slipmap needs --scene <scene.npz> (see fig_slipmap docstring).")
        else:
            print("  [slipmap skipped: pass --scene <scene.npz> to generate it]")


if __name__ == "__main__":
    main()
