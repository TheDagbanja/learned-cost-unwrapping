"""Combined two-event figure: (a) Ridgecrest and (b) Amatrice, framed subfigures.

Each block is a 2x3 panel set -- wrapped phase, coherence, learned-cost result /
SNAPHU, difference, uncertainty -- matching the single-event layout but stacked
into one figure with (a)/(b) labels and a dividing rule, so the two events occupy
one column-width figure in the manuscript.

Reads the HyP3 product for wrapped phase, coherence and the SNAPHU product, and
the outputs of ``scripts.unwrap_geotiff`` (``*_learned_unw.tif``,
``*_uncertainty.tif``) for our result. The product rasters are windowed to the
bounds of the learned raster, so the ROI matches whatever was unwrapped.

    python -m scripts.make_fig_events \\
        --a-product data/S1_data/<RIDGECREST> --a-out outputs/ridgecrest --a-label "(a) Ridgecrest M7.1" \\
        --b-product data/S1_data/<AMATRICE>  --b-out outputs/amatrice   --b-label "(b) Amatrice M6.2" \\
        --fig outputs/fig_events.png

Writes both PNG (300 dpi) and PDF.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

plt.rcParams.update({
    "font.family": "DejaVu Sans", "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.labelweight": "bold", "font.size": 8,
})


def _find(d, tag):
    h = glob.glob(os.path.join(d, f"*{tag}.tif"))
    if not h:
        raise FileNotFoundError(f"no *{tag}.tif in {d!r}")
    return h[0]


def load_event(product, out_prefix):
    """Return the six panel arrays for one event, on the learned raster's ROI."""
    import rasterio
    from rasterio.windows import from_bounds

    with rasterio.open(f"{out_prefix}_learned_unw.tif") as s:
        ours = s.read(1).astype(np.float64)
        bounds = s.bounds
    ucert = None
    up = f"{out_prefix}_uncertainty.tif"
    if os.path.exists(up):
        with rasterio.open(up) as s:
            ucert = s.read(1).astype(np.float64)

    def win_read(path):
        with rasterio.open(path) as s:
            w = from_bounds(*bounds, transform=s.transform)
            return s.read(1, window=w, boundless=True, fill_value=0).astype(np.float64)

    psi = win_read(_find(product, "_wrapped_phase"))
    coh = win_read(_find(product, "_corr"))
    unw = win_read(_find(product, "_unw_phase"))
    # crop to the smallest common shape (windowed reads can differ by a pixel)
    h = min(a.shape[0] for a in (ours, psi, coh, unw))
    w = min(a.shape[1] for a in (ours, psi, coh, unw))
    ours, psi, coh, unw = (a[:h, :w] for a in (ours, psi, coh, unw))
    if ucert is not None:
        ucert = ucert[:h, :w]
    mask = (coh > 0) & np.isfinite(psi) & (unw != 0)
    return dict(psi=psi, coh=coh, ours=ours, unw=unw, ucert=ucert, mask=mask)


def draw_block(subfig, ev, label):
    m = ev["mask"]
    nan = lambda a: np.where(m, a, np.nan)
    diff = nan(ev["ours"] - ev["unw"])
    panels = [
        ("Wrapped phase", nan(ev["psi"]), "twilight", None),
        ("Coherence", ev["coh"], "gray", None),
        ("Learned-cost (ours)", nan(ev["ours"]), "jet", None),
        ("SNAPHU", nan(ev["unw"]), "jet", None),
        ("Ours $-$ SNAPHU", diff, "RdBu", "sym"),
        ("Uncertainty (nats)", nan(ev["ucert"]) if ev["ucert"] is not None else None,
         "magma", None),
    ]
    axes = subfig.subplots(2, 3)
    # Fill the block without letting the rightmost colorbar tick labels run into
    # the outer frame (they sit outside the axes, so `right` must leave room for
    # them), and keep `top` low enough that panel titles clear the (a)/(b) label.
    # `top`/`bottom` also set the clearance from the outer frame: each subfigure
    # occupies half the figure, so bottom=0.08 in the lower block maps to ~0.04 in
    # figure coordinates, comfortably clear of the frame at 0.008.
    subfig.subplots_adjust(left=0.02, right=0.93, top=0.84, bottom=0.08,
                           wspace=0.30, hspace=0.18)
    for ax, (title, img, cmap, mode) in zip(axes.ravel(), panels):
        if img is None:
            ax.axis("off")
            continue
        kw = {}
        if mode == "sym":                       # symmetric limits about zero
            v = np.nanpercentile(np.abs(img), 99.5)
            kw = dict(vmin=-v, vmax=v)
        im = ax.imshow(img, cmap=cmap, **kw)
        ax.set_title(title, fontsize=8, fontweight="bold", pad=3)
        ax.axis("off")
        cb = subfig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=6)
        cb.outline.set_linewidth(0.4)
    subfig.text(0.012, 0.945, label, fontsize=11, fontweight="bold", va="top")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-product", required=True)
    ap.add_argument("--a-out", required=True, help="prefix of the unwrap_geotiff outputs")
    ap.add_argument("--a-label", default="(a)")   # panel letter only; name the event in the caption
    ap.add_argument("--b-product", required=True)
    ap.add_argument("--b-out", required=True)
    ap.add_argument("--b-label", default="(b)")   # panel letter only; name the event in the caption
    ap.add_argument("--fig", default="outputs/fig_events.png")
    args = ap.parse_args()

    a = load_event(args.a_product, args.a_out)
    b = load_event(args.b_product, args.b_out)

    fig = plt.figure(figsize=(7.16, 7.4))
    subfigs = fig.subfigures(2, 1, hspace=0.0)
    draw_block(subfigs[0], a, args.a_label)
    draw_block(subfigs[1], b, args.b_label)

    # outer frame + divider, in figure coordinates (so no tight bbox on save)
    fig.add_artist(Rectangle((0.008, 0.008), 0.984, 0.984, transform=fig.transFigure,
                             fill=False, lw=1.1, edgecolor="0.25"))
    fig.add_artist(Line2D([0.008, 0.992], [0.5, 0.5], transform=fig.transFigure,
                          lw=1.1, color="0.25"))

    os.makedirs(os.path.dirname(args.fig) or ".", exist_ok=True)
    fig.savefig(args.fig, dpi=300)
    fig.savefig(args.fig.replace(".png", ".pdf"))
    print(f"wrote {args.fig} and {args.fig.replace('.png', '.pdf')}")


if __name__ == "__main__":
    main()
