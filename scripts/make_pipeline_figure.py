"""Render the architecture pipeline figure (vertical, LaTeX-style) to disk.

Writes outputs/pipeline.pdf (vector), outputs/pipeline.png (300 dpi), and
outputs/pipeline.svg. Math is set with mathtext (Computer Modern) for a
LaTeX look.

    python -m scripts.make_pipeline_figure
"""

from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

INDIGO = ("#e8eaf6", "#5c6bc0")
CREAM = ("#fdf0e2", "#d99a5b")
SIDE = ("#f5f5f5", "#9e9e9e")

X, W, H = 50.0, 300.0, 46.0          # main column geometry (SVG-like coords)
CX = X + W / 2.0                     # 200

# (top-y, color, lines)  -- lines is a list of (text, fontsize)
MAIN = [
    (30,  INDIGO, [(r"Wrapped phase  $\psi$", 12)]),
    (104, INDIGO, [(r"Encode: $(\cos\psi,\ \sin\psi,\ \Delta^{x},\ \Delta^{y})$", 10.5)]),
    (178, INDIGO, [("Swin-T Tiny encoder", 11), ("(frozen backbone)", 11)]),
    (252, INDIGO, [("U-Net decoder, 4 stages", 12)]),
    (326, INDIGO, [(r"Softmax $\to\ p_e \in \Delta^{2}$", 12)]),
    (400, CREAM,  [(r"$c_e^{\pm} = -\log p_e(\pm 1) + \lambda\bar{\gamma}_e$", 12)]),
    (474, CREAM,  [("Min-cost-flow solver", 12)]),
    (548, CREAM,  [("DCT integration (6)", 12)]),
    (622, INDIGO, [(r"$\hat{\phi}$, residue-free", 12)]),
]

# (top-y, height, lines)  side annotations, connected into the box at cy
SIDE_BOXES = [
    (181, 40, [(r"LoRA  $r=8$", 10.5), (r"$\approx 140$k", 10.5)]),
    (329, 40, [(r"Entropy $H_e$", 10.5), (r"$\to$ map $U$", 10.5)]),
    (408, 30, [(r"Coherence $\bar{\gamma}_e$", 10.5)]),
]
SX, SW = 380.0, 150.0


def draw_box(ax, x, y, w, h, fc, ec, lines):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.3))
    cy = y + h / 2.0
    n = len(lines)
    for i, (txt, fs) in enumerate(lines):
        dy = (i - (n - 1) / 2.0) * (fs + 4)
        ax.text(x + w / 2.0, cy + dy, txt, ha="center", va="center",
                fontsize=fs, color="#1a1a1a")


def main():
    fig, ax = plt.subplots(figsize=(5.6, 7.0))
    ax.set_xlim(0, 560)
    ax.set_ylim(700, 0)                      # top-down
    ax.axis("off")

    for y, (fc, ec), lines in MAIN:
        draw_box(ax, X, y, W, H, fc, ec, lines)

    # vertical flow arrows between consecutive boxes
    for i in range(len(MAIN) - 1):
        y0 = MAIN[i][0] + H
        y1 = MAIN[i + 1][0]
        ax.annotate("", xy=(CX, y1), xytext=(CX, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.5,
                                    mutation_scale=12))

    # side annotation boxes + dashed connectors
    for y, h, lines in SIDE_BOXES:
        draw_box(ax, SX, y, SW, h, SIDE[0], SIDE[1], lines)
        cy = y + h / 2.0
        ax.annotate("", xy=(X + W, cy), xytext=(SX, cy),
                    arrowprops=dict(arrowstyle="-|>", color="#7a7a7a", lw=1.2,
                                    ls="--", mutation_scale=10))

    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{OUT}/pipeline.{ext}", dpi=300, bbox_inches="tight",
                    pad_inches=0.1)
    plt.close(fig)
    print(f"wrote {OUT}/pipeline.pdf, {OUT}/pipeline.png, {OUT}/pipeline.svg")


if __name__ == "__main__":
    main()
