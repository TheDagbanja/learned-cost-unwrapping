"""GPS validation of the Ridgecrest unwrapping: ours vs SNAPHU vs GPS LOS.

Both unwrappers operate on the SAME wrapped phase, so they differ only by integer
2pi cycles (~0 far-field, +/-cycles near the rupture). Independent GPS coseismic
offsets, projected to line-of-sight, test WHICH method got the near-fault cycles
right -- the experiment that can turn "matches SNAPHU" into "beats SNAPHU on real
data".

GPS CSV columns (meters, WGS84):  station,lon,lat,de,dn,du[,se,sn,su]
The offsets must span the interferogram (2019-07-04 13:51 -> 2019-07-16), i.e. the
cumulative M6.4 + M7.1 coseismic displacement. Sources: Nevada Geodetic Lab event
offsets, EarthScope/UNAVCO coseismic offsets, or a published Ridgecrest table.

    python -m scripts.gps_validate \\
        --product data/S1_data/S1AA_20190704T135158_20190716T135159_VVP012_INT80_G_ueF_1138 \\
        --ours outputs/ridgecrest_learned_unw.tif \\
        --gps ridgecrest_gps.csv --out outputs/gps
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np

# Radar wavelength (m). Set via --wavelength for non-C-band sensors:
#   Sentinel-1 C-band   0.055465763
#   ALOS-2 / NISAR L     0.2384   (1257.5 MHz)
#   ALOS-1 PALSAR L      0.2360
LAMBDA_C_BAND = 0.055465763
LAMBDA = LAMBDA_C_BAND                       # overridden by --wavelength in main()
FOUR_PI_OVER_LAMBDA = 4.0 * np.pi / LAMBDA


def _find(d, tag):
    h = glob.glob(os.path.join(d, f"*{tag}.tif"))
    if not h:
        raise FileNotFoundError(f"no *{tag}.tif in {d!r}")
    return h[0]


def read_gps(path):
    """Read one offset file (NGL whitespace `Sta Lon Lat de dn du ...` or CSV with a
    header). Returns {station: {lon,lat,de,dn,du}} in METERS."""
    with open(path) as f:
        lines = [l for l in f if l.strip() and not l.lstrip().startswith("#")]
    if not lines:
        return {}
    toks0 = lines[0].replace(",", " ").split()
    # normalise header names: 'de(m)' -> 'de', 'Lon' -> 'lon'
    norm = [t.lower().split("(")[0].strip() for t in toks0]
    has_hdr = any(t in ("lon", "longitude") for t in norm)
    hdr = norm if has_hdr else None
    rows = {}
    for line in lines[1:] if has_hdr else lines:
        t = line.replace(",", " ").split()
        if len(t) < 6:
            continue
        if hdr:
            d = {hdr[i]: t[i] for i in range(min(len(hdr), len(t)))}
            sta = d.get("sta", d.get("station", d.get("site", t[0])))
            lon, lat = float(d["lon"]), float(d["lat"])
            de, dn, du = float(d["de"]), float(d["dn"]), float(d["du"])
        else:                                            # NGL order: Sta Lon Lat de dn du
            sta, lon, lat, de, dn, du = t[0], float(t[1]), float(t[2]), float(t[3]), float(t[4]), float(t[5])
        rows[sta] = dict(station=sta, lon=lon, lat=lat, de=de, dn=dn, du=du)
    return rows


def load_offsets(paths):
    """Sum coseismic offsets across event files (cumulative displacement)."""
    total = {}
    for p in paths:
        for sta, r in read_gps(p).items():
            if sta in total:
                for k in ("de", "dn", "du"):
                    total[sta][k] += r[k]
            else:
                total[sta] = dict(r)
    return list(total.values())


def sample_at(src, lon, lat):
    """Sample raster band-1 at a WGS84 lon/lat (reproject to the raster CRS)."""
    import rasterio
    from rasterio.warp import transform as warp_transform

    xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
    row, col = src.index(xs[0], ys[0])
    if not (0 <= row < src.height and 0 <= col < src.width):
        return None
    return float(src.read(1)[row, col])


def main():
    import rasterio

    ap = argparse.ArgumentParser()
    ap.add_argument("--product", required=True, help="HyP3 product dir")
    ap.add_argument("--ours", default=None, help="our unwrapped GeoTIFF (shorthand for --method ours=PATH)")
    ap.add_argument("--method", action="append", default=[], metavar="NAME=PATH",
                    help="additional unwrapped GeoTIFF to score, repeatable "
                         "(e.g. --method phasenet2=outputs/rc_phasenet2.tif)")
    ap.add_argument("--gps", required=True, nargs="+",
                    help="one or more offset files (summed); NGL .txt or CSV. "
                         "Pass both Ridgecrest events to get the cumulative offset.")
    ap.add_argument("--out", default="outputs/gps")
    ap.add_argument("--wavelength", type=float, default=LAMBDA_C_BAND,
                    help="radar wavelength in m (C-band 0.0555 default; L-band ~0.2384)")
    args = ap.parse_args()
    global LAMBDA, FOUR_PI_OVER_LAMBDA
    LAMBDA = float(args.wavelength)
    FOUR_PI_OVER_LAMBDA = 4.0 * np.pi / LAMBDA

    methods = {}
    if args.ours:
        methods["ours"] = args.ours
    for spec in args.method:
        if "=" not in spec:
            raise SystemExit(f"--method expects NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        methods[name] = path
    if not methods:
        raise SystemExit("Give at least one --ours or --method NAME=PATH.")

    snaphu_f = _find(args.product, "_unw_phase")
    theta_f = _find(args.product, "_lv_theta")
    phi_f = _find(args.product, "_lv_phi")

    gps = load_offsets(args.gps)
    print(f"loaded {len(gps)} stations from {len(args.gps)} offset file(s) "
          f"({'summed cumulative' if len(args.gps) > 1 else 'single event'})")
    src_s = rasterio.open(snaphu_f)
    src_t = rasterio.open(theta_f)
    src_p = rasterio.open(phi_f)
    srcs = {n: rasterio.open(p) for n, p in methods.items()}

    recs = []
    for g in gps:
        th = sample_at(src_t, g["lon"], g["lat"])
        ph = sample_at(src_p, g["lon"], g["lat"])
        sn = sample_at(src_s, g["lon"], g["lat"])
        if None in (th, ph, sn) or th == 0 or sn == 0:
            continue                                    # outside scene / nodata
        # LOS unit vector toward sensor; d_los>0 = toward sensor
        le, ln, lu = np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), np.sin(th)
        d_los = g["de"] * le + g["dn"] * ln + g["du"] * lu
        gps_phase = -FOUR_PI_OVER_LAMBDA * d_los        # README sign: toward sensor -> -phase
        rec = dict(station=g["station"], gps=gps_phase, snaphu=sn)
        for n, sr in srcs.items():
            v = sample_at(sr, g["lon"], g["lat"])
            rec[n] = v if (v is not None and v != 0) else np.nan
        recs.append(rec)

    if not recs:
        raise SystemExit("No GPS stations fell within the scene/coverage.")
    gps_p = np.array([r["gps"] for r in recs])
    sn_p = np.array([r["snaphu"] for r in recs])

    # self-calibrate the sign convention using SNAPHU (the trusted baseline):
    # SNAPHU must correlate POSITIVELY with GPS LOS phase; flip if not.
    if np.corrcoef(sn_p, gps_p)[0, 1] < 0:
        gps_p = -gps_p
        print("note: flipped GPS phase sign to match SNAPHU convention.")

    def resid_rmse(pred, gt, m):
        d = pred[m] - gt[m]
        d = d - d.mean()                                # remove the reference constant
        return float(np.sqrt((d ** 2).mean()))

    n_all = len(recs)
    m_all = np.ones(n_all, bool)
    cm = LAMBDA / (4 * np.pi) * 100                     # rad -> cm

    print()
    print(f"=== GPS validation ({n_all} stations in scene) ===")
    # Fair head-to-head: every method (and SNAPHU) on the SAME stations, i.e. the
    # intersection of all method ROIs. SNAPHU alone also gets a whole-scene row,
    # clearly labelled, since it is the only method covering the full scene.
    common = np.ones(n_all, bool)
    for n in methods:
        common &= np.isfinite(np.array([r[n] for r in recs]))

    print(f"{'method':16s}{'n':>4}{'RMSE (rad)':>12}{'RMSE (cm)':>11}")
    rows = []
    if common.sum() >= 3:
        rows.append(("SNAPHU", int(common.sum()), resid_rmse(sn_p, gps_p, common)))
        for n in methods:
            v = np.array([r[n] for r in recs])
            rows.append((n, int(common.sum()), resid_rmse(v, gps_p, common)))
        for n, k, r in sorted(rows, key=lambda x: x[2]):
            print(f"{n:16s}{k:>4}{r:>12.3f}{r*cm:>11.1f}")
    else:
        print(f"  only {int(common.sum())} stations common to all methods -- "
              f"pass fewer --method args or a larger ROI")
    r_all = resid_rmse(sn_p, gps_p, m_all)
    print()
    print(f"  reference: SNAPHU over all {n_all} in-scene stations "
          f"{r_all:.3f} rad ({r_all*cm:.1f} cm)  [NOT comparable to the table: "
          f"different station set]")

    # per-station table + scatter
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    names = list(methods)
    with open(args.out + "_stations.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["station", "gps_phase", "snaphu_phase"] + [f"{n}_phase" for n in names])
        for r, gp in zip(recs, gps_p):
            w.writerow([r["station"], f"{gp:.3f}", f"{r['snaphu']:.3f}"]
                       + [f"{r[n]:.3f}" if np.isfinite(r[n]) else "" for n in names])
    print(f"  per-station: {args.out}_stations.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, a = plt.subplots(figsize=(6, 6))
        a.plot(gps_p, sn_p - (sn_p - gps_p).mean(), "o", label="SNAPHU", alpha=0.7)
        for n in names:
            v = np.array([r[n] for r in recs]); m = np.isfinite(v)
            if m.sum():
                a.plot(gps_p[m], v[m] - (v[m] - gps_p[m]).mean(), "s", label=n, alpha=0.7)
        lim = [gps_p.min(), gps_p.max()]
        a.plot(lim, lim, "k--", lw=1)
        a.set_xlabel("GPS LOS phase (rad)"); a.set_ylabel("InSAR unwrapped (rad, ref-aligned)")
        a.legend(frameon=False); fig.tight_layout()
        fig.savefig(args.out + "_scatter.png", dpi=130)
        print(f"  scatter: {args.out}_scatter.png")
    except Exception as e:
        print(f"  (scatter skipped: {e})")


if __name__ == "__main__":
    main()
