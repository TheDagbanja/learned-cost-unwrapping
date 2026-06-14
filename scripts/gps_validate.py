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
    ap.add_argument("--ours", required=True, help="our unwrapped GeoTIFF")
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

    snaphu_f = _find(args.product, "_unw_phase")
    theta_f = _find(args.product, "_lv_theta")
    phi_f = _find(args.product, "_lv_phi")

    gps = load_offsets(args.gps)
    print(f"loaded {len(gps)} stations from {len(args.gps)} offset file(s) "
          f"({'summed cumulative' if len(args.gps) > 1 else 'single event'})")
    src_s = rasterio.open(snaphu_f)
    src_o = rasterio.open(args.ours)
    src_t = rasterio.open(theta_f)
    src_p = rasterio.open(phi_f)

    recs = []
    for g in gps:
        th = sample_at(src_t, g["lon"], g["lat"])
        ph = sample_at(src_p, g["lon"], g["lat"])
        sn = sample_at(src_s, g["lon"], g["lat"])
        ou = sample_at(src_o, g["lon"], g["lat"])
        if None in (th, ph, sn) or th == 0 or sn == 0:
            continue                                    # outside scene / nodata
        # LOS unit vector toward sensor; d_los>0 = toward sensor
        le, ln, lu = np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), np.sin(th)
        d_los = g["de"] * le + g["dn"] * ln + g["du"] * lu
        gps_phase = -FOUR_PI_OVER_LAMBDA * d_los        # README sign: toward sensor -> -phase
        recs.append(dict(station=g["station"], gps=gps_phase, snaphu=sn,
                         ours=(ou if (ou is not None and ou != 0) else np.nan)))

    if not recs:
        raise SystemExit("No GPS stations fell within the scene/coverage.")
    gps_p = np.array([r["gps"] for r in recs])
    sn_p = np.array([r["snaphu"] for r in recs])
    ou_p = np.array([r["ours"] for r in recs])
    in_ours = np.isfinite(ou_p)

    # self-calibrate the sign convention using SNAPHU (the trusted baseline):
    # SNAPHU must correlate POSITIVELY with GPS LOS phase; flip if not.
    sign = 1.0
    if np.corrcoef(sn_p, gps_p)[0, 1] < 0:
        sign = -1.0
        gps_p = -gps_p
        print("note: flipped GPS phase sign to match SNAPHU convention.")

    def resid_rmse(pred, gt, m):
        d = pred[m] - gt[m]
        d = d - d.mean()                                # remove the reference constant
        return float(np.sqrt((d ** 2).mean()))

    n_all = len(recs)
    m_all = np.ones(n_all, bool)
    rmse_sn = resid_rmse(sn_p, gps_p, m_all)
    print(f"\n=== GPS validation ({n_all} stations in scene; "
          f"{int(in_ours.sum())} within our ROI) ===")
    print(f"  SNAPHU vs GPS LOS : rmse {rmse_sn:.3f} rad "
          f"({rmse_sn * LAMBDA / (4 * np.pi) * 100:.1f} cm)")
    if in_ours.sum() >= 3:
        m = in_ours
        rs = resid_rmse(sn_p, gps_p, m)
        ro = resid_rmse(ou_p, gps_p, m)
        print(f"  on the {int(m.sum())} stations inside our ROI:")
        print(f"    SNAPHU vs GPS : {rs:.3f} rad ({rs*LAMBDA/(4*np.pi)*100:.1f} cm)")
        print(f"    OURS   vs GPS : {ro:.3f} rad ({ro*LAMBDA/(4*np.pi)*100:.1f} cm)")
        better = "OURS" if ro < rs else "SNAPHU"
        print(f"    --> closer to GPS: {better}  (delta {abs(ro-rs):.3f} rad)")
    else:
        print(f"  WARNING: only {int(in_ours.sum())} GPS stations inside our ROI; "
              f"re-run unwrap_geotiff with a larger geotiff.size to cover more.")

    # per-station table + scatter
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + "_stations.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["station", "gps_phase", "snaphu_phase", "ours_phase"])
        for r, gp in zip(recs, gps_p):
            w.writerow([r["station"], f"{gp:.3f}", f"{r['snaphu']:.3f}",
                        f"{r['ours']:.3f}" if np.isfinite(r["ours"]) else ""])
    print(f"  per-station: {args.out}_stations.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        off_s = (sn_p - gps_p).mean()
        fig, a = plt.subplots(figsize=(6, 6))
        a.plot(gps_p, sn_p - off_s, "o", label="SNAPHU", alpha=0.7)
        if in_ours.sum():
            off_o = (ou_p[in_ours] - gps_p[in_ours]).mean()
            a.plot(gps_p[in_ours], ou_p[in_ours] - off_o, "s", label="ours", alpha=0.7)
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
