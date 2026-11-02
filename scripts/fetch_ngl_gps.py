"""Fetch coseismic GPS offsets from the Nevada Geodetic Laboratory (NGL) for a
second/independent GPS-validation event, in the exact format
``scripts/gps_validate.py`` already reads.

Uses the SAME source as the existing Ridgecrest validation (NGL reprocesses
GEONET, RING, and most global continuous networks under one consistent
reference frame), so a second event is methodologically consistent with the
first rather than mixing processing chains.

Pipeline: download NGL's master station list, keep stations within
``--radius-km`` of the epicenter with data spanning both the pre- and
post-event windows, download each candidate's daily ``.tenv3`` position time
series, and take the coseismic offset as (mean position in the post window)
minus (mean position in the pre window) -- the same six-hour-style
before/after differencing NGL itself uses, just windowed to match the SAR
acquisition dates instead of the earthquake origin time, so the GPS offset
spans exactly what the interferogram spans (cumulative foreshock+mainshock
displacement included, matching the Ridgecrest precedent).

    python -m scripts.fetch_ngl_gps \\
        --name kumamoto --lat 32.7546 --lon 130.7630 --radius-km 100 \\
        --pre-start 2016-04-01 --pre-end 2016-04-07 \\
        --post-start 2016-04-20 --post-end 2016-04-27 \\
        --out data/gps_kumamoto.txt

Output columns match ``gps_validate.read_gps``'s NGL whitespace format:
``Sta Lon Lat de dn du`` (meters, WGS84, de/dn/du = east/north/up offset).
"""

from __future__ import annotations

import argparse
import io
import math
import sys
import urllib.request

import numpy as np

HOLDINGS_URL = "https://geodesy.unr.edu/NGLStationPages/DataHoldings.txt"
TENV3_URL = "https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20/{sta}.tenv3"


def _get(url: str, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def find_candidates(lat0, lon0, radius_km, date_min, date_max):
    print(f"Downloading NGL station holdings list...")
    txt = _get(HOLDINGS_URL).decode("utf-8", errors="replace")
    cands = []
    for line in io.StringIO(txt).readlines()[1:]:
        parts = line.split()
        if len(parts) < 9:
            continue
        sta, lat, lon = parts[0], float(parts[1]), float(parts[2])
        dtbeg, dtend = parts[7], parts[8]
        if not (dtbeg <= date_min and dtend >= date_max):
            continue
        d = _haversine_km(lat0, lon0, lat, lon)
        if d <= radius_km:
            cands.append((sta, lat, lon, d))
    cands.sort(key=lambda r: r[3])
    print(f"  {len(cands)} candidate station(s) within {radius_km:.0f} km with "
          f"data spanning [{date_min}, {date_max}]")
    return cands


def _mjd_window_mean(tenv3_text: str, d0: str, d1: str):
    """Mean (east, north, up, n_days) over calendar dates in [d0, d1] (YYYY-MM-DD)."""
    import datetime
    t0 = datetime.date.fromisoformat(d0)
    t1 = datetime.date.fromisoformat(d1)
    es, ns, us = [], [], []
    lon_ref = lat_ref = None
    for line in tenv3_text.splitlines()[1:]:
        p = line.split()
        if len(p) < 21:
            continue
        yymmmdd = p[1]                          # e.g. 16APR10
        try:
            dt = datetime.datetime.strptime(yymmmdd, "%y%b%d").date()
        except ValueError:
            continue
        if not (t0 <= dt <= t1):
            continue
        es.append(float(p[8])); ns.append(float(p[10])); us.append(float(p[12]))
        lat_ref = float(p[-3]); lon_ref = float(p[-2])
    if not es:
        return None
    # NGL's raw longitude isn't consistently -180..180 or 0..360 (e.g. Japan
    # stations appear as -229.xx meaning 130.xx E); wrap to standard -180..180
    # rather than blindly adding 360, which would corrupt genuine West
    # longitudes (e.g. California's -117.x would become the wrong 242.4).
    lon = None if lon_ref is None else ((lon_ref + 180.0) % 360.0) - 180.0
    return dict(e=float(np.mean(es)), n=float(np.mean(ns)), u=float(np.mean(us)),
               n_days=len(es), lat=lat_ref, lon=lon)


def offset_for_station(sta, pre0, pre1, post0, post1):
    try:
        raw = _get(TENV3_URL.format(sta=sta)).decode("utf-8", errors="replace")
    except Exception as e:
        return None, f"download failed ({e})"
    if raw.lstrip().startswith("<"):
        return None, "no tenv3 file (404)"
    pre = _mjd_window_mean(raw, pre0, pre1)
    post = _mjd_window_mean(raw, post0, post1)
    if pre is None or post is None:
        return None, f"insufficient data in window (pre={pre is not None}, post={post is not None})"
    de, dn, du = post["e"] - pre["e"], post["n"] - pre["n"], post["u"] - pre["u"]
    return dict(lon=post["lon"], lat=post["lat"], de=de, dn=dn, du=du,
               n_pre=pre["n_days"], n_post=post["n_days"]), None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="event name, for logging only")
    ap.add_argument("--lat", type=float, required=True, help="epicenter latitude (deg)")
    ap.add_argument("--lon", type=float, required=True, help="epicenter longitude (deg, 0-360 or -180-180)")
    ap.add_argument("--radius-km", type=float, default=100.0)
    ap.add_argument("--pre-start", required=True); ap.add_argument("--pre-end", required=True)
    ap.add_argument("--post-start", required=True); ap.add_argument("--post-end", required=True)
    ap.add_argument("--min-days", type=int, default=2,
                    help="minimum days of data required in EACH window to keep a station")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cands = find_candidates(args.lat, args.lon, args.radius_km, args.pre_start, args.post_end)
    if not cands:
        raise SystemExit("No candidate stations found; widen --radius-km or check the date range.")

    rows = []
    for i, (sta, lat, lon, dist) in enumerate(cands):
        off, err = offset_for_station(sta, args.pre_start, args.pre_end, args.post_start, args.post_end)
        if err:
            print(f"  [{i+1}/{len(cands)}] {sta} ({dist:5.1f} km)  skip: {err}")
            continue
        if off["n_pre"] < args.min_days or off["n_post"] < args.min_days:
            print(f"  [{i+1}/{len(cands)}] {sta} ({dist:5.1f} km)  skip: too few days "
                  f"(pre={off['n_pre']}, post={off['n_post']})")
            continue
        mag = math.sqrt(off["de"] ** 2 + off["dn"] ** 2 + off["du"] ** 2)
        print(f"  [{i+1}/{len(cands)}] {sta} ({dist:5.1f} km)  "
              f"de={off['de']*100:6.1f}cm dn={off['dn']*100:6.1f}cm du={off['du']*100:6.1f}cm  "
              f"|d|={mag*100:.1f}cm  (pre={off['n_pre']}d, post={off['n_post']}d)")
        rows.append((sta, off["lon"], off["lat"], off["de"], off["dn"], off["du"]))

    if not rows:
        raise SystemExit("No usable stations after filtering -- widen the search or windows.")

    with open(args.out, "w") as f:
        f.write(f"# NGL coseismic offsets for {args.name}, source: {HOLDINGS_URL}\n")
        f.write(f"# pre=[{args.pre_start},{args.pre_end}] post=[{args.post_start},{args.post_end}]\n")
        f.write("Sta Lon Lat de dn du\n")
        for sta, lon, lat, de, dn, du in rows:
            f.write(f"{sta} {lon:.6f} {lat:.6f} {de:.4f} {dn:.4f} {du:.4f}\n")
    print(f"\nwrote {len(rows)} station offsets to {args.out}")


if __name__ == "__main__":
    main()
