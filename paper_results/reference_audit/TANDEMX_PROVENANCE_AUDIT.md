# TanDEM-X reference provenance audit

## Finding

The 100 InSAR-DLPU TanDEM-X targets should be described as **DEM-derived reference
absolute phase**, not as independently verified physical ground truth.  The available
evidence does not indicate that the targets were produced by SNAPHU or another phase
unwrapper: the later PUformer paper by the dataset authors identifies the displayed
TanDEM-X reference as obtained from SRTM DEM.  However, the released `.mat` files do
not include the geolocation, baseline, incidence geometry, phase-to-height conversion,
coregistration, filtering, uncertainty, or manual-correction record needed to reproduce
or independently validate that conversion.

## Source trail

1. The [official InSAR-DLPU repository](https://github.com/zhoulifan/InSAR-DLPU) describes 100 real
   TanDEM-X interferograms and paired absolute-phase patches, and labels the latter
   ground truth.  It does not document their construction.
2. The dataset article ([DOI 10.1109/MGRS.2024.3359691](https://doi.org/10.1109/MGRS.2024.3359691)) establishes
   the benchmark and its SRTM basis but the public abstract and repository do not
   supply scene-level provenance.
3. The later paper by the dataset authors, PUformer
   ([DOI 10.1109/JSTARS.2026.3677162](https://doi.org/10.1109/JSTARS.2026.3677162)), states in the Fig. 8 caption
   that the TanDEM-X absolute phase is obtained from SRTM DEM.  This is the clearest
   published provenance statement located in this audit.

## Local release audit

- Pair count: 100.
- Array shapes: [[[256, 256], [256, 256]]].
- Each wrapped file contains only the MATLAB variable `input`; each paired target
  contains only `output`.
- The release contains no paired coherence arrays, geolocation, acquisition metadata,
  sensor geometry, or per-scene reference-generation log.
- Mean circular discrepancy after removing the per-scene global circular offset between
  the released wrapped observation and the wrapped DEM-derived reference:
  1.270 rad (mean circular resultant length
  0.440).  This confirms
  that the observation is not merely an exact numerical rewrap of the target and is
  consistent with sensor/noise and processing differences, but it does not establish
  physical accuracy.
- Median target peak-to-peak range: 44.85 rad.
- File-level SHA-256 hashes and per-scene diagnostics are recorded in
  `tandemx_reference_inventory.csv`.

## Consequence for evaluation and wording

The subset can measure agreement with a DEM-derived topographic reference.  It cannot,
by itself, resolve which phase-unwrapping result is physically correct where the methods
and reference disagree.  Tables should use `Reference RMSE`, the text should use
`TanDEM-X/SRTM reference subset`, and claims of real-data superiority should rely on
independent measurements or be withdrawn.  The Sentinel-1 GPS experiment remains an
independent check, but its current precision supports comparability among the
solver-based methods rather than superiority of one method.
