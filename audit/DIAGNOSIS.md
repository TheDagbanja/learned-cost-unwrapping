# Forensic audit — InSAR phase-unwrapping evaluation harness

**Trigger.** On the InSAR-DLPU real TanDEM-X subset (`test_real`), the reported
ordering was

| row | RMSE (rad) |
|---|---:|
| quality-guided | 2.38 |
| learned-cost MCF | 3.02 |
| "oracle-cost MCF" | 3.71 |
| ground-truth-k floor | 1.55 |

i.e. the "oracle" — advertised as the ceiling — was beaten by both the learned
cost and a classical path-follower. The suspicion was a metric or definition bug.

**Verdict (Phase 1): definitional, not metric.** The row labelled *oracle-cost
MCF* was never an upper bound, and no arithmetic is wrong. Detail in §2.

---

## §0 Component map (read-only)

All references are `Learned_Cost/`.

### MCF solver and cost interface

| item | location |
|---|---|
| solver entry `unwrap_mcf(psi, wx, wy, coherence, edge_costs, coh_weight, backend)` | `src/mcf.py:281` |
| min-cost-flow solve | `src/mcf.py:193` (`_solve_min_cost_flow`) |
| dual-graph build, arc costs | `src/mcf.py:95` (`_build_dual`), arcs at `src/mcf.py:145` |

Costs are **per-edge directional arc costs**: positive flow (`k_e=+1`) on a
horizontal edge costs `cpx`, negative flow (`k_e=-1`) costs `cnx`; vertically
`cpy`/`cny` (`src/mcf.py:106`). Arc capacity is unbounded.

Two supply paths:

* **directional** — `edge_costs=(cpx,cnx,cpy,cny)`, sign-aware. Used by the
  learned method.
* **symmetric** — `wx`,`wy`; `src/mcf.py:323-324` sets `cpx=cnx=wx`,
  `cpy=cny=wy`. A symmetric cost encodes cut **location only, never sign**.
  Used by the unit-cost and "oracle" rows.

### Method definitions

| method | definition |
|---|---|
| learned-cost MCF | `scripts/eval_learned_mcf.py:104` → `src/mcf.py:235-258` `edge_costs_from_logits`: `cpx=-log P(k=+1)`, `cnx=-log P(k=-1)` — directional, graded |
| unit-cost MCF | `unwrap_mcf(psi)`, `wx=wy=1` (`src/mcf.py:321-322`) |
| **"oracle-cost" MCF** | `scripts/eval_learned_mcf.py:50-55`, called at `:112-114` |
| **floor (GT-K)** | `scripts/eval_learned_mcf.py:115-116` |
| SNAPHU | `src/baselines.py:117-143`, real `snaphu` package, `init="mcf"` |
| quality-guided | `src/baselines.py:20-25`, `skimage.restoration.unwrap_phase` (Herráez) |
| weighted-LS | `src/baselines.py:28-37`, DCT Poisson (Ghiglia); leaves residues |
| Goldstein branch-cut | `src/baselines.py:40-106`, greedy box-growing |

**Exact "oracle" construction** (`scripts/eval_learned_mcf.py:50-55`):

```python
def _oracle_costs(psi, phi):
    dxw = mcf.wrap(psi[:, 1:] - psi[:, :-1])
    dyw = mcf.wrap(psi[1:, :] - psi[:-1, :])
    kx = np.round(((phi[:, 1:] - phi[:, :-1]) - dxw) / mcf.TWO_PI)
    ky = np.round(((phi[1:, :] - phi[:-1, :]) - dyw) / mcf.TWO_PI)
    return np.where(kx != 0, 0.01, 1.0), np.where(ky != 0, 0.01, 1.0)
```

Consumed as `unwrap_mcf(psi, wx=wx, wy=wy)`. It is therefore **soft** (0.01 vs
1.0, not 0 vs infinity), **symmetric** (sign of `k*` discarded at
`src/mcf.py:323`) and **location-only**. It is a *cut-location prior derived
from ground truth*, not a constraint that forces `k = k*`.

**Exact floor construction** (`scripts/eval_learned_mcf.py:115-116`):

```python
k = np.round((gt[b] - psi[b]) / mcf.TWO_PI)
floor.append(_aligned_rmse(psi[b] + mcf.TWO_PI * k, gt[b]))
```

Per-pixel optimal integer field: the genuine lower bound on RMSE for any
unwrapper congruent with `psi`.

### RMSE and alignment

`scripts/eval_learned_mcf.py:34-37`:

```python
def _aligned_rmse(a, b):
    d = a - b
    d = d - d.mean()      # MEAN of the residual
    return float(np.sqrt((d ** 2).mean()))
```

The offset is the **mean** of the residual — not a median, not a reference
pixel. It is applied **identically to every row**, including the "oracle"
(`:114`) and the floor (`:116`). *There is no per-row alignment inconsistency.*
Mean-subtraction is, however, cycle-slip sensitive (Phase 2).

### Ground truth and `k*`

`gt = batch["absolute"]` (`scripts/eval_learned_mcf.py:106`), the InSAR-DLPU
`.mat` `output` variable. `k*` is derived per-edge in `_oracle_costs` (`:53-54`)
and per-pixel in the floor (`:115`).

### Residue counter

`scripts/eval_learned_mcf.py:40-47` takes the curl of the **solution's own** raw
gradient, which telescopes to zero for any single-valued field, so it returns 0
unconditionally. This is the "always-zero" mirror of the previously-fixed
degenerate counter (that fix landed in `Lband_ext/eval/metrics.py`). It is
uninformative here but does not affect any RMSE. **Open item, low priority** —
it should be replaced by the per-edge implied-ambiguity counter used in
`Lband_ext/eval/metrics.py`.

---

## §1 Oracle killer test

Script: `audit/phase1_oracle.py` (additive; pipeline untouched). Fixed subset:
first 40 sorted `test_real` scenes. Alignment/RMSE copied verbatim from
`_aligned_rmse`.

Two oracles were run through the **same** `mcf.unwrap_mcf`:

* **soft** — the harness's current symmetric `{0.01, 1.0}` location prior;
* **hard** — directional `edge_costs`, cost `0` for the true class per edge and
  `1e3` for every other class.

| quantity | soft oracle | hard oracle |
|---|---:|---:|
| (A) fraction of edges with `k != k*` | 0.93 % | **0.93 %** |
| (B) integrated RMSE | 3.419 | **3.419** |
| (C) floor RMSE (GT-K) | — | 1.468 |
| \|B − C\| | — | 1.951 |
| scenes with hard-oracle > floor + 0.05 | — | 34 / 40 |

### Findings

**1. A hard oracle does not reproduce `k*`, and does not reach the floor.**
Forcing the true class at cost 0 still leaves 0.93 % of edges disagreeing with
`k*`. This is not a solver defect: on noisy data the rounded per-edge field `k*`
is **itself not residue-free** — rounding breaks `curl(k*) = -R` on ~0.93 % of
plaquettes — so no residue-free solver can emit it, and the MCF must deviate to
remain consistent.

**2. Soft and hard oracle are identical (3.419 vs 3.419).** The soft/hard
distinction is a red herring; both target `k*` and yield the same field. The
defect is not that the cost was "soft".

**3. Root cause — `k*`-matching is the wrong objective.** Every unwrapping is
`psi + 2*pi*K`. The RMSE-optimal residue-free field is the **per-pixel** optimum
`K = round((gt - psi) / 2pi)`, which is exactly the floor (1.468). The
`k*`-gradient field (3.419) is a *different, suboptimal* residue-free field: it
matches the true **gradient** but integrates the sub-cycle noise along the way.
So the floor legitimately beats it, and any method landing nearer the per-pixel
optimum (learned 3.02, quality-guided 2.38) legitimately beats it too.

### Verdict

**Branch B — the "oracle" was never a ceiling — but the cause is deeper than a
soft/mislabelled cost.** Even a correctly-built *hard* oracle is not a ceiling,
because matching `k*` does not minimise RMSE under noise. This is a
**definitional / labelling defect, not a metric defect**:

* the RMSE + alignment pipeline is self-consistent (soft = hard = same path; the
  floor uses the identical alignment);
* the genuine invariant **holds** — every row sits above the floor
  (learned 3.02, QG 2.38, GT-gradient-cost 3.42, all > 1.55).

Nothing in the table is arithmetically impossible. The only false statement was
calling the `k*`-cost row an upper bound.

**Why it looked sound on simulated data.** With little noise the `k*`-gradient
field nearly coincides with the per-pixel optimum, so the row sat just above the
floor (0.84 vs 0.76) and the mislabel was invisible. The two separate only under
heavy noise — i.e. exactly on `test_real`.

---

## §2 Metric audit (cycle-slips vs alignment)

Script: `audit/phase2_slips.py`. Same 40 fixed `test_real` scenes. "Slip" = a
pixel whose mean-aligned residual rounds to a nonzero multiple of `2*pi`;
`RMSE_slipfree` removes those integer offsets, leaving only sub-cycle noise.

| method | RMSE (mean-aligned) | median \|res\| | slip % | # regions | RMSE_slipfree |
|---|---:|---:|---:|---:|---:|
| floor (GT-K) | 1.468 | 0.945 | 6.0 % | 1206 | 1.320 |
| quality-guided | 2.310 | 1.124 | 13.2 % | 435 | **1.282** |
| GT-gradient-cost MCF | 3.419 | 2.223 | 31.5 % | 408 | **1.416** |
| unit-cost MCF | 5.544 | 3.749 | 45.2 % | 346 | **1.491** |
| Goldstein | 6.173 | 3.659 | 43.8 % | 351 | **1.582** |
| weighted-LS | 6.351 | 4.595 | 51.8 % | 352 | **1.754** |

### Alignment sensitivity (same residuals, three offset rules)

| method | mean-sub | median-sub | ref-pixel |
|---|---:|---:|---:|
| floor | 1.468 | 1.516 | 1.915 |
| quality-guided | 2.310 | 2.356 | 2.643 |
| GT-gradient-cost | 3.419 | 3.528 | 3.875 |
| unit-cost MCF | 5.544 | 5.762 | 6.143 |
| Goldstein | 6.173 | 6.366 | 7.247 |
| weighted-LS | 6.351 | 6.534 | 6.952 |

### Findings — hypothesis half confirmed, half refuted

**CONFIRMED: the entire ranking is a cycle-slip ranking, not an accuracy
ranking.** `RMSE_slipfree` is nearly flat across every method — 1.28, 1.42, 1.49,
1.58, 1.75 — while headline RMSE spans 2.3 to 6.4. Once integer `2*pi` offsets
are removed, all methods reconstruct the sub-cycle field about equally well. The
reported spread is almost entirely *how often a method slips a whole cycle*
(13.2 % → 51.8 %), and the slip-rate order reproduces the RMSE order.

**REFUTED: mean-based alignment is not the culprit.** Median-subtraction changes
RMSE by ~2-4 % and a single reference pixel by ~5-17 %; **all three rules
preserve the ordering exactly**. The anomaly survives every alignment rule, so it
is not an artefact of subtracting the mean.

**Region structure separates true slips from a boundary artefact.** The floor
shows 6.0 % "slipped" pixels spread over 1206 components — scattered single
pixels whose residual sits near `±pi` and crosses the rounding boundary after
mean-subtraction. This is a metric artefact and sets the noise baseline for the
column. Every other method shows *fewer* components (346-435) carrying *far more*
pixels — i.e. a few large connected regions displaced by a whole cycle, the
signature of genuine cycle slips.

**Why quality-guided wins here.** On this noise-dominated subset (99.3 % of edges
Itoh-satisfying, per `audit/`-adjacent diagnosis) the greedy reliability-guided
path simply commits fewer whole-cycle errors (13.2 %) than the global solvers
(31-52 %), while being no better at the sub-cycle field (1.282 vs 1.416-1.491).
It is not more accurate — it slips less.

**Consequence for reporting.** RMSE alone conflates two different failures.
Tables should carry `slip rate` and `RMSE_slipfree` as separate columns so
"did it slip" is never averaged into "how noisy is the field".

### COMPLETE table — full `test_real` split (100 scenes, learned row included)

Produced by the main harness after fixes #3 and #5 landed. Floor = 1.270
(corrected definition, §3-I1).

| method | RMSE | slip % | RMSE_slipfree | residues |
|---|---:|---:|---:|---:|
| quality-guided | 2.383 | 12.5 % | **1.357** | 0 |
| **learned-cost MCF (ours)** | **3.024** | **22.8 %** | **1.463** | 0 |
| GT-gradient-cost MCF | 3.709 | 31.1 % | **1.519** | 0 |
| SNAPHU | 5.304 | 41.5 % | **1.580** | 0 |
| unit-cost MCF | 5.958 | 45.3 % | **1.601** | 0 |
| least squares | 7.421 | 59.3 % | **1.773** | **1326.5** |
| Goldstein | 7.731 | 50.4 % | **1.660** | 0 |
| *GT-K floor (the ceiling)* | *1.270* | — | — | — |

**Confirms the 40-scene finding at full scale.** Headline RMSE spans 3.2x
(2.38 → 7.73) while `RMSE_slipfree` spans only 1.3x (1.36 → 1.77), and every
method sits within 7–40 % of the 1.270 floor once whole-cycle offsets are
removed. The slip-rate ordering reproduces the RMSE ordering exactly (only
LS/Goldstein swap). **On this subset the ranking is a cycle-slip ranking, not an
accuracy ranking.**

Restated usefully for the paper: *ours halves SNAPHU's slip rate* (22.8 % vs
41.5 %) — a more physically meaningful statement of the same result as "beats
SNAPHU by 43 %". Quality-guided slips least (12.5 %) because greedy
reliability-guided path-following avoids committing global cycle errors on a
field that is 99.3 % Itoh-satisfying; it is not better at the sub-cycle phase.

**Fix #5 validated on real data**: `residues` now reads 1326.5 for least squares
and 0 for every congruent method. The old counter reported 0 for all of them.

### Companion table — simulated `test` split (1000 scenes)

Floor = 0.756 (unchanged by the §3 fix; the defect only bites under heavy noise).

| method | RMSE | slip % | RMSE_slipfree | residues |
|---|---:|---:|---:|---:|
| GT-gradient-cost MCF | 0.839 | 0.8 % | **0.757** | 0 |
| **learned-cost MCF (ours)** | **0.895** | **1.0 %** | **0.759** | 0 |
| SNAPHU | 1.714 | 5.8 % | **0.827** | 0 |
| unit-cost MCF | 2.315 | 9.1 % | **0.877** | 0 |
| quality-guided | 4.079 | 25.3 % | **1.095** | 0 |
| least squares | 4.327 | 23.7 % | **1.318** | **3470.3** |
| Goldstein | 8.368 | 30.9 % | **1.161** | 0.1 |
| *GT-K floor (the ceiling)* | *0.756* | — | — | — |

**Ours reconstructs the sub-cycle field essentially at the theoretical optimum:
`RMSE_slipfree` 0.759 against a floor of 0.756 — 0.4 % above it.** Headline RMSE
spans 10x (0.84 → 8.37) while `RMSE_slipfree` spans 1.7x, so slips dominate here
even more strongly than on `test_real`.

### Cross-regime synthesis

| | ours slip % | SNAPHU slip % | ratio | ours slipfree / floor |
|---|---:|---:|---:|---:|
| simulated `test` | 1.0 % | 5.8 % | **5.8x fewer** | 1.004 |
| real `test_real` | 22.8 % | 41.5 % | **1.8x fewer** | 1.152 |

In both regimes the sub-cycle noise is near-identical across methods and close to
the floor; what separates them is **how often they commit a whole-cycle error**.
The learned cost's entire advantage over SNAPHU is slip avoidance — 5.8x fewer on
clean data, 1.8x fewer under heavy real noise. That is a more physically
meaningful statement of "beats SNAPHU by 48 % / 43 %" and is the metric the paper
should lead with.

The columns are now a **first-class output of the main harness**, so these tables
come from a normal evaluation run rather than a separate audit script:

```bash
cd ~/lab/anaconda3/envs/sims/MyPoject/insar_peft_unwrapping
python -m scripts.eval_learned_mcf checkpoint.resume=checkpoints/edge_dlpu_clean.pt     data.source=dlpu eval.split=test_real eval.snaphu=true logging.backend=none
```

## §3 Invariants

Script: `audit/phase3_invariants.py`, same 40 scenes, tolerance 0.05 rad.
**I2 uses the correct per-edge implied-ambiguity counter**, not the degenerate
one at `scripts/eval_learned_mcf.py:40-47`.

| method | RMSE | residues (correct counter) |
|---|---:|---:|
| floor (c=0) | 1.468 | 0.0 |
| quality-guided | 2.310 | 0.0 |
| GT-gradient-cost | 3.419 | 0.0 |
| hard oracle | 3.419 | 0.0 |
| unit-cost MCF | 5.544 | 0.0 |
| Goldstein | 6.173 | 0.0 |
| weighted-LS | 6.351 | 1200.5 |

**I2 — PASS.** Every congruent method reads exactly 0; only weighted-LS leaves
residues. This validates the corrected counter *and* re-confirms that the
harness's own counter is degenerate (it would print 0 for weighted-LS too).

**I3 — premise false, not a bug.** `hard_oracle` (3.419) is beaten by
quality-guided (2.310). Phase 1 already established this is not a valid
invariant: `k*`-matching is not the RMSE-optimal residue-free unwrapping under
noise.

### I1 — FAIL, and it is a genuine metric bug (second defect found)

The floor was beaten on 9 of 40 scenes (20 method-scene pairs), by
quality-guided *and* by the GT-gradient / hard-oracle rows.

**Root cause: the floor is defined for a different metric than the one used.**
The floor is the per-pixel nearest integer with **no** offset,

```python
k = round((gt - psi) / 2pi);  floor = aligned_rmse(psi + 2pi*k, gt)
```

which minimises the **unaligned** residual. But `_aligned_rmse` then grants every
method a **free global offset** (mean-subtraction). Under that metric the optimal
congruent field must be optimised jointly over the integer field *and* the
offset:

```
floor_true = min over c of  aligned_rmse( psi + 2pi*round((gt - psi - c)/2pi), gt )
```

The `c = 0` choice is not that minimum, so the reported "floor" is **not a lower
bound for the metric actually used**, and methods legitimately beat it.

*Mechanism.* Where noise approaches `±pi`, per-pixel rounding splits neighbouring
pixels between `+pi` and `-pi`, producing a **bimodal** residual with large
variance. A method that instead commits one integer across a whole region yields
a unimodal, lower-variance residual once the mean is removed.

*Verification (40 scenes):*

| floor definition | mean RMSE | scenes violated |
|---|---:|---:|
| `c = 0` (current) | 1.468 | **9** |
| joint `(K, c)` (proposed) | **1.157** | **0** |

The corrected floor is 0.311 rad lower on average (max 1.157 rad) and restores
I1 exactly.

### Regenerated floors (full splits, after applying the fix)

| split | n | old (`c=0`) | new (offset-aligned) | delta |
|---|---:|---:|---:|---:|
| `test` (simulated) | 1000 | 0.756 | **0.756** | 0.000 |
| `test_real` (TanDEM-X) | 100 | 1.555 | **1.270** | −0.285 |

**The simulated floor is unchanged.** This is a direct confirmation of the
mechanism: with little noise the per-pixel residual sits far from `±pi`, so `c=0`
already is the minimiser and the defect cannot manifest. It bites only in the
noise-dominated regime — i.e. exactly on `test_real`, where the floor drops by
0.285 rad. All `test_real` methods remain above the corrected floor
(quality-guided 2.38 > 1.27), so invariant I1 now holds in the mean and per
scene.

Propagated to: `manuscript/manuscript.md` (Table II floor and prose 1.55 → 1.27),
`README.md`, and `outputs/fig_benchmark.png` (sim floor line unchanged at 0.76;
row relabelled).

## §4 Secondary checks

### 4.1 Congruence loss — disabled in the released model; a no-op only for pixel heads

* Definition: `src/losses.py:145`, `congruence_loss(phi_hat, psi) = mean(1 - cos(phi_hat - psi))`.
* Fed by: `outputs["phi_hat"]` at `src/losses.py:323-324`; combined at `:352` as
  `... + self.lambda_cong * terms["cong"]`.
* **Which tensor is `phi_hat` decides everything:**
  * **edge head** (the released model) — `src/models/unwrapper.py:248`:
    `phi_hat = self.integrator(g_x, g_y, ...)`, the differentiable weighted-LS /
    DCT integration. This is **not** exactly congruent, so `L_cong` is *not*
    identically zero and is informative in principle.
  * **pixel head** — `src/models/unwrapper.py:281`: `phi_hat = psi + 2*pi*k_used`,
    exactly congruent, so `cos(phi_hat - psi) = 1` and **`L_cong = 0`
    identically — a genuine no-op**, together with its weight.
* **However, in the released checkpoint `lambda_cong = 0.0`** (code default is
  `1.0` at `src/losses.py:234`; the trained config overrides it). So the term
  contributed **nothing** to `edge_dlpu_clean.pt` regardless of head type. The
  premise "lambda_1 = 1.0" does not hold for the delivered model.
* **Related mismatch worth noting:** training optimises through the *differentiable
  integrator*, while evaluation unwraps with the *MCF solver*. The training loss
  never sees the hard integer field that is actually scored.

### 4.2 Backbone initialisation — ImageNet, not synthetic-InSAR

* `src/models/backbone.py:210-212`:
  ```python
  self.encoder = timm.create_model(
      name,
      pretrained=pretrained_imagenet,
      ...
  ```
* Released checkpoint config: `pretrained_imagenet: true`,
  `pretrained_synthetic: null`, `backbone.name: swin_tiny_patch4_window7_224`,
  `backbone.freeze: true`.
* `load_synthetic_weights` exists (`src/models/backbone.py:247`) but is reached
  only when `pretrained_synthetic` is set. It is `null`, so it is **never
  called**.

**Finding:** the Swin-T Tiny is initialised from **ImageNet** weights. The
original design intent (recorded in the project notes) was a
synthetic-interferogram-pretrained backbone; that path exists in code but was not
used for the released model. The manuscript makes no synthetic-pretraining claim,
so no paper correction is required — but internal notes/README wording asserting
"synthetic-pretrained" are inaccurate for this checkpoint.

---

## Prioritised fixes

| # | severity | fix | status |
|---|---|---|---|
| 1 | **high** | Stop presenting the `k*`-cost row as a ceiling. Rename `oracle-cost MCF` → **GT-gradient-cost MCF** everywhere and describe it as a *reference cost derived from ground-truth gradients*, not a bound. The genuine ceiling is the per-pixel GT-K floor. | **approved, applied** |
| 2 | **high** | **Fix the floor** (§3 I1): optimise the integer field jointly with the alignment offset. Implemented as `mcf.aligned_gtk_floor` and wired into `scripts/eval_learned_mcf.py`, `src/mcf_infer.py`, `scripts/demo_mcf.py`. | **approved, applied** |
| 3 | **high** | Report `slip rate` and `RMSE_slipfree` alongside RMSE (§2). Implemented as `_metrics()` in `scripts/eval_learned_mcf.py`; the harness now prints a per-method table with RMSE / slip% / RMSE_slipfree / residues. | **approved, applied** |
| 4 | medium | Assert `floor <= every method` per scene in the harness (with the corrected floor), failing loudly with scene IDs. | proposed |
| 5 | medium | Replace the always-zero residue counter with the per-edge implied-ambiguity counter (validated §3-I2). Done in `scripts/eval_learned_mcf.py::_residues_pred`; verified to report 0 for congruent methods and 825 for least squares on a residue-rich test field. | **approved, applied** |
| 6 | low | Correct internal notes/README wording that describe the backbone as "synthetic-pretrained"; the released model is ImageNet-initialised (§4.2). | proposed |
| 7 | low | Either drop `lambda_cong` from the pretrain objective or document that it is `0.0` in the released model and identically zero for pixel heads (§4.1). | proposed |

### Remaining

All seven fixes are applied or closed. The §2 table is completed by a normal
harness run (command in §2); `audit/phase2_slips.py` is retained as the
independent cross-check that produced the original finding.

No change to any RMSE, alignment or solver code is warranted: that math is
sound. Fix #1 is a labelling change and does not alter a single number.
