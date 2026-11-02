"""Fast invariant tests for the audited evaluation rebuild."""

from __future__ import annotations

import numpy as np

from src import mcf
from src.dataset import _select_files
from src.metrics import decomposition_metrics, summarize_method_rows
from src.synthetic_engine import SyntheticConfig, SyntheticInterferogramGenerator
from scripts.eval_real_uncertainty import _capture, _cycle_disagreement, _top_fraction


def main() -> None:
    # Perfect reconstruction modulo a global constant has zero decomposition error.
    rng = np.random.default_rng(7)
    reference = rng.normal(size=(16, 16))
    wrapped = mcf.wrap(reference)
    metrics = decomposition_metrics(reference + 12.3, reference, wrapped)
    assert metrics["rmse"] < 1e-12
    assert metrics["slip"] == 0.0
    assert metrics["eps"] < 1e-12
    assert metrics["residues"] == 0.0

    # A regional 2pi offset is identified as a cycle slip. The shared global
    # alignment leaves a quarter-cycle slip-free offset in this construction.
    slipped = reference.copy(); slipped[:, 12:] += mcf.TWO_PI
    metrics = decomposition_metrics(slipped, reference, wrapped)
    assert 0.24 < metrics["slip"] < 0.26
    assert abs(metrics["eps"] - np.pi / 2) < 1e-12

    # Runtime distribution is carried into the same method summary as error tails.
    rows = [
        {"rmse": value, "slip": 0, "eps": 0, "residues": 0, "runtime_s": value / 10}
        for value in range(1, 101)
    ]
    summary = summarize_method_rows(rows)
    assert summary["p95"] == np.quantile(np.arange(1, 101), 0.95)
    assert summary["runtime_p95_s"] == np.quantile(np.arange(1, 101) / 10, 0.95)
    assert summary["cat_rmse_2pi"] == 0.94
    assert summary["cat_slip_005"] == 0.0

    # Classical-baseline calibration can use a reproducible, unaugmented
    # subset without changing the historical head-selection semantics.
    files = [f"scene-{index:02d}" for index in range(20)]
    assert _select_files(files, 3) == files[:3]
    assert _select_files(files, 5, "even") == [files[i] for i in (0, 4, 9, 14, 19)]
    assert _select_files(files, 5, "random", 17) == _select_files(files, 5, "random", 17)

    # Strict domain mode emits exactly the requested mechanism label.
    for domain in ("fault", "volcanic", "subsidence", "atmosphere"):
        cfg = SyntheticConfig(
            img_size=32, domain_protocol=True, domain_types=(domain,),
            domain_weights=(1.0,), forced_domain=domain,
        )
        sample = SyntheticInterferogramGenerator(cfg).generate(seed=11)
        assert sample["domain"] == domain
        assert sample["wrapped"].shape == (32, 32)

    # Fixed review coverage flags exactly the declared highest-uncertainty tail.
    score = np.arange(20, dtype=float)
    flagged = _top_fraction(score, np.ones(20, dtype=bool), 0.20)
    assert flagged.sum() == 4
    assert np.flatnonzero(flagged).tolist() == [16, 17, 18, 19]
    target = np.zeros(20, dtype=bool); target[[1, 18, 19]] = True
    assert _capture(target, flagged) == 2 / 3

    # Global phase constants are removed, but a regional whole-cycle difference
    # remains a solver-disagreement endpoint.
    base = rng.normal(size=(8, 8))
    shifted = base + 3.7
    shifted[:, 6:] += mcf.TWO_PI
    disagreement, valid, offsets = _cycle_disagreement(
        {"ours": base, "other": shifted}, np.ones_like(base, dtype=bool)
    )
    assert valid.all()
    assert np.isclose(offsets["other"], 3.7)
    assert disagreement[:, 6:].all() and not disagreement[:, :6].any()

    print("[test_evaluation_rebuild] PASSED")


if __name__ == "__main__":
    main()
