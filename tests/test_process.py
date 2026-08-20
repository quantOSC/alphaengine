"""Named processes, DGP stress, Grinold, and the chart contract.

Existing goldens in test_goldens.py stay frozen. These pin the NEW public
figures and the wire shape: paths, alpha vectors and variance paths stay in
the workspace.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from alphaengine.charts import CHART_KINDS, chart
from alphaengine.client import StepExecutor, UnsupportedOp
from alphaengine.client.executor import _guard
from alphaengine.core import (
    breadth_ir,
    dgp_stress,
    garch_calibrate,
    grinold_alpha,
    ou_calibrate,
    ou_simulate,
    variance_explained,
)
from alphaengine.core.process import histogram, sketch_curve
from alphaengine.demo import data as demo_data


def _ou_path(*, kappa=0.12, theta=1.0, sigma=0.05, n=800, seed=7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = theta
    for t in range(1, n):
        x[t] = x[t - 1] + kappa * (theta - x[t - 1]) + sigma * rng.normal()
    return x


def _garch_returns(*, omega=1e-5, alpha=0.08, beta=0.88, n=400, seed=3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    s2 = omega / max(1.0 - alpha - beta, 1e-6)
    for t in range(n):
        s2 = max(omega + alpha * (r[t - 1] ** 2 if t else s2) + beta * s2, 1e-18)
        r[t] = math.sqrt(s2) * rng.normal()
    return r


def _cs_panel(n_names: int = 12, n_obs: int = 60, seed: int = 7) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    return {f"N{i:02d}": rng.normal(i * 0.1, 1.0, n_obs).tolist() for i in range(n_names)}


def test_ou_recovers_kappa_on_a_planted_path():
    kappa = 0.12
    cal = ou_calibrate(_ou_path(kappa=kappa))
    assert cal["mean_reverting"] is True
    assert cal["kappa"] == pytest.approx(kappa, abs=0.06)
    assert cal["half_life"] is not None
    assert len(cal["path_sketch"]) <= 256


def test_garch_persistence_is_inside_the_unit_interval():
    cal = garch_calibrate(_garch_returns())
    persist = cal["persistence"]
    assert persist is not None
    assert 0.0 < persist < 1.0
    assert cal["vol_path"]
    assert "var_path" in cal  # local only; the executor pops this before the wire


def test_dgp_stress_trial_count_is_the_path_count():
    out = dgp_stress(_ou_path(), dgp="ou", n_paths=40, seed=7)
    assert out["n_trials"] == 40
    assert out["n_paths"] == 40
    assert out["n_trials_source"] == "monte_carlo"
    assert out["dgp"] == "ou"
    assert "paths" not in out
    hist = out["sharpe_hist"]
    assert hist["edges"] and hist["counts"]
    assert len(hist["counts"]) <= 24


def test_gbm_on_zero_mean_noise_does_not_claim_an_edge():
    """A DGP stress is not a strategy verdict. Zero-mean noise stays a coin flip."""
    rng = np.random.default_rng(7)
    r = rng.normal(0.0, 0.01, 800)
    r = r - r.mean()
    out = dgp_stress(r, dgp="gbm", n_paths=80, seed=7)
    assert "verdict" not in out
    assert out["n_trials"] == 80
    assert out["n_trials_source"] == "monte_carlo"
    assert out["frac_positive"] is not None
    assert abs(out["frac_positive"] - 0.5) < 0.25
    assert out["sharpe_q50"] is not None
    assert abs(out["sharpe_q50"]) < 1.5


def test_gbm_on_the_demo_walk_records_the_path_count():
    out = dgp_stress(demo_data["close"], dgp="gbm", n_paths=40, seed=7)
    assert out["n_trials"] == 40
    assert out["n_trials_source"] == "monte_carlo"
    assert "verdict" not in out
    assert "paths" not in out


def test_guard_refuses_a_raw_path_list():
    with pytest.raises(ValueError, match="is a series"):
        _guard({"paths": list(range(600))})


def test_sketch_and_histogram_stay_under_the_cap():
    x = np.linspace(0.0, 1.0, 4000)
    assert len(sketch_curve(x)) <= 256
    h = histogram(x, bins=24)
    assert len(h["counts"]) == 24
    assert len(h["edges"]) == 25


def test_ou_simulate_keeps_raw_paths_for_the_caller():
    sim = ou_simulate(kappa=0.1, theta=0.0, sigma=0.02, n_obs=80, n_paths=12, seed=1)
    assert sim["paths"].shape == (12, 80)
    assert sim["n_trials"] == 12
    assert sim["band"]


def test_chart_hints_are_kind_key_title():
    hint = chart("hist", "sharpe_hist", "Sharpe under the DGP")
    assert hint == {"kind": "hist", "key": "sharpe_hist", "title": "Sharpe under the DGP"}
    assert "curve" in CHART_KINDS
    with pytest.raises(ValueError, match="unknown chart kind"):
        chart("sparkline", "x", "no")


def test_grinold_alpha_vector_stays_with_the_caller():
    panel = _cs_panel()
    out = grinold_alpha(panel, ic=0.05)
    assert out["alpha"]
    assert out["n_names"] == 12
    ir = breadth_ir(ic=0.05, n_names=12)
    assert ir["transfer_coefficient"] == 1.0
    assert ir["implied_ir"] == pytest.approx(0.05 * math.sqrt(12), abs=1e-6)


def test_executor_garch_and_dgp_stress_do_not_emit_series():
    ex = StepExecutor(data=demo_data)
    g = ex.execute("compute.garch", {})
    assert "var_path" not in g
    assert "garch_var" in ex.workspace
    assert g["charts"][0]["kind"] == "curve"
    s = ex.execute("compute.dgp_stress", {"dgp": "gbm", "n_paths": 30})
    assert s["n_trials"] == 30
    assert s["n_trials_source"] == "monte_carlo"
    assert "paths" not in s
    assert s["charts"][0]["kind"] == "hist"


def test_executor_ou_simulate_strips_paths():
    ex = StepExecutor(data={"spread": _ou_path().tolist()})
    cal = ex.execute("compute.ou_calibrate", {})
    assert "paths" not in cal
    sim = ex.execute("compute.ou_simulate", {"n_paths": 20, "n_obs": 60})
    assert "paths" not in sim
    assert "ou_paths" in ex.workspace
    assert sim["charts"][0]["kind"] == "curve"


def test_executor_grinold_strips_alpha():
    ex = StepExecutor(data=_cs_panel())
    out = ex.execute("compute.grinold_alpha", {"ic": 0.04})
    assert "alpha" not in out
    assert "alpha" in ex.workspace
    assert out["n_names"] == 12


def test_executor_detone_emits_charts_not_the_matrix():
    rng = np.random.default_rng(9)
    data = {f"S{i}": rng.normal(0, 0.01, 80).tolist() for i in range(6)}
    ex = StepExecutor(data=data)
    out = ex.execute("compute.detone_cov", {})
    assert "cov" not in out
    assert "cov" in ex.workspace
    assert out["method"] == "detone"
    assert out["variance_explained"]
    assert any(h["kind"] == "triangle" for h in out["charts"])
    shares = [row["share"] for row in out["variance_explained"]]
    assert pytest.approx(sum(shares), abs=1e-5) == 1.0


def test_variance_explained_is_a_pca_sketch():
    cov = np.diag([4.0, 1.0, 0.5])
    rows = variance_explained(cov)
    assert rows[0]["k"] == 1
    assert rows[0]["share"] == pytest.approx(4.0 / 5.5, abs=1e-6)


def test_grinold_without_ic_is_refused():
    ex = StepExecutor(data=_cs_panel())
    with pytest.raises(UnsupportedOp, match="params.ic"):
        ex.execute("compute.grinold_alpha", {})
