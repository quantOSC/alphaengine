"""The sweep runner and the study artifact."""

from __future__ import annotations

import json

import numpy as np
import pytest

from alphaengine.study import SCHEMA_VERSION, Study, load, save
from alphaengine.sweep import sweep


def prices(seed: int = 42, n: int = 600) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100 * np.cumprod(1 + rng.normal(0.0004, 0.011, n))


def ma_cross(data=None, fast: int = 10, slow: int = 50):
    """A plain moving-average rule, standing in for a user's own backtest."""
    p = np.asarray(data, dtype=float)
    r = np.diff(p, prepend=p[0]) / p
    f = np.convolve(p, np.ones(fast) / fast, mode="same")
    s = np.convolve(p, np.ones(slow) / slow, mode="same")
    pos = (f > s).astype(float)
    return (np.r_[0.0, pos[:-1]] * r).tolist()


# ── the trial count ────────────────────────────────────────────────────────
def test_n_trials_is_derived_from_the_grid():
    """The reason this module exists. The count is the grid, not an argument."""
    r = sweep(ma_cross, {"fast": [5, 10, 15], "slow": [50, 100]}, data=prices())
    assert r.n_trials == 6
    assert r.verdict()["n_trials"] == 6
    assert r.verdict()["n_trials_source"] == "derived_from_grid"


def test_there_is_no_way_to_assert_a_trial_count():
    """`sweep()` must not accept an n_trials override.

    Pinned as a test because it is a design guarantee rather than an
    implementation detail: the moment a caller can supply this number, the
    deflation becomes self-reported and the library stops being worth using.
    """
    import inspect

    params = inspect.signature(sweep).parameters
    assert "n_trials" not in params
    assert not any("trial" in p.lower() for p in params)


def test_larger_grid_deflates_harder():
    p = prices()
    small = sweep(ma_cross, {"fast": [10], "slow": [50]}, data=p)
    large = sweep(ma_cross, {"fast": [5, 8, 10, 12, 15, 20], "slow": [40, 50, 80, 100, 150, 200]}, data=p)
    assert large.n_trials > small.n_trials
    # Same underlying rule family, more search, so the honest verdict cannot be
    # more flattering. This is the property a user is relying on.
    assert large.verdict()["deflated_sharpe"] <= small.verdict()["deflated_sharpe"] + 1e-9


# ── the surface ────────────────────────────────────────────────────────────
def test_surface_classifies_the_neighbourhood():
    r = sweep(ma_cross, {"fast": [5, 10, 15, 20], "slow": [50, 100, 200]}, data=prices())
    s = r.surface()
    assert s["shape"] in {"plateau", "ridge", "knife_edge"}
    assert s["n_ok"] + s["n_failed"] == r.n_trials
    assert 0.0 <= s["share_within_20pct_of_best"] <= 1.0
    assert s["reading"]


def test_matrix_is_the_full_trial_set():
    """PBO needs every column. A single-point run structurally cannot produce it."""
    r = sweep(ma_cross, {"fast": [5, 10, 15], "slow": [50, 100]}, data=prices())
    assert r.matrix.shape[1] == 6
    assert r.verdict()["selection"]["n_configs"] == 6


def test_single_configuration_reports_no_selection_reading():
    """One config means the choice among configs was not a choice."""
    r = sweep(ma_cross, {"fast": [10], "slow": [50]}, data=prices())
    assert "selection" not in r.verdict()


# ── parameters are treated as IP ───────────────────────────────────────────
def test_params_are_not_stored_by_default():
    """The grid is frequently bigger IP than the returns."""
    r = sweep(ma_cross, {"fast": [5, 10], "slow": [50, 100]}, data=prices())
    v = r.verdict()
    assert "best_params" not in v
    assert v["best_params_hash"]           # comparable without being disclosed

    r2 = sweep(ma_cross, {"fast": [5, 10], "slow": [50, 100]}, data=prices(), store_params=True)
    assert r2.verdict()["best_params"]


def test_data_identity_is_a_hash_not_a_label():
    a = sweep(ma_cross, {"fast": [10], "slow": [50]}, data=prices(seed=1))
    b = sweep(ma_cross, {"fast": [10], "slow": [50]}, data=prices(seed=1))
    c = sweep(ma_cross, {"fast": [10], "slow": [50]}, data=prices(seed=2))
    assert a.data_hash == b.data_hash, "same data must be the same segment"
    assert a.data_hash != c.data_hash, "different data must not collide"


# ── failure handling ───────────────────────────────────────────────────────
def test_one_bad_combination_does_not_lose_the_rest():
    def flaky(data=None, fast: int = 10, slow: int = 50):
        if fast == 15:
            raise ValueError("deliberate failure")
        return ma_cross(data=data, fast=fast, slow=slow)

    r = sweep(flaky, {"fast": [5, 10, 15], "slow": [50]}, data=prices())
    assert r.n_trials == 3
    assert r.surface()["n_failed"] == 1
    assert r.surface()["n_ok"] == 2


def test_all_failing_raises_rather_than_returning_a_verdict():
    def broken(data=None, **_):
        raise RuntimeError("no")

    with pytest.raises(RuntimeError, match="every one of"):
        sweep(broken, {"fast": [5, 10]}, data=prices())


def test_empty_grid_is_rejected():
    with pytest.raises(ValueError, match="grid is empty"):
        sweep(ma_cross, {}, data=prices())


# ── the study artifact ─────────────────────────────────────────────────────
def test_study_roundtrip(tmp_path):
    r = sweep(ma_cross, {"fast": [5, 10, 15], "slow": [50, 100]}, data=prices())
    p = r.save(tmp_path / "study.json", label="ma cross", data_description="synthetic")

    back = load(p)
    assert back.label == "ma cross"
    assert back.n_trials == 6
    assert back.n_trials_source == "derived_from_grid"
    assert back.data_hash == r.data_hash
    assert back.schema_version == SCHEMA_VERSION
    assert back.surface["shape"] in {"plateau", "ridge", "knife_edge"}


def test_study_holds_no_series(tmp_path):
    """A study is derived facts and references. The inputs stay with their owner."""
    r = sweep(ma_cross, {"fast": [5, 10], "slow": [50]}, data=prices())
    raw = json.loads(r.save(tmp_path / "s.json").read_text(encoding="utf8"))

    blob = json.dumps(raw)
    assert "matrix" not in raw
    assert "returns" not in raw
    assert "prices" not in raw
    # A price series would be a long run of floats; nothing here should be one.
    for value in raw.values():
        if isinstance(value, list):
            assert len(value) < 50, "a study should not carry a series"
    assert len(blob) < 20_000


def test_study_refuses_an_unknown_major_schema(tmp_path):
    """Refuse rather than half-parse. These figures get quoted to investors."""
    p = tmp_path / "future.json"
    save(Study(label="from the future"), p)
    raw = json.loads(p.read_text(encoding="utf8"))
    raw["schema_version"] = "99.0"
    p.write_text(json.dumps(raw), encoding="utf8")

    with pytest.raises(ValueError, match="cannot be read"):
        load(p)


def test_study_tolerates_unknown_minor_fields(tmp_path):
    """A newer minor version must stay readable by an older build."""
    p = tmp_path / "newer.json"
    save(Study(label="x", n_trials=3), p)
    raw = json.loads(p.read_text(encoding="utf8"))
    raw["schema_version"] = f"{SCHEMA_VERSION.split('.')[0]}.99"
    raw["a_field_from_the_future"] = {"anything": True}
    p.write_text(json.dumps(raw), encoding="utf8")

    back = load(p)
    assert back.label == "x"
    assert back.n_trials == 3
