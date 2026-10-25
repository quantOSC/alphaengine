"""The public surface: what a user gets from `import alphaengine`.

These are cheap tests guarding the first thirty seconds of somebody's
experience with the package, which is the part no internal test naturally
covers because internal tests import from the module that defines a thing.
"""

import alphaengine


def test_version_is_exposed():
    import importlib.metadata
    import inspect

    import alphaengine._version as ver_mod

    assert isinstance(alphaengine.__version__, str)
    assert alphaengine.__version__ == importlib.metadata.version("alphaengine")
    src = inspect.getsource(ver_mod)
    assert "importlib.metadata" in src


def test_combinatorial_oos_stability_and_deprecated_cpcv_alias():
    import pytest

    from alphaengine import combinatorial_oos_stability, cpcv_score

    returns = [0.002 * ((t % 7) - 3) + 0.0005 for t in range(120)]
    res = combinatorial_oos_stability(returns, n_groups=6, n_test_groups=2)
    assert "sharpe_annualized" in res
    with pytest.deprecated_call(match="combinatorial_oos_stability"):
        alias_res = cpcv_score(returns, n_groups=6, n_test_groups=2)
    assert alias_res["sharpe_annualized"] == res["sharpe_annualized"]


def test_documented_entry_point_is_callable():
    """`from alphaengine import sweep` must give the FUNCTION.

    Pinned because it silently was not. The subpackage `alphaengine.sweep`
    shadows the function unless __init__ re-exports it, so the exact line in
    our own README and module docstring raised "'module' object is not
    callable". It was caught by installing the built wheel and running the
    documented example, which is the only place a packaging-shaped bug like
    this shows up.
    """
    from alphaengine import sweep

    assert callable(sweep), "the documented top-level import must be the function"


def test_advertised_names_are_all_importable():
    """Everything in __all__ resolves. A promise in __all__ that raises on
    import is worse than not making the promise."""
    for name in alphaengine.__all__:
        assert hasattr(alphaengine, name), f"__all__ advertises {name!r}, which is not there"


def test_submodule_stays_reachable_explicitly():
    """Shadowing the subpackage name must not make the module unreachable."""
    from alphaengine.sweep import SweepResult
    from alphaengine.sweep.runner import sweep as runner_sweep

    assert SweepResult is alphaengine.SweepResult
    assert runner_sweep is alphaengine.sweep


def test_import_and_core_math_make_no_network_call(monkeypatch, tmp_path):
    """Importing alphaengine and running DSR, PBO via CSCV,
    combinatorial_oos_stability, metrics, sweep, and loaders must never touch
    a socket.

    Asserted with a real socket blocker rather than trusted as a convention:
    the offline guarantee is why a researcher can install and run the library
    without asking anyone.
    """
    import socket
    import sys

    def _forbid_network(*args, **kwargs):
        raise RuntimeError("network call forbidden in offline library path")

    monkeypatch.setattr(socket, "socket", _forbid_network)
    monkeypatch.setattr(socket, "create_connection", _forbid_network)

    from alphaengine import (
        combinatorial_oos_stability,
        compute_var_cvar,
        deflated_sharpe,
        pbo_cscv,
        performance_report,
        sweep,
    )
    from alphaengine.io import from_quantconnect, from_vectorbt, load_csv

    assert "alphaengine" in sys.modules
    for mod in ("requests", "httpx", "urllib3", "aiohttp"):
        assert mod not in sys.modules, f"importing alphaengine dragged in {mod}"

    # 1. DSR
    returns = [0.002 * ((t % 9) - 4) + 0.0006 for t in range(120)]
    dsr = deflated_sharpe(returns, n_trials=9)
    assert "deflated_sharpe" in dsr and 0.0 <= dsr["deflated_sharpe"] <= 1.0

    # 2. PBO via CSCV (T observations x N configs)
    trial_matrix = [[0.0015 * ((t + k) % 11 - 5) + 0.0002 * k for k in range(8)] for t in range(120)]
    pbo = pbo_cscv(trial_matrix, n_splits=8)
    assert "pbo" in pbo and 0.0 <= pbo["pbo"] <= 1.0

    # 3. combinatorial_oos_stability
    stab = combinatorial_oos_stability(returns, n_groups=6, n_test_groups=2, n_trials=9)
    assert "sharpe_annualized" in stab and stab["n_paths"] == 15

    # 4. Metrics & risk
    perf = performance_report(returns)
    risk = compute_var_cvar(returns)
    assert "sharpe_annualized" in perf and "cvar" in risk

    # 5. Sweep
    sr = sweep(
        lambda *, data, scale: [r * scale for r in data],
        {"scale": [0.8, 1.0, 1.2]},
        data=returns,
    )
    assert sr.n_trials == 3

    # 6. Loaders (CSV, QuantConnect, VectorBT)
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text(
        "date,AAPL,MSFT\n2025-01-01,100.0,200.0\n2025-01-02,101.0,199.0\n2025-01-03,102.5,202.0\n",
        encoding="utf-8",
    )
    loaded = load_csv(csv_path)
    assert set(loaded) == {"AAPL", "MSFT"}

    qc_path = tmp_path / "qc.json"
    qc_path.write_text(
        '{"Charts": {"Strategy Equity": {"Series": {"Equity": {"Values": ['
        '{"x": 1704067200, "y": 100000.0}, {"x": 1704153600, "y": 100500.0}, {"x": 1704240000, "y": 101002.5}'
        "]}}}}}",
        encoding="utf-8",
    )
    qc_study = from_quantconnect(qc_path, n_trials=3)
    assert len(qc_study.returns) == 2

    vbt_study = from_vectorbt([0.01, -0.005, 0.008], n_trials=3)
    assert len(vbt_study.returns) == 3


def test_report_refuses_a_series_before_it_leaves_the_machine():
    """The client-side half of the boundary, asserted rather than trusted.

    The guard is keyed on LENGTH, not on field name, so this must fail for a
    series hidden under any key at all. It is duplicated on the server on
    purpose: a check that runs only on the client is not a check, and one that
    runs only on the server tells you too late and without naming the field.

    SIZED OFF THE CAP. The fixture was 500 elements against a cap of 64; when
    the cap moved to 512 the payload became legal and this test would have gone
    on passing while asserting that the guard does nothing.
    """
    import pytest

    from alphaengine.study.report import MAX_FIGURE_LIST, _guard

    _guard({"performance": {"sharpe": 1.2}})  # figures pass

    over = list(range(MAX_FIGURE_LIST + 1))
    with pytest.raises(ValueError, match="is a series"):
        _guard({"performance": {"totally_not_returns": over}})


def test_report_needs_a_key_and_says_so():
    """No key is a message, not a traceback, and never a silent no-op."""
    import pytest

    from alphaengine import Study
    from alphaengine.study.report import ReportError

    with pytest.raises(ReportError, match="No API key"):
        Study(label="x").report(api_key="")
