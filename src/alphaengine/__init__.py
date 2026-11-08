"""AlphaEngine, validated research tooling for investment strategies.

Answers one question honestly: **is this result an edge, or did you try enough
things that something was bound to look good?**

    from alphaengine import sweep

    r = sweep(backtest_fn, grid, data=prices)   # runs every combination
    r.surface()                                 # stable plateau, or a knife edge
    r.verdict()                                 # deflated for the trials you ran
    r.save("study.json")                        # the artifact, on your disk

CODED TARGET-WEIGHTS ADAPTER (2.0)
    For engine-verified simulation with mandatory transaction costs, next-open
    fills, short borrow accrual, and causal `data_asof` slicing:

    from alphaengine import run_signal

    out = run_signal("my_signal.py", prices=prices, grid={"lookback": [10, 20, 40]})

OFFLINE BY CONSTRUCTION
    Importing this module makes no network call and needs no account. numpy and
    scipy, nothing else. Everything above runs on a laptop with the wifi off.
    Summary figures upload only when you explicitly run a firm CLI workflow,
    `ae push`, or `report()`; full return series upload only with `--share-series`.
"""

from typing import Any

from ._version import (
    MANIFEST_VERSION,
    SYNC_CONTRACT_VERSION,
    __version__,
    current_engine_version,
    engine_info,
)
from .coded import (
    MAX_CODED_GRID,
    CodedSignalError,
    Signal,
    check_signal_module,
    check_signal_source,
    run_signal,
)
from .core import (
    Verdict,
    combinatorial_oos_stability,
    compute_var_cvar,
    cpcv_score,
    deflated_sharpe,
    dsr_probability,
    expected_max_sharpe,
    harvey_liu_haircut,
    min_track_record_length,
    n_eff_participation,
    pbo_cscv,
    performance_report,
    probabilistic_sharpe_ratio,
    run_backtest,
    score_backtest,
    trials_for_deflated_sharpe,
)
from .core.conventions import (
    VOLATILITY_FLOOR,
    build_annualization_spec,
    build_costs_model_spec,
    build_risk_free_spec,
    normalize_verdict,
)
from .ledger import LedgerError
from .manifest import RunManifest, build_run_manifest
from .methods import METHODS_CATALOGUE, MethodSpec, get_method, list_methods
from .notebook import disable_notebook_hook, enable_notebook_hook, get_notebook_trials
from .study import SCHEMA_VERSION, SEAL_VERSION, Study, load, save
from .sweep import SweepResult, sweep

# `sweep` is bound here to the FUNCTION, deliberately shadowing the subpackage
# of the same name (`from alphaengine import sweep`).
__all__ = [
    "__version__",
    "current_engine_version",
    "SYNC_CONTRACT_VERSION",
    "MANIFEST_VERSION",
    "engine_info",
    "VOLATILITY_FLOOR",
    "Verdict",
    "normalize_verdict",
    "build_annualization_spec",
    "build_risk_free_spec",
    "build_costs_model_spec",
    "LedgerError",
    "Signal",
    "CodedSignalError",
    "MAX_CODED_GRID",
    "check_signal_module",
    "check_signal_source",
    "run_signal",
    "RunManifest",
    "build_run_manifest",
    "MethodSpec",
    "METHODS_CATALOGUE",
    "get_method",
    "list_methods",
    "sweep",
    "SweepResult",
    "Study",
    "save",
    "load",
    "report",
    "ReportError",
    "SCHEMA_VERSION",
    "SEAL_VERSION",
    "io",
    "validation",
    "combinatorial_oos_stability",
    "cpcv_score",
    "deflated_sharpe",
    "dsr_probability",
    "expected_max_sharpe",
    "harvey_liu_haircut",
    "probabilistic_sharpe_ratio",
    "min_track_record_length",
    "n_eff_participation",
    "pbo_cscv",
    "performance_report",
    "compute_var_cvar",
    "run_backtest",
    "score_backtest",
    "trials_for_deflated_sharpe",
    "enable_notebook_hook",
    "disable_notebook_hook",
    "get_notebook_trials",
]


def __getattr__(name: str) -> Any:  # PEP 562
    """`report`, `ReportError`, `io`, `validation`, and `notebook` load on first use."""
    if name in ("report", "ReportError"):
        import importlib

        mod = importlib.import_module(".study.report", __name__)
        return getattr(mod, name)
    if name in ("io", "validation", "notebook"):
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
