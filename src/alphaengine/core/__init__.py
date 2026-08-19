"""Deterministic quantitative primitives.

Pure functions over arrays. No I/O, no network, no global state, no clock, the
same inputs return the same outputs on every machine, which is the property
everything else here is built on. numpy and scipy only.

These are published methods, not proprietary formulae:

  probabilistic / deflated Sharpe   Bailey & Lopez de Prado (2014)
  PBO via CSCV                      Bailey, Borwein, Lopez de Prado & Zhu (2017)
  CPCV                              purged, embargoed cross-validation
  minimum track record length       how long a record must be before skill is
                                    distinguishable from luck

That they are published is the point. A referee whose reasoning you cannot
inspect is not a referee.

THE RETURN VALUES ARE A PUBLIC CONTRACT. A study written today has to reproduce
in two years, so a changed number is a breaking change requiring a major version
bump even when the signature is untouched. See tests/test_goldens.py.

Exports are listed explicitly rather than star-imported: on a published package
the difference between the API and an implementation detail should be a
decision, not an accident of which names happen to be module-level.

`factors` and `pairs` are NOT imported here. They need statsmodels, which is an
optional extra, and importing them eagerly would make the core install fail for
everyone who does not need them:

    from alphaengine.core.factors import decompose_factors   # pip install 'alphaengine[factors]'
    from alphaengine.core.pairs import find_cointegrated_pairs
"""

from .allocate import hrp_weights, risk_parity_weights, vol_target
from .backtest import run_backtest, score_backtest
from .covariance import denoise_cov, detone_cov, ewma_cov, ledoit_wolf_cov
from .cross_section import fama_macbeth, quantile_book
from .panel import cs_rank, cs_winsorize, cs_zscore, neutralize
from .performance import performance_report
from .profile import profile_data
from .risk import compute_var_cvar
from .screen import screen_universe
from .series_shapes import series_values
from .signals import information_coefficient, quantile_returns, signal_decay, signal_icir
from .stress import cost_ladder, drawdown_anatomy, overlap_stats, subperiod_stability
from .technical import technical_features
from .validation import (
    cpcv_score,
    deflated_sharpe,
    expected_max_sharpe,
    min_track_record_length,
    pbo_cscv,
    probabilistic_sharpe_ratio,
)

__all__ = [
    # validation, the honesty layer
    "deflated_sharpe",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "min_track_record_length",
    "pbo_cscv",
    "cpcv_score",
    # performance and risk
    "performance_report",
    "compute_var_cvar",
    # simulation and scoring
    "run_backtest",
    "score_backtest",
    # features
    "technical_features",
    # selection: rank a supplied universe, return a bounded shortlist
    "screen_universe",
    # signal evaluation: does a supplied signal carry information at all
    "information_coefficient",
    "signal_icir",
    "quantile_returns",
    "signal_decay",
    # cross-section: make a factor comparable, then ask what it is worth
    "cs_rank",
    "cs_zscore",
    "cs_winsorize",
    "neutralize",
    "fama_macbeth",
    "quantile_book",
    # covariance and allocation: overnight book construction
    "ewma_cov",
    "ledoit_wolf_cov",
    "denoise_cov",
    "detone_cov",
    "hrp_weights",
    "risk_parity_weights",
    "vol_target",
    # stress and overlap: where a validated result breaks, and whether an
    # idea is the book again
    "subperiod_stability",
    "cost_ladder",
    "drawdown_anatomy",
    "overlap_stats",
    # data hygiene: what is wrong with the file, before anything runs on it
    "profile_data",
    # the one reading of the three shapes a per-symbol series arrives in
    "series_values",
]
