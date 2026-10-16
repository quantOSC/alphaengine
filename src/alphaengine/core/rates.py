"""The cash rate used when a study does not name one.

This is a pinned H.15 print, not a download. Fetching the bill inside a
formula would make the same study a different number tomorrow, and the
offline demo would need a network. Pass ``risk_free_rate`` to use another
print. Pass ``0`` only when the study is genuinely excess of zero.

``DGS3MO`` is the 3-month Treasury constant maturity on an investment basis
(H.15, nominal, footnote 9). It is not the discount-basis Treasury bill,
which printed 4.04% the same day, and it is not SOFR.

The observation is the last business day on the October 7, 2026 H.15 release.
"""

from __future__ import annotations

TREASURY_BILL_RATE = 0.0421
TREASURY_BILL_AS_OF = "2026-10-06"
TREASURY_BILL_SERIES = "DGS3MO"
TREASURY_BILL_SOURCE = "H.15 3-month Treasury constant maturity, investment basis"


def cash_rate(rate: float | None) -> tuple[float, str, str | None]:
    """Resolve an omitted rate to the pinned bill.

    Returns ``(annual decimal, source, as_of)``. An explicit number, including
    zero, is the caller's and has no as-of date.
    """
    if rate is None:
        return TREASURY_BILL_RATE, TREASURY_BILL_SOURCE, TREASURY_BILL_AS_OF
    return float(rate), "caller", None


def rate_fields(rate: float | None) -> dict:
    """The three keys every cash-rate consumer stamps on its result."""
    annual, source, as_of = cash_rate(rate)
    return {
        "risk_free_rate": round(annual, 6),
        "risk_free_rate_source": source,
        "risk_free_rate_as_of": as_of,
        # rf/252, not (1+rf)^(1/252)-1. The simple split is the rate inside
        # mean(r)*ppy - rf, so the per-period excess and the annualized
        # Sharpe name the same bill.
        "risk_free_convention": "simple_annual_over_periods",
    }
