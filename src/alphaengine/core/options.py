"""Black-Scholes price and Greeks. The contract is whatever the caller states.

Black and Scholes, Journal of Political Economy 1973. No chain is fetched.
A Greek is a local slope. A large shock is a second price, not the slope
times the move.
"""

from __future__ import annotations

import math

from scipy.optimize import brentq
from scipy.stats import norm


def black_scholes(
    *,
    spot: float,
    strike: float,
    expiry_years: float,
    vol: float,
    rate: float,
    dividend_yield: float = 0.0,
    call: bool = True,
) -> dict:
    """Price one European option and its Greeks. Vol and rate are decimals."""
    spot = float(spot)
    strike = float(strike)
    t = float(expiry_years)
    vol = float(vol)
    rate = float(rate)
    q = float(dividend_yield)
    if spot <= 0 or strike <= 0 or t <= 0 or vol <= 0:
        return {"error": "spot, strike, expiry, and vol must be positive"}
    d1, d2 = _d(spot, strike, t, vol, rate, q)
    df_r = math.exp(-rate * t)
    df_q = math.exp(-q * t)
    n_d1 = float(norm.pdf(d1))
    if call:
        price = spot * df_q * float(norm.cdf(d1)) - strike * df_r * float(norm.cdf(d2))
        delta = df_q * float(norm.cdf(d1))
        rho = strike * t * df_r * float(norm.cdf(d2))
    else:
        price = strike * df_r * float(norm.cdf(-d2)) - spot * df_q * float(norm.cdf(-d1))
        delta = df_q * (float(norm.cdf(d1)) - 1.0)
        rho = -strike * t * df_r * float(norm.cdf(-d2))
    gamma = df_q * n_d1 / (spot * vol * math.sqrt(t))
    vega = spot * df_q * n_d1 * math.sqrt(t)
    theta = (
        -spot * df_q * n_d1 * vol / (2.0 * math.sqrt(t))
        - (rate * strike * df_r * float(norm.cdf(d2 if call else -d2)) * (1 if call else -1))
        + (q * spot * df_q * float(norm.cdf(d1 if call else -d1)) * (1 if call else -1))
    )
    return {
        "right": "call" if call else "put",
        "price": round(price, 6),
        "delta": round(delta, 6),
        "gamma": round(gamma, 6),
        "vega": round(vega, 6),
        "theta": round(theta / 365.0, 6),
        "rho": round(rho, 6),
        "spot": spot,
        "strike": strike,
        "expiry_years": t,
        "vol": vol,
        "rate": rate,
        "method": "black_scholes_1973",
    }


def implied_vol(
    *,
    price: float,
    spot: float,
    strike: float,
    expiry_years: float,
    rate: float,
    dividend_yield: float = 0.0,
    call: bool = True,
) -> dict:
    """The vol that reprices the contract. Refused when no vol in (0.1%, 500%) matches."""
    target = float(price)

    def gap(vol: float) -> float:
        got = black_scholes(
            spot=spot,
            strike=strike,
            expiry_years=expiry_years,
            vol=vol,
            rate=rate,
            dividend_yield=dividend_yield,
            call=call,
        )
        if "error" in got:
            return 1e9
        return float(got["price"]) - target

    try:
        vol = brentq(gap, 1e-3, 5.0)
    except ValueError:
        return {"error": "no implied vol in (0.001, 5) matches that price"}
    return {"implied_vol": round(float(vol), 6), "method": "black_scholes_1973"}


def reprice(
    *,
    spot: float,
    strike: float,
    expiry_years: float,
    vol: float,
    rate: float,
    spot_shock: float = 0.0,
    vol_shock: float = 0.0,
    call: bool = True,
) -> dict:
    """Full reprice under a spot return and an additive vol move. Not a Greek times the shock."""
    base = black_scholes(spot=spot, strike=strike, expiry_years=expiry_years, vol=vol, rate=rate, call=call)
    if "error" in base:
        return base
    shocked = black_scholes(
        spot=spot * (1.0 + float(spot_shock)),
        strike=strike,
        expiry_years=expiry_years,
        vol=max(vol + float(vol_shock), 1e-4),
        rate=rate,
        call=call,
    )
    if "error" in shocked:
        return shocked
    return {
        "base_price": base["price"],
        "shocked_price": shocked["price"],
        "pnl": round(shocked["price"] - base["price"], 6),
        "spot_shock": float(spot_shock),
        "vol_shock": float(vol_shock),
        "method": "full_reprice",
    }


def _d(spot: float, strike: float, t: float, vol: float, rate: float, q: float) -> tuple[float, float]:
    vol_sqrt = vol * math.sqrt(t)
    d1 = (math.log(spot / strike) + (rate - q + 0.5 * vol * vol) * t) / vol_sqrt
    return d1, d1 - vol_sqrt
