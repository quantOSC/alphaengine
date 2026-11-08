# Migrating from AlphaEngine 1.x to 2.0 (`QOS-170`)

AlphaEngine 2.0 unifies the research engine around single-source versioning, strict statistical honesty, causal coded backtesting, reproducible run manifests, and a hardened sync contract.

This guide details all breaking changes, deprecation mappings, and recommended upgrade steps for quant researchers and developers.

---

## 1. Unified Verdict Vocabulary & Stated Output Conventions

In AlphaEngine 1.x, different modules reported different verdict labels (`edge`, `inconclusive`, `acceptable`, `overfit`, `not_recorded`). AlphaEngine 2.0 consolidates all statistical verdicts across the Python library, CLI, and APIs into a single canonical `Verdict` enum:

```python
from alphaengine.core import Verdict

# Canonical 2.0 values:
# Verdict.ROBUST       == "robust"
# Verdict.MARGINAL     == "marginal"
# Verdict.LIKELY_NOISE == "likely_noise"
# Verdict.UNVERIFIED   == "unverified"
```

### Deprecated 1.x Alias Mapping

When parsing existing artifacts or calling functions with legacy strings, AlphaEngine 2.0 normalizes inputs automatically:

| Legacy 1.x Verdict | Canonical 2.0 Verdict | Note |
|---|---|---|
| `edge` | `robust` | Statistically defensible after multiple-testing deflation. |
| `inconclusive` | `marginal` | Plausible signal, but insufficient statistical evidence. |
| `acceptable` | `marginal` | Plausible signal, but insufficient statistical evidence. |
| `overfit` | `likely_noise` | Fails PBO ($\text{PBO} \ge 0.5$) or deflated Sharpe test. |
| `not_recorded` | `unverified` | Missing trial count or execution without verified transaction costs. |

### Explicit Output Conventions

All metric, validation, and backtest results now embed three standard machine-readable convention dictionaries:
1. `annualization`: describes `periods_per_year` (default 252) and square-root scaling factor $\sqrt{252}$.
2. `risk_free`: specifies `rate`, `as_of`, `source`, and `convention`.
3. `costs_model`: specifies `fill_timing` (`next_open` or `close`), `slippage_bps`, `commission_bps`, `borrow_bps`, and `liquidate_final`.

---

## 2. William's Trial Counting Rule

In AlphaEngine 1.1, identical re-runs of the same configuration within a project were deduplicated (`9 -> 9`). In AlphaEngine 2.0, William's rule is enforced:

> **William's Rule:** Every executed trial attempt counts toward project cumulative trials. Identical re-runs record lineage via `duplicate_of` while continuing to increment cumulative trials (`9 -> 18`), eliminating silent trial count deflation.

Furthermore, failed grid parameter combinations during a parameter sweep remain in the denominator `n_trials = len(self.trials)` rather than being silently dropped.

---

## 3. Coded Target-Weights Adapter (`alphaengine.coded`)

AlphaEngine 2.0 introduces the engine-verified coded signal runner (`run_signal`):

```python
from alphaengine import run_signal

def my_momentum_signal(*, data_asof: dict, params: dict) -> dict[str, float]:
    lookback = params.get("lookback", 20)
    prices = data_asof["prices"]
    weights = {}
    for ticker, closes in prices.items():
        if len(closes) > lookback:
            mom = (closes[-1] / closes[-1 - lookback]) - 1.0
            weights[ticker] = 0.5 if mom > 0 else -0.5
        else:
            weights[ticker] = 0.0
    return weights

res = run_signal(
    my_momentum_signal,
    prices={"AAPL": [...], "MSFT": [...]},
    grid={"lookback": [10, 20, 40]},
    slippage_bps=5.0,
    commission_bps=1.0,
)

print(res["verdict"])  # 'robust' | 'marginal' | 'likely_noise' | 'unverified'
```

### Guarantees Enforced:
- **Zero Look-Ahead Leak:** `data_asof["prices"]` and `data_asof["bars"]` are strictly sliced up to bar $t$.
- **Next-Open Execution:** Target weights formulated at bar $t$ are actioned at bar $t+1$'s open (or close fallback if open is unrecorded).
- **Mandatory Costs:** Includes slippage, commission, short borrow accruals, and final position liquidation drag.
- **Safety Cap:** Grids are capped at a maximum of 400 parameter combinations.

---

## 4. Reproducibility `RunManifest` (v1)

Every run within a project or executed through `run_signal` records a deterministic `manifest.json`:
- Exact 64-hex SHA-256 digest of signal code / project entry point.
- Canonical SHA-256 digests of all input datasets.
- Parameter configuration hashes.
- Observation horizon (`start`, `end`, `n_bars`).
- Stored as content-addressed CAS blobs under `.alphaengine/objects/`.

The `ae verify` and `ae rerun` commands now perform three-level verification:
1. `hash_verified`: content SHA-256 matches sealed digest.
2. `figures_reproduced`: re-execution reproduces headline Sharpe, return, and drawdown figures.
3. `provenance_verified`: manifest dataset hashes and parameters match recorded lineage.

---

## 5. Sync Contract v2 (`sync_contract = 2`)

For teams syncing research to quantOS:
- **Cursor Pagination:** `ae pull` utilizes cursor-based pagination (`/api/workspace/pull?since=<cursor>&limit=<n>`), persisting cursors in `.alphaengine/project.toml` under `[sync_cursors]`.
- **Per-Attempt Trial Events:** Each parameter evaluation carries a persistent UUID `attempt_id` ensuring idempotent ledger synchronization.
- **Clean Error Handling:** No silent fallback to older 1.x ingestion endpoints. Unauthenticated requests exit 4 (`EXIT_AUTH_ERROR`); remote 404 errors exit 5.
- **Series Sharing Opt-In:** Full return series are never sent over the wire unless explicitly opted in via `--share-series` or `ALPHAENGINE_SHARE_SERIES=1`.

---

## 6. Keyring-First Authentication & Diagnostics

Credentials (`QUANTOS_API_KEY`, LLM provider keys) are managed with security-first defaults:
- **OS Keychain First:** `keyring` is used whenever a supported OS backend is available (Windows Credential Manager, macOS Keychain, Freedesktop SecretService).
- **No Plaintext Fallback:** If keyring is available, no plaintext `credentials.json` is written to disk.
- **Disclosure:** If keyring is unavailable, keys are stored in `~/.config/alphaengine/credentials.json` (mode 0600) with an explicit disclosure warning.
- **Keyless Diagnostics:**
  - `ae version [--json]`: prints engine version, sync contract version, and runtime info.
  - `ae whoami`: inspects active credentials, masked keys, and keyring backend status.
  - `ae doctor`: verifies local project integrity, credential hygiene, and remote API connectivity.
