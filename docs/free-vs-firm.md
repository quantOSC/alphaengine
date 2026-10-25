# Free Open-Source Library vs. Firm CLI Workflows (`Connect`)

AlphaEngine has a strict, unambiguous boundary between what is **free and offline forever** for every researcher, and what requires a **firm `ae_live_` key** from **Connect** in the [quantOS portal](https://qosai.app).

There is **no retail CLI key** and **no retail CLI path**. Retail ledger import happens by web upload in the quantOS portal, not through the CLI.

---

## 1. Free, Keyless, and 100% Offline

Installing `pip install alphaengine` gives you the complete validation and research library with **no account, no API key, and zero network calls**:

| Surface | Commands / Entry Points | Network Calls | Key Required |
|---|---|---|---|
| **Python Library** | `import alphaengine`, `sweep()`, `Study`, `save()`, `load()` | **None** | **No** |
| **Validation Math** | `deflated_sharpe` (DSR), `probabilistic_sharpe_ratio` (PSR), `min_track_record_length`, `pbo_cscv` (PBO via CSCV), `combinatorial_oos_stability` | **None** | **No** |
| **Portfolio & Process Math** | `performance_report`, `compute_var_cvar`, `alphaengine.core.panel`, `signals`, `cross_section`, `covariance`, `allocate`, `process`, `active` | **None** | **No** |
| **Loaders** | `load_csv`, `load_delimited`, `from_quantconnect`, `from_vectorbt` | **None** | **No** |
| **Offline CLI Demos** | `alphaengine demo`, `alphaengine process` | **None** | **No** |
| **Study Verification** | `ae verify <study.json> [--data <file>]` | **None** | **No** |

Anyone receiving a sealed `study.json` (whether an academic referee, a peer, or an allocator) can run `ae verify study.json --data prices.csv` offline without an account to verify both the canonical `content_sha256` seal (including legacy seals) and the 64-hex `data_hash`.

### Quickstart: Offline First (No Account Needed)

```bash
pip install alphaengine

# 1. Run the built-in 9-trial parameter sweep and inspect the surface & DSR verdict
alphaengine demo

# 2. Fit OU, GBM, jump-diffusion, and GARCH(1,1) processes and run DGP stress offline
alphaengine process

# 3. Verify any sealed Study artifact (Schema 1.2 or legacy)
ae verify study.json
```

```python
from alphaengine import (
    combinatorial_oos_stability,
    compute_var_cvar,
    deflated_sharpe,
    pbo_cscv,
    performance_report,
    sweep,
)
```

---

## 2. Connect for Firms (`ae_live_` Key Required)

CLI workflows and the local-first Research OS (`ae init`, project ledgers, dataset registry, hypothesis tracking, sealing, memos, and workspace sync) are built for quantitative teams and require an `ae_live_` key issued to your firm through **Connect** in the quantOS portal:

- **Core workflows:** `diagnose`, `screen`, `signal`, `validate`, `stress`, `overlap`, `size`, `monitor`, `run`, `workflows`, `runs`, `gaps`, `tonight`
- **Research OS (`ae`):** `init`, `log`, `show`, `diff`, `rerun`, `watch`, `data`, `idea`, `seal`, `report`, `remote`, `push`, `pull`, `plugin`, `shell`

Every gated command checks `alphaengine.auth.require_firm_key()` before executing. The server is the authority: the key's firm tier is verified server-side on every request, and a key stops working if its firm leaves an active firm tier.

### Quickstart: Connect for Firms

1. Open your firm's [quantOS portal](https://qosai.app) → **Connect** and issue an `ae_live_...` key.
2. Sign in once from your terminal (stored in your OS keychain via `keyring` when available, or `~/.config/alphaengine/credentials.json` with `0600` permissions as fallback):

```bash
ae login
# or set: export QUANTOS_API_KEY="ae_live_..."
```

3. Initialize your firm project and run workflows with cumulative project-wide trial accounting:

```bash
ae init --name equity-momentum
ae data add prices.csv --name equities --asof-col date
ae idea new "Cross-sectional 20d momentum holds net of costs" --kill-if "DSR < 0.9"
ae validate --project research.momentum --idea idea_1
ae seal HEAD
ae push
```

---

## 3. What Uses the Network (and What Never Leaves Your Machine)

- **Makes zero network calls:** The Python library (`import alphaengine`), all math functions, `sweep()`, `Study.save()` / `load()`, `alphaengine demo`, `alphaengine process`, and `ae verify`. Enforced in CI by a socket-blocked test (`tests/test_smoke.py`).
- **Uses the network:** `ae login` / `whoami`, CLI workflows (`validate`, `screen`, `diagnose`, etc.), `Study.report()` / `ae report`, and workspace sync (`ae push`, `ae pull`).
- **Return series boundary:** By default, only summary figures and canonical SHA-256 hashes cross the wire; any list longer than 512 elements is refused by both client and server guards. Full return series upload **only** when you explicitly pass `--share-series` (or set `ALPHAENGINE_SHARE_SERIES=1`).
