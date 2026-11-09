# Free Open-Source Library & Local Research OS vs. Firm Workflows (`Connect`)

AlphaEngine 2.0 has a strict, unambiguous boundary: **everything local is free and offline forever** for every researcher, and a **firm `ae_live_` key** from **Connect** in the [quantOS portal](https://qosai.app) is used exclusively for network synchronization, hosted workflows, and cloud workspace integration.

The command-line tool `ae` is the standard alias for `alphaengine` (`ae` and `alphaengine` execute the identical CLI binary). License is Apache-2.0.

There is **no retail CLI key** and **no retail CLI path**. Retail ledger import happens by web upload in the quantOS portal, not through the CLI.

---

## 1. Free, Keyless, and 100% Offline

Installing `pip install alphaengine` gives you the complete quantitative validation engine, local research OS, and coded backtesting system with **no account, no API key, and zero network calls**:

| Surface | Commands / Entry Points | Network Calls | Key Required |
|---|---|---|---|
| **Python Library** | `import alphaengine`, `sweep()`, `Study`, `save()`, `load()` | **None** | **No** |
| **Validation Math** | `deflated_sharpe` (DSR), `probabilistic_sharpe_ratio` (PSR), `min_track_record_length`, `pbo_cscv` (PBO via CSCV), `combinatorial_oos_stability` | **None** | **No** |
| **Portfolio & Process Math** | `performance_report`, `compute_var_cvar`, `alphaengine.core.panel`, `signals`, `cross_section`, `covariance`, `allocate`, `process`, `active` | **None** | **No** |
| **Loaders** | `load_csv`, `load_delimited`, `from_quantconnect`, `from_vectorbt` | **None** | **No** |
| **Coded Signal Runner** | `from alphaengine.coded import run_signal, Signal`, `ae run --signal` | **None** | **No** |
| **Offline CLI Demos** | `ae demo` (`alphaengine demo`), `ae process` (`alphaengine process`) | **None** | **No** |
| **Local Research Workspace** | `ae init`, `ae log`, `ae show`, `ae diff`, `ae rerun`, `ae watch`, `ae shell`, `ae plugin` | **None** | **No** |
| **Hypotheses & Datasets** | `ae idea new|list|status|show`, `ae data add|list|show|lineage` | **None** | **No** |
| **Sealing & Verification** | `ae seal <run|study>`, `ae verify <study.json> [--data <file>]` | **None** | **No** |
| **Local Cited Memos** | `ae report memo <idea|run> --format md|pdf` | **None** | **No** |
| **Keyless Diagnostics** | `ae version`, `ae whoami`, `ae doctor`, `ae push --dry-run` | **None** | **No** |
| **Local Workflow Runs** | `ae run <recipe.yaml>`, `ae validate`, `ae screen`, `ae diagnose`, `ae size`, etc. (without `--url`) | **None** | **No** |

Anyone receiving a sealed `study.json` (whether an academic referee, an institutional peer, or an allocator) can run `ae verify study.json --data prices.csv` offline without an account to verify both the canonical `content_sha256` seal and dataset lineage hashes.

### Quickstart: Offline First (No Account Needed)

```bash
pip install alphaengine

# 1. Initialize a local .alphaengine/ workspace (free, keyless)
ae init --name momentum-research

# 2. Register local prices and track a hypothesis with an explicit kill rule
ae data add prices.csv --name equities --asof-col date
ae idea new "Cross-sectional 20d momentum holds net of costs" --kill-rule "DSR < 0.9"

# 3. Run parameter sweeps, coded backtests, and pipelines locally
ae run --signal research/momentum.py --data prices.csv
alphaengine demo
alphaengine process

# 4. Inspect cumulative trials per William's rule, diff runs, and export cited memos
ae log
ae show HEAD
ae report memo HEAD --format md

# 5. Verify any sealed Study artifact offline
ae seal HEAD
ae verify .alphaengine/studies/HEAD.json
```

---

## 2. Connect for Firms (`ae_live_` Key Checked Server-Side on Network Paths)

Firm keys issued through **Connect** in your firm's [quantOS portal](https://qosai.app) connect your local research to team infrastructure. Per AlphaEngine 2.0 policy, **all local operations are free** and firm authentication is strictly enforced **server-side on network requests**:

- **Workspace Sync (`ae push`, `ae pull`):** `push` sends this project's summaries up. `pull` copies team runs, ideas, and sealed studies down, marked read-only. Code and price files never move. The portal at `https://qosai.app` is not the sync API; `ae remote add quantos` with no URL stores the firm API.
- **Hosted Portal Workflows (`ae run --url <url>`, `ae workflows`, `ae runs`, `ae gaps`, `ae tonight`):** Query remote workflow catalogues, trigger remote execution runs on cloud runners, or inspect portal execution gaps.
- **Report Ingestion (`ae report <study.json>`):** Ingest sealed studies into the quantOS web portal for firm-wide committee review and institutional story pages.

Every network request transmits the credential in the Authorization header (`Bearer ae_live_...`). The server is the sole authority: invalid, missing, or expired credentials return `401 Unauthorized` / `403 Forbidden` (`EXIT_AUTH_ERROR = 4`).

### Signing In with Your Firm Key

```bash
# Stored securely in OS keychain (keyring) with 0600 file fallback
ae login

# Or pass via environment variable:
export QUANTOS_API_KEY="ae_live_..."

# Push research summaries to the firm API. Leave the URL off:
# `qosai.app` is the portal, not the sync API.
ae remote add quantos
ae push --yes
ae pull
```

---

## 3. What Uses the Network (and What Never Leaves Your Machine)

- **Makes zero network calls (100% offline):** The Python library (`import alphaengine`), all math functions, `sweep()`, `run_signal()`, `Study.save()` / `load()`, `ae demo`, `ae process`, `ae init`, `ae log`, `ae show`, `ae diff`, `ae rerun`, `ae idea`, `ae data`, `ae seal`, `ae verify`, `ae report memo`, `ae version`, `ae whoami`, `ae doctor`, and `ae push --dry-run`. Enforced in CI by socket-blocked tests (`tests/test_smoke.py`).
- **Uses the network:** `ae login` (remote credential check), live workspace sync (`ae push`, `ae pull`), portal report upload (`ae report study.json`), remote workflow execution (`ae run --url ...`), and portal queries (`ae workflows`, `ae runs`, `ae gaps`, `ae tonight`).
- **Return series boundary:** By default, only summary figures and canonical SHA-256 hashes cross the wire; any list longer than 512 elements is refused by both client and server guards. Full return series upload **only** when you explicitly pass `--share-series` (or set `ALPHAENGINE_SHARE_SERIES=1`).
