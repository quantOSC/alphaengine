# AlphaEngine Project On-Disk Format (`v1.0`)

**Format Version:** `1.0` (`PROJECT_FORMAT_VERSION = "1.0"`)

AlphaEngine stores a local research workspace inside a `.alphaengine/` directory at the root of your project, discovered by walking up from the current working directory (the same way `git` locates `.git/`). Outside a project directory, every CLI command and Python function behaves as a standalone call.

No database process or third-party storage library is required: all state is plain TOML, append-only JSON Lines (`.jsonl`), and content-addressed JSON blobs named by their full 64-character SHA-256 digest.

---

## Directory Layout

Running `ae init` (or `alphaengine init`) creates:

```text
.alphaengine/
├── project.toml          # Project metadata and configured remotes
├── ledger/
│   ├── runs.jsonl        # Append-only run records (every command/step executed)
│   └── trials.jsonl      # Append-only trial records across sweeps and ideas
├── data/
│   └── datasets.jsonl    # Append-only dataset registry and hygiene diagnostics
├── studies/
│   └── <id>.json         # Sealed Study artifacts (schema 1.2)
├── ideas/
│   └── ideas.jsonl       # Append-only hypothesis definitions and status transitions
├── universes/
│   └── <name>.json       # Local universe definitions
└── objects/
    └── <sha256>.json     # Content-addressed immutable JSON objects
```

---

## 1. `project.toml`

Stores the project's human-readable name, UTC creation timestamp, on-disk format version, and optional sync remotes:

```toml
# AlphaEngine local research project configuration
name = "equity-momentum"
created = "2026-10-24T18:00:00+00:00"
schema_version = "1.0"

[remotes]
quantos = "https://alpha-backend-production-51df.up.railway.app"
```

---

## 2. Content-Addressed Object Store (`objects/<sha256>.json`)

Every structured record written to the ledger, dataset registry, or idea log is also serialized into canonical JSON:
- Dictionary keys sorted lexicographically (`sort_keys=True`)
- Compact separators (`(",", ":")`)
- Finite floats rounded deterministically; non-finite values mapped to `null`
- Encoded as UTF-8

The 64-character hex SHA-256 of those bytes determines the file path `.alphaengine/objects/<sha256>.json`. Reading an object via `Project.get_object(sha256)` re-hashes the file on disk and raises `ValueError` if the content was modified.

---

## 3. Append-Only Logs (`.jsonl`)

Each `.jsonl` file under `ledger/`, `data/`, and `ideas/` contains one UTF-8 JSON object per line, in chronological order:

- **`ledger/runs.jsonl`**: Every command or pipeline step executed inside the project (`run_id`, `timestamp`, `command`, `idea_id`, `family`, `params_hash`, `dataset_hashes`, `n_trials`, `n_trials_source`, `cumulative_trials`, `figures`, `verdict`, `engine_version`, `object_sha256`).
- **`ledger/trials.jsonl`**: Per-trial entries (`run_id`, `idea_id`, `family`, `index`, `params_hash`, `sharpe`, `failed`, `timestamp`).
- **`data/datasets.jsonl`**: Registered local datasets (`name`, `path`, `sha256`, `format`, `n_rows`, `start_date`, `end_date`, `columns`, `checks`, `registered_at`, `object_sha256`).
- **`ideas/ideas.jsonl`**: Hypothesis creation and status updates (`idea_id`, `thesis`, `family`, `kill_if`, `status`, `created_at`, `updated_at`, `object_sha256`).

---

## 4. Discovery Rule & Network Boundary

When any `ae` / `alphaengine` command runs:
1. It checks the current working directory and each parent directory for `.alphaengine/project.toml`.
2. If found, the command attaches to that `Project` instance.
3. If not found, commands that do not strictly require a project (`demo`, `process`, `verify <study.json>`, `version`, etc.) run in standalone mode without touching the filesystem.

**Network boundary:** The Python library (`import alphaengine`, `sweep`, DSR, PBO via CSCV, `combinatorial_oos_stability`, metrics, loaders), `alphaengine demo`, `alphaengine process`, and `ae verify` make no network calls. Network calls occur only during `login`/`whoami`, firm CLI workflows, `report`, and `seal`/`push`/`pull` against the configured remote. Full return series upload only when `--share-series` is explicitly passed; otherwise the 512-element wire guard rejects any series-length list.
