# AlphaEngine Result Envelope (`v1.0`) & Declarative Pipelines

**Envelope Schema Version:** `1.0` (`ENVELOPE_SCHEMA_VERSION = "1.0"`)

Every AlphaEngine CLI command can emit a structured JSON result envelope with `--json` and read an upstream envelope from standard input with `--stdin`.

```bash
ae screen --data prices.csv --json | ae validate --stdin --json | ae size --stdin --json
```

---

## 1. JSON Result Envelope Schema (`v1.0`)

```json
{
  "schema_version": "1.0",
  "engine_version": "1.0.3",
  "run_id": "run_a1b2c3d4e5f6",
  "timestamp": "2026-10-24T18:30:00+00:00",
  "command": "screen_universe",
  "idea_id": "idea_01",
  "family": "cross_sectional_momentum",
  "params_hash": "8f2a9c...",
  "dataset_hashes": [
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
  ],
  "n_trials": 60,
  "n_trials_source": "derived_from_universe",
  "cumulative_trials": 60,
  "project_trials": 60,
  "verdict": "ranked",
  "figures": {
    "universe_size": 60,
    "n_evaluated": 60,
    "n_passing": 60,
    "n_insufficient": 0,
    "cumulative_trials": 60,
    "project_trials": 60
  },
  "context": {
    "data_path": "prices.csv",
    "project_module": null,
    "top_symbol": "SYM00",
    "shortlist": ["SYM00", "SYM07", "SYM14"],
    "returns": [0.0012, -0.0034, 0.0041]
  }
}
```

### Field Reference

| Field | Type | Description |
|---|---|---|
| `schema_version` | `string` | Envelope format version (`"1.0"`). |
| `engine_version` | `string` | AlphaEngine package version (`__version__`). |
| `run_id` | `string` | Unique run identifier (`run_<12-hex>`). |
| `timestamp` | `string` | ISO-8601 UTC timestamp when the run completed. |
| `command` | `string` | Canonical command or workflow name executed. |
| `idea_id` | `string \| null` | Linked hypothesis id (`--idea`), if attached. |
| `family` | `string` | Strategy or experiment family used for cumulative trial counting. |
| `params_hash` | `string` | 64-hex SHA-256 of the run's input parameters. |
| `dataset_hashes` | `list[string]` | 64-hex SHA-256 digests of the input datasets. |
| `n_trials` | `int \| null` | Trial count for this individual run (`null` when `"not_recorded"`). |
| `n_trials_source` | `string` | Provenance of `n_trials` (`derived_from_grid`, `asserted`, `monte_carlo`, `not_recorded`, etc.). |
| `cumulative_trials` | `int` | Accumulated trial count across all project runs on the same `idea_id` or `family`. |
| `project_trials` | `int` | Total trials accumulated across the entire `.alphaengine/` project. |
| `verdict` | `string \| null` | Statistical or diagnostic verdict (`robust`, `marginal`, `likely_noise`, `clean`, `sized`, etc.; `null` when refused). |
| `figures` | `object` | Computed scalar figures and bounded tables (never > 512 elements). |
| `context` | `object` | Local pipe handoff context (`top_symbol`, `shortlist`, `returns`, `data_path`) consumed when piping into `--stdin`. |

---

## 2. Declarative Pipelines (`ae run pipeline.yaml`)

Multi-step research recipes can be declared in YAML and executed with `ae run <pipeline.yaml>`. Each step receives the upstream envelope and writes its own run record to `.alphaengine/ledger/runs.jsonl`.

```yaml
name: momentum-validation-pipeline
family: equity_momentum
steps:
  - command: diagnose
  - command: screen
    inputs:
      rank_by: return_pct
      top_n: 5
  - command: validate
  - command: size
```
