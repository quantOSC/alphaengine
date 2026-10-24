# Authoring AlphaEngine Plugins (`alphaengine.plugins`)

AlphaEngine discovers plugins through standard Python package entry points under the group `"alphaengine.plugins"`.

Three plugin kinds are supported:

1. **`loader`**: Reads external data formats or backtest artifacts (for example, the built-in `quantconnect`, `vectorbt`, and `csv` loaders).
2. **`method`**: Runs a statistical validation or diagnostic function and returns a versioned `v1.0` JSON result envelope (or a `figures` dictionary that AlphaEngine wraps into an envelope).
3. **`template`**: Renders a study or project run into a custom report or memo (for example, `committee_memo`).

Inspect installed plugins at any time with:

```bash
ae plugin list
```

---

## Minimal Example

### 1. Write your plugin module (`my_ae_plugin.py`)

```python
from alphaengine.plugins import PluginSpec


def tail_ratio_diagnostic(returns: list[float], *, n_trials: int = 1) -> dict:
    """95th-percentile gain divided by 5th-percentile loss magnitude."""
    import numpy as np

    arr = np.asarray(returns, dtype=float)
    q95 = float(np.quantile(arr, 0.95))
    q05 = abs(float(np.quantile(arr, 0.05)))
    ratio = round(q95 / max(q05, 1e-9), 4)
    return {
        "tail_ratio": ratio,
        "n_trials": n_trials,
        "n_trials_source": "asserted",
        "verdict": "robust" if ratio >= 1.2 else "marginal",
    }


def register() -> PluginSpec:
    return PluginSpec(
        name="tail_ratio",
        kind="method",
        description="95th/5th percentile tail ratio diagnostic",
        fn=tail_ratio_diagnostic,
        source="my-ae-plugin",
    )
```

### 2. Declare the entry point in `pyproject.toml`

```toml
[project.entry-points."alphaengine.plugins"]
tail_ratio = "my_ae_plugin:register"
```

After `pip install -e .`, `ae plugin list` will display `tail_ratio` under `method`, and `run_method_plugin("tail_ratio", returns)` will execute it and return a standard `v1.0` result envelope.
