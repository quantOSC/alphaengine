"""Entry-point plugin system (`alphaengine.plugins`).

Supports three plugin kinds under the `alphaengine.plugins` entry-point group:
- `loader`: ingests external data or backtest artifacts (e.g. `quantconnect`, `vectorbt`, `csv`)
- `method`: validation or diagnostic functions returning a `v1.0` result envelope
- `template`: custom report/memo renderers (e.g. `committee_memo`)

Built-in plugins are always available and third-party packages can register
additional plugins via `[project.entry-points."alphaengine.plugins"]`.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .envelope import ENVELOPE_SCHEMA_VERSION, build_envelope

ENTRYPOINT_GROUP = "alphaengine.plugins"
VALID_PLUGIN_KINDS = ("loader", "method", "template")

__all__ = [
    "ENTRYPOINT_GROUP",
    "VALID_PLUGIN_KINDS",
    "PluginSpec",
    "discover_plugins",
    "get_plugin",
    "register_plugin",
    "load_with_plugin",
    "run_method_plugin",
    "render_template_plugin",
    "quantconnect_plugin",
    "vectorbt_plugin",
    "committee_memo_plugin",
]


@dataclass(frozen=True)
class PluginSpec:
    """Descriptor for an AlphaEngine loader, method, or report template plugin."""

    name: str
    kind: str  # "loader" | "method" | "template"
    description: str
    fn: Callable[..., Any]
    source: str = "builtin"

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "source": self.source,
        }


_CUSTOM_REGISTRY: dict[str, PluginSpec] = {}


def register_plugin(spec: PluginSpec) -> PluginSpec:
    """Register a plugin in-process (for user scripts or tests)."""
    if spec.kind not in VALID_PLUGIN_KINDS:
        raise ValueError(f"Invalid plugin kind {spec.kind!r}; expected one of {VALID_PLUGIN_KINDS}")
    _CUSTOM_REGISTRY[spec.name] = spec
    return spec


def quantconnect_plugin() -> PluginSpec:
    from .io import _from_quantconnect_impl

    return PluginSpec(
        name="quantconnect",
        kind="loader",
        description="Load a QuantConnect backtest result JSON into a sealed Study",
        fn=_from_quantconnect_impl,
        source="builtin",
    )


def vectorbt_plugin() -> PluginSpec:
    from .io import _from_vectorbt_impl

    return PluginSpec(
        name="vectorbt",
        kind="loader",
        description="Load a VectorBT Portfolio object or CSV export into a sealed Study",
        fn=_from_vectorbt_impl,
        source="builtin",
    )


def committee_memo_plugin() -> PluginSpec:
    from .memo import generate_memo

    return PluginSpec(
        name="committee_memo",
        kind="template",
        description="Committee-style Markdown/PDF memo with per-figure run citations",
        fn=generate_memo,
        source="builtin",
    )


def _deflated_sharpe_method(returns: list[float], *, n_trials: int = 1, **kwargs: Any) -> dict[str, Any]:
    from .core import deflated_sharpe

    res = deflated_sharpe(returns, n_trials=n_trials, **kwargs)
    return build_envelope(
        command="deflated_sharpe",
        figures=res,
        verdict=res.get("verdict"),
        n_trials=n_trials,
        n_trials_source="asserted",
    )


def _combinatorial_oos_method(returns: list[float], *, n_trials: int = 1, **kwargs: Any) -> dict[str, Any]:
    from .core import combinatorial_oos_stability

    res = combinatorial_oos_stability(returns, n_trials=n_trials, **kwargs)
    return build_envelope(
        command="combinatorial_oos_stability",
        figures=res,
        verdict=res.get("verdict"),
        n_trials=n_trials,
        n_trials_source="asserted",
    )


def _builtin_plugins() -> dict[str, PluginSpec]:
    from .loaders import load_csv

    builtins = [
        quantconnect_plugin(),
        vectorbt_plugin(),
        PluginSpec(
            name="csv",
            kind="loader",
            description="Load wide, long, or single-series CSV panels",
            fn=load_csv,
            source="builtin",
        ),
        PluginSpec(
            name="deflated_sharpe",
            kind="method",
            description="Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) returning a v1.0 envelope",
            fn=_deflated_sharpe_method,
            source="builtin",
        ),
        PluginSpec(
            name="combinatorial_oos_stability",
            kind="method",
            description="Combinatorial out-of-sample stability across purged held-out partitions",
            fn=_combinatorial_oos_method,
            source="builtin",
        ),
        committee_memo_plugin(),
    ]
    return {p.name: p for p in builtins}


def _coerce_entrypoint(ep: Any) -> PluginSpec | None:
    try:
        loaded = ep.load()
    except Exception:  # noqa: BLE001
        return None
    dist_name = getattr(getattr(ep, "dist", None), "name", None) or "entrypoint"
    if isinstance(loaded, PluginSpec):
        return loaded
    if callable(loaded) and not hasattr(loaded, "plugin_kind"):
        # Factory function returning PluginSpec (like quantconnect_plugin)
        try:
            maybe = loaded()
            if isinstance(maybe, PluginSpec):
                return maybe
        except TypeError:
            pass
    if isinstance(loaded, dict) and callable(loaded.get("fn")):
        kind = str(loaded.get("kind") or "method")
        if kind not in VALID_PLUGIN_KINDS:
            kind = "method"
        return PluginSpec(
            name=str(loaded.get("name") or getattr(ep, "name", "plugin")),
            kind=kind,
            description=str(loaded.get("description") or ""),
            fn=loaded["fn"],
            source=str(dist_name),
        )
    if callable(loaded):
        kind = str(getattr(loaded, "plugin_kind", "method"))
        if kind not in VALID_PLUGIN_KINDS:
            kind = "method"
        desc = str(
            getattr(loaded, "plugin_description", None) or (loaded.__doc__ or "").strip().splitlines()[0]
            if loaded.__doc__
            else "Installed plugin"
        )
        return PluginSpec(
            name=str(getattr(ep, "name", getattr(loaded, "__name__", "plugin"))),
            kind=kind,
            description=desc,
            fn=loaded,
            source=str(dist_name),
        )
    return None


def discover_plugins(*, kind: str | None = None) -> list[PluginSpec]:
    """Discover all built-in, entry-point (`alphaengine.plugins`), and registered plugins."""
    merged = _builtin_plugins()
    try:
        eps = importlib.metadata.entry_points(group=ENTRYPOINT_GROUP)
        for ep in eps:
            spec = _coerce_entrypoint(ep)
            if spec is not None and spec.name not in merged:
                merged[spec.name] = spec
    except Exception:  # noqa: BLE001
        pass
    merged.update(_CUSTOM_REGISTRY)
    items = list(merged.values())
    if kind:
        items = [p for p in items if p.kind == kind]
    return sorted(items, key=lambda s: (s.kind, s.name))


def get_plugin(name: str, *, kind: str | None = None) -> PluginSpec:
    """Look up a plugin by `name` (and optional `kind`)."""
    wanted = (name or "").strip()
    for p in discover_plugins(kind=kind):
        if p.name == wanted:
            return p
    raise KeyError(f"No plugin named {name!r}" + (f" of kind {kind!r}" if kind else ""))


def load_with_plugin(name: str, *args: Any, **kwargs: Any) -> Any:
    """Invoke a `loader` plugin by name."""
    spec = get_plugin(name, kind="loader")
    return spec.fn(*args, **kwargs)


def run_method_plugin(name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Invoke a `method` plugin and ensure the result is wrapped in a `v1.0` result envelope."""
    spec = get_plugin(name, kind="method")
    res = spec.fn(*args, **kwargs)
    if isinstance(res, dict) and res.get("schema_version") == ENVELOPE_SCHEMA_VERSION and "figures" in res:
        return res
    figures = res if isinstance(res, dict) else {"result": res}
    verdict = figures.get("verdict") if isinstance(figures, dict) else None
    n_trials = figures.get("n_trials") if isinstance(figures, dict) else None
    return build_envelope(
        command=spec.name,
        figures=figures,
        verdict=str(verdict) if verdict is not None else None,
        n_trials=int(n_trials) if isinstance(n_trials, int) else None,
        n_trials_source=str(figures.get("n_trials_source") or "plugin"),
    )


def render_template_plugin(name: str, *args: Any, **kwargs: Any) -> Any:
    """Invoke a `template` plugin by name."""
    spec = get_plugin(name, kind="template")
    return spec.fn(*args, **kwargs)
