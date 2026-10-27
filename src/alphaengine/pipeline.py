"""Declarative multi-step research pipelines (`ae run pipeline.yaml`).

Parses declarative YAML/JSON pipeline recipes using the Python standard library
(no PyYAML runtime dependency required) and executes each step in sequence,
forwarding the upstream result envelope into the next step and recording every
step in the local `.alphaengine/ledger/` when inside a project.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .envelope import build_envelope, returns_from_panel_or_series
from .ledger import execute_local_workflow, hash_dataset_any, record_run, record_trials
from .project import Project, find_project

__all__ = ["load_pipeline_file", "parse_simple_yaml", "run_pipeline"]


def _parse_scalar(text: str) -> Any:
    val = text.strip()
    if not val:
        return ""
    if val[0] in ("{", "[") or (val[0] in ('"', "'") and val[-1] == val[0]):
        try:
            return json.loads(val if val[0] != "'" else '"' + val[1:-1].replace('"', '\\"') + '"')
        except ValueError:
            return val.strip("\"'")
    low = val.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "none", "~"):
        return None
    try:
        if "." in val or "e" in low:
            return float(val)
        return int(val)
    except ValueError:
        return val


def _strip_comment(line: str) -> str:
    in_quote: str | None = None
    for i, ch in enumerate(line):
        if ch in ('"', "'"):
            if in_quote is None:
                in_quote = ch
            elif in_quote == ch:
                in_quote = None
        elif ch == "#" and in_quote is None:
            return line[:i]
    return line


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse a declarative pipeline YAML document using stdlib only."""
    stripped = text.strip()
    if stripped.startswith("{"):
        loaded = json.loads(stripped)
        if isinstance(loaded, dict):
            return loaded

    raw_lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        cleaned = _strip_comment(raw).rstrip()
        if not cleaned.strip():
            continue
        indent = len(cleaned) - len(cleaned.lstrip(" "))
        raw_lines.append((indent, cleaned.strip()))

    root: dict[str, Any] = {}
    steps: list[dict[str, Any]] = []
    in_steps = False
    current_step: dict[str, Any] | None = None
    current_submap: str | None = None

    for indent, content in raw_lines:
        if indent == 0 and not content.startswith("-"):
            in_steps = False
            current_step = None
            current_submap = None
            if ":" in content:
                k, _, rhs = content.partition(":")
                key = k.strip()
                rhs_val = rhs.strip()
                if key == "steps" and not rhs_val:
                    in_steps = True
                    root["steps"] = steps
                else:
                    root[key] = _parse_scalar(rhs_val) if rhs_val else {}
            continue

        if in_steps:
            if content.startswith("-"):
                current_step = {}
                steps.append(current_step)
                current_submap = None
                after_dash = content[1:].strip()
                if after_dash and ":" in after_dash:
                    k, _, rhs = after_dash.partition(":")
                    k_clean = k.strip()
                    rhs_val = rhs.strip()
                    if rhs_val:
                        current_step[k_clean] = _parse_scalar(rhs_val)
                    else:
                        current_step[k_clean] = {}
                        current_submap = k_clean
            elif current_step is not None and ":" in content:
                k, _, rhs = content.partition(":")
                k_clean = k.strip()
                rhs_val = rhs.strip()
                if current_submap and indent >= 6:
                    sub = current_step.setdefault(current_submap, {})
                    if isinstance(sub, dict):
                        sub[k_clean] = _parse_scalar(rhs_val)
                elif not rhs_val:
                    current_step[k_clean] = {}
                    current_submap = k_clean
                else:
                    current_submap = None
                    current_step[k_clean] = _parse_scalar(rhs_val)

    return root


def load_pipeline_file(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"No pipeline file at {p}")
    return parse_simple_yaml(p.read_text(encoding="utf-8"))


def run_pipeline(
    pipeline_path: str | Path,
    *,
    proj: Project | None = None,
) -> list[dict[str, Any]]:
    """Execute every step in `pipeline_path` and return the list of step envelopes."""
    from .cli import QUESTION_VERBS, load_project, project_grid
    from .connectors import load_panel
    from .core.process import dgp_stress

    p = Path(pipeline_path).expanduser().resolve()
    spec = load_pipeline_file(p)
    active_proj = proj if proj is not None else find_project(p.parent) or find_project()
    base_dir = active_proj.root if active_proj is not None else p.parent

    default_family = str(spec.get("family") or spec.get("name") or p.stem)
    default_idea = str(spec.get("idea_id") or spec.get("idea") or "") or None
    default_data_ref = spec.get("data")
    default_proj_mod = spec.get("project")

    steps = spec.get("steps") or []
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"Pipeline {p} defines no steps.")

    envelopes: list[dict[str, Any]] = []
    upstream_env: dict[str, Any] | None = None

    for idx, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        raw_cmd = str(step.get("command") or step.get("workflow") or step.get("step") or "").strip()
        if not raw_cmd:
            raise ValueError(f"Step {idx} in {p.name} has no `command` or `workflow` field.")
        wf = QUESTION_VERBS.get(raw_cmd, raw_cmd)
        step_inputs = dict(step.get("inputs") or step.get("params") or {})
        step_grid = step.get("grid") if isinstance(step.get("grid"), dict) else step_inputs.get("grid")
        step_family = str(step.get("family") or default_family)
        step_idea = str(step.get("idea_id") or step.get("idea") or default_idea or "") or None
        data_ref = step.get("data") or default_data_ref
        proj_mod = step.get("project") or default_proj_mod
        symbol = step.get("symbol") or step_inputs.get("symbol")

        data: Any = None
        backtest_fn: Any = None
        if proj_mod:
            data, backtest_fn = load_project(str(proj_mod))
            if step_grid is None:
                step_grid = project_grid()

        if data_ref and data is None:
            cand = Path(str(data_ref))
            if not cand.is_absolute():
                cand = base_dir / cand
            if not cand.exists() and active_proj is not None:
                from .datasets import get_dataset

                ds_info = get_dataset(active_proj, str(data_ref))
                cand = Path(
                    str(ds_info.get("abs_path") or (active_proj.root / str(ds_info.get("path") or "")))
                )
            if cand.exists():
                data = load_panel(cand)

        # Carry context from upstream envelope if needed
        if upstream_env is not None:
            up_ctx = upstream_env.get("context") or {}
            if symbol is None and up_ctx.get("top_symbol"):
                symbol = up_ctx["top_symbol"]
            if data is None and up_ctx.get("returns"):
                data = list(up_ctx["returns"])
            elif (
                wf in ("validate_study", "size_position", "stress_study", "monitor_sleeve")
                and backtest_fn is None
                and up_ctx.get("returns")
            ):
                data = list(up_ctx["returns"])
                if "n_trials" not in step_inputs and upstream_env.get("n_trials"):
                    step_inputs["n_trials"] = int(upstream_env["n_trials"])

        if data is None:
            from .demo_universe import data as demo_u

            data = demo_u

        # Narrow a universe dict to a 1-D return series when the workflow expects returns
        needs_1d = wf in ("size_position", "stress_study", "monitor_sleeve") or (
            wf == "validate_study" and backtest_fn is None
        )
        if needs_1d and isinstance(data, dict) and "returns" not in data:
            derived_rets, chosen_sym = returns_from_panel_or_series(data, str(symbol) if symbol else None)
            if derived_rets:
                data = derived_rets
                symbol = chosen_sym

        if wf == "process":
            from .cli import _process_series

            series = _process_series(data)
            dgp = str(step.get("dgp") or step_inputs.get("dgp") or "gbm")
            n_paths = int(step_inputs.get("n_paths") or 80)
            figures = dgp_stress(series, dgp=dgp, n_paths=n_paths, seed=7)
            verdict: str | None = (
                "positive" if float(figures.get("sharpe_q50") or 0.0) > 0 else "likely_noise"
            )
            n_trials: int | None = 0
            source = "monte_carlo"
            returns_col, _ = returns_from_panel_or_series(series.tolist())
            trial_rows: list[dict[str, Any]] = []
        else:
            figures, verdict, n_trials, source, returns_col, trial_rows = execute_local_workflow(
                wf,
                data=data,
                backtest_fn=backtest_fn,
                grid=step_grid if isinstance(step_grid, dict) else None,
                inputs=step_inputs,
            )

        # Build downstream context (e.g. top_symbol and returns from a screen)
        ctx: dict[str, Any] = {
            "data_path": str(data_ref) if data_ref else None,
            "project_module": str(proj_mod) if proj_mod else None,
        }
        rows = figures.get("rows")
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            top_sym = str(rows[0].get("symbol") or rows[0].get("ticker") or "")
            ctx["top_symbol"] = top_sym
            ctx["shortlist"] = [str(r.get("symbol") or r.get("ticker") or "") for r in rows[:10]]
            rets_top, _ = returns_from_panel_or_series(data, top_sym)
            if rets_top:
                ctx["returns"] = rets_top
        elif returns_col:
            ctx["returns"] = returns_col
        elif isinstance(data, list):
            ctx["returns"] = [float(x) for x in data[:512]]

        if active_proj is not None:
            data_sha = None
            try:
                data_sha = active_proj.put_object(data)
            except Exception:  # noqa: BLE001
                data_sha = None
            rec = record_run(
                active_proj,
                command=wf,
                params={"workflow": wf, "step": idx, **step_inputs},
                dataset_hashes=hash_dataset_any(data),
                n_trials=n_trials,
                n_trials_source=source,
                figures=figures,
                verdict=verdict,
                idea_id=step_idea,
                family=step_family,
                repro_spec={
                    "kind": "process" if wf == "process" else "workflow",
                    "workflow": wf,
                    "dgp": step.get("dgp") or step_inputs.get("dgp"),
                    "project_module": proj_mod,
                    "data_path": str(data_ref) if data_ref else None,
                    "data_object_sha256": data_sha,
                    "grid": step_grid,
                    "inputs": step_inputs,
                },
                returns_for_deflation=returns_col,
                already_counted_in_ledger=bool(
                    upstream_env is not None
                    and upstream_env.get("n_trials")
                    and backtest_fn is None
                    and wf in ("validate_study", "size_position", "stress_study", "monitor_sleeve")
                ),
            )
            if trial_rows:
                record_trials(
                    active_proj,
                    run_id=str(rec["run_id"]),
                    trials=trial_rows,
                    idea_id=step_idea,
                    family=step_family,
                    timestamp=str(rec["timestamp"]),
                )
            env = build_envelope(
                command=wf,
                figures=rec["figures"],
                verdict=rec["verdict"],
                run_id=str(rec["run_id"]),
                timestamp=str(rec["timestamp"]),
                idea_id=step_idea,
                family=step_family,
                params_hash=str(rec["params_hash"]),
                dataset_hashes=rec["dataset_hashes"],
                n_trials=rec["n_trials"],
                n_trials_source=str(rec["n_trials_source"]),
                cumulative_trials=int(rec["cumulative_trials"]),
                project_trials=int(rec["project_trials"]),
                context=ctx,
            )
        else:
            env = build_envelope(
                command=wf,
                figures=figures,
                verdict=verdict,
                idea_id=step_idea,
                family=step_family,
                dataset_hashes=hash_dataset_any(data),
                n_trials=n_trials,
                n_trials_source=source,
                context=ctx,
            )

        envelopes.append(env)
        upstream_env = env

    return envelopes
