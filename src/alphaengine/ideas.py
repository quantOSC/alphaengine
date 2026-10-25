"""Hypotheses as first-class project objects (`ae idea`) and prior-work checks.

Stores hypotheses in `.alphaengine/ideas/ideas.jsonl` (backed by content-addressed
SHA-256 objects), links runs via `--idea <id>`, evaluates `--kill-if` rules against
the latest and cumulative figures, and warns before a run when the ledger already
contains prior work on the same family or overlapping parameter sets.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any

from .project import Project

VALID_STATUSES = ("open", "testing", "killed", "promoted")

__all__ = [
    "VALID_STATUSES",
    "create_idea",
    "list_ideas",
    "resolve_idea",
    "update_idea_status",
    "idea_summary",
    "evaluate_kill_rule",
    "check_prior_work",
]

_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "dsr": ("cumulative_deflated_sharpe", "deflated_sharpe"),
    "deflated_sharpe": ("cumulative_deflated_sharpe", "deflated_sharpe"),
    "pbo": ("pbo",),
    "psr": ("psr_vs_zero",),
    "psr_vs_zero": ("psr_vs_zero",),
    "sharpe": ("sharpe_annualized", "sharpe_ratio", "sharpe_per_period"),
    "sharpe_annualized": ("sharpe_annualized",),
    "drawdown": ("max_drawdown_pct",),
    "max_drawdown": ("max_drawdown_pct",),
    "max_drawdown_pct": ("max_drawdown_pct",),
    "var": ("var_pct",),
    "var_pct": ("var_pct",),
    "cvar": ("cvar_pct",),
    "cvar_pct": ("cvar_pct",),
    "ic": ("mean_ic",),
    "ic_t_stat": ("ic_t_stat",),
}

_CLAUSE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(<=|>=|==|!=|<|>)\s*(-?\d+(?:\.\d+)?)\s*$")


def _slug_family(thesis: str) -> str:
    words = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", thesis) if len(w) > 2]
    return "_".join(words[:3]) if words else "hypothesis"


def create_idea(
    proj: Project,
    thesis: str,
    *,
    kill_if: str = "",
    family: str | None = None,
    idea_id: str | None = None,
) -> dict[str, Any]:
    """Create a new hypothesis in `.alphaengine/ideas/ideas.jsonl`."""
    clean_thesis = (thesis or "").strip()
    if not clean_thesis:
        raise ValueError("Hypothesis thesis cannot be empty.")
    now = datetime.now(timezone.utc).isoformat()
    iid = idea_id or f"idea_{uuid.uuid4().hex[:8]}"
    fam = (family or "").strip() or _slug_family(clean_thesis)
    record: dict[str, Any] = {
        "idea_id": iid,
        "thesis": clean_thesis,
        "kill_if": (kill_if or "").strip(),
        "family": fam,
        "status": "open",
        "created_at": now,
        "updated_at": now,
    }
    obj_sha = proj.append_jsonl("ideas/ideas.jsonl", record)
    record["object_sha256"] = obj_sha
    return record


def list_ideas(proj: Project) -> list[dict[str, Any]]:
    """Return the current state of every idea in `proj` (folded in chronological order)."""
    rows = proj.read_jsonl("ideas/ideas.jsonl")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        iid = str(row.get("idea_id") or "")
        if not iid:
            continue
        if iid not in by_id:
            by_id[iid] = dict(row)
        else:
            by_id[iid].update(row)
    return list(by_id.values())


def resolve_idea(proj: Project, idea_ref: str) -> dict[str, Any]:
    """Resolve an idea by exact `idea_id` or unambiguous prefix."""
    q = (idea_ref or "").strip()
    ideas = list_ideas(proj)
    if not ideas:
        raise KeyError("No ideas recorded in this project yet.")
    exact = [i for i in ideas if str(i.get("idea_id")) == q]
    if len(exact) == 1:
        return exact[0]
    prefix = [i for i in ideas if str(i.get("idea_id") or "").startswith(q)]
    if len(prefix) == 1:
        return prefix[0]
    if len(prefix) > 1:
        names = ", ".join(str(i.get("idea_id")) for i in prefix[:5])
        raise KeyError(f"Ambiguous idea prefix {q!r}; matches: {names}")
    raise KeyError(f"No idea matching {q!r} in {proj.dot_dir / 'ideas/ideas.jsonl'}")


def update_idea_status(proj: Project, idea_ref: str, status: str) -> dict[str, Any]:
    """Update an idea's status (`open`, `testing`, `killed`, `promoted`) by appending a state record."""
    clean_status = (status or "").strip().lower()
    if clean_status not in VALID_STATUSES:
        raise ValueError(f"Invalid status {status!r}; expected one of {', '.join(VALID_STATUSES)}")
    current = resolve_idea(proj, idea_ref)
    updated = dict(current)
    updated["status"] = clean_status
    updated["updated_at"] = datetime.now(timezone.utc).isoformat()
    updated.pop("object_sha256", None)
    obj_sha = proj.append_jsonl("ideas/ideas.jsonl", updated)
    updated["object_sha256"] = obj_sha
    return updated


def _lookup_metric(metric_name: str, figures: dict[str, Any]) -> tuple[str, float | None]:
    key_low = metric_name.strip().lower()
    candidates = _METRIC_ALIASES.get(key_low, (key_low, metric_name.strip()))
    for cand in candidates:
        val = figures.get(cand)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return cand, float(val)
        # Also check nested dicts one level deep
        for sub in figures.values():
            if (
                isinstance(sub, dict)
                and isinstance(sub.get(cand), (int, float))
                and not isinstance(sub.get(cand), bool)
            ):
                return cand, float(sub[cand])
    return candidates[0], None


def evaluate_kill_rule(kill_if: str, figures: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a `--kill-if` rule (e.g. `DSR < 0.9`) against `figures`."""
    rule = (kill_if or "").strip()
    if not rule:
        return {"rule": "", "triggered": None, "status": "no_rule", "detail": "no kill rule specified"}

    # Split on `or` / `,` (any triggered clause triggers the kill rule)
    clauses = [c.strip() for c in re.split(r"\b(?:or|OR)\b|;", rule) if c.strip()]
    details: list[str] = []
    any_triggered = False
    any_evaluated = False

    for clause in clauses:
        m = _CLAUSE_RE.match(clause)
        if not m:
            details.append(f"{clause} (unparsed)")
            continue
        raw_metric, op, rhs_str = m.group(1), m.group(2), m.group(3)
        threshold = float(rhs_str)
        resolved_key, observed = _lookup_metric(raw_metric, figures)
        if observed is None:
            details.append(f"{raw_metric} not yet measured")
            continue
        any_evaluated = True
        hit = (
            (op == "<" and observed < threshold)
            or (op == "<=" and observed <= threshold)
            or (op == ">" and observed > threshold)
            or (op == ">=" and observed >= threshold)
            or (op == "==" and abs(observed - threshold) <= 1e-9)
            or (op == "!=" and abs(observed - threshold) > 1e-9)
        )
        if hit:
            any_triggered = True
            details.append(f"TRIGGERED: {resolved_key}={observed} {op} {threshold}")
        else:
            details.append(f"ok: {resolved_key}={observed} (kill if {op} {threshold})")

    if not any_evaluated:
        return {"rule": rule, "triggered": None, "status": "undetermined", "detail": "; ".join(details)}
    return {
        "rule": rule,
        "triggered": any_triggered,
        "status": "kill_triggered" if any_triggered else "passing",
        "detail": "; ".join(details),
    }


def idea_summary(proj: Project, idea_ref: str) -> dict[str, Any]:
    """Assemble an idea, its linked runs, cumulative trials, and current verdict vs. its kill rule."""
    idea = resolve_idea(proj, idea_ref)
    iid = str(idea.get("idea_id"))
    fam = str(idea.get("family") or "")
    all_runs = proj.read_jsonl("ledger/runs.jsonl")
    linked = [
        r
        for r in all_runs
        if str(r.get("idea_id") or "") == iid
        or (not r.get("idea_id") and fam and str(r.get("family") or "") == fam)
    ]
    from .ledger import cumulative_trials_for

    _fam_t, cum_trials = cumulative_trials_for(proj, idea_id=iid, family=fam)
    latest_run = linked[-1] if linked else None
    latest_figs = dict(latest_run.get("figures") or {}) if latest_run else {}
    current_verdict = latest_run.get("verdict") if latest_run else None
    kill_eval = evaluate_kill_rule(str(idea.get("kill_if") or ""), latest_figs)
    return {
        **idea,
        "linked_runs": linked,
        "n_linked_runs": len(linked),
        "cumulative_trials": cum_trials,
        "current_verdict": current_verdict,
        "kill_evaluation": kill_eval,
    }


def _param_key_set(params: dict[str, Any] | None) -> set[str]:
    if not isinstance(params, dict):
        return set()
    keys: set[str] = set()
    for k, v in params.items():
        if k in ("workflow", "step", "label"):
            continue
        if k == "grid" and isinstance(v, dict) or k == "grid_keys" and isinstance(v, list):
            keys.update(f"grid.{gk}" for gk in v)
        else:
            keys.add(str(k))
    return keys


def check_prior_work(
    proj: Project,
    *,
    family: str | None = None,
    idea_id: str | None = None,
    params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Find prior runs in the ledger with the same family/idea or overlapping parameter keys."""
    runs = proj.read_jsonl("ledger/runs.jsonl")
    if not runs:
        return []
    wanted_keys = _param_key_set(params)
    target_fam = (family or "").strip()
    target_idea = (idea_id or "").strip()

    matches: list[dict[str, Any]] = []
    for r in reversed(runs):
        r_fam = str(r.get("family") or "")
        r_idea = str(r.get("idea_id") or "")
        r_keys = _param_key_set(r.get("params"))
        same_idea = bool(target_idea and r_idea == target_idea)
        same_fam = bool(target_fam and r_fam == target_fam)
        overlap = sorted(wanted_keys & r_keys) if (wanted_keys and r_keys) else []
        if same_idea or same_fam or overlap:
            matches.append(
                {
                    "run_id": r.get("run_id"),
                    "date": str(r.get("timestamp") or "")[:10],
                    "command": r.get("command"),
                    "family": r_fam,
                    "verdict": r.get("verdict"),
                    "cumulative_trials": r.get("cumulative_trials"),
                    "overlapping_keys": overlap,
                }
            )
            if len(matches) >= 3:
                break
    return matches
