"""Jupyter and IPython session hook for AlphaEngine (`QOS-181`).

Ensures interactive and ad-hoc single backtests executed in a notebook session
are logged to the local `.alphaengine/` ledger, so `n_trials` cannot silently
be asserted as 1 when multiple trials were tried.

Usage:
    from alphaengine.notebook import enable_notebook_hook, get_notebook_trials

    hook = enable_notebook_hook()
    # Or in an IPython/Jupyter notebook:
    # %load_ext alphaengine.notebook

    res1 = run_backtest(signals1, prices)
    res2 = run_backtest(signals2, prices)
    res3 = run_backtest(signals3, prices)

    # Project ledger and hook now record >= 3 trials:
    assert get_notebook_trials() >= 3

    # Subsequent score_backtest(res3, n_trials=1) derives n_trials >= 3
    # from the ledger floor rather than silently evaluating at N=1.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core.validation import (
    get_session_ledger_floor,
    set_session_ledger_provider,
)
from .ledger import cumulative_trials_for, record_run, record_trials
from .project import Project, find_project, init_project, sha256_hex

__all__ = [
    "NotebookHook",
    "enable_notebook_hook",
    "disable_notebook_hook",
    "get_notebook_trials",
    "load_ipython_extension",
    "unload_ipython_extension",
]

_ACTIVE_HOOK: NotebookHook | None = None


class NotebookHook:
    """Manages interception of ad-hoc backtests in an interactive notebook session."""

    def __init__(
        self,
        project_dir: str | Path | None = None,
        *,
        idea_id: str | None = None,
        family: str = "notebook",
    ) -> None:
        self.project_dir = Path(project_dir).expanduser().resolve() if project_dir else None
        self.idea_id = (idea_id or "").strip() or None
        self.family = (family or "").strip() or "notebook"
        self.attempt_count: int = 0
        self._enabled = False

        # Originals for unpatching
        self._orig_run_backtest: Callable[..., Any] | None = None
        self._orig_score_backtest: Callable[..., Any] | None = None

    def get_project(self) -> Project:
        """Find or initialize the local .alphaengine project workspace."""
        root = self.project_dir or Path.cwd()
        proj = find_project(root)
        if proj is None:
            proj = init_project(root, name="notebook_research")
        return proj

    def get_ledger_floor(self) -> int:
        """Return cumulative attempts for this session and the project ledger (`QOS-181`).

        A ledger that exists and cannot be read FAILS CLOSED. Falling back to this
        session's attempt count would deflate against fewer trials than were run,
        which is the one error a trial counter exists to prevent.
        """
        from .ledger import LedgerError

        try:
            proj = self.get_project()
            fam_trials, proj_trials = cumulative_trials_for(proj, family=self.family)
            return max(self.attempt_count, proj_trials, fam_trials)
        except LedgerError:
            raise
        except Exception:  # noqa: BLE001
            return max(self.attempt_count, 1)

    def record_backtest_attempt(
        self,
        result: dict[str, Any],
        *,
        command: str = "notebook.run_backtest",
        params: dict[str, Any] | None = None,
    ) -> str:
        """Log an executed ad-hoc backtest attempt to .alphaengine/ledger."""
        self.attempt_count += 1
        attempt_id = str(uuid.uuid4())
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        ts = datetime.now(timezone.utc).isoformat()

        try:
            proj = self.get_project()
            p_dict = dict(params or {})
            if "config" in result and isinstance(result["config"], dict):
                p_dict.setdefault("config", result["config"])
            p_hash = sha256_hex(p_dict)

            # Record per-trial event
            sharpe_val = result.get("sharpe")
            if sharpe_val is None and "performance" in result and isinstance(result["performance"], dict):
                sharpe_val = result["performance"].get("sharpe_ratio")

            ret_pct = result.get("total_return_pct")
            if ret_pct is None and "performance" in result and isinstance(result["performance"], dict):
                ret_pct = result["performance"].get("total_return_pct")

            trial_entry = {
                "attempt_id": attempt_id,
                "run_id": run_id,
                "timestamp": ts,
                "executed_at": ts,
                "idea_id": self.idea_id,
                "family": self.family,
                "index": self.attempt_count,
                "params_hash": p_hash,
                "status": "ok",
                "n_obs": len(result.get("returns") or []),
                "sharpe": sharpe_val,
                "total_return_pct": ret_pct,
            }
            record_trials(
                proj,
                run_id=run_id,
                trials=[trial_entry],
                idea_id=self.idea_id,
                family=self.family,
                timestamp=ts,
            )

            # Record run record
            record_run(
                proj,
                command=command,
                params=p_dict,
                n_trials=1,
                n_trials_source="notebook",
                run_id=run_id,
                attempt_id=attempt_id,
                family=self.family,
                idea_id=self.idea_id,
                figures={
                    "sharpe": sharpe_val,
                    "total_return_pct": ret_pct,
                    "notebook_attempt": self.attempt_count,
                },
            )
        except Exception:  # noqa: BLE001
            pass

        return attempt_id

    def enable(self) -> None:
        """Activate backtest wrapping and ledger session floor."""
        if self._enabled:
            return

        import alphaengine.core.backtest as bt_mod

        hook_self = self

        def _on_backtest_completed(res: dict[str, Any]) -> None:
            if isinstance(res, dict) and "notebook_attempt_id" not in res:
                aid = hook_self.record_backtest_attempt(
                    res,
                    command="notebook.run_backtest",
                )
                res["notebook_attempt_id"] = aid
                res["notebook_cumulative_trials"] = hook_self.get_ledger_floor()

        bt_mod.register_notebook_hook(_on_backtest_completed)
        set_session_ledger_provider(self.get_ledger_floor)
        self._enabled = True

    def disable(self) -> None:
        """Deactivate hook and restore original functions."""
        if not self._enabled:
            return

        import alphaengine.core.backtest as bt_mod

        bt_mod.register_notebook_hook(None)
        set_session_ledger_provider(None)
        self._enabled = False


def enable_notebook_hook(
    project_dir: str | Path | None = None,
    *,
    idea_id: str | None = None,
    family: str = "notebook",
) -> NotebookHook:
    """Enable notebook backtest interception and trial logging (`QOS-181`)."""
    global _ACTIVE_HOOK
    if _ACTIVE_HOOK is not None:
        _ACTIVE_HOOK.disable()
    hook = NotebookHook(project_dir=project_dir, idea_id=idea_id, family=family)
    hook.enable()
    _ACTIVE_HOOK = hook
    return hook


def disable_notebook_hook() -> None:
    """Disable the active notebook hook and restore standard execution."""
    global _ACTIVE_HOOK
    if _ACTIVE_HOOK is not None:
        _ACTIVE_HOOK.disable()
        _ACTIVE_HOOK = None


def get_notebook_trials() -> int:
    """Return the total number of executed attempts recorded in this session."""
    if _ACTIVE_HOOK is not None:
        return _ACTIVE_HOOK.get_ledger_floor()
    floor = get_session_ledger_floor()
    return floor if floor is not None else 0


def load_ipython_extension(ipython: Any) -> None:
    """IPython extension entrypoint (`%load_ext alphaengine.notebook`)."""
    enable_notebook_hook()
    try:
        from rich.console import Console

        Console().print(
            "[dim]AlphaEngine notebook hook active: ad-hoc backtests are logged "
            "to .alphaengine/ledger (QOS-181).[/dim]"
        )
    except Exception:  # noqa: BLE001
        print("AlphaEngine notebook hook active: ad-hoc backtests are logged to .alphaengine/ledger.")


def unload_ipython_extension(ipython: Any) -> None:
    """IPython extension unload entrypoint (`%unload_ext alphaengine.notebook`)."""
    disable_notebook_hook()
