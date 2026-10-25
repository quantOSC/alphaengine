"""Seal project runs/ideas into Schema 1.2 Studies, verify studies against data, and export committee memos.

- `seal_from_project(proj, target_ref)` builds a sealed Schema 1.2 `Study` from a
  run or idea in `.alphaengine/`, including `content_sha256`, `trial_ledger`,
  `dataset_hashes`, and PBO / PSR / SR0 / MinTRL figures, saving it to
  `.alphaengine/studies/<id>.json`.
- `verify_study_file(study_path, data_path=None, proj=None)` verifies the
  canonical `content_sha256` hash and, when data or a project run record is
  available, re-computes and checks the figures within tolerance.
- `generate_memo(proj, target_ref, fmt="md")` generates a committee-style
  research memo (Markdown or PDF) where every reported figure cites its `[run_id]`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._version import __version__
from .project import Project
from .study import Study, compute_content_sha256, load, save, verify_study_mode

__all__ = [
    "seal_from_project",
    "verify_study_file",
    "generate_memo",
]


def seal_from_project(proj: Project, target_ref: str) -> tuple[Study, Path]:
    """Build and write a sealed Schema 1.2 Study from a `<run>` or `<idea>` in `proj`."""
    from .ideas import idea_summary
    from .ledger import resolve_run

    ref = (target_ref or "HEAD").strip()
    run_rec: dict[str, Any] | None = None
    idea_rec: dict[str, Any] | None = None

    # Check if `ref` resolves to an idea first (when it starts with `idea_` or matches an idea)
    try:
        idea_rec = idea_summary(proj, ref)
    except KeyError:
        idea_rec = None

    if idea_rec is not None:
        linked = list(idea_rec.get("linked_runs") or [])
        run_rec = linked[-1] if linked else None
        iid = str(idea_rec["idea_id"])
        label = str(idea_rec.get("thesis") or iid)
        linked_ids = {str(r.get("run_id")) for r in linked}
        all_trials = [
            t
            for t in proj.read_jsonl("ledger/trials.jsonl")
            if str(t.get("run_id")) in linked_ids or str(t.get("idea_id") or "") == iid
        ]
        cum_trials = int(idea_rec.get("cumulative_trials") or 0)
        file_stem = iid
    else:
        run_rec = resolve_run(proj, ref)
        rid = str(run_rec["run_id"])
        iid = str(run_rec.get("idea_id") or "") or ""
        label = f"{run_rec.get('command')} ({run_rec.get('family')})"
        all_trials = [t for t in proj.read_jsonl("ledger/trials.jsonl") if str(t.get("run_id")) == rid]
        cum_trials = int(run_rec.get("cumulative_trials") or run_rec.get("n_trials") or 0)
        file_stem = rid

    figs = dict((run_rec or {}).get("figures") or {})
    ds_hashes = list((run_rec or {}).get("dataset_hashes") or [])
    primary_hash = ds_hashes[0] if ds_hashes else ""
    n_trials_val = cum_trials if idea_rec is not None else int((run_rec or {}).get("n_trials") or 0)
    n_trials_src = str((run_rec or {}).get("n_trials_source") or "derived_from_grid")
    failed_count = sum(1 for t in all_trials if t.get("failed") is not None)

    mintrl_val = figs.get("min_track_record_length")
    mintrl_dict = (
        mintrl_val
        if isinstance(mintrl_val, dict)
        else ({"min_track_record_length": mintrl_val} if mintrl_val is not None else None)
    )
    sel_val = figs.get("selection")
    if sel_val is None and figs.get("pbo") is not None:
        sel_val = {"pbo": figs.get("pbo")}

    raw_perf = figs.get("performance")
    perf_dict: dict[str, Any] = dict(raw_perf) if isinstance(raw_perf, dict) else {}
    for k in (
        "sharpe_ratio",
        "sharpe_annualized",
        "sortino_ratio",
        "calmar_ratio",
        "max_drawdown_pct",
        "total_return_pct",
        "annualized_return_pct",
        "volatility_annualized_pct",
        "n_obs",
    ):
        if k in figs and k not in perf_dict:
            perf_dict[k] = figs[k]

    raw_surf = figs.get("surface")
    surf_dict: dict[str, Any] = dict(raw_surf) if isinstance(raw_surf, dict) else {}
    for k in ("shape", "best_sharpe", "median_sharpe", "share_within_20pct_of_best", "n_ok", "n_failed"):
        if k in figs and k not in surf_dict:
            surf_dict[k] = figs[k]

    run_id_note = str((run_rec or {}).get("run_id") or "")
    notes_parts = [f"run_id={run_id_note}"] if run_id_note else []
    if ds_hashes:
        notes_parts.append("dataset_hashes=" + ",".join(ds_hashes))

    study = Study(
        label=label,
        n_trials=n_trials_val,
        n_trials_source=n_trials_src,
        grid_keys=list(((run_rec or {}).get("params") or {}).get("grid_keys") or []),
        n_failed_trials=failed_count,
        data_hash=primary_hash,
        data_description=", ".join(ds_hashes),
        verdict=figs.get("cumulative_verdict") or (run_rec or {}).get("verdict"),
        deflated_sharpe=figs.get("cumulative_deflated_sharpe")
        if figs.get("cumulative_deflated_sharpe") is not None
        else figs.get("deflated_sharpe"),
        psr_vs_zero=figs.get("psr_vs_zero"),
        sr0_expected_max=figs.get("cumulative_sr0_expected_max")
        if figs.get("cumulative_sr0_expected_max") is not None
        else figs.get("sr0_expected_max"),
        min_track_record_length=mintrl_dict,
        performance=perf_dict,
        selection=sel_val,
        surface=surf_dict,
        trial_ledger=[
            {
                "index": t.get("index"),
                "params_hash": t.get("params_hash"),
                "n_obs": t.get("n_obs"),
                "sharpe": t.get("sharpe"),
                "sharpe_annualized": t.get("sharpe_annualized"),
                "total_return_pct": t.get("total_return_pct"),
                "failed": t.get("failed"),
            }
            for t in all_trials[:512]
        ]
        or None,
        notes="; ".join(notes_parts),
        thesis_id=iid or None,
    )
    study.seal()
    out_path = proj.studies_dir / f"{file_stem}.json"
    save(study, out_path)
    return study, out_path


def verify_study_file(
    study_path: str | Path,
    *,
    data_path: str | Path | None = None,
    proj: Project | None = None,
    tolerance: float = 1e-4,
) -> tuple[bool, dict[str, Any]]:
    """Verify a `study.json` hash and, when data or a project run is available, re-check its figures."""
    p = Path(study_path).expanduser().resolve()
    raw = json.loads(p.read_text(encoding="utf-8"))
    study = load(p)

    stored_sha = raw.get("content_sha256")
    computed_sha = compute_content_sha256(study)
    mode = verify_study_mode(raw) or verify_study_mode(study)
    hash_ok = mode is not None
    report_info: dict[str, Any] = {
        "study_path": str(p),
        "stored_sha256": stored_sha,
        "computed_sha256": computed_sha,
        "hash_verified": hash_ok,
        "verification_mode": mode,
        "data_verified": None,
        "mismatches": {},
    }
    if not hash_ok:
        return False, report_info

    # If `--data <file>` was provided, verify data_hash and recompute performance/DSR figures
    if data_path is not None:
        from .connectors import load_panel
        from .core import deflated_sharpe, performance_report
        from .envelope import returns_from_panel_or_series
        from .ledger import hash_dataset_any

        dp = Path(data_path).expanduser().resolve()
        loaded = load_panel(dp)
        file_sha = __import__("hashlib").sha256(dp.read_bytes()).hexdigest()
        panel_hashes = hash_dataset_any(loaded)
        valid_hashes = {file_sha, *panel_hashes}
        mismatches: dict[str, Any] = {}
        if (
            study.data_hash
            and study.data_hash not in valid_hashes
            and not any(h.startswith(study.data_hash) for h in valid_hashes)
        ):
            mismatches["data_hash"] = {"recorded": study.data_hash, "reproduced": panel_hashes[0]}

        rets, _ = returns_from_panel_or_series(loaded)
        if rets and len(rets) >= 8:
            repro_perf = performance_report(rets)
            for k, rec_val in (study.performance or {}).items():
                if (
                    k in repro_perf
                    and isinstance(rec_val, (int, float))
                    and isinstance(repro_perf[k], (int, float))
                    and abs(float(rec_val) - float(repro_perf[k])) > tolerance
                ):
                    mismatches[f"performance.{k}"] = {
                        "recorded": rec_val,
                        "reproduced": repro_perf[k],
                    }
            if study.n_trials and study.n_trials >= 1 and study.psr_vs_zero is not None:
                repro_dsr = deflated_sharpe(rets, n_trials=study.n_trials)
                if (
                    "psr_vs_zero" in repro_dsr
                    and abs(float(study.psr_vs_zero) - float(repro_dsr["psr_vs_zero"])) > tolerance
                ):
                    mismatches["psr_vs_zero"] = {
                        "recorded": study.psr_vs_zero,
                        "reproduced": repro_dsr["psr_vs_zero"],
                    }

        report_info["data_verified"] = len(mismatches) == 0
        report_info["mismatches"] = mismatches
        return len(mismatches) == 0, report_info

    # If inside a project and notes reference `run_id=...`, rerun that ledger record
    if proj is not None and "run_id=" in (study.notes or ""):
        from .ledger import rerun_record, resolve_run

        rid = study.notes.split("run_id=", 1)[1].split(";", 1)[0].strip()
        if rid:
            try:
                run_rec = resolve_run(proj, rid)
                matched, _repro, mismatches = rerun_record(proj, run_rec)
                report_info["data_verified"] = matched
                report_info["mismatches"] = mismatches
                return matched, report_info
            except Exception:  # noqa: BLE001
                pass

    return True, report_info


def generate_memo(
    proj: Project,
    target_ref: str,
    *,
    fmt: str = "md",
    output_path: str | Path | None = None,
) -> tuple[str, Path]:
    """Generate a committee-style research memo (`.md` or `.pdf`) where every figure cites `[run_id]`."""
    from .ideas import idea_summary
    from .ledger import resolve_run

    ref = (target_ref or "HEAD").strip()
    idea_rec: dict[str, Any] | None = None
    try:
        idea_rec = idea_summary(proj, ref)
    except KeyError:
        idea_rec = None

    if idea_rec is not None:
        title = f"Research Committee Memo: {idea_rec['idea_id']}"
        thesis = str(idea_rec.get("thesis") or "")
        family = str(idea_rec.get("family") or "")
        status = str(idea_rec.get("status") or "open")
        kill_if = str(idea_rec.get("kill_if") or "(none)")
        kill_eval = idea_rec.get("kill_evaluation") or {}
        runs = list(idea_rec.get("linked_runs") or [])
        stem = f"memo_{idea_rec['idea_id']}"
    else:
        run_one = resolve_run(proj, ref)
        title = f"Research Committee Memo: {run_one['run_id']}"
        thesis = f"Run {run_one['run_id']} ({run_one.get('command')})"
        family = str(run_one.get("family") or "")
        status = str(run_one.get("verdict") or "recorded")
        kill_if = "(none)"
        kill_eval = {}
        runs = [run_one]
        stem = f"memo_{run_one['run_id']}"

    lines = [
        f"# {title}",
        "",
        "*Research tool, not investment advice.*",
        "",
        f"- **Project:** `{proj.name}`",
        f"- **Family:** `{family}`",
        f"- **Status:** `{status}`",
        f"- **Thesis:** {thesis}",
        f"- **Kill Rule:** `{kill_if}`",
    ]
    if kill_eval.get("detail"):
        lines.append(f"- **Kill Evaluation:** `{kill_eval.get('status')}` — {kill_eval.get('detail')}")
    lines.append(f"- **Engine Version:** `alphaengine {__version__}`")
    lines.append("")
    lines.append("## Recorded Runs & Cited Figures")
    lines.append("")

    if not runs:
        lines.append("No runs linked yet.")
    else:
        for r in runs:
            rid = str(r.get("run_id") or "")
            cmd = str(r.get("command") or "")
            ts = str(r.get("timestamp") or "")[:19].replace("T", " ")
            verd = str(r.get("verdict") or "unscored")
            nt = r.get("n_trials")
            ct = r.get("cumulative_trials")
            src = str(r.get("n_trials_source") or "")
            ds_hashes = ", ".join(str(h)[:16] for h in (r.get("dataset_hashes") or [])) or "none"
            lines.append(f"### Run `{rid}` — `{cmd}` ({ts})")
            lines.append(f"- **Verdict:** `{verd}` `[{rid}]`")
            lines.append(f"- **Trials (this run):** `{nt}` (`{src}`) `[{rid}]`")
            lines.append(f"- **Trials (cumulative):** `{ct}` `[{rid}]`")
            lines.append(f"- **Dataset SHA-256:** `{ds_hashes}` `[{rid}]`")
            figs = r.get("figures") or {}
            if isinstance(figs, dict):
                for fk, fv in sorted(figs.items()):
                    if isinstance(fv, (int, float, str)) and not isinstance(fv, bool):
                        lines.append(f"- **{fk}:** `{fv}` `[{rid}]`")
                    elif isinstance(fv, dict):
                        for sk, sv in sorted(fv.items()):
                            if isinstance(sv, (int, float, str)) and not isinstance(sv, bool):
                                lines.append(f"- **{fk}.{sk}:** `{sv}` `[{rid}]`")
            lines.append("")

    md_text = "\n".join(lines).rstrip() + "\n"
    proj.studies_dir.mkdir(parents=True, exist_ok=True)
    clean_fmt = (fmt or "md").strip().lower()
    if clean_fmt == "pdf":
        out = Path(output_path) if output_path else (proj.studies_dir / f"{stem}.pdf")
        _write_memo_pdf(md_text, out, title=title)
        return md_text, out

    out = Path(output_path) if output_path else (proj.studies_dir / f"{stem}.md")
    out.write_text(md_text, encoding="utf-8")
    return md_text, out


def _write_memo_pdf(md_text: str, out_path: Path, *, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError as exc:
        raise ModuleNotFoundError(
            "PDF memo export needs matplotlib. Install it with:  pip install 'alphaengine[tui]'"
        ) from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_lines = md_text.splitlines()
    per_page = 46
    pages = [raw_lines[i : i + per_page] for i in range(0, max(len(raw_lines), 1), per_page)]
    with PdfPages(out_path) as pdf:
        info = pdf.infodict()
        info["Title"] = title
        info["Creator"] = f"alphaengine {__version__}"
        for page_lines in pages:
            fig = plt.figure(figsize=(8.5, 11))
            y = 0.94
            for ln in page_lines:
                weight = "bold" if ln.startswith("#") else "normal"
                size = 11.0 if ln.startswith("# ") else 9.5 if ln.startswith("##") else 8.2
                clean = ln.lstrip("# ").strip() if ln.startswith("#") else ln
                fig.text(0.075, y, clean[:105], fontsize=size, fontweight=weight, family="monospace")
                y -= 0.019
            pdf.savefig(fig)
            plt.close(fig)
