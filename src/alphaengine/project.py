"""Local-first research project workspace (`.alphaengine/`).

Like git for quantitative research: commands walk up from the current working
directory to find `.alphaengine/project.toml`. Inside a project, runs, trials,
datasets, ideas, and sealed studies are recorded in append-only JSON Lines logs
backed by content-addressed SHA-256 objects. Outside a project, every command
behaves as a standalone invocation.

Pure Python stdlib (`hashlib`, `json`, `pathlib`). No database dependency.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .study.schema import canonical_json, canonical_sha256

PROJECT_FORMAT_VERSION = "1.0"
PROJECT_DIR_NAME = ".alphaengine"
SUBDIRS: tuple[str, ...] = (
    "ledger",
    "data",
    "studies",
    "ideas",
    "universes",
    "objects",
)

__all__ = [
    "PROJECT_FORMAT_VERSION",
    "PROJECT_DIR_NAME",
    "SUBDIRS",
    "Project",
    "ProjectNotFoundError",
    "canonical_bytes",
    "canonical_sha256",
    "sha256_hex",
    "init_project",
    "find_project",
]


class ProjectNotFoundError(RuntimeError):
    """Raised when a project-only operation is invoked outside `.alphaengine/`."""


def _sanitize_json_numbers(node: Any) -> Any:
    """Replace non-finite floats with None while keeping finite numbers numeric for JSON storage."""
    if isinstance(node, dict):
        return {str(k): _sanitize_json_numbers(node[k]) for k in sorted(node.keys(), key=str)}
    if isinstance(node, (list, tuple)):
        return [_sanitize_json_numbers(v) for v in node]
    if isinstance(node, float) and not math.isfinite(node):
        return None
    return node


def canonical_bytes(payload: Any) -> bytes:
    """Deterministic UTF-8 JSON bytes using the single study schema 1.2 canonical format."""
    return canonical_json(payload).encode("utf-8")


def sha256_hex(payload: Any) -> str:
    """64-character hex SHA-256 of ``payload`` via ``canonical_sha256``."""
    return canonical_sha256(payload)


def _format_toml(
    *,
    name: str,
    created: str,
    schema_version: str,
    remotes: dict[str, str] | None = None,
) -> str:
    lines = [
        "# AlphaEngine local research project configuration",
        f"name = {json.dumps(name)}",
        f"created = {json.dumps(created)}",
        f"schema_version = {json.dumps(schema_version)}",
    ]
    if remotes:
        lines.append("")
        lines.append("[remotes]")
        for k, v in sorted(remotes.items()):
            lines.append(f"{k} = {json.dumps(v)}")
    return "\n".join(lines) + "\n"


def _parse_toml(text: str) -> dict[str, Any]:
    """Parse `project.toml` without requiring external dependencies on Python 3.10."""
    out: dict[str, Any] = {"remotes": {}}
    section: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            if section not in out or not isinstance(out.get(section), dict):
                out[section] = {}
            continue
        if "=" not in line:
            continue
        key, _, rhs = line.partition("=")
        k = key.strip()
        val_str = rhs.strip()
        if val_str.startswith(('"', "'")):
            try:
                val: Any = json.loads(val_str)
            except ValueError:
                val = val_str.strip("\"'")
        else:
            val = val_str.split("#", 1)[0].strip()
        if section is None:
            out[k] = val
        else:
            sec_map = out.setdefault(section, {})
            if isinstance(sec_map, dict):
                sec_map[k] = val
    return out


@dataclass
class Project:
    """Handle to a local `.alphaengine/` project on disk."""

    root: Path
    name: str
    created: str
    schema_version: str = PROJECT_FORMAT_VERSION
    remotes: dict[str, str] = field(default_factory=dict)

    @property
    def dot_dir(self) -> Path:
        return self.root / PROJECT_DIR_NAME

    @property
    def config_file(self) -> Path:
        return self.dot_dir / "project.toml"

    @property
    def ledger_dir(self) -> Path:
        return self.dot_dir / "ledger"

    @property
    def data_dir(self) -> Path:
        return self.dot_dir / "data"

    @property
    def studies_dir(self) -> Path:
        return self.dot_dir / "studies"

    @property
    def ideas_dir(self) -> Path:
        return self.dot_dir / "ideas"

    @property
    def universes_dir(self) -> Path:
        return self.dot_dir / "universes"

    @property
    def objects_dir(self) -> Path:
        return self.dot_dir / "objects"

    def save_config(self) -> None:
        self.dot_dir.mkdir(parents=True, exist_ok=True)
        self.config_file.write_text(
            _format_toml(
                name=self.name,
                created=self.created,
                schema_version=self.schema_version,
                remotes=self.remotes,
            ),
            encoding="utf-8",
        )

    def put_object(self, payload: Any) -> str:
        """Write a content-addressed JSON object under `.alphaengine/objects/<sha256>.json`."""
        digest = canonical_sha256(payload)
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        obj_path = self.objects_dir / f"{digest}.json"
        if not obj_path.exists():
            stored_bytes = json.dumps(
                _sanitize_json_numbers(payload),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            obj_path.write_bytes(stored_bytes)
        return digest

    def get_object(self, digest: str) -> Any:
        """Read and verify a content-addressed object by its 64-hex SHA-256."""
        clean = str(digest).strip().lower()
        obj_path = self.objects_dir / f"{clean}.json"
        if not obj_path.is_file():
            raise FileNotFoundError(f"No object {clean} in {self.objects_dir}")
        raw = obj_path.read_bytes()
        parsed = json.loads(raw.decode("utf-8"))
        actual = canonical_sha256(parsed)
        if actual != clean:
            raise ValueError(f"Object {clean} failed integrity check (computed {actual})")
        return parsed

    def append_jsonl(self, rel_path: str, record: dict[str, Any]) -> str:
        """Store `record` as a content-addressed object and append a line to `.alphaengine/<rel_path>`."""
        entry = dict(record)
        digest = self.put_object(entry)
        entry.setdefault("object_sha256", digest)
        target = self.dot_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(_sanitize_json_numbers(entry), sort_keys=True, ensure_ascii=False)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return digest

    def read_jsonl(self, rel_path: str) -> list[dict[str, Any]]:
        """Read all records from an append-only JSON Lines file under `.alphaengine/`."""
        target = self.dot_dir / rel_path
        if not target.is_file():
            return []
        out: list[dict[str, Any]] = []
        for raw in target.read_text(encoding="utf-8").splitlines():
            text = raw.strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                out.append(parsed)
        return out


def init_project(
    directory: str | Path | None = None,
    *,
    name: str | None = None,
) -> Project:
    """Initialize `.alphaengine/` in `directory` (defaults to current working directory)."""
    root = Path(directory or Path.cwd()).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    dot = root / PROJECT_DIR_NAME
    dot.mkdir(parents=True, exist_ok=True)
    for sub in SUBDIRS:
        (dot / sub).mkdir(parents=True, exist_ok=True)

    # Touch append-only log files so the layout is immediately inspectable
    for log_rel in ("ledger/runs.jsonl", "ledger/trials.jsonl", "data/datasets.jsonl", "ideas/ideas.jsonl"):
        log_path = dot / log_rel
        if not log_path.exists():
            log_path.write_text("", encoding="utf-8")

    cfg = dot / "project.toml"
    if cfg.is_file():
        existing = _load_from_root(root)
        if name and name.strip() and existing.name != name.strip():
            existing.name = name.strip()
            existing.save_config()
        return existing

    proj_name = (name or "").strip() or root.name or "research"
    created = datetime.now(timezone.utc).isoformat()
    proj = Project(
        root=root,
        name=proj_name,
        created=created,
        schema_version=PROJECT_FORMAT_VERSION,
    )
    proj.save_config()
    return proj


def _load_from_root(root: Path) -> Project:
    cfg = root / PROJECT_DIR_NAME / "project.toml"
    parsed = _parse_toml(cfg.read_text(encoding="utf-8"))
    raw_remotes = parsed.get("remotes")
    remotes = {str(k): str(v) for k, v in raw_remotes.items()} if isinstance(raw_remotes, dict) else {}
    return Project(
        root=root,
        name=str(parsed.get("name") or root.name or "research"),
        created=str(parsed.get("created") or ""),
        schema_version=str(parsed.get("schema_version") or PROJECT_FORMAT_VERSION),
        remotes=remotes,
    )


def find_project(start: str | Path | None = None) -> Project | None:
    """Walk up from `start` (default `Path.cwd()`) to find the enclosing `.alphaengine/` project."""
    cur = Path(start or Path.cwd()).expanduser().resolve()
    if cur.is_file():
        cur = cur.parent
    for candidate in (cur, *cur.parents):
        cfg = candidate / PROJECT_DIR_NAME / "project.toml"
        if cfg.is_file():
            try:
                return _load_from_root(candidate)
            except OSError:
                return None
    return None
