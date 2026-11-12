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
import numbers
import os
import re
from collections.abc import Mapping
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

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,127}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

__all__ = [
    "PROJECT_FORMAT_VERSION",
    "PROJECT_DIR_NAME",
    "SUBDIRS",
    "Project",
    "ProjectNotFoundError",
    "canonical_bytes",
    "canonical_sha256",
    "sha256_hex",
    "validate_safe_id",
    "init_project",
    "find_project",
]


class ProjectNotFoundError(RuntimeError):
    """Raised when a project-only operation is invoked outside `.alphaengine/`."""


def validate_safe_id(value: str, kind: str = "identifier") -> str:
    """Validate an identifier used in file stems, URLs, or ledger references (`[A-Za-z0-9._-]`, no `..`)."""
    clean = str(value or "").strip()
    if not clean or ".." in clean or not _SAFE_ID_RE.fullmatch(clean):
        raise ValueError(
            f"Invalid {kind} {value!r}: expected 1..128 alphanumeric, '.', '_', or '-' characters "
            "without path separators or '..'."
        )
    return clean


def _sanitize_json_numbers(node: Any) -> Any:
    """Replace non-finite floats with None and unwrap NumPy scalars/arrays for RFC 8259 JSON storage."""
    if hasattr(node, "dtype") and hasattr(node, "tolist") and callable(node.tolist):
        node = node.tolist()
    if isinstance(node, Mapping):
        return {str(k): _sanitize_json_numbers(node[k]) for k in sorted(node.keys(), key=str)}
    if isinstance(node, (list, tuple)):
        return [_sanitize_json_numbers(v) for v in node]
    if isinstance(node, bool) or node is None or isinstance(node, (int, str)):
        return node
    if isinstance(node, numbers.Integral):
        return int(node)
    if isinstance(node, (float, numbers.Real)):
        fval = float(node)
        return fval if math.isfinite(fval) else None
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
    sync_cursors: dict[str, str] | None = None,
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
    if sync_cursors:
        lines.append("")
        lines.append("[sync_cursors]")
        for k, v in sorted(sync_cursors.items()):
            if v:
                lines.append(f"{k} = {json.dumps(v)}")
    return "\n".join(lines) + "\n"


def _parse_toml(text: str) -> dict[str, Any]:
    """Parse `project.toml` without requiring external dependencies on Python 3.10."""
    out: dict[str, Any] = {"remotes": {}, "sync_cursors": {}}
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
    sync_cursors: dict[str, str] = field(default_factory=dict)

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
                sync_cursors=self.sync_cursors,
            ),
            encoding="utf-8",
        )

    def get_sync_cursor(self, remote_name: str = "quantos") -> str | None:
        val = self.sync_cursors.get((remote_name or "quantos").strip())
        return str(val).strip() if val and str(val).strip() else None

    def set_sync_cursor(self, remote_name: str, cursor: str | None) -> None:
        key = (remote_name or "quantos").strip()
        if cursor and cursor.strip():
            self.sync_cursors[key] = cursor.strip()
        else:
            self.sync_cursors.pop(key, None)
        self.save_config()

    def _resolve_dot_child(self, rel_path: str) -> Path:
        base = self.dot_dir.resolve()
        target = (self.dot_dir / rel_path).resolve()
        if target == base or not target.is_relative_to(base):
            raise ValueError(f"Refusing path outside project workspace: {rel_path!r}")
        return target

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
                allow_nan=False,
            ).encode("utf-8")
            obj_path.write_bytes(stored_bytes)
        return digest

    def get_object(self, digest: str) -> Any:
        """Read and verify a content-addressed object by its 64-hex SHA-256."""
        clean = str(digest).strip().lower()
        if not _HEX64_RE.fullmatch(clean):
            raise ValueError(f"Invalid CAS object digest {digest!r}: expected 64 lowercase hex chars.")
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
        target = self._resolve_dot_child(rel_path)
        entry = dict(record)
        entry.pop("object_sha256", None)
        digest = self.put_object(entry)
        entry["object_sha256"] = digest
        target.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(_sanitize_json_numbers(entry), sort_keys=True, ensure_ascii=False, allow_nan=False)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return digest

    def write_jsonl(self, rel_path: str, records: list[dict[str, Any]]) -> None:
        """Atomically write `records` to `.alphaengine/<rel_path>`, storing each record in CAS."""
        target = self._resolve_dot_child(rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for rec in records:
            entry = dict(rec)
            entry.pop("object_sha256", None)
            digest = self.put_object(entry)
            entry["object_sha256"] = digest
            lines.append(
                json.dumps(_sanitize_json_numbers(entry), sort_keys=True, ensure_ascii=False, allow_nan=False)
            )
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        tmp.replace(target)

    def read_jsonl(self, rel_path: str, *, strict: bool = False) -> list[dict[str, Any]]:
        """Read all records from an append-only JSON Lines file under `.alphaengine/`."""
        target = self._resolve_dot_child(rel_path)
        if not target.exists():
            return []
        if not target.is_file():
            raise OSError(f"Expected file at {target}, found directory or special path.")
        out: list[dict[str, Any]] = []
        for lineno, raw in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                if strict or rel_path.startswith("ledger/"):
                    raise ValueError(f"Corrupted JSONL at {target}:{lineno}: {exc}") from exc
                continue
            if isinstance(parsed, dict):
                out.append(parsed)
            elif strict or rel_path.startswith("ledger/"):
                raise ValueError(f"Expected JSON object at {target}:{lineno}, got {type(parsed).__name__}")
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
    raw_cursors = parsed.get("sync_cursors")
    cursors = {str(k): str(v) for k, v in raw_cursors.items() if v} if isinstance(raw_cursors, dict) else {}
    return Project(
        root=root,
        name=str(parsed.get("name") or root.name or "research"),
        created=str(parsed.get("created") or ""),
        schema_version=str(parsed.get("schema_version") or PROJECT_FORMAT_VERSION),
        remotes=remotes,
        sync_cursors=cursors,
    )


def find_project(start: str | Path | None = None) -> Project | None:
    """Walk up from `start` (default `Path.cwd()`) to find the enclosing `.alphaengine/` project.

    Honours `ALPHAENGINE_CEILING_DIRS` (`os.pathsep`-separated directory paths):
    when walking upward reaches any ceiling directory, discovery stops without
    reading `.alphaengine/` at or above that ceiling.
    """
    cur = Path(start or Path.cwd()).expanduser().resolve()
    if cur.is_file():
        cur = cur.parent
    raw_ceilings = os.environ.get("ALPHAENGINE_CEILING_DIRS", "")
    ceilings: set[Path] = set()
    if raw_ceilings:
        for entry in raw_ceilings.split(os.pathsep):
            clean = entry.strip()
            if clean:
                ceilings.add(Path(clean).expanduser().resolve())
    for candidate in (cur, *cur.parents):
        if candidate in ceilings:
            return None
        cfg = candidate / PROJECT_DIR_NAME / "project.toml"
        if cfg.is_file():
            try:
                return _load_from_root(candidate)
            except OSError:
                return None
    return None
