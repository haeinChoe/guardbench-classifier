"""Run manifest: what was run, under which conditions. Written once; checked on resume."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from guardbench_classifier.config import Settings
from guardbench_classifier.runner._files import atomic_write_bytes, atomic_write_json


class RunnerError(RuntimeError):
    """Base class for runner failures."""


class DirtyWorktreeError(RunnerError):
    """The git working tree has uncommitted changes and allow_dirty was not set."""


class ManifestMismatchError(RunnerError):
    """A resume was attempted with a config, concurrency or input different from the first run."""


@dataclass(frozen=True)
class GitState:
    commit: str
    dirty: bool


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _git(root: Path, *args: str) -> bytes:
    try:
        proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = exc.stderr.decode(errors="replace").strip() if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        raise RunnerError(f"git {' '.join(args)} failed in {root}: {detail}") from exc
    return proc.stdout


def git_state(root: Path) -> GitState:
    """Current commit and whether the tree has uncommitted changes (ignored files do not count)."""
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    dirty = bool(_git(root, "status", "--porcelain").strip())
    return GitState(commit=commit, dirty=dirty)


def input_hash(items: list[tuple[str, object]]) -> str:
    """sha256 over the full (id, payload) list, in the given order."""
    canonical = json.dumps([[i, p] for i, p in items], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize(obj: object) -> object:
    """JSON round trip, so a config compares equal to its stored form (tuples become lists, etc.)."""
    return json.loads(json.dumps(obj, ensure_ascii=False, sort_keys=True))


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def prepare(
    run_dir: Path,
    *,
    run_id: str,
    settings: Settings,
    git: GitState,
    config: object,
    concurrency: int,
    digest: str,
) -> dict:
    """Write the manifest on the first run, or verify the existing one on resume. Returns it."""
    path = run_dir / "manifest.json"
    config = normalize(config)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        expected = {"config": config, "concurrency": concurrency, "input_hash": digest}
        diffs = [k for k, v in expected.items() if existing.get(k) != v]
        if diffs:
            raise ManifestMismatchError(
                f"run {run_id!r} was started with a different {', '.join(diffs)}; start a new run instead of resuming"
            )
        return existing
    if (run_dir / "records.jsonl").exists() or (run_dir / "errors.jsonl").exists():
        raise RunnerError(f"{run_dir} has records but no manifest.json; cannot verify the run, refusing to resume")
    if git.dirty:  # saved before the manifest, so a manifest with dirty=true always has its diff
        atomic_write_bytes(run_dir / "git.diff", _git(settings.repo_root, "diff", "HEAD"))
    manifest = {
        "run_id": run_id,
        "created_at": now_utc(),
        "git_commit": git.commit,
        "dirty": git.dirty,
        "settings": settings.public_snapshot(),
        "versions": {
            "python": platform.python_version(),
            "torch": _version("torch"),
            "transformers": _version("transformers"),
        },
        "config": config,
        "concurrency": concurrency,
        "input_hash": digest,
    }
    atomic_write_json(path, manifest)
    return manifest
