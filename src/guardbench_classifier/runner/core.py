"""Run a list of ``(id, payload)`` rows through a processing function and record the results.

The runner knows nothing about the experiment: data, prompts, parsing and metrics all live
in the caller. Layout of ``<runs_dir>/<run_id>/``:

    worker.lock     flock held while a run is active
    manifest.json   conditions of the first run (checked on resume)
    git.diff        ``git diff HEAD``, only when started with allow_dirty on a dirty tree
    git.status      ``git status --porcelain`` (lists untracked files), same condition
    records.jsonl   one ``{"id", "result"}`` line per completed row (append-only)
    errors.jsonl    one line per technical failure (exception); these rows are retried on resume
    status.json     progress, replaced atomically
"""
from __future__ import annotations

import fcntl
import re
import traceback
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

from guardbench_classifier.config import Settings
from guardbench_classifier.runner import manifest
from guardbench_classifier.runner._files import JsonlAppender, dumps_line, read_jsonl
from guardbench_classifier.runner.manifest import DirtyWorktreeError, RunnerError
from guardbench_classifier.runner.status import StatusFile

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_MAX_ERROR_MESSAGE = 2000


class RunLockedError(RunnerError):
    """Another process holds worker.lock for this run."""


@dataclass(frozen=True)
class RunSummary:
    run_dir: Path
    state: str  # "complete" or "failed"
    total: int
    done: int
    error_rows: int  # rows that raised in this invocation and are still not done


class _Lock:
    def __init__(self, path: Path) -> None:
        self._f = open(path, "a")
        try:
            fcntl.flock(self._f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._f.close()
            raise RunLockedError(f"{path} is held by another process") from None

    def release(self) -> None:
        self._f.close()  # closing the descriptor drops the flock


def run(
    run_id: str,
    items: Iterable[tuple[str, object]],
    process: Callable[[str, object], object],
    *,
    config: object,
    concurrency: int = 1,
    allow_dirty: bool = False,
    settings: Settings | None = None,
) -> RunSummary:
    """Process the rows of ``items`` that are not yet in ``records.jsonl``.

    ``process(id, payload)`` is called once per row, from up to ``concurrency`` threads, and
    returns a JSON-serializable result. The runner records it as is (an invalid parse is a
    result, not an error). If it raises ``Exception`` the row goes to ``errors.jsonl``, is not
    counted as done, and is retried on the next run. Other exceptions (KeyboardInterrupt, ...)
    abort the run after marking it failed.

    ``config`` is the experiment's settings, including measurement conditions; it is stored in
    the manifest. Resuming with a different config, concurrency, input or code state (commit, dirty diff) fails.
    ``allow_dirty`` permits running with uncommitted changes (saved as ``git.diff`` and ``git.status``).
    """
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError(f"run_id must match {_RUN_ID.pattern}: {run_id!r}")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    rows = list(items)
    ids = [i for i, _ in rows]
    if not all(isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("item ids must be unique strings")

    settings = settings or Settings.from_env()
    git = manifest.git_state(settings.repo_root)
    if git.dirty and not allow_dirty:
        raise DirtyWorktreeError("working tree has uncommitted changes; commit them or pass allow_dirty=True")

    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = _Lock(run_dir / "worker.lock")
    try:
        manifest.prepare(
            run_dir,
            run_id=run_id,
            settings=settings,
            git=git,
            config=config,
            concurrency=concurrency,
            digest=manifest.input_hash(rows),
        )
        return _execute(run_dir, rows, process, concurrency)
    finally:
        lock.release()


def _execute(run_dir: Path, rows: list[tuple[str, object]], process: Callable[[str, object], object], concurrency: int) -> RunSummary:
    completed = {r["id"] for r in read_jsonl(run_dir / "records.jsonl") if "result" in r}
    todo = [(i, p) for i, p in rows if i not in completed]

    status = StatusFile(run_dir, total=len(rows))
    status.done = len(rows) - len(todo)
    status.write("running")
    failed_ids: set[str] = set()
    records = JsonlAppender(run_dir / "records.jsonl")
    errors = JsonlAppender(run_dir / "errors.jsonl")
    state = "failed"
    try:
        # Only this thread writes files; workers just compute. At most `concurrency` rows are in flight.
        pending: dict[Future, str] = {}
        queue = iter(todo)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            while True:
                while len(pending) < concurrency and (row := next(queue, None)) is not None:
                    pending[pool.submit(process, *row)] = row[0]
                if not pending:
                    break
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    row_id = pending.pop(future)
                    try:
                        line = dumps_line({"id": row_id, "result": future.result()})
                    except Exception as exc:  # BaseException propagates and aborts the run
                        errors.append(dumps_line(_error_row(row_id, exc)))
                        failed_ids.add(row_id)
                        status.error_rows = len(failed_ids)
                    else:
                        records.append(line)
                        status.done += 1
                    status.write("running")
        state = "complete" if status.done == status.total else "failed"
        return RunSummary(run_dir, state, status.total, status.done, status.error_rows)
    finally:
        records.close()
        errors.close()
        status.write(state)


def _error_row(row_id: str, exc: Exception) -> dict:
    return {
        "id": row_id,
        "at": manifest.now_utc(),
        "error_type": type(exc).__name__,
        "message": str(exc)[:_MAX_ERROR_MESSAGE],
        "traceback": "".join(traceback.format_exception(exc))[-_MAX_ERROR_MESSAGE:],
    }
