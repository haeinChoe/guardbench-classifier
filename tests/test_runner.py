import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from guardbench_classifier.config import Settings
from guardbench_classifier.runner import (
    DirtyWorktreeError,
    ManifestMismatchError,
    RunLockedError,
    run,
)
from guardbench_classifier.runner._files import JsonlAppender, read_jsonl


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Settings:
    """A clean temporary git repo whose runs/ directory is ignored, like the real one."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    (root / ".gitignore").write_text("runs/\n")
    (root / "tracked.txt").write_text("v1\n")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "init")
    return Settings.from_env(env={}, root=root)


ITEMS = [(f"r{i}", {"text": f"payload {i}"}) for i in range(10)]


def echo(row_id, payload):
    return {"label": "COMPLY", "text": payload["text"]}


def record_ids(run_dir: Path) -> list[str]:
    return [r["id"] for r in read_jsonl(run_dir / "records.jsonl")]


class Abort(BaseException):
    pass


def test_runs_all_rows_and_writes_manifest(repo):
    summary = run("exp1", ITEMS, echo, config={"prompt": "v1", "gpu_shared": True}, concurrency=2, settings=repo)
    assert (summary.state, summary.total, summary.done, summary.error_rows) == ("complete", 10, 10, 0)
    assert sorted(record_ids(summary.run_dir)) == sorted(i for i, _ in ITEMS)
    m = json.loads((summary.run_dir / "manifest.json").read_text())
    assert m["run_id"] == "exp1" and m["dirty"] is False and len(m["git_commit"]) == 40
    assert m["concurrency"] == 2 and m["config"] == {"prompt": "v1", "gpu_shared": True}
    assert len(m["input_hash"]) == 64 and m["settings"] == repo.public_snapshot()
    assert {"python", "torch", "transformers"} <= m["versions"].keys() and m["created_at"].endswith("+00:00")
    assert not (summary.run_dir / "git.diff").exists()


def test_resume_after_abort_records_every_id_exactly_once(repo):
    calls = []

    def abort_at_r5(row_id, payload):
        calls.append(row_id)
        if row_id == "r5":
            raise Abort
        return echo(row_id, payload)

    with pytest.raises(Abort):
        run("exp", ITEMS, abort_at_r5, config={}, settings=repo)
    run_dir = repo.runs_dir / "exp"
    assert json.loads((run_dir / "status.json").read_text())["state"] == "failed"
    assert record_ids(run_dir) == ["r0", "r1", "r2", "r3", "r4"]

    calls.clear()
    summary = run("exp", ITEMS, lambda i, p: (calls.append(i), echo(i, p))[1], config={}, settings=repo)
    assert calls == ["r5", "r6", "r7", "r8", "r9"]  # only the missing rows
    assert summary.state == "complete"
    assert sorted(record_ids(run_dir)) == sorted(i for i, _ in ITEMS)


def test_resume_with_truncated_last_line(repo):
    run("exp", ITEMS[:6], echo, config={}, settings=repo)
    # simulate a crash in the middle of writing the last row's line
    run_dir = repo.runs_dir / "exp"
    lines = (run_dir / "records.jsonl").read_bytes().splitlines(keepends=True)
    (run_dir / "records.jsonl").write_bytes(b"".join(lines[:-1]) + lines[-1][:15])
    lost = json.loads(lines[-1])["id"]
    assert lost not in record_ids(run_dir)  # a cut-off line is not complete

    calls = []
    summary = run("exp", ITEMS[:6], lambda i, p: (calls.append(i), echo(i, p))[1], config={}, settings=repo)
    assert calls == [lost] and summary.state == "complete"
    ids = record_ids(run_dir)
    assert sorted(ids) == sorted(i for i, _ in ITEMS[:6])
    raw = (run_dir / "records.jsonl").read_bytes()
    assert raw.endswith(b"\n")
    unparsable = [ln for ln in raw.split(b"\n") if ln and not _parses(ln)]
    assert len(unparsable) == 1  # the cut-off fragment stays on its own line; the new record starts on a fresh line


def _parses(line: bytes) -> bool:
    try:
        json.loads(line)
    except ValueError:
        return False
    return True


def test_complete_last_line_missing_only_newline_is_not_duplicated(repo):
    run("exp", ITEMS[:3], echo, config={}, settings=repo)
    path = repo.runs_dir / "exp" / "records.jsonl"
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    calls = []
    run("exp", ITEMS[:3], lambda i, p: (calls.append(i), echo(i, p))[1], config={}, settings=repo)
    assert calls == []
    assert sorted(record_ids(path.parent)) == ["r0", "r1", "r2"]


def test_appender_does_not_merge_with_fragment(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_bytes(b'{"id":"a","result":1}\n{"id":"b","res')
    appender = JsonlAppender(path)
    appender.append('{"id":"b","result":2}')
    appender.close()
    assert [r["id"] for r in read_jsonl(path)] == ["a", "b"]
    assert path.read_bytes().startswith(b'{"id":"a","result":1}\n{"id":"b","res\n')


def test_lock_held_by_another_process_fails_immediately(repo):
    run_dir = repo.runs_dir / "exp"
    run_dir.mkdir(parents=True)
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import fcntl,sys;f=open(sys.argv[1],'a');fcntl.flock(f,fcntl.LOCK_EX);print('ready',flush=True);sys.stdin.read()",
            str(run_dir / "worker.lock"),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "ready"
        with pytest.raises(RunLockedError):
            run("exp", ITEMS, echo, config={}, settings=repo)
        assert not (run_dir / "manifest.json").exists()  # nothing was written
    finally:
        holder.stdin.close()
        holder.wait(timeout=10)
    run("exp", ITEMS, echo, config={}, settings=repo)  # the lock is free again


def test_lock_is_released_after_run_and_after_failure(repo):
    with pytest.raises(Abort):
        run("exp", ITEMS, lambda i, p: (_ for _ in ()).throw(Abort()), config={}, settings=repo)
    assert run("exp", ITEMS, echo, config={}, settings=repo).state == "complete"


@pytest.mark.parametrize(
    "change",
    [
        {"config": {"prompt": "v2"}},
        {"concurrency": 4},
        {"items": ITEMS[:-1]},
        {"items": [*ITEMS[:-1], ("r9", {"text": "edited"})]},
    ],
    ids=["config", "concurrency", "fewer-ids", "edited-payload"],
)
def test_resume_with_different_conditions_fails(repo, change):
    base = {"items": ITEMS, "config": {"prompt": "v1"}, "concurrency": 2}
    run("exp", base["items"], echo, config=base["config"], concurrency=base["concurrency"], settings=repo)
    args = {**base, **change}
    with pytest.raises(ManifestMismatchError):
        run("exp", args["items"], echo, config=args["config"], concurrency=args["concurrency"], settings=repo)


def test_config_with_tuple_resumes(repo):
    run("exp", ITEMS, echo, config={"stop": ("a", "b")}, settings=repo)
    assert run("exp", ITEMS, echo, config={"stop": ("a", "b")}, settings=repo).state == "complete"


def test_technical_errors_are_logged_and_retried(repo):
    flaky = {"r2", "r7"}

    def process(row_id, payload):
        if row_id in flaky:
            raise ConnectionError(f"server down for {row_id}")
        return echo(row_id, payload)

    summary = run("exp", ITEMS, process, config={}, concurrency=3, settings=repo)
    run_dir = summary.run_dir
    assert (summary.state, summary.done, summary.error_rows) == ("failed", 8, 2)
    assert set(record_ids(run_dir)) == {i for i, _ in ITEMS} - flaky
    errs = read_jsonl(run_dir / "errors.jsonl")
    assert {e["id"] for e in errs} == flaky and all(e["error_type"] == "ConnectionError" for e in errs)
    status = json.loads((run_dir / "status.json").read_text())
    assert (status["state"], status["done"], status["error_rows"]) == ("failed", 8, 2)

    flaky.clear()  # server is back
    summary = run("exp", ITEMS, process, config={}, concurrency=3, settings=repo)
    assert (summary.state, summary.done, summary.error_rows) == ("complete", 10, 0)
    assert sorted(record_ids(run_dir)) == sorted(i for i, _ in ITEMS)


def test_invalid_results_are_recorded_as_given(repo):
    summary = run("exp", ITEMS[:2], lambda i, p: {"invalid": "parse_failure", "raw_len": 3}, config={}, settings=repo)
    assert summary.error_rows == 0
    assert [r["result"] for r in read_jsonl(summary.run_dir / "records.jsonl")] == [{"invalid": "parse_failure", "raw_len": 3}] * 2


def test_unserializable_result_is_a_technical_error(repo):
    summary = run("exp", ITEMS[:2], lambda i, p: {"x": object()} if i == "r0" else echo(i, p), config={}, settings=repo)
    assert (summary.done, summary.error_rows) == (1, 1)
    assert read_jsonl(summary.run_dir / "errors.jsonl")[0]["id"] == "r0"


def test_status_json_is_always_complete_json(repo):
    run_dir = repo.runs_dir / "exp"
    stop = threading.Event()
    seen: list[dict] = []
    bad: list[str] = []

    def watch():
        while not stop.is_set():
            try:
                text = (run_dir / "status.json").read_text()
            except FileNotFoundError:
                continue
            try:
                seen.append(json.loads(text))
            except ValueError:
                bad.append(text)

    watcher = threading.Thread(target=watch)
    watcher.start()
    try:
        items = [(f"r{i}", i) for i in range(300)]
        run("exp", items, lambda i, p: p, config={}, concurrency=4, settings=repo)
    finally:
        stop.set()
        watcher.join()
    assert bad == [] and seen
    keys = {"state", "total", "done", "error_rows", "pid", "started_at", "updated_at"}
    assert all(keys <= s.keys() for s in seen)
    final = json.loads((run_dir / "status.json").read_text())
    assert (final["state"], final["total"], final["done"]) == ("complete", 300, 300)
    assert not (run_dir / "status.json.tmp").exists()


def test_dirty_tree_is_refused_without_allow_dirty(repo):
    (repo.repo_root / "tracked.txt").write_text("v2\n")
    with pytest.raises(DirtyWorktreeError):
        run("exp", ITEMS, echo, config={}, settings=repo)
    assert not (repo.runs_dir / "exp").exists()  # nothing was created


def test_untracked_file_counts_as_dirty_but_ignored_does_not(repo):
    (repo.repo_root / "runs").mkdir(exist_ok=True)
    (repo.repo_root / "runs" / "junk.txt").write_text("ignored")
    (repo.repo_root / ".venv").mkdir()
    (repo.repo_root / "scratch.py").write_text("untracked")
    with pytest.raises(DirtyWorktreeError):
        run("exp", ITEMS, echo, config={}, settings=repo)
    (repo.repo_root / "scratch.py").unlink()
    assert run("exp", ITEMS, echo, config={}, settings=repo).state == "complete"


def test_allow_dirty_records_flag_and_diff(repo):
    (repo.repo_root / "tracked.txt").write_text("v2\n")
    summary = run("exp", ITEMS, echo, config={}, allow_dirty=True, settings=repo)
    assert summary.state == "complete"
    assert json.loads((summary.run_dir / "manifest.json").read_text())["dirty"] is True
    diff = (summary.run_dir / "git.diff").read_text()
    assert "-v1" in diff and "+v2" in diff


@pytest.mark.parametrize("bad", ["", "../x", "a/b", ".hidden"])
def test_invalid_run_id(repo, bad):
    with pytest.raises(ValueError):
        run(bad, ITEMS, echo, config={}, settings=repo)


def test_duplicate_ids_rejected(repo):
    with pytest.raises(ValueError):
        run("exp", [("a", 1), ("a", 2)], echo, config={}, settings=repo)
