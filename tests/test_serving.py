import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import httpx
import pytest

from guardbench_classifier.config import Settings
from guardbench_classifier import serving
from guardbench_classifier.serving import (
    ContainerInfo, ServingError, VllmClient, check_gpu_memory, compose_env, configured_servers, down, dtype_of,
    format_status, parse_gpu_memory, status, status_all, up, wait_server,
)


def _transport(seen: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.16.0"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "Qwen/Qwen3-4B", "root": "Qwen/Qwen3-4B", "max_model_len": 8192}]})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={"choices": [{"message": {"content": "REFUSE"}}]})
        return httpx.Response(404)
    return httpx.MockTransport(handler)


def test_status_and_greedy_chat_request() -> None:
    seen: list[httpx.Request] = []
    client = VllmClient("http://server", transport=_transport(seen))
    assert status(client) == {"reachable": True, "version": "0.16.0",
                              "models": [{"id": "Qwen/Qwen3-4B", "root": "Qwen/Qwen3-4B", "max_model_len": 8192}]}
    client.chat("Qwen/Qwen3-4B", [{"role": "user", "content": "x"}], max_tokens=8,
                chat_template_kwargs={"enable_thinking": False})
    body = json.loads(seen[-1].content)
    assert body["temperature"] == 0.0 and body["max_tokens"] == 8
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_unreachable_server() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)
    assert status(VllmClient("http://server", transport=httpx.MockTransport(down))) == {"reachable": False}


def test_base_url_from_port() -> None:
    assert Settings.from_env({"VLLM_PORT": "9001"}).vllm_base_url == "http://127.0.0.1:9001"


QWEN_URL = "http://127.0.0.1:8000"
ENCODER_URL = "http://127.0.0.1:8001"


CONTAINERS = {"vllm": "guardbench-vllm", "vllm-encoder": "guardbench-vllm-encoder"}
DTYPES = {"vllm": "bfloat16", "vllm-encoder": "float32"}


class FakeStack:
    """Stands in for docker compose and the servers it would start. Nothing real is run.

    ``events`` records every docker call and log save in order, e.g. "up:vllm", "logs:guardbench-vllm", "rm:vllm".
    """

    def __init__(self, healthy: set[str] | None = None, fail_services: set[str] = frozenset(),
                 never_healthy: set[str] = frozenset(), exit_services: set[str] = frozenset(),
                 fail_rm: bool = False, logs_available: bool = True, leave_container: bool = False) -> None:
        self.healthy = set(healthy or ())
        self.fail_services = set(fail_services)
        self.never_healthy = set(never_healthy)  # starts and keeps running, but never gets healthy
        self.exit_services = set(exit_services)  # starts, then the container exits (never healthy)
        self.fail_rm = fail_rm
        self.logs_available = logs_available
        self.leave_container = leave_container  # a failed `up -d` still leaves a created container
        self.calls: list[tuple[list[str], dict[str, str]]] = []
        self.events: list[str] = []
        self.containers: dict[str, ContainerInfo] = {}
        self.saved: dict[Path, str] = {}
        self.gpu_reads = 0
        self.urls = {"vllm": QWEN_URL, "vllm-encoder": ENCODER_URL}

    def run(self, cmd: Sequence[str], env: Mapping[str, str]) -> int:
        self.calls.append((list(cmd), dict(env)))
        verb = next(a for a in cmd if a in {"up", "down", "rm"})
        service = cmd[-1]
        self.events.append(f"{verb}:{service}")
        if verb == "up":
            command = ("vllm", "serve", f"--dtype={DTYPES[service]}")
            if service in self.fail_services:
                if self.leave_container:
                    self.containers[CONTAINERS[service]] = ContainerInfo("created", None, command)
                return 1
            if service in self.exit_services:
                self.containers[CONTAINERS[service]] = ContainerInfo("exited", 1, command)
                return 0
            self.containers[CONTAINERS[service]] = ContainerInfo("running", None, command)
            if service not in self.never_healthy:
                self.healthy.add(self.urls[service])
        elif verb == "rm":
            self.healthy.discard(self.urls[service])
            self.containers.pop(CONTAINERS[service], None)
            return 1 if self.fail_rm else 0
        return 0

    def gpu(self, total: int, free: int):
        def read() -> tuple[int, int]:
            self.gpu_reads += 1
            return total, free
        return read

    def inspect(self, container: str) -> ContainerInfo | None:
        if container in self.containers:
            return self.containers[container]
        # a server that was already healthy before this run has a container too
        service = next(s for s, c in CONTAINERS.items() if c == container)
        if self.urls[service] in self.healthy:
            return ContainerInfo("running", None, ("vllm", "serve", f"--dtype={DTYPES[service]}"))
        return None

    def save_logs(self, container: str, path: Path) -> bool:
        self.events.append(f"logs:{container}")
        if not self.logs_available:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"logs of {container}\n")
        self.saved[path] = container
        return True

    def client_for(self, base_url: str) -> VllmClient:
        def handler(request: httpx.Request) -> httpx.Response:
            if base_url not in self.healthy:
                return httpx.Response(503)
            if request.url.path == "/version":
                return httpx.Response(200, json={"version": "0.16.0"})
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "m", "root": "m", "max_model_len": 8}]})
            return httpx.Response(200)
        return VllmClient(base_url, transport=httpx.MockTransport(handler))

    def verbs(self) -> list[str]:
        return [next(a for a in cmd if a in {"up", "down", "rm"}) + ":" + cmd[-1] for cmd, _ in self.calls]


def _settings(tmp_path: Path, **env: str) -> Settings:
    return Settings.from_env(env, root=tmp_path)


def _encoder_settings(tmp_path: Path, **env: str) -> Settings:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(exist_ok=True)
    return _settings(tmp_path, VLLM_ENCODER_CHECKPOINT_DIR=str(ckpt), **env)


def _up(settings: Settings, stack: FakeStack, name: str = "qwen", total: int = 16000, free: int = 16000, **kw):
    kw.setdefault("timeout", 0)
    return up(settings, name, run=stack.run, gpu_memory=stack.gpu(total, free), client_for=stack.client_for,
              inspect=stack.inspect, save_logs=stack.save_logs, base_env={"PATH": "/bin"}, interval=0, **kw)


# --- memory check ---

def test_parse_gpu_memory_uses_first_gpu() -> None:
    assert parse_gpu_memory("16376, 12530\n24000, 100\n") == (16376, 12530)
    with pytest.raises(ServingError):
        parse_gpu_memory("not csv")
    with pytest.raises(ServingError):
        parse_gpu_memory("")


def test_memory_boundary_is_inclusive() -> None:
    servers = configured_servers(Settings.from_env({"VLLM_ENCODER_CHECKPOINT_DIR": "x"}, root=Path("/r")))
    # (0.65 * 16000 + 1024) + (0.10 * 16000 + 1024) = 14048: the margin is counted per server
    check_gpu_memory(servers, 1024, 16000, 14048)
    with pytest.raises(ServingError, match="need 14048 MiB.*free 14047 MiB"):
        check_gpu_memory(servers, 1024, 16000, 14047)


def test_insufficient_memory_starts_nothing(tmp_path: Path) -> None:
    stack = FakeStack()
    with pytest.raises(ServingError, match="not enough free GPU memory"):
        _up(_encoder_settings(tmp_path), stack, total=16000, free=10000)
    assert stack.calls == []


def test_margin_is_counted(tmp_path: Path) -> None:
    # 0.65 * 16000 = 10400; free 10500 fits without margin but not with a 200 MiB margin.
    _up(_settings(tmp_path, VLLM_GPU_MEMORY_MARGIN_MIB="0"), FakeStack(), free=10500)
    stack = FakeStack()
    with pytest.raises(ServingError):
        _up(_settings(tmp_path, VLLM_GPU_MEMORY_MARGIN_MIB="200"), stack, free=10500)
    assert stack.calls == []


def test_memory_counts_only_the_target_server(tmp_path: Path) -> None:
    # Qwen needs 0.65 * 16000 + 1024 = 11424 MiB, the encoder 0.10 * 16000 + 1024 = 2624 MiB.
    stack = FakeStack(healthy={QWEN_URL})
    _up(_encoder_settings(tmp_path), stack, "encoder", free=2624)  # a running Qwen is not counted again
    assert stack.verbs() == ["up:vllm-encoder"]
    refused = FakeStack()
    with pytest.raises(ServingError, match="need 2624 MiB.*free 2623 MiB"):
        _up(_encoder_settings(tmp_path), refused, "encoder", free=2623)
    assert refused.calls == []
    with pytest.raises(ServingError, match="need 11424 MiB"):
        _up(_encoder_settings(tmp_path), refused, "qwen", free=11423)
    assert refused.calls == []


# --- compose command and environment ---

def test_compose_gets_checked_values_over_dotenv(tmp_path: Path) -> None:
    settings = _encoder_settings(tmp_path, VLLM_GPU_MEMORY_UTILIZATION="0.5", VLLM_PORT="9001", VLLM_MODEL="m/x",
                                 VLLM_MODEL_REVISION="rev1", VLLM_MAX_MODEL_LEN="4096",
                                 VLLM_ENCODER_PORT="9002", VLLM_ENCODER_GPU_MEMORY_UTILIZATION="0.2",
                                 VLLM_ENCODER_DTYPE="bfloat16")
    stack = FakeStack()
    stack.urls = {"vllm": "http://127.0.0.1:9001", "vllm-encoder": "http://127.0.0.1:9002"}
    # base env carries stale values like a shell that exported .env with different numbers
    stale = {"PATH": "/bin", "VLLM_GPU_MEMORY_UTILIZATION": "0.9", "VLLM_ENCODER_PORT": "1",
              "VLLM_MODEL_REVISION": "stale", "VLLM_MAX_MODEL_LEN": "1", "VLLM_ENCODER_DTYPE": "float16"}
    for name in ("qwen", "encoder"):
        up(settings, name, run=stack.run, gpu_memory=stack.gpu(16000, 16000), client_for=stack.client_for,
           inspect=stack.inspect, save_logs=stack.save_logs, base_env=stale, timeout=0, interval=0)
    assert stack.verbs() == ["up:vllm", "up:vllm-encoder"]
    assert [c for c, _ in stack.calls][0][:3] == ["docker", "compose", "--project-directory"]
    for cmd, env in stack.calls:
        assert env["VLLM_GPU_MEMORY_UTILIZATION"] == "0.5"
        assert env["VLLM_PORT"] == "9001" and env["VLLM_MODEL"] == "m/x"
        assert env["VLLM_MODEL_REVISION"] == "rev1" and env["VLLM_MAX_MODEL_LEN"] == "4096"
        assert env["VLLM_ENCODER_PORT"] == "9002" and env["VLLM_ENCODER_GPU_MEMORY_UTILIZATION"] == "0.2"
        assert env["VLLM_ENCODER_DTYPE"] == "bfloat16"
        assert env["VLLM_ENCODER_CHECKPOINT_DIR"] == str(tmp_path / "ckpt")
        assert env["PATH"] == "/bin"
        assert cmd[cmd.index("--project-directory") + 1] == str(tmp_path)
        assert cmd[cmd.index("-f") + 1] == str(tmp_path / "compose.yaml")


def test_qwen_defaults_unchanged_in_compose_env(tmp_path: Path) -> None:
    env = compose_env(_settings(tmp_path), {})
    assert env == {"VLLM_MODEL": "Qwen/Qwen3-4B", "VLLM_MODEL_REVISION": "1cfa9a7208912126459214e8b04321603b3df60c",
                   "VLLM_MAX_MODEL_LEN": "8192", "VLLM_PORT": "8000", "VLLM_GPU_MEMORY_UTILIZATION": "0.65"}


# Variables compose.yaml interpolates that are not serving parameters (credentials, cache, host paths).
NON_SERVING_COMPOSE_VARS = {"HF_TOKEN", "HF_HUB_OFFLINE", "HF_CACHE_DIR", "HOME"}


def test_every_serving_compose_variable_is_passed_by_python(tmp_path: Path) -> None:
    import re
    compose = Path(__file__).resolve().parents[1] / "compose.yaml"
    used = set(re.findall(r"\$\{([A-Z_]+)[:}-]", compose.read_text()))
    passed = set(compose_env(_encoder_settings(tmp_path), {}))
    assert used - NON_SERVING_COMPOSE_VARS == passed


def test_encoder_dtype_is_pinned_in_compose_and_passed_by_python(tmp_path: Path) -> None:
    import re
    compose = (Path(__file__).resolve().parents[1] / "compose.yaml").read_text()
    encoder = compose[compose.index("  vllm-encoder:"):]
    match = re.search(r'"--dtype=\$\{VLLM_ENCODER_DTYPE:-([a-z0-9]+)\}"', encoder)
    assert match, "encoder service must pin --dtype"
    assert match.group(1) == Settings.from_env({}, root=tmp_path).vllm_encoder_dtype == "float32"
    env = compose_env(_encoder_settings(tmp_path, VLLM_ENCODER_DTYPE="bfloat16"), {})
    assert env["VLLM_ENCODER_DTYPE"] == "bfloat16"


# --- server selection ---

def test_up_defaults_to_qwen_and_starts_only_it(tmp_path: Path) -> None:
    for settings in (_settings(tmp_path), _encoder_settings(tmp_path)):
        stack = FakeStack()
        up(settings, run=stack.run, gpu_memory=stack.gpu(16000, 16000), client_for=stack.client_for,
           inspect=stack.inspect, save_logs=stack.save_logs, base_env={}, timeout=0, interval=0)
        assert stack.verbs() == ["up:vllm"]
        assert stack.healthy == {QWEN_URL}


def test_up_encoder_starts_only_the_encoder(tmp_path: Path) -> None:
    stack = FakeStack()
    _up(_encoder_settings(tmp_path), stack, "encoder")
    assert stack.verbs() == ["up:vllm-encoder"]
    assert stack.healthy == {ENCODER_URL}
    assert stack.calls[0][1]["VLLM_ENCODER_CHECKPOINT_DIR"] == str(tmp_path / "ckpt")


def test_qwen_env_does_not_carry_encoder_variables_when_unconfigured(tmp_path: Path) -> None:
    stack = FakeStack()
    _up(_settings(tmp_path), stack)
    assert "VLLM_ENCODER_CHECKPOINT_DIR" not in stack.calls[0][1]


def test_encoder_not_configured_starts_nothing(tmp_path: Path) -> None:
    stack = FakeStack()
    with pytest.raises(ServingError, match="encoder is not configured"):
        _up(_settings(tmp_path), stack, "encoder")
    assert stack.calls == [] and stack.gpu_reads == 0


def test_missing_checkpoint_dir_starts_no_encoder(tmp_path: Path) -> None:
    stack = FakeStack()
    with pytest.raises(ServingError, match="not a directory"):
        _up(_settings(tmp_path, VLLM_ENCODER_CHECKPOINT_DIR=str(tmp_path / "missing")), stack, "encoder")
    assert stack.calls == [] and stack.gpu_reads == 0


def test_checkpoint_path_that_is_a_file_fails(tmp_path: Path) -> None:
    (tmp_path / "f").write_text("x")
    with pytest.raises(ServingError):
        _up(_settings(tmp_path, VLLM_ENCODER_CHECKPOINT_DIR=str(tmp_path / "f")), FakeStack(), "encoder")


def test_qwen_does_not_need_the_encoder_checkpoint(tmp_path: Path) -> None:
    stack = FakeStack()
    _up(_settings(tmp_path, VLLM_ENCODER_CHECKPOINT_DIR=str(tmp_path / "missing")), stack, "qwen")
    assert stack.verbs() == ["up:vllm"]


def test_unknown_server_name_is_rejected(tmp_path: Path) -> None:
    stack = FakeStack()
    with pytest.raises(ServingError, match="unknown server"):
        _up(_settings(tmp_path), stack, "both")
    assert stack.calls == []


# --- already running ---

def test_healthy_server_is_not_restarted(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stack = FakeStack(healthy={QWEN_URL, ENCODER_URL})
    _up(_encoder_settings(tmp_path), stack, "qwen")
    _up(_encoder_settings(tmp_path), stack, "encoder")
    assert stack.calls == [] and stack.gpu_reads == 0
    assert capsys.readouterr().out.count("already healthy, not restarted") == 2


# --- failure handling ---

def test_compose_up_failure_removes_the_service(tmp_path: Path) -> None:
    stack = FakeStack(fail_services={"vllm"})
    with pytest.raises(ServingError, match="docker compose up failed for vllm.*no container logs were available"):
        _up(_settings(tmp_path), stack)
    assert stack.events == ["up:vllm", "rm:vllm"]


def test_exited_container_fails_at_once_and_keeps_its_logs(tmp_path: Path) -> None:
    stack = FakeStack(exit_services={"vllm"})
    with pytest.raises(ServingError, match="qwen container exited with code 1 before it was healthy") as exc:
        _up(_settings(tmp_path), stack, timeout=2)  # far shorter than a real 600 s wait, still never reached
    # the logs are saved before the container is removed
    assert stack.events == ["up:vllm", "logs:guardbench-vllm", "rm:vllm"]
    [(path, container)] = stack.saved.items()
    assert container == "guardbench-vllm"
    assert path.parent == tmp_path / "runs" / "vllm-logs" and path.name.startswith("qwen-") and path.suffix == ".log"
    assert str(path) in str(exc.value)
    assert path.read_text() == "logs of guardbench-vllm\n"


def test_container_that_never_gets_healthy_times_out_with_logs(tmp_path: Path) -> None:
    stack = FakeStack(never_healthy={"vllm"})
    with pytest.raises(ServingError, match="not healthy within 0s; logs: .*qwen-"):
        _up(_settings(tmp_path), stack)
    assert stack.events == ["up:vllm", "logs:guardbench-vllm", "rm:vllm"]


def test_unavailable_logs_do_not_hide_the_failure(tmp_path: Path) -> None:
    stack = FakeStack(exit_services={"vllm"}, logs_available=False)
    with pytest.raises(ServingError, match="exited with code 1.*no container logs were available"):
        _up(_settings(tmp_path), stack)
    assert stack.events == ["up:vllm", "logs:guardbench-vllm", "rm:vllm"]


def test_failure_leaves_the_other_server_running(tmp_path: Path) -> None:
    stack = FakeStack(healthy={QWEN_URL}, exit_services={"vllm-encoder"})
    with pytest.raises(ServingError, match="encoder container exited"):
        _up(_encoder_settings(tmp_path), stack, "encoder", timeout=2)
    assert stack.events == ["up:vllm-encoder", "logs:guardbench-vllm-encoder", "rm:vllm-encoder"]
    assert stack.healthy == {QWEN_URL}


def test_failed_removal_is_reported(tmp_path: Path) -> None:
    stack = FakeStack(fail_services={"vllm"}, fail_rm=True)
    with pytest.raises(ServingError, match="could not remove vllm, check `docker ps`") as exc:
        _up(_settings(tmp_path), stack)
    assert "removed the service" not in str(exc.value)


def test_failed_start_that_left_a_container_keeps_its_logs(tmp_path: Path) -> None:
    stack = FakeStack(fail_services={"vllm"}, leave_container=True)
    with pytest.raises(ServingError, match="docker compose up failed for vllm; logs: .*qwen-.*removed the service"):
        _up(_settings(tmp_path), stack)
    assert stack.events == ["up:vllm", "logs:guardbench-vllm", "rm:vllm"]


# --- the wait loop (fake clock: no real sleeping, every poll is visible) ---

class Clock:
    def __init__(self) -> None:
        self.now = 0.0


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    fake = Clock()
    monkeypatch.setattr(serving.time, "monotonic", lambda: fake.now)
    monkeypatch.setattr(serving.time, "sleep", lambda seconds: setattr(fake, "now", fake.now + seconds))
    return fake


def _scripted(states: list[ContainerInfo | None], healthy_at: int | None = None):
    """A server that answers /health with 503 until its ``healthy_at``-th check, and a container whose
    state follows ``states`` poll by poll (the last state repeats). Returns (client_for, inspect, polls)."""
    polls = {"health": 0, "inspect": 0}

    def client_for(url: str) -> VllmClient:
        def handler(request: httpx.Request) -> httpx.Response:
            polls["health"] += 1
            return httpx.Response(200 if healthy_at is not None and polls["health"] >= healthy_at else 503)
        return VllmClient(url, transport=httpx.MockTransport(handler))

    def inspect(name: str) -> ContainerInfo | None:
        state = states[min(polls["inspect"], len(states) - 1)]
        polls["inspect"] += 1
        return state
    return client_for, inspect, polls


RUNNING = ContainerInfo("running", None, ())


def _qwen(tmp_path: Path) -> serving.Server:
    return configured_servers(_settings(tmp_path))[0]


def test_wait_polls_until_the_server_is_healthy(tmp_path: Path, clock: Clock) -> None:
    client_for, inspect, polls = _scripted([RUNNING], healthy_at=4)
    assert serving._wait_until_up(_qwen(tmp_path), client_for, inspect, 600, 5) == ("healthy", None)
    assert polls["health"] == 4 and clock.now == 15


def test_wait_stops_when_the_container_exits_midway(tmp_path: Path, clock: Clock) -> None:
    exited = ContainerInfo("exited", 137, ())
    client_for, inspect, polls = _scripted([RUNNING, RUNNING, RUNNING, exited])
    assert serving._wait_until_up(_qwen(tmp_path), client_for, inspect, 600, 5) == ("exited", exited)
    assert clock.now == 15  # not 600


def test_wait_treats_a_dead_container_like_an_exited_one(tmp_path: Path, clock: Clock) -> None:
    dead = ContainerInfo("dead", None, ())
    client_for, inspect, _ = _scripted([dead])
    assert serving._wait_until_up(_qwen(tmp_path), client_for, inspect, 600, 5) == ("exited", dead)
    assert clock.now == 0


def test_wait_keeps_waiting_while_docker_cannot_inspect(tmp_path: Path, clock: Clock) -> None:
    client_for, inspect, _ = _scripted([None])
    assert serving._wait_until_up(_qwen(tmp_path), client_for, inspect, 20, 5) == ("timeout", None)
    assert clock.now == 20


@pytest.mark.parametrize("timeout,interval", [(7, 5), (1, 5), (12, 5)])
def test_wait_never_sleeps_past_the_timeout(tmp_path: Path, clock: Clock, timeout: float, interval: float) -> None:
    client_for, inspect, _ = _scripted([RUNNING])
    assert serving._wait_until_up(_qwen(tmp_path), client_for, inspect, timeout, interval)[0] == "timeout"
    assert clock.now == timeout


def test_up_fails_when_the_container_exits_while_waiting(tmp_path: Path, clock: Clock) -> None:
    stack = FakeStack(never_healthy={"vllm"})
    client_for, inspect, _ = _scripted([RUNNING, RUNNING, ContainerInfo("exited", 1, ())])
    with pytest.raises(ServingError, match="exited with code 1 before it was healthy; logs: "):
        up(_settings(tmp_path), run=stack.run, gpu_memory=stack.gpu(16000, 16000), client_for=client_for,
           inspect=inspect, save_logs=stack.save_logs, base_env={}, timeout=600, interval=5)
    assert clock.now == 10  # two sleeps, then the exit is seen; the 600 s timeout is never reached
    assert stack.events == ["up:vllm", "logs:guardbench-vllm", "rm:vllm"]


def test_unknown_exit_code_is_shown_as_a_question_mark(tmp_path: Path, clock: Clock,
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    client_for, inspect, _ = _scripted([ContainerInfo("exited", None, ())])
    assert not wait_server(_settings(tmp_path), "qwen", 600, client_for=client_for, inspect=inspect)
    assert capsys.readouterr().out == "qwen: exited (exit code ?)\n"


# --- down, status, wait ---

def test_down_runs_compose_down_with_encoder_profile(tmp_path: Path) -> None:
    stack = FakeStack()
    down(_settings(tmp_path), run=stack.run, base_env={})
    cmd, env = stack.calls[0]
    assert cmd[-1] == "down" and "--profile" in cmd and cmd[cmd.index("--profile") + 1] == "encoder"
    assert env["VLLM_PORT"] == "8000"


def test_down_failure_raises(tmp_path: Path) -> None:
    with pytest.raises(ServingError):
        down(_settings(tmp_path), run=lambda cmd, env: 1, base_env={})


def test_status_reports_encoder_not_configured_and_dtype(tmp_path: Path) -> None:
    stack = FakeStack(healthy={QWEN_URL})
    text = format_status(status_all(_settings(tmp_path), client_for=stack.client_for, inspect=stack.inspect)).splitlines()
    assert text[0].startswith('qwen: {"reachable": true, "version": "0.16.0"')
    assert '"dtype": "bfloat16"' in text[0]
    assert text[-1] == "encoder: not configured"


def test_status_lists_both_when_configured(tmp_path: Path) -> None:
    stack = FakeStack()
    report = status_all(_encoder_settings(tmp_path), client_for=stack.client_for, inspect=stack.inspect)
    assert report == {"qwen": {"reachable": False}, "encoder": {"reachable": False}}


def test_status_of_one_server(tmp_path: Path) -> None:
    stack = FakeStack(healthy={QWEN_URL, ENCODER_URL})
    report = status_all(_encoder_settings(tmp_path), "encoder", client_for=stack.client_for, inspect=stack.inspect)
    assert list(report) == ["encoder"] and report["encoder"]["dtype"] == "float32"
    report = status_all(_encoder_settings(tmp_path), "qwen", client_for=stack.client_for, inspect=stack.inspect)
    assert list(report) == ["qwen"] and report["qwen"]["dtype"] == "bfloat16"


def test_status_of_unconfigured_encoder(tmp_path: Path) -> None:
    stack = FakeStack(healthy={QWEN_URL})
    assert status_all(_settings(tmp_path), "encoder", client_for=stack.client_for, inspect=stack.inspect) \
        == {"encoder": "not configured"}


def test_status_reports_dtype_only_for_a_running_container(tmp_path: Path) -> None:
    stack = FakeStack(healthy={QWEN_URL})
    left_over = ContainerInfo("exited", 1, ("vllm", "serve", "--dtype=float16"))
    report = status_all(_settings(tmp_path), "qwen", client_for=stack.client_for, inspect=lambda name: left_over)
    assert report["qwen"]["reachable"] is True and "dtype" not in report["qwen"]


def test_status_rejects_an_unknown_server_name(tmp_path: Path) -> None:
    with pytest.raises(ServingError, match="unknown server"):
        status_all(_settings(tmp_path), "typo", client_for=FakeStack().client_for, inspect=FakeStack().inspect)


def test_status_omits_dtype_when_the_container_is_unknown(tmp_path: Path) -> None:
    stack = FakeStack(healthy={QWEN_URL})
    report = status_all(_settings(tmp_path), "qwen", client_for=stack.client_for, inspect=lambda name: None)
    assert report["qwen"]["reachable"] is True and "dtype" not in report["qwen"]


def test_dtype_of_command_forms() -> None:
    assert dtype_of(("/models/encoder", "--dtype=float32", "--seed=0")) == "float32"
    assert dtype_of(("--dtype", "bfloat16")) == "bfloat16"
    assert dtype_of(("--dtype",)) is None
    assert dtype_of(("--seed=0",)) is None


def test_wait_server_returns_when_healthy(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stack = FakeStack(healthy={ENCODER_URL})
    assert wait_server(_encoder_settings(tmp_path), "encoder", 0, client_for=stack.client_for, inspect=stack.inspect,
                       interval=0)
    assert capsys.readouterr().out == "encoder: healthy\n"


def test_wait_server_times_out_when_the_container_is_unknown(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stack = FakeStack()
    assert not wait_server(_settings(tmp_path), "qwen", 0, client_for=stack.client_for, inspect=stack.inspect,
                           interval=0)
    assert capsys.readouterr().out == "qwen: timeout\n"


def test_wait_server_fails_at_once_when_the_container_exited(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stack = FakeStack(exit_services={"vllm"})
    stack.run(["docker", "compose", "up", "-d", "vllm"], {})
    assert not wait_server(_settings(tmp_path), "qwen", 2, client_for=stack.client_for, inspect=stack.inspect,
                           interval=0)
    assert capsys.readouterr().out == "qwen: exited (exit code 1)\n"


def test_wait_server_needs_a_configured_encoder(tmp_path: Path) -> None:
    with pytest.raises(ServingError, match="encoder is not configured"):
        wait_server(_settings(tmp_path), "encoder", 0, client_for=FakeStack().client_for, inspect=FakeStack().inspect)


# --- CLI exit codes and nvidia-smi ---

def test_cli_exits_non_zero_on_failure(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def boom(settings: Settings, name: str, **kw: object) -> None:
        raise ServingError("nope")
    monkeypatch.setattr(serving, "up", boom)
    monkeypatch.setattr("sys.argv", ["guardbench-vllm", "up"])
    with pytest.raises(SystemExit) as exc:
        serving.main()
    assert exc.value.code == 1 and "error: nope" in capsys.readouterr().err


def test_cli_passes_the_chosen_server(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[tuple[str, float]] = []
    monkeypatch.setattr(serving, "up", lambda settings, name, timeout: started.append((name, timeout)))
    monkeypatch.setattr(serving, "status_all", lambda settings, name=None: {"x": name or "all"})
    for argv in (["up"], ["up", "encoder"], ["up", "qwen", "--timeout", "30"]):
        monkeypatch.setattr("sys.argv", ["guardbench-vllm", *argv])
        serving.main()
    assert started == [("qwen", 600.0), ("encoder", 600.0), ("qwen", 30.0)]


def test_cli_status_and_down(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    asked: list[str | None] = []

    def fake_status(settings: Settings, name: str | None = None) -> dict[str, str]:
        asked.append(name)
        return {name or "all": "ok"}
    monkeypatch.setattr(serving, "status_all", fake_status)
    for argv in (["status"], ["status", "encoder"], ["status", "qwen"]):
        monkeypatch.setattr("sys.argv", ["guardbench-vllm", *argv])
        serving.main()
    assert asked == [None, "encoder", "qwen"]
    assert capsys.readouterr().out.splitlines() == ["all: ok", "encoder: ok", "qwen: ok"]

    stopped: list[Settings] = []
    monkeypatch.setattr(serving, "down", stopped.append)
    monkeypatch.setattr("sys.argv", ["guardbench-vllm", "down"])
    serving.main()
    assert len(stopped) == 1 and capsys.readouterr().out == "down\n"


def test_cli_up_prints_the_status_of_that_server(monkeypatch: pytest.MonkeyPatch,
                                                 capsys: pytest.CaptureFixture[str]) -> None:
    asked: list[str | None] = []
    monkeypatch.setattr(serving, "up", lambda settings, name, timeout: None)
    monkeypatch.setattr(serving, "status_all", lambda settings, name=None: asked.append(name) or {name: "ok"})
    monkeypatch.setattr("sys.argv", ["guardbench-vllm", "up", "encoder"])
    serving.main()
    assert asked == ["encoder"] and capsys.readouterr().out == "encoder: ok\n"


def test_cli_rejects_an_unknown_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["guardbench-vllm", "up", "both"])
    with pytest.raises(SystemExit) as exc:
        serving.main()
    assert exc.value.code == 2


def test_cli_wait_exit_code_follows_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serving, "wait_server", lambda settings, name, timeout: name == "qwen")
    for argv, code in ((["wait"], 0), (["wait", "encoder"], 1)):
        monkeypatch.setattr("sys.argv", ["guardbench-vllm", *argv])
        with pytest.raises(SystemExit) as exc:
            serving.main()
        assert exc.value.code == code


def test_cli_rejects_invalid_setting_with_non_zero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_GPU_MEMORY_UTILIZATION", "2")
    monkeypatch.setattr("sys.argv", ["guardbench-vllm", "up"])
    with pytest.raises(SystemExit) as exc:
        serving.main()
    assert exc.value.code == 1


def test_read_gpu_memory_runs_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="16376, 12530\n", stderr="")
    monkeypatch.setattr(serving.subprocess, "run", fake_run)
    assert serving.read_gpu_memory_mib() == (16376, 12530)
    assert seen[0][0] == "nvidia-smi" and "--query-gpu=memory.total,memory.free" in seen[0]


def test_read_gpu_memory_failure_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(cmd: list[str], **kw: object) -> None:
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(serving.subprocess, "run", missing)
    with pytest.raises(ServingError):
        serving.read_gpu_memory_mib()


def test_inspect_container_reads_state_and_command(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess
    payload = json.dumps([{"State": {"Status": "exited", "ExitCode": 1},
                           "Config": {"Cmd": ["/models/encoder", "--dtype=float32"]}}])
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=payload, stderr="")
    monkeypatch.setattr(serving.subprocess, "run", fake_run)
    assert serving.inspect_container("guardbench-vllm-encoder") == ContainerInfo(
        "exited", 1, ("/models/encoder", "--dtype=float32"))
    assert seen == [["docker", "inspect", "guardbench-vllm-encoder"]]


@pytest.mark.parametrize("returncode,stdout", [(1, ""), (0, "not json"), (0, "[]"), (0, "[{}]")])
def test_inspect_container_returns_none_when_docker_cannot_say(monkeypatch: pytest.MonkeyPatch, returncode: int,
                                                               stdout: str) -> None:
    import subprocess
    monkeypatch.setattr(serving.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=""))
    assert serving.inspect_container("x") is None


def test_inspect_container_returns_none_without_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(cmd: list[str], **kw: object) -> None:
        raise FileNotFoundError("docker")
    monkeypatch.setattr(serving.subprocess, "run", missing)
    assert serving.inspect_container("x") is None


def test_save_container_logs_writes_stdout_and_stderr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import subprocess

    def fake_run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[bytes]:
        assert cmd == ["docker", "logs", "guardbench-vllm"] and kw["stderr"] == subprocess.STDOUT
        kw["stdout"].write(b"engine failed\n")  # type: ignore[union-attr]
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(serving.subprocess, "run", fake_run)
    path = tmp_path / "runs" / "vllm-logs" / "qwen-1.log"
    assert serving.save_container_logs("guardbench-vllm", path)
    assert path.read_text() == "engine failed\n"


def test_save_container_logs_removes_the_file_when_docker_cannot_run(monkeypatch: pytest.MonkeyPatch,
                                                                     tmp_path: Path) -> None:
    def missing(cmd: list[str], **kw: object) -> None:
        raise FileNotFoundError("docker")
    monkeypatch.setattr(serving.subprocess, "run", missing)
    path = tmp_path / "vllm-logs" / "qwen-1.log"
    assert not serving.save_container_logs("guardbench-vllm", path)
    assert not path.exists()


def test_save_container_logs_removes_the_file_when_docker_fails(monkeypatch: pytest.MonkeyPatch,
                                                                tmp_path: Path) -> None:
    import subprocess
    monkeypatch.setattr(serving.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1))
    path = tmp_path / "vllm-logs" / "qwen-1.log"
    assert not serving.save_container_logs("guardbench-vllm", path)
    assert not path.exists()


@pytest.mark.integration
def test_live_server_greedy_is_repeatable() -> None:
    client = VllmClient(Settings.from_env().vllm_base_url)
    if not client.healthy():
        pytest.skip("vLLM server is not running")
    model = client.models()[0]["id"]
    messages = [{"role": "user", "content": "Reply with exactly one word: COMPLY or REFUSE."}]
    outs = [client.chat(model, messages, max_tokens=8, chat_template_kwargs={"enable_thinking": False})
            ["choices"][0]["message"]["content"] for _ in range(3)]
    assert len(set(outs)) == 1
