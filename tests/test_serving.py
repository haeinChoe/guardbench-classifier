import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import httpx
import pytest

from guardbench_classifier.config import Settings
from guardbench_classifier import serving
from guardbench_classifier.serving import (
    ServingError, VllmClient, check_gpu_memory, compose_env, configured_servers, down, format_status,
    parse_gpu_memory, status, status_all, up, wait_all,
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


class FakeStack:
    """Stands in for docker compose and the servers it would start. Nothing real is run."""

    def __init__(self, healthy: set[str] | None = None, fail_services: set[str] = frozenset(),
                 never_healthy: set[str] = frozenset(), fail_rm: bool = False) -> None:
        self.healthy = set(healthy or ())
        self.fail_services = set(fail_services)
        self.never_healthy = set(never_healthy)
        self.fail_rm = fail_rm
        self.calls: list[tuple[list[str], dict[str, str]]] = []
        self.gpu_reads = 0
        self.urls = {"vllm": QWEN_URL, "vllm-encoder": ENCODER_URL}

    def run(self, cmd: Sequence[str], env: Mapping[str, str]) -> int:
        self.calls.append((list(cmd), dict(env)))
        verb = next(a for a in cmd if a in {"up", "down", "rm"})
        service = cmd[-1]
        if verb == "up":
            if service in self.fail_services:
                return 1
            if service not in self.never_healthy:
                self.healthy.add(self.urls[service])
        elif verb == "rm":
            self.healthy.discard(self.urls[service])
            return 1 if self.fail_rm else 0
        return 0

    def gpu(self, total: int, free: int):
        def read() -> tuple[int, int]:
            self.gpu_reads += 1
            return total, free
        return read

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


def _up(settings: Settings, stack: FakeStack, total: int = 16000, free: int = 16000, **kw):
    return up(settings, run=stack.run, gpu_memory=stack.gpu(total, free), client_for=stack.client_for,
              base_env={"PATH": "/bin"}, timeout=0, interval=0, **kw)


# --- memory check ---

def test_parse_gpu_memory_uses_first_gpu() -> None:
    assert parse_gpu_memory("16376, 12530\n24000, 100\n") == (16376, 12530)
    with pytest.raises(ServingError):
        parse_gpu_memory("not csv")
    with pytest.raises(ServingError):
        parse_gpu_memory("")


def test_memory_boundary_is_inclusive() -> None:
    servers = configured_servers(Settings.from_env({"VLLM_ENCODER_CHECKPOINT_DIR": "x"}, root=Path("/r")))
    # (0.65 + 0.10) * 16000 + 1024 = 13024
    check_gpu_memory(servers, 1024, 16000, 13024)
    with pytest.raises(ServingError, match="need 13024 MiB.*free 13023 MiB"):
        check_gpu_memory(servers, 1024, 16000, 13023)


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


def test_memory_only_counts_servers_to_start(tmp_path: Path) -> None:
    stack = FakeStack(healthy={QWEN_URL})
    # Qwen already runs; only the encoder (0.10 * 16000 + 1024 = 2624) must fit.
    _up(_encoder_settings(tmp_path), stack, free=2624)
    assert stack.verbs() == ["up:vllm-encoder"]


# --- compose command and environment ---

def test_compose_gets_checked_values_over_dotenv(tmp_path: Path) -> None:
    settings = _encoder_settings(tmp_path, VLLM_GPU_MEMORY_UTILIZATION="0.5", VLLM_PORT="9001", VLLM_MODEL="m/x",
                                 VLLM_ENCODER_PORT="9002", VLLM_ENCODER_GPU_MEMORY_UTILIZATION="0.2")
    stack = FakeStack()
    stack.urls = {"vllm": "http://127.0.0.1:9001", "vllm-encoder": "http://127.0.0.1:9002"}
    # base env carries stale values like a shell that exported .env with different numbers
    up(settings, run=stack.run, gpu_memory=stack.gpu(16000, 16000), client_for=stack.client_for,
       base_env={"PATH": "/bin", "VLLM_GPU_MEMORY_UTILIZATION": "0.9", "VLLM_ENCODER_PORT": "1"},
       timeout=0, interval=0)
    assert len(stack.calls) == 2
    assert [c for c, _ in stack.calls][0][:3] == ["docker", "compose", "--project-directory"]
    for cmd, env in stack.calls:
        assert env["VLLM_GPU_MEMORY_UTILIZATION"] == "0.5"
        assert env["VLLM_PORT"] == "9001" and env["VLLM_MODEL"] == "m/x"
        assert env["VLLM_ENCODER_PORT"] == "9002" and env["VLLM_ENCODER_GPU_MEMORY_UTILIZATION"] == "0.2"
        assert env["VLLM_ENCODER_CHECKPOINT_DIR"] == str(tmp_path / "ckpt")
        assert env["PATH"] == "/bin"
        assert cmd[cmd.index("--project-directory") + 1] == str(tmp_path)
        assert cmd[cmd.index("-f") + 1] == str(tmp_path / "compose.yaml")


def test_qwen_defaults_unchanged_in_compose_env(tmp_path: Path) -> None:
    env = compose_env(_settings(tmp_path), {})
    assert env == {"VLLM_MODEL": "Qwen/Qwen3-4B", "VLLM_PORT": "8000", "VLLM_GPU_MEMORY_UTILIZATION": "0.65"}


# --- encoder selection ---

def test_encoder_unconfigured_starts_qwen_only(tmp_path: Path) -> None:
    stack = FakeStack()
    _up(_settings(tmp_path), stack)
    assert stack.verbs() == ["up:vllm"]
    assert "VLLM_ENCODER_CHECKPOINT_DIR" not in stack.calls[0][1]


def test_encoder_configured_starts_both_in_order(tmp_path: Path) -> None:
    stack = FakeStack()
    _up(_encoder_settings(tmp_path), stack)
    assert stack.verbs() == ["up:vllm", "up:vllm-encoder"]
    assert stack.healthy == {QWEN_URL, ENCODER_URL}


def test_missing_checkpoint_dir_starts_nothing(tmp_path: Path) -> None:
    stack = FakeStack()
    with pytest.raises(ServingError, match="not a directory"):
        _up(_settings(tmp_path, VLLM_ENCODER_CHECKPOINT_DIR=str(tmp_path / "missing")), stack)
    assert stack.calls == [] and stack.gpu_reads == 0


def test_checkpoint_path_that_is_a_file_fails(tmp_path: Path) -> None:
    (tmp_path / "f").write_text("x")
    with pytest.raises(ServingError):
        _up(_settings(tmp_path, VLLM_ENCODER_CHECKPOINT_DIR=str(tmp_path / "f")), FakeStack())


# --- already running ---

def test_healthy_servers_are_not_restarted(tmp_path: Path) -> None:
    stack = FakeStack(healthy={QWEN_URL, ENCODER_URL})
    _up(_encoder_settings(tmp_path), stack)
    assert stack.calls == [] and stack.gpu_reads == 0


# --- partial failure ---

def test_encoder_start_failure_removes_what_this_run_started(tmp_path: Path) -> None:
    stack = FakeStack(fail_services={"vllm-encoder"})
    with pytest.raises(ServingError, match="vllm-encoder"):
        _up(_encoder_settings(tmp_path), stack)
    assert stack.verbs() == ["up:vllm", "up:vllm-encoder", "rm:vllm-encoder", "rm:vllm"]
    assert stack.healthy == set()


def test_unhealthy_encoder_rolls_back_both(tmp_path: Path) -> None:
    stack = FakeStack(never_healthy={"vllm-encoder"})
    with pytest.raises(ServingError, match="not healthy"):
        _up(_encoder_settings(tmp_path), stack)
    assert stack.verbs() == ["up:vllm", "up:vllm-encoder", "rm:vllm-encoder", "rm:vllm"]


def test_rollback_keeps_servers_that_were_already_running(tmp_path: Path) -> None:
    stack = FakeStack(healthy={QWEN_URL}, fail_services={"vllm-encoder"})
    with pytest.raises(ServingError):
        _up(_encoder_settings(tmp_path), stack)
    assert stack.verbs() == ["up:vllm-encoder", "rm:vllm-encoder"]
    assert stack.healthy == {QWEN_URL}


def test_qwen_failure_never_starts_encoder(tmp_path: Path) -> None:
    stack = FakeStack(fail_services={"vllm"})
    with pytest.raises(ServingError):
        _up(_encoder_settings(tmp_path), stack)
    assert stack.verbs() == ["up:vllm", "rm:vllm"]


def test_failed_rollback_still_raises(tmp_path: Path) -> None:
    stack = FakeStack(fail_services={"vllm"}, fail_rm=True)
    with pytest.raises(ServingError):
        _up(_settings(tmp_path), stack)


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


def test_status_reports_encoder_not_configured(tmp_path: Path) -> None:
    stack = FakeStack(healthy={QWEN_URL})
    text = format_status(status_all(_settings(tmp_path), stack.client_for)).splitlines()
    assert text[0].startswith('qwen: {"reachable": true, "version": "0.16.0"')
    assert text[-1] == "encoder: not configured"


def test_status_lists_both_when_configured(tmp_path: Path) -> None:
    report = status_all(_encoder_settings(tmp_path), FakeStack().client_for)
    assert report == {"qwen": {"reachable": False}, "encoder": {"reachable": False}}


def test_wait_all_requires_every_server(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    settings = _encoder_settings(tmp_path)
    assert wait_all(settings, 0, client_for=FakeStack(healthy={QWEN_URL, ENCODER_URL}).client_for, interval=0)
    assert not wait_all(settings, 0, client_for=FakeStack(healthy={QWEN_URL}).client_for, interval=0)
    assert "encoder: timeout" in capsys.readouterr().out


def test_wait_all_qwen_only_mentions_encoder(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert wait_all(_settings(tmp_path), 0, client_for=FakeStack(healthy={QWEN_URL}).client_for, interval=0)
    assert "encoder: not configured" in capsys.readouterr().out


# --- CLI exit codes and nvidia-smi ---

def test_cli_exits_non_zero_on_failure(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def boom(settings: Settings, **kw: object) -> None:
        raise ServingError("nope")
    monkeypatch.setattr(serving, "up", boom)
    monkeypatch.setattr("sys.argv", ["guardbench-vllm", "up"])
    with pytest.raises(SystemExit) as exc:
        serving.main()
    assert exc.value.code == 1 and "error: nope" in capsys.readouterr().err


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
