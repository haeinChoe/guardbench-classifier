"""Client and lifecycle commands for the local vLLM servers (see compose.yaml).

Qwen (service ``vllm``) is always managed; the encoder (service ``vllm-encoder``) only when
VLLM_ENCODER_CHECKPOINT_DIR is set.

  guardbench-vllm up [--timeout 600]     # check free GPU memory, start, wait for health, print status
  guardbench-vllm down                   # stop and remove the compose services
  guardbench-vllm status                 # version and served models per server, or "unreachable"
  guardbench-vllm wait [--timeout 600]   # block until /health is OK (exit 1 on timeout)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from guardbench_classifier.config import Settings


class VllmClient:
    def __init__(self, base_url: str, timeout: float = 600.0, transport: httpx.BaseTransport | None = None) -> None:
        self._http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, transport=transport)

    def healthy(self) -> bool:
        try:
            return self._http.get("/health", timeout=5.0).status_code == 200
        except httpx.HTTPError:
            return False

    def version(self) -> str:
        return self._http.get("/version").raise_for_status().json()["version"]

    def models(self) -> list[dict[str, Any]]:
        return self._http.get("/v1/models").raise_for_status().json()["data"]

    def chat(self, model: str, messages: list[dict[str, str]], *, max_tokens: int, temperature: float = 0.0,
             **extra: Any) -> dict[str, Any]:
        """One chat completion. Greedy by default; ``extra`` passes vLLM fields such as chat_template_kwargs."""
        body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, **extra}
        return self._http.post("/v1/chat/completions", json=body).raise_for_status().json()

    def close(self) -> None:
        self._http.close()


def status(client: VllmClient) -> dict[str, Any]:
    if not client.healthy():
        return {"reachable": False}
    return {"reachable": True, "version": client.version(),
            "models": [{"id": m["id"], "root": m.get("root"), "max_model_len": m.get("max_model_len")} for m in client.models()]}


def wait(client: VllmClient, timeout: float, interval: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        if client.healthy():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)


class ServingError(RuntimeError):
    """A pre-flight check or a docker call failed; ``up``/``down`` exit non-zero."""


@dataclass(frozen=True)
class Server:
    name: str  # label used in output
    service: str  # compose service name
    base_url: str
    memory_fraction: float


# (command, environment) -> exit code. Tests inject a fake instead of running docker.
Runner = Callable[[Sequence[str], Mapping[str, str]], int]
# () -> (total_mib, free_mib) of the GPU compose will use. Tests inject a fake instead of nvidia-smi.
GpuMemory = Callable[[], tuple[int, int]]
ClientFactory = Callable[[str], VllmClient]

ENCODER_PROFILE = "encoder"


def configured_servers(settings: Settings) -> list[Server]:
    servers = [Server("qwen", "vllm", settings.vllm_base_url, settings.vllm_gpu_memory_utilization)]
    if settings.vllm_encoder_checkpoint_dir is not None:
        servers.append(Server("encoder", "vllm-encoder", f"http://127.0.0.1:{settings.vllm_encoder_port}",
                              settings.vllm_encoder_gpu_memory_utilization))
    return servers


def check_encoder_checkpoint(settings: Settings) -> None:
    path = settings.vllm_encoder_checkpoint_dir
    if path is not None and not path.is_dir():
        raise ServingError(f"VLLM_ENCODER_CHECKPOINT_DIR is not a directory: {path}")


def compose_env(settings: Settings, base: Mapping[str, str]) -> dict[str, str]:
    """Process environment for docker compose.

    Process variables take precedence over .env, so the values Python checked are the values
    compose starts with, whether or not --env-file was used. Every serving variable that
    compose.yaml interpolates is passed here; the encoder ones only when it is configured.
    """
    env = dict(base)
    env.update(
        VLLM_MODEL=settings.vllm_model,
        VLLM_MODEL_REVISION=settings.vllm_model_revision,
        VLLM_MAX_MODEL_LEN=str(settings.vllm_max_model_len),
        VLLM_PORT=str(settings.vllm_port),
        VLLM_GPU_MEMORY_UTILIZATION=str(settings.vllm_gpu_memory_utilization),
    )
    if settings.vllm_encoder_checkpoint_dir is not None:
        env.update(
            VLLM_ENCODER_CHECKPOINT_DIR=str(settings.vllm_encoder_checkpoint_dir),
            VLLM_ENCODER_PORT=str(settings.vllm_encoder_port),
            VLLM_ENCODER_GPU_MEMORY_UTILIZATION=str(settings.vllm_encoder_gpu_memory_utilization),
        )
    return env


def compose_command(settings: Settings, *args: str) -> list[str]:
    """``docker compose`` rooted at the repo, with the encoder profile enabled so its container
    can be addressed (started or removed). Which services start is decided by the service names."""
    root = settings.repo_root
    return ["docker", "compose", "--project-directory", str(root), "-f", str(root / "compose.yaml"),
            "--profile", ENCODER_PROFILE, *args]


def run_command(cmd: Sequence[str], env: Mapping[str, str]) -> int:
    try:
        return subprocess.run(list(cmd), env=dict(env), check=False).returncode
    except OSError as exc:
        print(f"cannot run {cmd[0]}: {exc}", file=sys.stderr)
        return 127


def parse_gpu_memory(output: str) -> tuple[int, int]:
    """Parse ``nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits``.

    Uses the first GPU listed, the one docker's ``count: 1`` reservation takes by default.
    """
    lines = [line for line in output.splitlines() if line.strip()]
    try:
        total, free = (int(part) for part in lines[0].split(","))
    except (IndexError, ValueError):
        raise ServingError(f"cannot parse nvidia-smi output: {output!r}") from None
    return total, free


def read_gpu_memory_mib() -> tuple[int, int]:
    cmd = ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise ServingError(f"cannot read GPU memory with nvidia-smi: {exc}") from None
    if proc.returncode != 0:
        raise ServingError(f"nvidia-smi failed ({proc.returncode}): {proc.stderr.strip()}")
    return parse_gpu_memory(proc.stdout)


def check_gpu_memory(servers: Sequence[Server], margin_mib: int, total_mib: int, free_mib: int) -> None:
    """Fail if sum(fraction * total + margin) over the servers exceeds the free memory.

    The margin is per server: each vLLM process also holds memory outside its fraction (CUDA context).
    """
    needed = round(sum(server.memory_fraction * total_mib + margin_mib for server in servers), 3)
    if needed > free_mib:
        names = "+".join(server.name for server in servers)
        fractions = sum(server.memory_fraction for server in servers)
        raise ServingError(f"not enough free GPU memory for {names}: need {needed:g} MiB "
                           f"(fractions {fractions:g} x total {total_mib} MiB + {len(servers)} x margin {margin_mib} MiB), "
                           f"free {free_mib} MiB")


def status_all(settings: Settings, client_for: ClientFactory = VllmClient) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for server in configured_servers(settings):
        client = client_for(server.base_url)
        try:
            report[server.name] = status(client)
        finally:
            client.close()
    report.setdefault("encoder", "not configured")
    return report


def format_status(report: Mapping[str, Any]) -> str:
    return "\n".join(f"{name}: {value if isinstance(value, str) else json.dumps(value)}" for name, value in report.items())


def _healthy(server: Server, client_for: ClientFactory) -> bool:
    client = client_for(server.base_url)
    try:
        return client.healthy()
    finally:
        client.close()


def _wait_healthy(server: Server, client_for: ClientFactory, timeout: float, interval: float) -> bool:
    client = client_for(server.base_url)
    try:
        return wait(client, timeout, interval)
    finally:
        client.close()


def up(settings: Settings, *, run: Runner = run_command, gpu_memory: GpuMemory = read_gpu_memory_mib,
       client_for: ClientFactory = VllmClient, base_env: Mapping[str, str] | None = None,
       timeout: float = 600.0, interval: float = 5.0) -> None:
    """Start the configured servers that are not already healthy, one at a time, and wait for each.

    Raises ServingError (after removing what this call started) on any failure. Nothing is started
    if the encoder checkpoint directory is missing or free GPU memory is insufficient.
    """
    check_encoder_checkpoint(settings)
    servers = configured_servers(settings)
    to_start = []
    for server in servers:
        if _healthy(server, client_for):
            print(f"{server.name}: already healthy, not restarted")
        else:
            to_start.append(server)
    if to_start:
        total, free = gpu_memory()
        check_gpu_memory(to_start, settings.vllm_gpu_memory_margin_mib, total, free)
    env = compose_env(settings, os.environ if base_env is None else base_env)
    started: list[Server] = []
    for server in to_start:
        started.append(server)  # a failed `up -d` can still leave a created container to remove
        print(f"{server.name}: starting service {server.service}")
        if run(compose_command(settings, "up", "-d", server.service), env) != 0:
            _rollback(settings, started, run, env)
            raise ServingError(f"docker compose up failed for {server.service}; removed services started by this run")
        if not _wait_healthy(server, client_for, timeout, interval):
            _rollback(settings, started, run, env)
            raise ServingError(f"{server.name} not healthy within {timeout:g}s; removed services started by this run")
        print(f"{server.name}: healthy")


def _rollback(settings: Settings, started: Sequence[Server], run: Runner, env: Mapping[str, str]) -> None:
    for server in reversed(started):
        cmd = compose_command(settings, "rm", "--stop", "--force", server.service)
        if run(cmd, env) != 0:
            print(f"warning: could not remove {server.service}; check `docker ps`", file=sys.stderr)


def down(settings: Settings, *, run: Runner = run_command, base_env: Mapping[str, str] | None = None) -> None:
    env = compose_env(settings, os.environ if base_env is None else base_env)
    if run(compose_command(settings, "down"), env) != 0:
        raise ServingError("docker compose down failed")


def wait_all(settings: Settings, timeout: float, *, client_for: ClientFactory = VllmClient,
             interval: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    ok = True
    for server in configured_servers(settings):
        healthy = _wait_healthy(server, client_for, max(0.0, deadline - time.monotonic()), interval)
        print(f"{server.name}: {'healthy' if healthy else 'timeout'}")
        ok = ok and healthy
    if len(configured_servers(settings)) == 1:
        print("encoder: not configured")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(prog="guardbench-vllm")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("down")
    for name in ("up", "wait"):
        sub.add_parser(name).add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    try:
        settings = Settings.from_env()
        if args.cmd == "status":
            print(format_status(status_all(settings)))
        elif args.cmd == "wait":
            sys.exit(0 if wait_all(settings, args.timeout) else 1)
        elif args.cmd == "up":
            up(settings, timeout=args.timeout)
            print(format_status(status_all(settings)))
        else:
            down(settings)
            print("down")
    except (ServingError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
