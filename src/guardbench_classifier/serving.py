"""Client and lifecycle commands for the local vLLM servers (see compose.yaml).

Servers run one at a time: Qwen (compose service ``vllm``) and the encoder (``vllm-encoder``, needs
VLLM_ENCODER_CHECKPOINT_DIR) do not fit in the GPU memory left by other jobs together.

  guardbench-vllm up [qwen|encoder] [--timeout 600]    # check free GPU memory, start one server, wait for health
  guardbench-vllm down                                 # stop and remove every compose service
  guardbench-vllm status [qwen|encoder]                # version, served models and dtype, or "unreachable"
  guardbench-vllm wait [qwen|encoder] [--timeout 600]  # block until /health is OK (exit 1 on timeout or exit)

``up`` and ``wait`` default to qwen; ``status`` without a name reports every configured server.
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
from datetime import datetime, timezone
from pathlib import Path
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


class ServingError(RuntimeError):
    """A pre-flight check or a docker call failed; ``up``/``down`` exit non-zero."""


SERVER_NAMES = ("qwen", "encoder")
DEFAULT_SERVER = "qwen"
ENCODER_PROFILE = "encoder"


@dataclass(frozen=True)
class Server:
    name: str  # label used in output and in the CLI
    service: str  # compose service name
    container: str  # container_name fixed in compose.yaml
    base_url: str
    memory_fraction: float


@dataclass(frozen=True)
class ContainerInfo:
    status: str  # docker State.Status: created, running, exited, dead, ...
    exit_code: int | None
    command: tuple[str, ...]  # Config.Cmd: the arguments the container was started with


# (command, environment) -> exit code. Tests inject a fake instead of running docker.
Runner = Callable[[Sequence[str], Mapping[str, str]], int]
# () -> (total_mib, free_mib) of the GPU compose will use. Tests inject a fake instead of nvidia-smi.
GpuMemory = Callable[[], tuple[int, int]]
ClientFactory = Callable[[str], VllmClient]
# container name -> its state, or None when there is no such container (or docker cannot say).
Inspect = Callable[[str], ContainerInfo | None]
# (container name, file) -> True when the container's logs were written to the file.
SaveLogs = Callable[[str, Path], bool]


def configured_servers(settings: Settings) -> list[Server]:
    servers = [Server("qwen", "vllm", "guardbench-vllm", settings.vllm_base_url, settings.vllm_gpu_memory_utilization)]
    if settings.vllm_encoder_checkpoint_dir is not None:
        servers.append(Server("encoder", "vllm-encoder", "guardbench-vllm-encoder",
                              f"http://127.0.0.1:{settings.vllm_encoder_port}", settings.vllm_encoder_gpu_memory_utilization))
    return servers


def select_server(settings: Settings, name: str) -> Server:
    """The configured server called ``name``. The encoder also needs an existing checkpoint directory."""
    if name not in SERVER_NAMES:
        raise ServingError(f"unknown server {name!r}; choose one of {', '.join(SERVER_NAMES)}")
    for server in configured_servers(settings):
        if server.name == name:
            check_encoder_checkpoint(settings, name)
            return server
    raise ServingError("encoder is not configured: set VLLM_ENCODER_CHECKPOINT_DIR to a checkpoint directory")


def check_encoder_checkpoint(settings: Settings, name: str) -> None:
    path = settings.vllm_encoder_checkpoint_dir
    if name == "encoder" and path is not None and not path.is_dir():
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
            VLLM_ENCODER_DTYPE=settings.vllm_encoder_dtype,
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


def inspect_container(name: str) -> ContainerInfo | None:
    try:
        proc = subprocess.run(["docker", "inspect", name], capture_output=True, text=True, check=False)
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout)[0]
        return ContainerInfo(status=data["State"]["Status"], exit_code=data["State"].get("ExitCode"),
                             command=tuple(data["Config"].get("Cmd") or ()))
    except (ValueError, IndexError, KeyError, TypeError):
        return None


def save_container_logs(name: str, path: Path) -> bool:
    """Write ``docker logs`` (stdout and stderr) of container ``name`` to ``path``."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as out:
            returncode = subprocess.run(["docker", "logs", name], stdout=out, stderr=subprocess.STDOUT,
                                        check=False).returncode
    except OSError:
        path.unlink(missing_ok=True)
        return False
    if returncode != 0:
        path.unlink(missing_ok=True)
    return returncode == 0


def dtype_of(command: Sequence[str]) -> str | None:
    """The ``--dtype`` value a vllm serve command line was started with."""
    for index, arg in enumerate(command):
        if arg.startswith("--dtype="):
            return arg.split("=", 1)[1]
        if arg == "--dtype" and index + 1 < len(command):
            return command[index + 1]
    return None


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


def status_all(settings: Settings, name: str | None = None, *, client_for: ClientFactory = VllmClient,
               inspect: Inspect = inspect_container) -> dict[str, Any]:
    """Status of server ``name``, or of every configured server (plus "encoder: not configured") when None."""
    if name is not None and name not in SERVER_NAMES:
        raise ServingError(f"unknown server {name!r}; choose one of {', '.join(SERVER_NAMES)}")
    report: dict[str, Any] = {}
    for server in configured_servers(settings):
        if name not in (None, server.name):
            continue
        client = client_for(server.base_url)
        try:
            report[server.name] = status(client)
        finally:
            client.close()
        info = inspect(server.container) if report[server.name]["reachable"] else None
        dtype = dtype_of(info.command) if info is not None and info.status == "running" else None
        if dtype:
            report[server.name]["dtype"] = dtype
    if name in (None, "encoder"):
        report.setdefault("encoder", "not configured")
    return report


def format_status(report: Mapping[str, Any]) -> str:
    return "\n".join(f"{name}: {value if isinstance(value, str) else json.dumps(value)}" for name, value in report.items())


def _exit_code(info: ContainerInfo | None) -> str:
    return "?" if info is None or info.exit_code is None else str(info.exit_code)


def _healthy(server: Server, client_for: ClientFactory) -> bool:
    client = client_for(server.base_url)
    try:
        return client.healthy()
    finally:
        client.close()


def _wait_until_up(server: Server, client_for: ClientFactory, inspect: Inspect, timeout: float,
                   interval: float) -> tuple[str, ContainerInfo | None]:
    """Poll until the server is healthy: returns ("healthy" | "exited" | "timeout", last container state).

    A container that has exited will never become healthy (compose sets no restart policy), so
    that ends the wait at once instead of after the timeout. A container that docker cannot
    inspect is treated as unknown: the wait continues until the timeout.
    """
    client = client_for(server.base_url)
    try:
        deadline = time.monotonic() + timeout
        while True:
            if client.healthy():
                return "healthy", None
            info = inspect(server.container)
            if info is not None and info.status in ("exited", "dead"):
                return "exited", info
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "timeout", info
            time.sleep(min(interval, remaining))
    finally:
        client.close()


def _save_logs(settings: Settings, server: Server, inspect: Inspect, save_logs: SaveLogs) -> Path | None:
    """Keep the container's logs before it is removed. None when there is no container or no logs."""
    if inspect(server.container) is None:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = settings.runs_dir / "vllm-logs" / f"{server.name}-{stamp}.log"
    return path if save_logs(server.container, path) else None


def up(settings: Settings, name: str = DEFAULT_SERVER, *, run: Runner = run_command,
       gpu_memory: GpuMemory = read_gpu_memory_mib, client_for: ClientFactory = VllmClient,
       inspect: Inspect = inspect_container, save_logs: SaveLogs = save_container_logs,
       base_env: Mapping[str, str] | None = None, timeout: float = 600.0, interval: float = 5.0) -> None:
    """Start server ``name`` (qwen or encoder) unless it is already healthy, and wait for it.

    Raises ServingError on any failure. Nothing is started if the encoder is not configured, its
    checkpoint directory is missing, or free GPU memory is insufficient. If the container fails to
    start, exits, or is not healthy within ``timeout``, its logs are saved under
    ``<runs_dir>/vllm-logs/`` and the container this call started is removed.
    """
    server = select_server(settings, name)
    if _healthy(server, client_for):
        print(f"{server.name}: already healthy, not restarted")
        return
    total, free = gpu_memory()
    check_gpu_memory([server], settings.vllm_gpu_memory_margin_mib, total, free)
    env = compose_env(settings, os.environ if base_env is None else base_env)
    print(f"{server.name}: starting service {server.service}")
    failure = ""
    if run(compose_command(settings, "up", "-d", server.service), env) != 0:
        failure = f"docker compose up failed for {server.service}"
    else:
        outcome, info = _wait_until_up(server, client_for, inspect, timeout, interval)
        if outcome == "exited":
            failure = f"{server.name} container exited with code {_exit_code(info)} before it was healthy"
        elif outcome == "timeout":
            failure = f"{server.name} not healthy within {timeout:g}s"
    if not failure:
        print(f"{server.name}: healthy")
        return
    log = _save_logs(settings, server, inspect, save_logs)  # before the container is removed
    removed = _remove(settings, server, run, env)
    where = f"logs: {log}" if log else "no container logs were available"
    cleanup = "removed the service started by this run" if removed else f"could not remove {server.service}, check `docker ps`"
    raise ServingError(f"{failure}; {where}; {cleanup}")


def _remove(settings: Settings, server: Server, run: Runner, env: Mapping[str, str]) -> bool:
    # A failed `up -d` can still leave a created container, so remove it in every failure case.
    return run(compose_command(settings, "rm", "--stop", "--force", server.service), env) == 0


def down(settings: Settings, *, run: Runner = run_command, base_env: Mapping[str, str] | None = None) -> None:
    env = compose_env(settings, os.environ if base_env is None else base_env)
    if run(compose_command(settings, "down"), env) != 0:
        raise ServingError("docker compose down failed")


def wait_server(settings: Settings, name: str, timeout: float, *, client_for: ClientFactory = VllmClient,
                inspect: Inspect = inspect_container, interval: float = 5.0) -> bool:
    server = select_server(settings, name)
    outcome, info = _wait_until_up(server, client_for, inspect, timeout, interval)
    detail = f" (exit code {_exit_code(info)})" if outcome == "exited" else ""
    print(f"{server.name}: {'healthy' if outcome == 'healthy' else outcome}{detail}")
    return outcome == "healthy"


def main() -> None:
    parser = argparse.ArgumentParser(prog="guardbench-vllm")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").add_argument("server", nargs="?", choices=SERVER_NAMES, default=None)
    sub.add_parser("down")
    for cmd in ("up", "wait"):
        command = sub.add_parser(cmd)
        command.add_argument("server", nargs="?", choices=SERVER_NAMES, default=DEFAULT_SERVER)
        command.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    try:
        settings = Settings.from_env()
        if args.cmd == "status":
            print(format_status(status_all(settings, args.server)))
        elif args.cmd == "wait":
            sys.exit(0 if wait_server(settings, args.server, args.timeout) else 1)
        elif args.cmd == "up":
            up(settings, args.server, timeout=args.timeout)
            print(format_status(status_all(settings, args.server)))
        else:
            down(settings)
            print("down")
    except (ServingError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
