"""Minimal client for the local vLLM OpenAI-compatible server (see compose.yaml).

  guardbench-vllm status       # server version and served models, or "unreachable"
  guardbench-vllm wait [--timeout 600]   # block until /health is OK (exit 1 on timeout)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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
    while time.monotonic() < deadline:
        if client.healthy():
            return True
        time.sleep(interval)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(prog="guardbench-vllm")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    w = sub.add_parser("wait")
    w.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    client = VllmClient(Settings.from_env().vllm_base_url)
    try:
        if args.cmd == "status":
            print(json.dumps(status(client), indent=2))
        else:
            ok = wait(client, args.timeout)
            print("healthy" if ok else "timeout")
            sys.exit(0 if ok else 1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
