import json

import httpx
import pytest

from guardbench_classifier.config import Settings
from guardbench_classifier.serving import VllmClient, status


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
