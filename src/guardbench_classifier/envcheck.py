"""Print the local experiment environment: versions, GPU, and non-secret settings."""
from __future__ import annotations

import json
import platform
from importlib import metadata

from guardbench_classifier.config import Settings
from guardbench_classifier.serving import VllmClient, status


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def collect() -> dict[str, object]:
    import torch

    settings = Settings.from_env()
    gpu: dict[str, object] = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu.update(name=props.name, total_memory_mib=props.total_memory // 2**20, device_count=torch.cuda.device_count())
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": _version("transformers"),
        "huggingface_hub": _version("huggingface_hub"),
        "gpu": gpu,
        "vllm_server": _server(settings),
        "settings": settings.public_snapshot(),
    }


def _server(settings: Settings) -> dict[str, object]:
    client = VllmClient(settings.vllm_base_url, timeout=10.0)
    try:
        return status(client)
    finally:
        client.close()


def main() -> None:
    print(json.dumps(collect(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
