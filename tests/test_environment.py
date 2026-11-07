import pytest
import torch

from guardbench_classifier.envcheck import collect


def test_pinned_torch() -> None:
    assert torch.__version__.startswith("2.9.1")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device in this environment")
def test_cuda_visible() -> None:
    info = collect()
    assert info["gpu"]["cuda_available"] is True
    assert info["gpu"]["total_memory_mib"] > 0
