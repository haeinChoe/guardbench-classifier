from pathlib import Path

import pytest

from guardbench_classifier.config import MissingSettingError, Settings


def test_relative_paths_resolve_from_repo_root(tmp_path: Path) -> None:
    settings = Settings.from_env({"GUARDBENCH_DATA_DIR": "d", "GUARDBENCH_RUNS_DIR": "r"}, root=tmp_path)
    assert settings.data_dir == tmp_path / "d"
    assert settings.runs_dir == tmp_path / "r"


def test_defaults_when_unset(tmp_path: Path) -> None:
    settings = Settings.from_env({}, root=tmp_path)
    assert settings.data_dir == tmp_path / "data"
    assert settings.runs_dir == tmp_path / "runs"
    assert settings.hf_hub_offline is False


def test_absolute_path_kept(tmp_path: Path) -> None:
    settings = Settings.from_env({"GUARDBENCH_DATA_DIR": str(tmp_path / "abs")}, root=Path("/nonexistent"))
    assert settings.data_dir == tmp_path / "abs"


def test_token_never_exposed(tmp_path: Path) -> None:
    secret = "hf_should_not_leak"
    settings = Settings.from_env({"HF_TOKEN": secret}, root=tmp_path)
    assert secret not in repr(settings)
    assert secret not in str(settings.public_snapshot())
    assert settings.public_snapshot()["hf_token_present"] is True
    assert settings.require_hf_token() == secret


def test_missing_token_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(MissingSettingError):
        Settings.from_env({}, root=tmp_path).require_hf_token()


def test_vllm_defaults_keep_qwen_values(tmp_path: Path) -> None:
    settings = Settings.from_env({}, root=tmp_path)
    assert settings.vllm_model == "Qwen/Qwen3-4B"
    assert settings.vllm_model_revision == "1cfa9a7208912126459214e8b04321603b3df60c"
    assert settings.vllm_max_model_len == 8192
    assert settings.vllm_port == 8000
    assert settings.vllm_gpu_memory_utilization == 0.65
    assert settings.vllm_encoder_checkpoint_dir is None
    assert settings.vllm_encoder_dtype == "float32"


def test_encoder_settings_from_env(tmp_path: Path) -> None:
    settings = Settings.from_env({"VLLM_ENCODER_CHECKPOINT_DIR": "ckpt/enc", "VLLM_ENCODER_PORT": "9002",
                                  "VLLM_ENCODER_GPU_MEMORY_UTILIZATION": "0.2",
                                  "VLLM_GPU_MEMORY_MARGIN_MIB": "512"}, root=tmp_path)
    assert settings.vllm_encoder_checkpoint_dir == tmp_path / "ckpt" / "enc"
    assert (settings.vllm_encoder_port, settings.vllm_encoder_gpu_memory_utilization) == (9002, 0.2)
    assert settings.vllm_gpu_memory_margin_mib == 512


@pytest.mark.parametrize("name,value", [("VLLM_GPU_MEMORY_UTILIZATION", "1.5"), ("VLLM_GPU_MEMORY_UTILIZATION", "abc"),
                                        ("VLLM_ENCODER_GPU_MEMORY_UTILIZATION", "0"), ("VLLM_PORT", "x"), ("VLLM_MAX_MODEL_LEN", "0"),
                                        ("VLLM_ENCODER_PORT", "70000"), ("VLLM_GPU_MEMORY_MARGIN_MIB", "-1")])
def test_invalid_vllm_numbers_fail_fast(tmp_path: Path, name: str, value: str) -> None:
    with pytest.raises(ValueError, match=name):
        Settings.from_env({name: value}, root=tmp_path)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
def test_encoder_dtype_accepts_supported_values(tmp_path: Path, dtype: str) -> None:
    settings = Settings.from_env({"VLLM_ENCODER_DTYPE": dtype}, root=tmp_path)
    assert settings.vllm_encoder_dtype == dtype
    assert settings.public_snapshot()["vllm_encoder_dtype"] == dtype


@pytest.mark.parametrize("value", ["fp32", "auto", "half", "FLOAT32"])
def test_encoder_dtype_rejects_other_values(tmp_path: Path, value: str) -> None:
    with pytest.raises(ValueError, match="VLLM_ENCODER_DTYPE"):
        Settings.from_env({"VLLM_ENCODER_DTYPE": value}, root=tmp_path)
