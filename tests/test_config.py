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
