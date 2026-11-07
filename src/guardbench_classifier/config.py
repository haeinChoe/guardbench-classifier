"""Single entry point for environment-based settings.

Experiment code reads environment variables only through this module.
Secrets (HF_TOKEN) are never included in ``public_snapshot``.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


class MissingSettingError(RuntimeError):
    """Raised when a setting required by the current command is not set."""


def find_repo_root(start: Path | None = None) -> Path:
    """Return the nearest ancestor of ``start`` (default: cwd) containing pyproject.toml."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return current


def _flag(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _path(value: str | None, default: str, root: Path) -> Path:
    raw = (value or "").strip() or default
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    data_dir: Path
    runs_dir: Path
    hf_home: str | None
    hf_hub_offline: bool
    hf_hub_disable_telemetry: bool
    cuda_visible_devices: str | None
    vllm_base_url: str
    _hf_token: str | None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, root: Path | None = None) -> Settings:
        env = os.environ if env is None else env
        root = root or find_repo_root()
        return cls(
            repo_root=root,
            data_dir=_path(env.get("GUARDBENCH_DATA_DIR"), "data", root),
            runs_dir=_path(env.get("GUARDBENCH_RUNS_DIR"), "runs", root),
            hf_home=(env.get("HF_HOME") or "").strip() or None,
            hf_hub_offline=_flag(env.get("HF_HUB_OFFLINE")),
            hf_hub_disable_telemetry=_flag(env.get("HF_HUB_DISABLE_TELEMETRY")),
            cuda_visible_devices=(env.get("CUDA_VISIBLE_DEVICES") or "").strip() or None,
            vllm_base_url=(env.get("VLLM_BASE_URL") or "").strip() or f"http://127.0.0.1:{(env.get('VLLM_PORT') or '8000').strip()}",
            _hf_token=(env.get("HF_TOKEN") or "").strip() or None,
        )

    def require_hf_token(self) -> str:
        """Return HF_TOKEN or fail immediately; call only where gated access is needed."""
        if not self._hf_token:
            raise MissingSettingError("HF_TOKEN is required for gated Hugging Face resources; set it in .env")
        return self._hf_token

    def public_snapshot(self) -> dict[str, object]:
        """Non-secret settings for run manifests. The token is reported only as present/absent."""
        return {
            "data_dir": str(self.data_dir),
            "runs_dir": str(self.runs_dir),
            "hf_home": self.hf_home,
            "hf_hub_offline": self.hf_hub_offline,
            "hf_hub_disable_telemetry": self.hf_hub_disable_telemetry,
            "cuda_visible_devices": self.cuda_visible_devices,
            "vllm_base_url": self.vllm_base_url,
            "hf_token_present": self._hf_token is not None,
        }

    def __repr__(self) -> str:  # keep the token out of logs and tracebacks
        return f"Settings({self.public_snapshot()!r})"
