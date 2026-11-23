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


def _ratio(env: Mapping[str, str], name: str, default: float) -> float:
    raw = (env.get(name) or "").strip()
    try:
        value = float(raw) if raw else default
    except ValueError:
        raise ValueError(f"{name} must be a number in (0, 1], got {raw!r}") from None
    if not 0 < value <= 1:
        raise ValueError(f"{name} must be in (0, 1], got {value}")
    return value


def _port(env: Mapping[str, str], name: str, default: int) -> int:
    raw = (env.get(name) or "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        raise ValueError(f"{name} must be an integer port, got {raw!r}") from None
    if not 0 < value < 65536:
        raise ValueError(f"{name} must be in 1..65535, got {value}")
    return value


def _mib(env: Mapping[str, str], name: str, default: int) -> int:
    raw = (env.get(name) or "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        raise ValueError(f"{name} must be a non-negative integer (MiB), got {raw!r}") from None
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = (env.get(name) or "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from None
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


ENCODER_DTYPES = ("float32", "bfloat16", "float16")


def _choice(env: Mapping[str, str], name: str, default: str, choices: tuple[str, ...]) -> str:
    value = (env.get(name) or "").strip() or default
    if value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}, got {value!r}")
    return value


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
    vllm_model: str
    vllm_model_revision: str
    vllm_max_model_len: int
    vllm_port: int
    vllm_gpu_memory_utilization: float
    vllm_gpu_memory_margin_mib: int
    vllm_encoder_checkpoint_dir: Path | None  # None = encoder not configured
    vllm_encoder_port: int
    vllm_encoder_gpu_memory_utilization: float
    vllm_encoder_dtype: str
    _hf_token: str | None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, root: Path | None = None) -> Settings:
        env = os.environ if env is None else env
        root = root or find_repo_root()
        vllm_port = _port(env, "VLLM_PORT", 8000)
        encoder_dir = (env.get("VLLM_ENCODER_CHECKPOINT_DIR") or "").strip()
        return cls(
            repo_root=root,
            data_dir=_path(env.get("GUARDBENCH_DATA_DIR"), "data", root),
            runs_dir=_path(env.get("GUARDBENCH_RUNS_DIR"), "runs", root),
            hf_home=(env.get("HF_HOME") or "").strip() or None,
            hf_hub_offline=_flag(env.get("HF_HUB_OFFLINE")),
            hf_hub_disable_telemetry=_flag(env.get("HF_HUB_DISABLE_TELEMETRY")),
            cuda_visible_devices=(env.get("CUDA_VISIBLE_DEVICES") or "").strip() or None,
            vllm_base_url=(env.get("VLLM_BASE_URL") or "").strip() or f"http://127.0.0.1:{vllm_port}",
            vllm_model=(env.get("VLLM_MODEL") or "").strip() or "Qwen/Qwen3-4B",
            vllm_model_revision=(env.get("VLLM_MODEL_REVISION") or "").strip() or "1cfa9a7208912126459214e8b04321603b3df60c",
            vllm_max_model_len=_positive_int(env, "VLLM_MAX_MODEL_LEN", 8192),
            vllm_port=vllm_port,
            vllm_gpu_memory_utilization=_ratio(env, "VLLM_GPU_MEMORY_UTILIZATION", 0.65),
            vllm_gpu_memory_margin_mib=_mib(env, "VLLM_GPU_MEMORY_MARGIN_MIB", 1024),
            vllm_encoder_checkpoint_dir=_path(encoder_dir, "", root) if encoder_dir else None,
            vllm_encoder_port=_port(env, "VLLM_ENCODER_PORT", 8001),
            vllm_encoder_gpu_memory_utilization=_ratio(env, "VLLM_ENCODER_GPU_MEMORY_UTILIZATION", 0.10),
            vllm_encoder_dtype=_choice(env, "VLLM_ENCODER_DTYPE", "float32", ENCODER_DTYPES),
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
            "vllm_model": self.vllm_model,
            "vllm_model_revision": self.vllm_model_revision,
            "vllm_max_model_len": self.vllm_max_model_len,
            "vllm_gpu_memory_utilization": self.vllm_gpu_memory_utilization,
            "vllm_encoder_checkpoint_dir": str(self.vllm_encoder_checkpoint_dir) if self.vllm_encoder_checkpoint_dir else None,
            "vllm_encoder_gpu_memory_utilization": self.vllm_encoder_gpu_memory_utilization,
            "vllm_encoder_dtype": self.vllm_encoder_dtype,
            "hf_token_present": self._hf_token is not None,
        }

    def __repr__(self) -> str:  # keep the token out of logs and tracebacks
        return f"Settings({self.public_snapshot()!r})"
