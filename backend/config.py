"""Central configuration for the Browser Assistant backend."""

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
MODELS_DIR = BACKEND_DIR / "models"

# Phi-4 mini ONNX variants published by Microsoft.
CPU_SUBFOLDER = "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
GPU_SUBFOLDER = "gpu/gpu-int4-rtn-block-32"

PHI4_DIR = MODELS_DIR / "phi4"
EMBEDDING_DIR = MODELS_DIR / "embedding"


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def model_path_for(device: str) -> Path:
    """Return the Phi-4 ONNX folder for a resolved device."""
    subfolder = GPU_SUBFOLDER if device == "gpu" else CPU_SUBFOLDER
    return PHI4_DIR / subfolder


def _is_downloaded(device: str) -> bool:
    return (model_path_for(device) / "genai_config.json").is_file()


def resolve_device(requested: str | None = None) -> str:
    """Normalise the requested device to 'cpu' or 'gpu'.

    ``auto`` (the default) prefers the GPU build when its model files are
    present, and silently falls back to CPU otherwise.
    """
    value = (requested or _env("DEVICE", "auto")).lower()
    if value in ("cuda", "gpu"):
        return "gpu"
    if value == "cpu":
        return "cpu"
    return "gpu" if _is_downloaded("gpu") else "cpu"


DEVICE = resolve_device()
MODEL_PATH = Path(_env("MODEL_PATH", str(model_path_for(DEVICE))))
EMBEDDING_MODEL_PATH = Path(_env("EMBEDDING_MODEL_PATH", str(EMBEDDING_DIR)))

HOST = _env("HOST", "127.0.0.1")
PORT = _env_int("PORT", 5000)

# Retrieval / generation budgets. Lower values mean faster answers.
CHUNK_SIZE = _env_int("CHUNK_SIZE", 1000)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 100)
TOP_K = _env_int("TOP_K", 7)
MAX_CONTEXT_CHARS = _env_int("MAX_CONTEXT_CHARS", 20000)
MAX_NEW_TOKENS = _env_int("MAX_NEW_TOKENS", 256)

# Pages shorter than this skip retrieval entirely and are passed in full.
FULL_CONTEXT_THRESHOLD = _env_int("FULL_CONTEXT_THRESHOLD", 10000)

# Number of page indexes kept in memory (keyed by content hash).
INDEX_CACHE_SIZE = _env_int("INDEX_CACHE_SIZE", 8)

REQUEST_TIMEOUT = _env_int("REQUEST_TIMEOUT", 20)
USER_AGENT = _env(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
)
