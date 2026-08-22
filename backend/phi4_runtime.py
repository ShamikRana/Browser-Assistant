"""ONNX Runtime GenAI wrapper for Phi-4 mini, supporting CPU and GPU."""

import logging
import os
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

_CUDA_DLL_DIRECTORIES: set[str] = set()
_CUDA_DLL_HANDLES: list = []


def _configure_cuda_dll_search_path() -> None:
    """Make pip-installed CUDA DLLs discoverable on Windows."""
    if sys.platform != "win32":
        return

    nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not nvidia_root.is_dir():
        return

    for dll_path in nvidia_root.rglob("*.dll"):
        directory = str(dll_path.parent)
        if directory not in _CUDA_DLL_DIRECTORIES:
            _CUDA_DLL_DIRECTORIES.add(directory)
            _CUDA_DLL_HANDLES.append(os.add_dll_directory(directory))

    os.environ["PATH"] = os.pathsep.join(
        [*_CUDA_DLL_DIRECTORIES, os.environ.get("PATH", "")]
    )


_configure_cuda_dll_search_path()
import onnxruntime_genai as og  # noqa: E402  (must follow the DLL setup above)

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise assistant that answers questions about a web page. "
    "Use only the provided context. If the context does not contain the answer, "
    "say you could not find it on the page."
)


class Phi4Runner:
    """Loads a Phi-4 mini ONNX model and generates text on CPU or GPU.

    ONNX Runtime GenAI is not thread-safe, so generation is serialised behind a
    lock; FastAPI stays responsive because generation runs in a worker thread.
    """

    def __init__(self, model_dir: str | Path, device: str = "cpu"):
        self.device = "gpu" if device.lower() in ("gpu", "cuda") else "cpu"
        self.model_dir = str(model_dir)
        self._lock = threading.Lock()

        self.model = self._load_model()
        self.tokenizer = og.Tokenizer(self.model)
        self.tokenizer_stream = self.tokenizer.create_stream()

    def _build_config(self, device: str):
        config = og.Config(self.model_dir)
        config.clear_providers()
        if device == "gpu":
            config.append_provider("cuda")
        # The CPU execution provider is the default and needs no registration.
        return config

    def _load_model(self):
        try:
            model = og.Model(self._build_config(self.device))
            logger.info("Loaded Phi-4 on %s from %s", self.device, self.model_dir)
            return model
        except Exception as exc:
            if self.device != "gpu":
                raise
            logger.warning("GPU load failed (%s); falling back to CPU.", exc)
            self.device = "cpu"
            return og.Model(self._build_config("cpu"))

    @staticmethod
    def _build_prompt(message: str, system_prompt: str) -> str:
        return (
            f"<|system|>\n{system_prompt}<|end|>\n"
            f"<|user|>\n{message}<|end|>\n"
            f"<|assistant|>\n"
        )

    def _make_generator(
        self,
        message: str,
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ):
        input_ids = self.tokenizer.encode(self._build_prompt(message, system_prompt))

        params = og.GeneratorParams(self.model)
        search_options = {"max_length": len(input_ids) + max_new_tokens}
        if temperature <= 0:
            # Greedy decoding: faster and deterministic, which suits grounded QA.
            search_options["do_sample"] = False
        else:
            search_options.update(
                do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k
            )
        params.set_search_options(**search_options)

        generator = og.Generator(self.model, params)
        generator.append_tokens(input_ids)
        return generator

    def stream(
        self,
        message: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_new_tokens: int = 320,
        temperature: float = 0.0,
        top_p: float = 0.9,
        top_k: int = 40,
    ) -> Iterator[str]:
        """Yield decoded text pieces as they are generated."""
        with self._lock:
            generator = self._make_generator(
                message, system_prompt, max_new_tokens, temperature, top_p, top_k
            )
            while not generator.is_done():
                generator.generate_next_token()
                token_ids = generator.get_next_tokens()
                if token_ids:
                    yield self.tokenizer_stream.decode(token_ids[0])

    def generate(
        self,
        message: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_new_tokens: int = 320,
        temperature: float = 0.0,
        top_p: float = 0.9,
        top_k: int = 40,
    ) -> str:
        """Generate a complete response."""
        pieces = list(
            self.stream(message, system_prompt, max_new_tokens, temperature, top_p, top_k)
        )
        return "".join(pieces).strip()

    def warm_up(self) -> None:
        """Run a tiny generation so the first real request isn't slowed by init."""
        try:
            self.generate("Reply with OK.", max_new_tokens=4)
            logger.info("Phi-4 warm-up complete.")
        except Exception as exc:
            logger.warning("Phi-4 warm-up failed: %s", exc)
