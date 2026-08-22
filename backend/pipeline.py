"""Query pipeline: extract page text, retrieve context, generate an answer."""

import logging
from collections.abc import Iterator

import config
from extraction import get_page_content
from phi4_runtime import Phi4Runner
from retrieval import index_cache, load_embedder

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """Answer the question using only the page context below.

Page title: {title}

Context:
{context}

Question: {question}

Answer concisely (under 120 words). If the context does not contain the answer, say so."""


class Pipeline:
    """Holds the loaded models and answers questions about a page."""

    def __init__(self, device: str | None = None):
        self.device = config.resolve_device(device)
        # config.MODEL_PATH honours an explicit MODEL_PATH override for the
        # resolved device; any other device falls back to its standard folder.
        self.model_path = (
            config.MODEL_PATH
            if self.device == config.DEVICE
            else config.model_path_for(self.device)
        )
        self.runner = Phi4Runner(self.model_path, self.device)
        # The runner may have fallen back to CPU if the GPU build failed to load.
        self.device = self.runner.device
        self.embedding_device = self._resolve_embedding_device()
        load_embedder(self.embedding_device)

    def _resolve_embedding_device(self) -> str:
        if self.device != "gpu":
            return "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            logger.info("CUDA unavailable in PyTorch; embeddings will use CPU.")
        except Exception as exc:
            logger.info("Could not query PyTorch CUDA support (%s); using CPU.", exc)
        return "cpu"

    def warm_up(self) -> None:
        self.runner.warm_up()

    def build_context(self, question: str, text: str) -> tuple[str, int]:
        """Return ``(context, chunks_used)`` for a question."""
        # Short pages fit whole: skip retrieval for better answers and less latency.
        if len(text) <= config.FULL_CONTEXT_THRESHOLD:
            return text, 1

        index = index_cache.get_or_build(text)
        if index is None:
            return text[: config.MAX_CONTEXT_CHARS], 1

        chunks = index.search(question, k=config.TOP_K)
        context = ""
        for chunk in chunks:
            candidate = f"{context}\n\n{chunk.with_heading}" if context else chunk.with_heading
            if len(candidate) > config.MAX_CONTEXT_CHARS:
                break
            context = candidate
        return context, len(chunks)

    def prepare(self, url: str, question: str, html: str | None, text: str | None):
        """Resolve page content and build the prompt."""
        title, page_text = get_page_content(url, html=html, text=text)
        if not page_text:
            return None

        context, used = self.build_context(question, page_text)
        prompt = _PROMPT_TEMPLATE.format(
            title=title or url, context=context, question=question
        )
        return prompt, title, used

    def answer(self, url: str, question: str, html: str | None = None, text: str | None = None) -> dict:
        prepared = self.prepare(url, question, html, text)
        if prepared is None:
            return {"error": "Could not extract readable content from this page."}

        prompt, title, used = prepared
        answer = self.runner.generate(prompt, max_new_tokens=config.MAX_NEW_TOKENS)
        return {"answer": answer, "title": title, "chunks_used": used}

    def answer_stream(
        self, url: str, question: str, html: str | None = None, text: str | None = None
    ) -> Iterator[tuple[str, dict | str]]:
        """Yield ``(event, payload)`` pairs for server-sent events."""
        prepared = self.prepare(url, question, html, text)
        if prepared is None:
            yield "error", {"message": "Could not extract readable content from this page."}
            return

        prompt, title, used = prepared
        yield "meta", {"title": title, "chunks_used": used, "device": self.device}
        try:
            for piece in self.runner.stream(prompt, max_new_tokens=config.MAX_NEW_TOKENS):
                yield "token", piece
        except Exception as exc:
            logger.exception("Generation failed")
            yield "error", {"message": str(exc)}
            return
        yield "done", {}
