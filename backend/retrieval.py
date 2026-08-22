"""Hybrid retrieval over a single web page.

Two retrievers run in parallel and their rankings are merged with Reciprocal
Rank Fusion:

* **Dense** (gte-small embeddings) - captures paraphrases and meaning.
* **BM25** (lexical) - reliably finds names, numbers and rare terms that
  embeddings tend to smooth away.

For a page-sized corpus a plain NumPy cosine similarity is faster than building
a FAISS index, so no external vector store is needed. Built indexes are cached
by content hash, which makes follow-up questions on the same page near-instant.
"""

import hashlib
import logging
import re
import threading
from collections import OrderedDict, defaultdict

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from chunking import Chunk, chunk_text
from config import EMBEDDING_MODEL_PATH, INDEX_CACHE_SIZE, TOP_K

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Questions that need whole-document coverage rather than the closest chunks.
_BROAD_QUERY_RE = re.compile(
    r"\b(summar\w*|tl;?dr|overview|key (points|takeaways)|main (points|ideas|topics)"
    r"|what is this (page|article|about)|gist|conclusion|outline)\b",
    re.IGNORECASE,
)

_embedder: SentenceTransformer | None = None
_embedder_lock = threading.Lock()


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def load_embedder(device: str = "cpu") -> SentenceTransformer:
    """Load the sentence-transformer once, on the requested device."""
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            logger.info("Loading embedding model on %s", device)
            _embedder = SentenceTransformer(str(EMBEDDING_MODEL_PATH), device=device)
    return _embedder


def is_broad_query(question: str) -> bool:
    """True for summary-style questions that shouldn't use top-k similarity."""
    return bool(_BROAD_QUERY_RE.search(question))


class PageIndex:
    """Dense + lexical index over the chunks of one page."""

    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray):
        self.chunks = chunks
        self.embeddings = embeddings
        self.bm25 = BM25Okapi([_tokenize(c.with_heading) for c in chunks])

    def __len__(self) -> int:
        return len(self.chunks)

    def _dense_ranking(self, question: str, limit: int) -> list[int]:
        query_vector = load_embedder().encode(
            question, normalize_embeddings=True, convert_to_numpy=True
        )
        scores = self.embeddings @ query_vector
        return np.argsort(scores)[::-1][:limit].tolist()

    def _lexical_ranking(self, question: str, limit: int) -> list[int]:
        scores = self.bm25.get_scores(_tokenize(question))
        return np.argsort(scores)[::-1][:limit].tolist()

    def search(self, question: str, k: int = TOP_K) -> list[Chunk]:
        """Return the best chunks, restored to original reading order."""
        if len(self.chunks) <= k:
            return list(self.chunks)

        if is_broad_query(question):
            # Evenly sample the whole page so summaries aren't biased to one section.
            step = len(self.chunks) / k
            picked = sorted({int(i * step) for i in range(k)})
            return [self.chunks[i] for i in picked]

        pool = max(k * 3, 10)
        rankings = [self._dense_ranking(question, pool), self._lexical_ranking(question, pool)]
        fused = _reciprocal_rank_fusion(rankings)[:k]
        return [self.chunks[i] for i in sorted(fused)]


def _reciprocal_rank_fusion(rankings: list[list[int]], damping: int = 60) -> list[int]:
    """Merge rankings so a chunk ranked well by either retriever surfaces."""
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, index in enumerate(ranking):
            scores[index] += 1.0 / (damping + rank + 1)
    return sorted(scores, key=lambda index: scores[index], reverse=True)


class IndexCache:
    """Small LRU cache of page indexes keyed by content hash."""

    def __init__(self, maxsize: int = INDEX_CACHE_SIZE):
        self._store: OrderedDict[str, PageIndex] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()

    def get_or_build(self, text: str) -> PageIndex | None:
        cache_key = self.key(text)
        with self._lock:
            cached = self._store.get(cache_key)
            if cached is not None:
                self._store.move_to_end(cache_key)
                return cached

        index = _build_index(text)
        if index is None:
            return None

        with self._lock:
            self._store[cache_key] = index
            self._store.move_to_end(cache_key)
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)
        return index


def _build_index(text: str) -> PageIndex | None:
    chunks = chunk_text(text)
    if not chunks:
        return None
    embeddings = load_embedder().encode(
        [chunk.with_heading for chunk in chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=64,
    )
    logger.info("Indexed page into %d chunks", len(chunks))
    return PageIndex(chunks, embeddings)


index_cache = IndexCache()
