"""FastAPI application for the Browser Assistant backend."""

import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import config
from pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("browser-assistant")

pipeline: Pipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Loading models (device=%s)...", config.DEVICE)
    started = time.perf_counter()
    pipeline = Pipeline(config.DEVICE)
    logger.info(
        "Ready in %.1fs (phi4=%s, embeddings=%s)",
        time.perf_counter() - started,
        pipeline.device,
        pipeline.embedding_device,
    )
    pipeline.warm_up()
    yield
    pipeline = None


app = FastAPI(title="Browser Assistant (Phi-4)", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(chrome-extension|moz-extension)://.*$|^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class QueryBody(BaseModel):
    url: str = ""
    question: str
    # Rendered DOM from the extension. Preferred over a server-side fetch because
    # it captures JavaScript-rendered, logged-in and paywalled pages.
    html: str | None = Field(default=None, repr=False)
    text: str | None = Field(default=None, repr=False)


@app.get("/health")
async def health():
    ready = pipeline is not None
    return {
        "status": "healthy" if ready else "loading",
        "model_loaded": ready,
        "device": pipeline.device if ready else config.DEVICE,
        "embedding_device": pipeline.embedding_device if ready else None,
        "version": app.version,
    }


@app.post("/query")
async def query(body: QueryBody):
    """Answer a question about a page and return the full response at once."""
    if pipeline is None:
        return {"error": "Model is still loading, please retry shortly."}

    started = time.perf_counter()
    result = pipeline.answer(body.url, body.question, body.html, body.text)
    result["elapsed"] = round(time.perf_counter() - started, 2)
    return result


@app.post("/query/stream")
async def query_stream(body: QueryBody):
    """Answer a question, streaming tokens as server-sent events."""

    def event_source():
        if pipeline is None:
            yield _sse("error", {"message": "Model is still loading, please retry shortly."})
            return
        for event, payload in pipeline.answer_stream(
            body.url, body.question, body.html, body.text
        ):
            yield _sse(event, payload)

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: str, payload) -> str:
    # JSON-encode every payload so newlines inside tokens survive the SSE framing.
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"
