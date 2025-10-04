import re, trafilatura
from fastapi import FastAPI
from pydantic import BaseModel

from onnx_genai_runner import ONNXGenAIRunner
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = FastAPI()

# model paths
MODEL_PATH = (
    r".\models\phi_models\models--microsoft--Phi-3.5-mini-instruct-onnx\snapshots\aded733f0b665ac2e21ffd8a008f82eb4278a134\cpu_and_mobile\cpu-int4-awq-block-128-acc-level-4"
)
runner = ONNXGenAIRunner(model_path=MODEL_PATH, execution_provider="cpu")

embeddings = HuggingFaceEmbeddings(
    model_name=r".\models\embedding_models\models--thenlper--gte-small\snapshots\17e1f347d17fe144873b1201da91788898c639cd"
)


def fetch_page_text(url: str) -> str:
    """Fetch and extract the main text content from a webpage using Trafilatura only."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return ""
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=True,
            favor_recall=True,
        )
        return text if text else ""
    except Exception as e:
        print("Error fetching page:", e)
        return ""


def build_vectorstore(text: str):
    """Split text into chunks and build FAISS index."""
    if not text:
        return None
    docs = [Document(page_content=text)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, embeddings)


def generate_answer(question: str, text: str) -> str:
    """Generate context-aware answer using ONNX model."""
    if not text or not question:
        return "No text or question provided."

    vectorstore = build_vectorstore(text)
    if not vectorstore:
        return "Failed to build vector store."

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "lambda_mult": 0.7}
    )
    retrieved_docs = retriever.invoke(question)
    context_text = "\n".join(doc.page_content for doc in retrieved_docs)

    prompt = (
        "You are an expert assistant. Use the following context to answer the question. "
        "If the context doesn't contain the answer, say \"I don't know\".\n\n"
        f"Context: {context_text}\n"
        f"Question: {question}\n"
        "Answer in less than 100 words:"
    )

    return runner.generate(prompt, max_length=2048, temperature=0.7)


# Request Model
class QueryBody(BaseModel):
    url: str
    question: str


# API Endpoint
@app.post("/query")
async def query(body: QueryBody):
    """Fetch URL text using Trafilatura and generate an answer."""
    print(f"Fetching content from URL: {body.url}")
    page_text = fetch_page_text(body.url)

    if not page_text:
        return {"error": "Failed to extract content from the webpage."}

    answer = generate_answer(body.question, page_text)
    return {"answer": answer}
