"""
answer.py
---------
RAG answer pipeline for code_assistant_rag.

Pipeline per query:
  1. Rewrite  – rephrase the user question to maximise KB recall
  2. Retrieve – dual retrieval (original + rewritten query), merge, rerank
  3. Answer   – build RAG prompt with history and call gpt-4.1
  4. Evaluate – LLM-as-judge scores accuracy / relevance / completeness (1-5)
               if overall score < EVAL_THRESHOLD → retry once with broader context

Public API (used by app.py):
    answer_question(question, history) -> (answer, chunks, eval_result)
"""

from pathlib import Path

from chromadb import PersistentClient
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential

load_dotenv(override=True)

# ── Models ─────────────────────────────────────────────────────────────────────
ANSWER_MODEL = "gpt-4.1"
RERANK_MODEL = "gpt-4.1-nano"
EVAL_MODEL   = "gpt-4.1-nano"
EMBEDDING_MODEL = "text-embedding-3-large"

# ── ChromaDB ───────────────────────────────────────────────────────────────────
DB_PATH = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "codebase"

# ── Retrieval config ───────────────────────────────────────────────────────────
RETRIEVAL_K = 20   # candidates fetched per query
FINAL_K     = 10   # top chunks passed to the LLM after reranking
RETRY_K     = 15   # extra chunks used on eval-triggered retry

# ── Evaluator config ───────────────────────────────────────────────────────────
EVAL_THRESHOLD = 3.5   # overall score (1-5); below this triggers a retry

# ── Retry config ───────────────────────────────────────────────────────────────
wait = wait_exponential(multiplier=1, min=10, max=240)

openai_client = OpenAI()
chroma = PersistentClient(path=DB_PATH)
collection = chroma.get_or_create_collection(COLLECTION_NAME)


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ══════════════════════════════════════════════════════════════════════════════

class RetrievedChunk(BaseModel):
    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    order: list[int] = Field(
        description=(
            "All chunk IDs re-ordered from most relevant to least relevant. "
            "Must include every provided ID exactly once."
        )
    )


class EvalResult(BaseModel):
    accuracy: float = Field(
        ge=1, le=5,
        description="How factually correct is the answer? (1=wrong, 5=perfect)"
    )
    relevance: float = Field(
        ge=1, le=5,
        description="How well does the answer address the question? (1=off-topic, 5=spot-on)"
    )
    completeness: float = Field(
        ge=1, le=5,
        description="Is the answer sufficiently complete? (1=barely, 5=fully)"
    )
    reasoning: str = Field(
        description="One or two sentences explaining the scores."
    )

    @property
    def overall(self) -> float:
        return round((self.accuracy + self.relevance + self.completeness) / 3, 2)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — QUERY REWRITER
# ══════════════════════════════════════════════════════════════════════════════

REWRITE_SYSTEM = """You are a code-search query optimizer.
You receive a developer's question and optionally a conversation history.
Your task: produce ONE short, precise search query (max 15 words) that will surface
the most relevant code chunks in a vector database.
Focus on: function names, class names, module names, or specific behaviour.
Reply ONLY with the rewritten query — no explanation, no punctuation at the end."""


@retry(wait=wait)
def rewrite_query(question: str, history: list[dict]) -> str:
    """Rewrite the user question for better KB retrieval."""
    prompt = (
        f"Conversation so far:\n{history}\n\n"
        f"Current question: {question}\n\n"
        "Rewritten search query:"
    )
    response = openai_client.chat.completions.create(
        model=RERANK_MODEL,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=40,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — RETRIEVER + RERANKER
# ══════════════════════════════════════════════════════════════════════════════

def _embed(text: str) -> list[float]:
    return openai_client.embeddings.create(
        model=EMBEDDING_MODEL, input=[text]
    ).data[0].embedding


def _fetch_chunks(query: str, k: int) -> list[RetrievedChunk]:
    """Vector-search ChromaDB and return k chunks."""
    vector = _embed(query)
    results = collection.query(query_embeddings=[vector], n_results=k)
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(RetrievedChunk(page_content=doc, metadata=meta))
    return chunks


def _merge(primary: list[RetrievedChunk], secondary: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Merge two chunk lists, deduplicating by content."""
    seen = {c.page_content for c in primary}
    merged = primary[:]
    for chunk in secondary:
        if chunk.page_content not in seen:
            merged.append(chunk)
            seen.add(chunk.page_content)
    return merged


@retry(wait=wait)
def rerank(question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Ask gpt-4.1-nano to re-order chunks by relevance to the question."""
    chunk_text = "\n\n".join(
        f"# CHUNK {i+1}\n{c.page_content}" for i, c in enumerate(chunks)
    )
    response = openai_client.beta.chat.completions.parse(
        model=RERANK_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a code-search re-ranker. "
                    "Order the provided chunks from most to least relevant to the question. "
                    "Return ALL chunk IDs (1-based). Most relevant first."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Chunks:\n{chunk_text}\n\n"
                    "Return the re-ranked order of ALL chunk IDs."
                ),
            },
        ],
        response_format=RankOrder,
    )
    order: RankOrder = response.choices[0].message.parsed
    # Guard against out-of-range or duplicate ids
    seen = set()
    reranked = []
    for idx in order.order:
        if 1 <= idx <= len(chunks) and idx not in seen:
            reranked.append(chunks[idx - 1])
            seen.add(idx)
    # Append any chunks not mentioned (safety net)
    for i, chunk in enumerate(chunks, start=1):
        if i not in seen:
            reranked.append(chunk)
    return reranked


def fetch_context(question: str, history: list[dict], k: int = FINAL_K) -> list[RetrievedChunk]:
    """
    Full retrieval pipeline:
      rewrite → dual-fetch → merge → rerank → top-k
    """
    rewritten = rewrite_query(question, history)
    chunks_original  = _fetch_chunks(question, RETRIEVAL_K)
    chunks_rewritten = _fetch_chunks(rewritten, RETRIEVAL_K)
    merged  = _merge(chunks_original, chunks_rewritten)
    reranked = rerank(question, merged)
    return reranked[:k], rewritten


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — RESPONDER
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert AI assistant for software developers.
You have been given access to a codebase via a retrieval system.
Your answers should be accurate, concise, and grounded in the retrieved code.

Guidelines:
- Reference specific files and line-level details when relevant.
- If a question cannot be answered from the context, say so clearly.
- Use markdown with code blocks where helpful.
- Never invent code that is not present in the context.

Relevant context from the codebase:
{context}
"""


def _build_messages(
    question: str,
    history: list[dict],
    chunks: list[RetrievedChunk],
) -> list[dict]:
    context = "\n\n---\n\n".join(
        f"**File:** `{c.metadata.get('source', 'unknown')}`\n\n{c.page_content}"
        for c in chunks
    )
    system = SYSTEM_PROMPT.format(context=context)
    return (
        [{"role": "system", "content": system}]
        + history
        + [{"role": "user", "content": question}]
    )


@retry(wait=wait)
def _call_llm(messages: list[dict]) -> str:
    response = openai_client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=messages,
    )
    return response.choices[0].message.content


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — EVALUATOR (LLM-as-judge)
# ══════════════════════════════════════════════════════════════════════════════

EVAL_SYSTEM = """You are a strict but fair evaluator of AI-generated answers about code.
Score each dimension honestly from 1 (very poor) to 5 (excellent).

Dimensions:
  accuracy     – Is every factual claim in the answer correct given the context?
  relevance    – Does the answer directly address what was asked?
  completeness – Does the answer cover all important aspects of the question?

Be concise in your reasoning (1-2 sentences)."""


@retry(wait=wait)
def evaluate_answer(
    question: str,
    rewritten_question: str,
    answer: str,
    chunks: list[RetrievedChunk],
) -> EvalResult:
    """Score an answer on accuracy, relevance, and completeness (1-5 each)."""
    context_summary = "\n".join(
        f"- {c.metadata.get('source', '?')}: {c.page_content[:200]}..."
        for c in chunks[:5]
    )
    prompt = (
        f"Question asked by the developer:\n{question}\n\n"
        f"Interpreted as: {rewritten_question}\n\n"
        f"Top retrieved context (abbreviated):\n{context_summary}\n\n"
        f"Answer to evaluate:\n{answer}"
    )
    response = openai_client.beta.chat.completions.parse(
        model=EVAL_MODEL,
        messages=[
            {"role": "system", "content": EVAL_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        response_format=EvalResult,
    )
    return response.choices[0].message.parsed


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def answer_question(
    question: str,
    history: list[dict] | None = None,
) -> tuple[str, list[RetrievedChunk], EvalResult]:
    """
    Full RAG pipeline: retrieve → answer → evaluate → (retry if poor).

    Args:
        question: the developer's question
        history:  list of {"role": "user"|"assistant", "content": str}

    Returns:
        (answer_text, retrieved_chunks, eval_result)
    """
    if history is None:
        history = []

    # ── First attempt ─────────────────────────────────────────────────────────
    chunks, rewritten_question  = fetch_context(question, history, k=FINAL_K)
    messages = _build_messages(question, history, chunks)
    answer  = _call_llm(messages)
    result  = evaluate_answer(question, rewritten_question, answer, chunks)

    # ── Retry if quality is below threshold ───────────────────────────────────
    if result.overall < EVAL_THRESHOLD:
        chunks, rewritten_question   = fetch_context(question, history, k=RETRY_K)
        messages = _build_messages(question, history, chunks)
        answer   = _call_llm(messages)
        result   = evaluate_answer(question, rewritten_question, answer, chunks)

    return answer, chunks, result