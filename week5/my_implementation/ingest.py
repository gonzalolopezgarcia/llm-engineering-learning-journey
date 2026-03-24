"""
ingest.py
---------
Ingestion pipeline for code_assistant_rag.

Steps:
  1. Scan  – recursively walk a repo, filter files by extension and ignored folders
  2. Chunk – LLM-based chunking (headline + summary + original code) via gpt-4.1-nano
  3. Embed – generate embeddings with text-embedding-3-large and persist to ChromaDB

Usage:
    python ingest.py --repo /path/to/your/project
    python ingest.py --repo /path/to/your/project --reset   # wipe DB first
"""

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

from chromadb import PersistentClient
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential
from tqdm import tqdm

load_dotenv(override=True)

# ── Models ─────────────────────────────────────────────────────────────────────
CHUNK_MODEL = "gpt-4.1-nano"           # cheap LLM for chunking
EMBEDDING_MODEL = "text-embedding-3-large"

# ── ChromaDB ───────────────────────────────────────────────────────────────────
DB_PATH = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "codebase"

# ── Scanning config ────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".md", ".json", ".yaml", ".yml",
    ".html", ".css", ".scss",
    ".sh", ".toml", ".cfg", ".ini", ".env.example",
    ".sql", ".rs", ".go", ".java", ".cpp", ".c", ".h",
}

IGNORED_FOLDERS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__",
    "build", "dist", ".next", ".nuxt", "coverage",
    ".pytest_cache", ".mypy_cache", ".tox", "eggs",
    ".eggs", "*.egg-info", ".idea", ".vscode",
}

MAX_FILE_SIZE_KB = 500  # skip files larger than this (generated/minified)
WORKERS = 3             # parallel LLM chunking workers; set to 1 if rate-limited

# ── Retry config ───────────────────────────────────────────────────────────────
wait = wait_exponential(multiplier=1, min=10, max=240)

openai_client = OpenAI()


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ══════════════════════════════════════════════════════════════════════════════

class Chunk(BaseModel):
    headline: str = Field(
        description=(
            "A brief, specific heading for this chunk (a few words) that is most likely "
            "to be matched when a developer searches for this piece of code."
        )
    )
    summary: str = Field(
        description=(
            "2-4 sentences describing what this chunk does, its purpose, "
            "inputs/outputs, and any important details a developer would want to know."
        )
    )
    original_code: str = Field(
        description="The exact original text of this chunk, unchanged."
    )

    def as_document(self, metadata: dict) -> dict:
        """Combine headline + summary + code into a single searchable document."""
        page_content = (
            f"### {self.headline}\n\n"
            f"{self.summary}\n\n"
            f"```{metadata.get('language', '')}\n{self.original_code}\n```"
        )
        return {"page_content": page_content, "metadata": metadata}


class Chunks(BaseModel):
    chunks: list[Chunk]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — SCANNER
# ══════════════════════════════════════════════════════════════════════════════

def _is_ignored(path: Path, repo_root: Path) -> bool:
    """Return True if any part of the path is an ignored folder."""
    relative_parts = path.relative_to(repo_root).parts
    return bool(set(relative_parts) & IGNORED_FOLDERS)


def _detect_language(path: Path) -> str:
    """Map file extension to a language string for syntax highlighting."""
    mapping = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".jsx": "jsx", ".tsx": "tsx", ".md": "markdown", ".json": "json",
        ".yaml": "yaml", ".yml": "yaml", ".html": "html", ".css": "css",
        ".scss": "scss", ".sh": "bash", ".toml": "toml", ".sql": "sql",
        ".rs": "rust", ".go": "go", ".java": "java",
        ".cpp": "cpp", ".c": "c", ".h": "c",
    }
    return mapping.get(path.suffix.lower(), "text")


def _classify_type(path: Path) -> str:
    """Classify a file as code, config, or docs."""
    if path.suffix in {".md", ".rst", ".txt"}:
        return "docs"
    if path.suffix in {".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".env.example"}:
        return "config"
    return "code"


def scan_repository(repo_path: str) -> list[dict]:
    """
    Recursively scan a repository and return a list of file documents.

    Each document is a dict with keys:
        source   – relative path from repo root (str)
        filename – file name
        language – detected language
        folder   – parent folder (relative)
        type     – code | config | docs
        text     – raw file content
    """
    repo_root = Path(repo_path).resolve()
    documents = []
    skipped = 0

    for file_path in repo_root.rglob("*"):
        if not file_path.is_file():
            continue
        if _is_ignored(file_path, repo_root):
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if file_path.stat().st_size > MAX_FILE_SIZE_KB * 1024:
            skipped += 1
            continue

        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            skipped += 1
            continue

        if not text.strip():
            continue

        relative = file_path.relative_to(repo_root)
        documents.append({
            "source":   str(relative),
            "filename": file_path.name,
            "language": _detect_language(file_path),
            "folder":   str(relative.parent),
            "type":     _classify_type(file_path),
            "text":     text,
        })

    print(f"Scanned {len(documents)} files ({skipped} skipped — too large or unreadable)")
    return documents


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — CHUNKER (LLM-based)
# ══════════════════════════════════════════════════════════════════════════════

def _build_chunk_prompt(document: dict) -> str:
    avg_chunk_size = 80  # lines
    estimated_chunks = max(1, len(document["text"].splitlines()) // avg_chunk_size)

    return f"""You are a code knowledge-base builder.
Your job is to split the following source file into overlapping chunks for a RAG system.
A developer will query this knowledge base with questions like:
  - "Where is function X defined?"
  - "How does module Y work?"
  - "What does this class do?"

File metadata:
  - Path:     {document['source']}
  - Language: {document['language']}
  - Type:     {document['type']}

Rules:
  1. Each chunk must correspond to a LOGICAL unit: a function, class, section, or block of related code.
  2. Include ~25% overlap with adjacent chunks so context is not lost at boundaries.
  3. Aim for roughly {estimated_chunks} chunks, but use more or fewer as the content demands.
  4. Cover the ENTIRE file — do not skip anything.
  5. The `original_code` field must be the exact text from the file, unchanged.
  6. The `headline` should be specific enough to surface in a keyword search.
  7. The `summary` should explain purpose, behaviour, inputs/outputs in plain English.

Here is the file:

{document['text']}
"""


@retry(wait=wait)
def _chunk_document(document: dict) -> list[dict]:
    """Call gpt-4.1-nano to split one file into structured chunks."""
    from openai import OpenAI
    client = OpenAI()  # each worker process needs its own client

    response = client.beta.chat.completions.parse(
        model=CHUNK_MODEL,
        messages=[{"role": "user", "content": _build_chunk_prompt(document)}],
        response_format=Chunks,
    )
    chunks_obj: Chunks = response.choices[0].message.parsed

    metadata = {
        "source":   document["source"],
        "filename": document["filename"],
        "language": document["language"],
        "folder":   document["folder"],
        "type":     document["type"],
    }
    return [chunk.as_document(metadata) for chunk in chunks_obj.chunks]


def create_chunks(documents: list[dict]) -> list[dict]:
    """
    Chunk all documents in parallel using a process pool.
    Set WORKERS=1 if you hit rate limits.
    """
    all_chunks = []
    with Pool(processes=WORKERS) as pool:
        for result in tqdm(
            pool.imap_unordered(_chunk_document, documents),
            total=len(documents),
            desc="Chunking files",
        ):
            all_chunks.extend(result)

    print(f"Created {len(all_chunks)} chunks from {len(documents)} files")
    return all_chunks


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — EMBEDDER
# ══════════════════════════════════════════════════════════════════════════════

EMBED_BATCH_SIZE = 100  # OpenAI allows up to 2048 inputs per request


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def create_embeddings(chunks: list[dict], reset: bool = False) -> None:
    """
    Generate embeddings for all chunks and persist them to ChromaDB.

    Args:
        chunks: list of {"page_content": str, "metadata": dict}
        reset:  if True, delete the existing collection before inserting
    """
    chroma = PersistentClient(path=DB_PATH)

    if reset and COLLECTION_NAME in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'")

    collection = chroma.get_or_create_collection(COLLECTION_NAME)

    texts = [c["page_content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    all_vectors = []
    for i in tqdm(
        range(0, len(texts), EMBED_BATCH_SIZE),
        desc="Embedding chunks",
    ):
        batch = texts[i: i + EMBED_BATCH_SIZE]
        all_vectors.extend(_embed_batch(batch))

    ids = [str(i) for i in range(len(chunks))]
    collection.add(
        ids=ids,
        embeddings=all_vectors,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"ChromaDB collection '{COLLECTION_NAME}' now has {collection.count()} documents")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def ingest(repo_path: str, reset: bool = False) -> None:
    """Full ingestion pipeline: scan → chunk → embed."""
    print(f"\n{'='*60}")
    print(f"  code_assistant_rag — Ingestion Pipeline")
    print(f"  Repository : {repo_path}")
    print(f"  Reset DB   : {reset}")
    print(f"{'='*60}\n")

    documents = scan_repository(repo_path)
    if not documents:
        print("No files found. Check your path and SUPPORTED_EXTENSIONS.")
        return

    chunks = create_chunks(documents)
    create_embeddings(chunks, reset=reset)

    print("\n✅ Ingestion complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a code repository into ChromaDB")
    parser.add_argument("--repo", required=True, help="Path to the repository to ingest")
    parser.add_argument(
        "--reset", action="store_true",
        help="Wipe the existing ChromaDB collection before ingesting"
    )
    args = parser.parse_args()
    ingest(args.repo, reset=args.reset)