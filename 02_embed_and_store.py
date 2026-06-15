"""
Phase 2: Embed chunks with LaBSE and store in ChromaDB.

LaBSE (Language-agnostic BERT Sentence Embeddings) maps Hindi and English
into the SAME embedding space — so a Hindi query can retrieve English chunks
and vice versa. This is the core of cross-lingual RAG.

Run AFTER 01_load_and_chunk.py.

Install first:
    pip install langchain langchain-community chromadb sentence-transformers
"""

import json
import time
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# ── Configuration ─────────────────────────────────────────────────────────────

# LaBSE: best for cross-lingual Hindi↔English retrieval
# Alternative: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_MODEL = "sentence-transformers/LaBSE"

CHROMA_DIR = "./chroma_db"       # persisted vector store location
COLLECTION_NAME = "hindi_rag"
CHUNKS_FILE = "./data/chunks/chunks.jsonl"
BATCH_SIZE = 64                  # how many chunks to embed at once (Colab memory)


# ── Load saved chunks ─────────────────────────────────────────────────────────

def load_chunks_from_disk(path: str = CHUNKS_FILE) -> List[Document]:
    """Load chunks saved by Phase 1."""
    chunks_path = Path(path)
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {path}\n"
            "Run 01_load_and_chunk.py first."
        )

    docs = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            docs.append(Document(
                page_content=record["content"],
                metadata=record["metadata"]
            ))

    print(f"Loaded {len(docs)} chunks from disk")
    return docs


# ── Embedding model ───────────────────────────────────────────────────────────

def load_embedding_model(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """
    Load LaBSE from HuggingFace.
    First run downloads ~1.8GB — cached after that.
    In Colab: use GPU by setting device='cuda' if available.
    """
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model: {model_name}")
    print(f"Using device: {device}")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,  # cosine similarity works better normalized
            "batch_size": BATCH_SIZE
        }
    )

    print("Embedding model loaded.")
    return embeddings


# ── Quick sanity check ────────────────────────────────────────────────────────

def sanity_check_crosslingual(embeddings: HuggingFaceEmbeddings):
    """
    Verify cross-lingual property: Hindi and English sentences about
    the same topic should have high cosine similarity.
    This is your thesis in miniature.
    """
    import numpy as np

    print("\nSanity check — cross-lingual embedding similarity:")

    pairs = [
        ("भारत का संविधान", "Constitution of India"),
        ("महात्मा गांधी", "Mahatma Gandhi"),
        ("दिल्ली भारत की राजधानी है", "Delhi is the capital of India"),
    ]

    for hindi, english in pairs:
        hi_vec = embeddings.embed_query(hindi)
        en_vec = embeddings.embed_query(english)

        # Cosine similarity
        hi_arr = np.array(hi_vec)
        en_arr = np.array(en_vec)
        similarity = float(
            np.dot(hi_arr, en_arr) /
            (np.linalg.norm(hi_arr) * np.linalg.norm(en_arr))
        )
        status = "GOOD" if similarity > 0.7 else "LOW — check model"
        print(f"  [{status}] '{hindi}' ↔ '{english}' → sim={similarity:.3f}")


# ── Build ChromaDB ────────────────────────────────────────────────────────────

def build_vectorstore(
    docs: List[Document],
    embeddings: HuggingFaceEmbeddings,
    persist_dir: str = CHROMA_DIR
) -> Chroma:
    """
    Embed all chunks and store in ChromaDB.
    ChromaDB persists to disk — you only run this once.
    """
    print(f"\nBuilding ChromaDB at: {persist_dir}")
    print(f"Embedding {len(docs)} chunks (this takes ~2–5 min on Colab GPU)...")

    start = time.time()

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
        collection_metadata={
            "description": "Multilingual Hindi-English RAG corpus",
            "embedding_model": EMBEDDING_MODEL,
        }
    )

    elapsed = time.time() - start
    print(f"Done. Embedded {len(docs)} chunks in {elapsed:.1f}s")

    return vectorstore


# ── Test retrieval ────────────────────────────────────────────────────────────

def test_retrieval(vectorstore: Chroma):
    """
    Run a few test queries — Hindi and English — to verify retrieval works.
    These results are what your RAG chain will pass to the LLM.
    """
    print("\nTesting retrieval (cross-lingual queries):")

    test_queries = [
        ("hi", "भारतीय संविधान में मौलिक अधिकार क्या हैं?"),   # Hindi query
        ("en", "What are the fundamental rights in Indian Constitution?"),
        ("hi", "गांधीजी का जन्म कब हुआ था?"),                  # Hindi query
        ("en", "When was Mahatma Gandhi born?"),
    ]

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}      # top-3 chunks per query
    )

    for lang, query in test_queries:
        print(f"\n  Query [{lang}]: {query}")
        results = retriever.invoke(query)
        for i, doc in enumerate(results):
            src_lang = doc.metadata.get("language", "?")
            source = doc.metadata.get("source", "unknown")
            preview = doc.page_content[:120].replace("\n", " ")
            print(f"    [{i+1}] lang={src_lang} | {preview}...")


# ── Load existing vectorstore ─────────────────────────────────────────────────

def load_vectorstore(
    embeddings: HuggingFaceEmbeddings,
    persist_dir: str = CHROMA_DIR
) -> Chroma:
    """Load an already-built ChromaDB (skip re-embedding)."""
    print(f"Loading existing ChromaDB from: {persist_dir}")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir
    )
    count = vectorstore._collection.count()
    print(f"Loaded collection with {count} chunks")
    return vectorstore


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("Phase 2: Embedding + ChromaDB vector store")
    print("=" * 55)

    # Step 1: Load chunks from Phase 1
    docs = load_chunks_from_disk()

    # Step 2: Load embedding model
    embeddings = load_embedding_model()

    # Step 3: Cross-lingual sanity check (important for your thesis!)
    sanity_check_crosslingual(embeddings)

    # Step 4: Build or load vectorstore
    chroma_path = Path(CHROMA_DIR)
    if chroma_path.exists() and any(chroma_path.iterdir()):
        print(f"\nChromaDB already exists at {CHROMA_DIR}.")
        print("Loading existing store (delete the folder to rebuild).")
        vectorstore = load_vectorstore(embeddings)
    else:
        vectorstore = build_vectorstore(docs, embeddings)

    # Step 5: Test retrieval
    test_retrieval(vectorstore)

    print("\n" + "=" * 55)
    print("Phase 2 complete!")
    print(f"ChromaDB saved at: {CHROMA_DIR}")
    print("Next: build the RAG chain in app/rag_chain.py")
    print("=" * 55)

    return vectorstore


if __name__ == "__main__":
    main()
