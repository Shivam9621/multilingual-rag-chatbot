"""
Phase 1: Load Hindi/English documents and chunk them.

Run this in Google Colab or locally.
Supports: Wikipedia articles, PDF files, plain text files.

Install first:
    pip install langchain langchain-community pypdf wikipedia datasets
"""

import os
import json
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ── Configuration ─────────────────────────────────────────────────────────────

CHUNK_SIZE = 500        # tokens per chunk (smaller = better faithfulness)
CHUNK_OVERLAP = 50      # overlap between chunks to preserve context
OUTPUT_DIR = Path("./data/chunks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Source 1: Wikipedia (Hindi + English) ─────────────────────────────────────

def load_from_wikipedia(queries: List[str], lang: str = "hi") -> List[Document]:
    """
    Load Wikipedia articles for given queries.
    lang='hi' for Hindi, lang='en' for English.
    """
    from langchain_community.document_loaders import WikipediaLoader

    docs = []
    for query in queries:
        print(f"  Fetching Wikipedia [{lang}]: {query}")
        try:
            loader = WikipediaLoader(
                query=query,
                lang=lang,
                load_max_docs=2,          # 2 articles per query is enough
                doc_content_chars_max=5000
            )
            loaded = loader.load()
            # Tag each doc with its language for later analysis
            for doc in loaded:
                doc.metadata["language"] = lang
                doc.metadata["source_type"] = "wikipedia"
            docs.extend(loaded)
            print(f"    Loaded {len(loaded)} articles")
        except Exception as e:
            print(f"    Skipped '{query}': {e}")

    return docs


# ── Source 2: PDF files ────────────────────────────────────────────────────────

def load_from_pdfs(pdf_dir: str = "./data/pdfs") -> List[Document]:
    """
    Load all PDFs from a local folder.
    Put your Hindi government docs / news PDFs here.
    """
    from langchain_community.document_loaders import PyPDFLoader

    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        print(f"  PDF folder not found: {pdf_dir} — skipping")
        return []

    docs = []
    for pdf_file in pdf_path.glob("*.pdf"):
        print(f"  Loading PDF: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["language"] = "hi"   # adjust if mixed
                doc.metadata["source_type"] = "pdf"
            docs.extend(loaded)
            print(f"    Loaded {len(loaded)} pages")
        except Exception as e:
            print(f"    Failed on {pdf_file.name}: {e}")

    return docs


# ── Source 3: Plain text files ─────────────────────────────────────────────────

def load_from_text_files(text_dir: str = "./data/texts") -> List[Document]:
    """
    Load plain .txt files. Useful for any custom Hindi corpus.
    """
    from langchain_community.document_loaders import TextLoader

    text_path = Path(text_dir)
    if not text_path.exists():
        print(f"  Text folder not found: {text_dir} — skipping")
        return []

    docs = []
    for txt_file in text_path.glob("*.txt"):
        print(f"  Loading text: {txt_file.name}")
        try:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["language"] = "hi"
                doc.metadata["source_type"] = "text"
            docs.extend(loaded)
        except Exception as e:
            print(f"    Failed on {txt_file.name}: {e}")

    return docs


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Split documents into chunks.
    RecursiveCharacterTextSplitter respects sentence/paragraph boundaries,
    which is important for Hindi text (Devanagari doesn't use Latin punctuation).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",   # paragraph break (universal)
            "\n",     # line break
            "।",      # Hindi full stop (Devanagari danda)
            ".",      # English full stop
            " ",      # word boundary fallback
            ""        # character fallback
        ]
    )

    chunks = splitter.split_documents(docs)

    # Add chunk index to metadata for traceability
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_length"] = len(chunk.page_content)

    return chunks


# ── Save chunks to disk ───────────────────────────────────────────────────────

def save_chunks(chunks: List[Document], filename: str = "chunks.jsonl"):
    """Save chunks as JSONL so you can inspect them without reprocessing."""
    out_path = OUTPUT_DIR / filename
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {
                "content": chunk.page_content,
                "metadata": chunk.metadata
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(chunks)} chunks to {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("Phase 1: Loading and chunking documents")
    print("=" * 55)

    all_docs = []

    # --- Wikipedia: Hindi topics ---
    hindi_topics = [
        "भारतीय संविधान",      # Indian Constitution
        "महात्मा गांधी",         # Mahatma Gandhi
        "भारत का इतिहास",       # History of India
        "आयुर्वेद",              # Ayurveda
        "हिंदी साहित्य",         # Hindi literature
    ]
    print("\n[1/3] Loading Hindi Wikipedia...")
    hindi_wiki_docs = load_from_wikipedia(hindi_topics, lang="hi")
    all_docs.extend(hindi_wiki_docs)

    # --- Wikipedia: English topics (cross-lingual test) ---
    english_topics = [
        "Indian Constitution",
        "Mahatma Gandhi",
        "History of India",
    ]
    print("\n[2/3] Loading English Wikipedia (for cross-lingual retrieval)...")
    english_wiki_docs = load_from_wikipedia(english_topics, lang="en")
    all_docs.extend(english_wiki_docs)

    # --- PDFs (optional) ---
    print("\n[3/3] Loading PDFs (if any)...")
    pdf_docs = load_from_pdfs("./data/pdfs")
    all_docs.extend(pdf_docs)

    print(f"\nTotal raw documents loaded: {len(all_docs)}")

    if not all_docs:
        print("No documents loaded. Check your internet connection or add PDFs.")
        return

    # --- Chunk ---
    print("\nChunking documents...")
    chunks = chunk_documents(all_docs)
    print(f"Total chunks created: {len(chunks)}")

    # --- Stats ---
    hindi_chunks = [c for c in chunks if c.metadata.get("language") == "hi"]
    english_chunks = [c for c in chunks if c.metadata.get("language") == "en"]
    avg_len = sum(c.metadata["chunk_length"] for c in chunks) / len(chunks)
    print(f"  Hindi chunks   : {len(hindi_chunks)}")
    print(f"  English chunks : {len(english_chunks)}")
    print(f"  Avg chunk size : {avg_len:.0f} chars")

    # --- Save ---
    save_chunks(chunks)

    print("\nPhase 1 complete. Run 02_embed_and_store.py next.")
    return chunks


if __name__ == "__main__":
    main()
