---
title: Multilingual RAG Chatbot
emoji: 🤖
colorFrom: orange
colorTo: green
sdk: docker
pinned: false
---
# Multilingual RAG Chatbot â€” Hindi + English

Cross-lingual RAG system that answers questions in Hindi and English,
retrieving from a shared multilingual corpus using LaBSE embeddings.

## Architecture

- **Embeddings**: LaBSE (cross-lingual â€” Hindi and English share the same vector space)
- **Vector store**: ChromaDB (persisted)
- **LLM**: Groq Llama-3.1-70B (free tier)
- **Evaluation**: RAGAS faithfulness + answer relevancy
- **Stack**: LangChain Â· FastAPI Â· Streamlit Â· Render

## Quickstart

### Option A â€” Google Colab (recommended for Phase 1 & 2)

1. Open `notebooks/phase1_2_colab.ipynb` in Colab
2. Set runtime to **GPU (T4)**
3. Run all cells top to bottom
4. ChromaDB saves to your Google Drive

### Option B â€” Local (VS Code)

```bash
git clone https://github.com/YOUR_USERNAME/multilingual-rag
cd multilingual-rag

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Phase 1: Load and chunk documents
python ingest/01_load_and_chunk.py

# Phase 2: Embed and store in ChromaDB
python ingest/02_embed_and_store.py
```

## Project structure

```
multilingual-rag/
â”œâ”€â”€ ingest/
â”‚   â”œâ”€â”€ 01_load_and_chunk.py      # Phase 1
â”‚   â””â”€â”€ 02_embed_and_store.py     # Phase 2
â”œâ”€â”€ app/
â”‚   â”œâ”€â”€ main.py                   # FastAPI routes (Phase 3)
â”‚   â”œâ”€â”€ rag_chain.py              # LangChain RAG logic
â”‚   â””â”€â”€ evaluator.py              # RAGAS metrics (Phase 4)
â”œâ”€â”€ frontend/
â”‚   â””â”€â”€ streamlit_app.py          # UI (Phase 5)
â”œâ”€â”€ notebooks/
â”‚   â””â”€â”€ phase1_2_colab.ipynb      # Colab notebook (Phase 1+2)
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ chunks/                   # Saved chunks (JSONL)
â”‚   â””â”€â”€ pdfs/                     # Your custom PDFs go here
â”œâ”€â”€ chroma_db/                    # Persisted vector store
â”œâ”€â”€ .env.example
â””â”€â”€ requirements.txt
```

## Environment variables

Copy `.env.example` to `.env` and fill in your keys:

```
GROQ_API_KEY=your_groq_key_here
```

Get a free Groq API key at: https://console.groq.com

## Evaluation metrics (RAGAS)

| Metric | Description |
|--------|-------------|
| Faithfulness | Is the answer grounded in retrieved context? |
| Answer relevancy | Does the answer address the question? |
| Context recall | Did retrieval find the right chunks? |

## Resume line

> "Built and deployed a cross-lingual RAG chatbot over Hindi corpora using
> LaBSE embeddings and Llama-3.1-70B; integrated RAGAS faithfulness
> evaluation achieving X% grounding score on Hindi QA benchmarks."


