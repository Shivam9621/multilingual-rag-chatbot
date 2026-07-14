---
title: Multilingual RAG Chatbot
emoji: 🤖
colorFrom: red
colorTo: green
sdk: docker
pinned: false
---

# Cross-Lingual RAG Chatbot — Hindi + English

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://multilingual-rag-chatbot-huhtbme9lhgyevmb8hcc9p.streamlit.app)
[![Backend API](https://img.shields.io/badge/Backend%20API-HuggingFace-FFD21E?style=for-the-badge&logo=huggingface)](https://shivamj18-rag-deploy.hf.space/docs)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/Shivam9621/multilingual-rag-chatbot)

> Ask questions in **Hindi or English** — answers grounded in retrieved sources with **0.95+ faithfulness score**

---

## What makes this different

Standard RAG systems fail at cross-lingual retrieval — a Hindi query won't match English documents because the embeddings live in different spaces. This project solves that by using **LaBSE (Language-agnostic BERT Sentence Embeddings)**, which maps Hindi and English into the **same vector space**. A Hindi question retrieves relevant English chunks, and vice versa — seamlessly.

---

## Demo

| Hindi query | English answer |
|---|---|
| भारतीय संविधान में मौलिक अधिकार क्या हैं? | Grounded answer from retrieved Hindi + English chunks |
| What is the significance of Mahatma Gandhi? | Hindi answer generated from cross-lingual retrieval |

---

## Architecture

```
User Query (Hindi or English)
        |
        v
LaBSE Embedding (cross-lingual — same vector space for both languages)
        |
        v
ChromaDB Semantic Search (top-k chunks retrieved)
        |
        v
Prompt Builder (context + query + language instruction)
        |
        v
Groq Llama-3.3-70B (grounded generation, same-language response)
        |
        v
RAGAS Evaluation (faithfulness, answer relevancy, context precision)
```

---

## Evaluation Results (RAGAS)

| Metric | Score | What it measures |
|--------|-------|-----------------|
| **Faithfulness** | **0.95+** | Is every claim in the answer grounded in retrieved context? |
| **Answer Relevancy** | **0.92** | Does the answer actually address the question? |
| **Context Precision** | **0.96** | Did retrieval find the right chunks? |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Embeddings | LaBSE (`sentence-transformers/LaBSE`) |
| Vector Store | ChromaDB (persisted) |
| LLM | Groq Llama-3.3-70B (free tier) |
| Evaluation | RAGAS |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Backend Deployment | Hugging Face Spaces (Docker) |
| Frontend Deployment | Streamlit Community Cloud |
| Orchestration | LangChain |

---

## Project Structure

```
multilingual-rag-chatbot/
├── app/
│   ├── main.py                # FastAPI routes (/chat, /health)
│   └── rag_chain.py           # LangChain RAG pipeline
├── frontend/
│   └── streamlit_app.py       # Streamlit chat UI
├── ingest/
│   ├── 01_load_and_chunk.py   # Phase 1: Wikipedia + PDF loading
│   └── 02_embed_and_store.py  # Phase 2: LaBSE embedding + ChromaDB
├── evaluation/
│   └── evaluate_rag.py        # RAGAS faithfulness evaluation
├── chroma_db/                 # Persisted vector store (254 chunks)
├── data/chunks/               # Saved JSONL chunks
├── Dockerfile                 # HF Spaces deployment
├── requirements-deploy.txt    # Slim backend requirements
├── requirements-streamlit.txt # Slim frontend requirements
└── requirements.txt           # Full local requirements
```

---

## Quickstart (Local)

**Prerequisites:** Python 3.10+, free [Groq API key](https://console.groq.com)

```bash
git clone https://github.com/Shivam9621/multilingual-rag-chatbot
cd multilingual-rag-chatbot

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_groq_key_here
```

```bash
# Phase 1 + 2: Build the knowledge base (run once)
python ingest/01_load_and_chunk.py
python ingest/02_embed_and_store.py

# Phase 3: Start the API
uvicorn app.main:app --reload

# Phase 5: Start the UI (new terminal)
streamlit run frontend/streamlit_app.py
```

Open `http://localhost:8501` and start chatting.

---

## API Reference

**Base URL:** `https://shivamj18-rag-deploy.hf.space`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Returns chunk count + model info |
| `/chat` | POST | Ask a question |
| `/docs` | GET | Interactive Swagger UI |

**Example request:**
```bash
curl -X POST https://shivamj18-rag-deploy.hf.space/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Ayurveda?", "top_k": 3}'
```

---

## Run Evaluation

```bash
python evaluation/evaluate_rag.py
```

Runs RAGAS faithfulness, answer relevancy, and context precision across Hindi + English test questions. Results saved to `evaluation/results/`.

---

## Key Design Decisions

**Why LaBSE over mBERT?** LaBSE is specifically trained for cross-lingual sentence similarity across 109 languages, giving much better retrieval performance than general multilingual models.

**Why Groq?** Free tier, sub-second inference on Llama-3.3-70B — ideal for a deployed demo with real latency.

**Why ChromaDB?** Persistent, lightweight, no infrastructure needed. The entire vector store is 7.9MB and ships with the repo.

**Why RAGAS?** Industry-standard RAG evaluation framework. Faithfulness score directly measures hallucination rate — the most critical metric for grounded systems.

---

## Built by

[Shivam Jaiswal](https://github.com/Shivam9621) — Pre-final year IDD (B.Tech CSE + M.Tech AI) at RGIPT Bengaluru
