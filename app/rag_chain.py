"""
Phase 3: RAG chain — retrieval + prompt building + LLM generation.

Uses Groq's free Llama-3.1-70B for generation, ChromaDB (built in Phase 2)
for retrieval, and LaBSE embeddings for cross-lingual matching.

Get a free Groq API key at: https://console.groq.com
"""

import os
from typing import List, Dict, Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


# ── Configuration ─────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "sentence-transformers/LaBSE"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "hindi_rag"
LLM_MODEL = "llama-3.3-70b-versatile"
TOP_K = 3   # number of chunks to retrieve


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful multilingual assistant that answers questions \
using ONLY the provided context.

Rules:
1. Respond in the SAME language as the user's question (Hindi question -> Hindi answer, \
English question -> English answer).
2. Base your answer ONLY on the context provided below. Do not use outside knowledge.
3. If the context does not contain enough information to answer, say so honestly \
(in the same language as the question) instead of guessing.
4. Be concise — 2-4 sentences unless the question requires more detail.
5. Do not mention "the context" or "the provided text" in your answer — just answer naturally.
"""


# ── RAG Chain class ───────────────────────────────────────────────────────────

class MultilingualRAGChain:
    """
    Cross-lingual RAG: embeds query with LaBSE, retrieves from ChromaDB,
    builds a grounded prompt, and generates an answer with Groq.
    """

    def __init__(self):
        print("Initializing RAG chain...")

        # Load embedding model (same as used in Phase 2 — must match!)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        # Load existing ChromaDB
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DIR
        )
        count = self.vectorstore._collection.count()
        print(f"Loaded ChromaDB with {count} chunks")

        # Groq client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Create a .env file with GROQ_API_KEY=your_key"
            )
        self.groq_client = Groq(api_key=api_key)

        print("RAG chain ready.\n")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
        """
        Embed the query (any language) and retrieve top-k similar chunks
        from ChromaDB. Returns chunks with similarity scores.
        """
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=k
        )

        retrieved = []
        for doc, score in results:
            retrieved.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": round(float(score), 4)
            })

        return retrieved

    # ── Prompt building ──────────────────────────────────────────────────────

    def build_prompt(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Combine retrieved chunks into a context block for the LLM."""
        if not retrieved_chunks:
            context = "(No relevant context found)"
        else:
            context_parts = []
            for i, chunk in enumerate(retrieved_chunks, 1):
                lang = chunk["metadata"].get("language", "?")
                context_parts.append(f"[Source {i} - lang:{lang}]\n{chunk['content']}")
            context = "\n\n".join(context_parts)

        prompt = f"""Context:
{context}

Question: {query}

Answer:"""
        return prompt

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Call Groq LLM with the grounded prompt."""
        user_prompt = self.build_prompt(query, retrieved_chunks)

        response = self.groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,   # low temperature -> more grounded, less creative
            max_tokens=512
        )

        return response.choices[0].message.content.strip()

    # ── End-to-end query ─────────────────────────────────────────────────────

    def query(self, question: str, k: int = TOP_K) -> Dict[str, Any]:
        """
        Full RAG pipeline: retrieve -> build prompt -> generate.
        Returns answer + source chunks (for transparency / faithfulness eval).
        """
        retrieved = self.retrieve(question, k=k)
        answer = self.generate(question, retrieved)

        return {
            "question": question,
            "answer": answer,
            "sources": retrieved
        }


# ── Quick test (run this file directly) ───────────────────────────────────────

if __name__ == "__main__":
    rag = MultilingualRAGChain()

    test_questions = [
        "भारतीय संविधान में मौलिक अधिकार क्या हैं?",
        "When was Mahatma Gandhi born?",
        "हिंदी साहित्य का इतिहास क्या है?",
    ]

    for q in test_questions:
        print("=" * 60)
        print(f"Q: {q}")
        result = rag.query(q)
        print(f"\nA: {result['answer']}")
        print(f"\nSources used ({len(result['sources'])}):")
        for i, src in enumerate(result["sources"], 1):
            lang = src["metadata"].get("language", "?")
            score = src["score"]
            preview = src["content"][:80].replace("\n", " ")
            print(f"  [{i}] score={score} lang={lang} | {preview}...")
        print()
