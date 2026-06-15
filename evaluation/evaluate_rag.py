"""
Phase 4: RAGAS evaluation — measure faithfulness, answer relevancy, and
context precision of the RAG pipeline.

RAGAS uses an LLM as a "judge" to score these metrics. By default RAGAS
expects OpenAI, but we configure it to use Groq (free) instead via
LangChain's ChatGroq wrapper, and LaBSE for embedding-based relevancy.

Install first:
    pip install ragas langchain-groq

Run:
    python evaluation/evaluate_rag.py
"""

import os
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithoutReference

# Import your RAG chain
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.rag_chain import MultilingualRAGChain


load_dotenv()


# ── Configuration ─────────────────────────────────────────────────────────────

# Use a smaller/faster Groq model as the JUDGE to save rate limits
# (separate from the model that generates RAG answers)
JUDGE_MODEL = "llama-3.1-8b-instant"

OUTPUT_DIR = Path("./evaluation/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Test questions (Hindi + English) ───────────────────────────────────────────

TEST_QUESTIONS = [
    "भारतीय संविधान कब लागू हुआ था?",
    "What is Ayurveda?",
    "महात्मा गांधी का जन्म कहाँ हुआ था?",
    "What are the parts of the Indian Constitution?",
    "हिंदी साहित्य का महत्व क्या है?",
]


# ── Build evaluation dataset from your RAG chain ─────────────────────────────

def build_eval_dataset(rag_chain: MultilingualRAGChain, questions: list) -> list:
    """
    Run each question through the RAG pipeline and collect:
    - user_input (question)
    - response (generated answer)
    - retrieved_contexts (chunks used)
    """
    records = []
    for i, question in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {question}")
        result = rag_chain.query(question, k=3)

        records.append({
            "user_input": result["question"],
            "response": result["answer"],
            "retrieved_contexts": [src["content"] for src in result["sources"]],
        })

    return records


# ── Run RAGAS evaluation ──────────────────────────────────────────────────────

def run_evaluation(records: list):
    """
    Score the dataset using RAGAS metrics, with Groq as the judge LLM
    and LaBSE as the embedding model for relevancy scoring.
    """
    print("\nSetting up evaluator LLM (Groq) and embeddings (LaBSE)...")

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in .env")

    judge_llm = ChatGroq(
        model=JUDGE_MODEL,
        api_key=groq_api_key,
        temperature=0.0
    )
    evaluator_llm = LangchainLLMWrapper(judge_llm)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/LaBSE",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(embedding_model)

    dataset = EvaluationDataset.from_list(records)

    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        LLMContextPrecisionWithoutReference(llm=evaluator_llm),
    ]

    print("Running RAGAS evaluation (this calls the judge LLM multiple times "
          "per question — may take a few minutes)...\n")

    # Lower concurrency + higher timeout avoids Groq free-tier rate-limit
    # timeouts that were causing many jobs to fail and skew the average.
    run_config = RunConfig(
        timeout=120,        # seconds per LLM call
        max_workers=2,      # low concurrency to respect Groq rate limits
        max_retries=5,      # retry on transient failures
        max_wait=60         # max backoff between retries
    )

    result = evaluate(dataset=dataset, metrics=metrics, run_config=run_config)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 4: RAGAS Faithfulness Evaluation")
    print("=" * 60)

    print("\nLoading RAG chain...")
    rag = MultilingualRAGChain()

    print(f"\nRunning {len(TEST_QUESTIONS)} test questions through RAG pipeline...")
    records = build_eval_dataset(rag, TEST_QUESTIONS)

    # Save raw Q&A pairs for inspection / your README
    qa_path = OUTPUT_DIR / "qa_pairs.json"
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\nSaved Q&A pairs to {qa_path}")

    # Run evaluation
    result = run_evaluation(records)

    # ── Results ──
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    df = result.to_pandas()

    # Per-question breakdown
    csv_path = OUTPUT_DIR / "ragas_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Per-question results saved to {csv_path}")

    # Overall averages
    print("\nOverall scores (averaged across all questions):")
    summary = {}
    for metric in ["faithfulness", "answer_relevancy", "llm_context_precision_without_reference"]:
        if metric in df.columns:
            avg_score = df[metric].mean()
            summary[metric] = round(float(avg_score), 4)
            print(f"  {metric:45s}: {avg_score:.4f}")

    # Save summary
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    print("\n" + "=" * 60)
    print("Phase 4 complete!")
    print("Use the faithfulness score in your resume / README:")
    if "faithfulness" in summary:
        pct = summary["faithfulness"] * 100
        print(f'  "...achieved {pct:.1f}% faithfulness on Hindi-English RAG QA"')
    print("=" * 60)


if __name__ == "__main__":
    main()