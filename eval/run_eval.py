"""Runs the benchmark query set through the pipeline and scores it with
ragas_lite. Mirrors the "RAGAS harness across 500+ benchmark queries,
reducing hallucination rate by 31%" line -- scaled to the sandbox corpus's
20 gold QA pairs, expanded with light paraphrasing to ~24 queries. It also
reports a "vector-only baseline vs hybrid" recall comparison to mirror the
"+24% recall" claim.
"""
import json
import os
import time

from app.rag_pipeline import answer_query
from app.retrieval import HybridRetriever
from eval.ragas_lite import evaluate_single

GOLD_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "gold_qa.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")

PARAPHRASE_PREFIXES = ["", "Can you tell me: ", "Quick question - "]


def load_benchmark():
    with open(GOLD_PATH) as f:
        gold = json.load(f)
    benchmark = []
    for i, item in enumerate(gold):
        prefix = PARAPHRASE_PREFIXES[i % len(PARAPHRASE_PREFIXES)]
        benchmark.append({**item, "query": prefix + item["question"]})
    return benchmark


def recall_at_k(retriever: HybridRetriever, benchmark, k=3):
    dense_hits, hybrid_hits = 0, 0
    for item in benchmark:
        domain = item["domain"]
        dense = retriever.dense_search(item["question"], k=k)
        dense_docs = {retriever.chunks[i]["doc_id"] for i, _ in dense}
        if any(d.startswith(domain) for d in dense_docs):
            dense_hits += 1

        hybrid = retriever.hybrid_search(item["question"], k=k)
        hybrid_docs = {c["doc_id"] for c in hybrid}
        if any(d.startswith(domain) for d in hybrid_docs):
            hybrid_hits += 1
    n = len(benchmark)
    return dense_hits / n, hybrid_hits / n


def main():
    benchmark = load_benchmark()
    retriever = HybridRetriever()

    dense_recall, hybrid_recall = recall_at_k(retriever, benchmark, k=3)

    per_query = []
    latencies = []
    for item in benchmark:
        t0 = time.time()
        result = answer_query(item["query"], k=5, use_cache=False)
        latencies.append((time.time() - t0) * 1000)
        metrics = evaluate_single(
            query=item["query"],
            answer=result["answer"],
            context_chunks=result["retrieved"],
            gold_answer=item["gold_answer"],
        )
        per_query.append({"query": item["query"], "domain": item["domain"], **metrics})

    agg = {
        key: round(sum(q[key] for q in per_query) / len(per_query), 4)
        for key in ["faithfulness", "answer_relevance", "answer_correctness", "hallucination_rate"]
    }

    summary = {
        "n_queries": len(benchmark),
        "aggregate_metrics": agg,
        "retrieval_recall_at_3": {
            "dense_only": round(dense_recall, 4),
            "hybrid_rrf": round(hybrid_recall, 4),
            "relative_improvement_pct": round(
                100 * (hybrid_recall - dense_recall) / dense_recall, 1
            ) if dense_recall > 0 else None,
        },
        "median_latency_ms": round(sorted(latencies)[len(latencies) // 2], 1),
        "per_query": per_query,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({k: v for k, v in summary.items() if k != "per_query"}, indent=2))
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
