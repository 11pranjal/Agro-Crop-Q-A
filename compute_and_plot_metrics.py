"""
Compute common RAG evaluation metrics and plot them.

Run with: python compute_and_plot_metrics.py
Saves charts to `images/evaluation_metrics/`.
"""
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def precision_at_k(relevances: List[List[int]], k: int) -> float:
    """Average Precision@k. `relevances` is list per query of 0/1 for retrieved items."""
    vals = []
    for r in relevances:
        vals.append(sum(r[:k]) / float(k))
    return float(np.mean(vals))


def recall_at_k(relevances: List[List[int]], k: int, total_relevants: List[int]) -> float:
    """Average Recall@k. `total_relevants` is total relevant items for each query."""
    vals = []
    for r, total in zip(relevances, total_relevants):
        if total <= 0:
            vals.append(0.0)
        else:
            vals.append(sum(r[:k]) / float(total))
    return float(np.mean(vals))


def hit_rate_at_k(relevances: List[List[int]], k: int) -> float:
    """Fraction of queries with at least one relevant item in top-k."""
    hits = [1 if any(r[:k]) else 0 for r in relevances]
    return float(np.mean(hits))


def mrr_at_k(relevances: List[List[int]], k: int) -> float:
    """Mean reciprocal rank at k."""
    rr = []
    for r in relevances:
        rank = 0
        for i, v in enumerate(r[:k]):
            if v:
                rank = i + 1
                break
        rr.append(1.0 / rank if rank > 0 else 0.0)
    return float(np.mean(rr))


def mean_semantic_similarity(similarities: List[float]) -> float:
    return float(np.mean(similarities)) if similarities else 0.0


def hallucination_rate(hall_flags: List[int]) -> float:
    """Fraction of answers flagged as hallucinations (1) vs not (0)."""
    return float(np.mean(hall_flags)) if hall_flags else 0.0


def avg_latency(latencies: List[float]) -> float:
    return float(np.mean(latencies)) if latencies else 0.0


def avg_context_used(context_counts: List[float]) -> float:
    return float(np.mean(context_counts)) if context_counts else 0.0


def rag_efficiency(precision: float, avg_latency_sec: float) -> float:
    """A simple RAG efficiency proxy: precision divided by latency (higher is better)."""
    return precision / avg_latency_sec if avg_latency_sec > 0 else 0.0


def plot_bar(metrics: dict, title: str, output_path: str) -> None:
    labels = list(metrics.keys())
    values = [metrics[k] for k in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color="#2a5d84")
    ax.set_title(title, fontsize=16)
    ax.set_ylim(0, max(values) * 1.2 if values else 1)
    ax.set_ylabel("Value")

    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02 * max(values), f"{v:.2f}", ha="center")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    # Example synthetic data (could be replaced by real evaluation traces)
    # For retrieval relevances we use lists of 0/1 for top-10 retrieved items per query
    relevances = [
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    ]
    total_relevants = [2, 1, 2, 1]
    similarities = [0.92, 0.87, 0.90, 0.86]
    halluc_flags = [0, 0, 0, 1]
    latencies = [1.3, 1.1, 1.0, 1.4]
    context_counts = [4, 5, 3, 5]

    k = 5
    p_at_k = precision_at_k(relevances, k)
    r_at_k = recall_at_k(relevances, k, total_relevants)
    hit_at_k = hit_rate_at_k(relevances, k)
    mrr = mrr_at_k(relevances, k)
    sem_sim = mean_semantic_similarity(similarities)
    hall_rate = hallucination_rate(halluc_flags)
    avg_lat = avg_latency(latencies)
    avg_ctx = avg_context_used(context_counts)
    efficiency = rag_efficiency(p_at_k, avg_lat)

    # Prepare groups and plots
    retrieval_metrics = {
        "Precision@5": p_at_k,
        "Recall@5": r_at_k,
        "HitRate@5": hit_at_k,
        "MRR@5": mrr,
    }

    answer_metrics = {
        "SemanticSim": sem_sim,
        "HallucinationRate": hall_rate,
    }

    system_metrics = {
        "AvgLatency(s)": avg_lat,
        "AvgContextUsed(chunks)": avg_ctx,
        "RAG_Efficiency": efficiency,
    }

    out_dir = os.path.join("images", "evaluation_metrics")
    plot_bar(retrieval_metrics, "Retrieval Metrics", os.path.join(out_dir, "retrieval_metrics.png"))
    plot_bar(answer_metrics, "Answer Quality Metrics", os.path.join(out_dir, "answer_quality_metrics.png"))
    plot_bar(system_metrics, "System Performance Metrics", os.path.join(out_dir, "system_performance_metrics.png"))

    # Also write a simple summary to stdout
    print("Computed metrics summary:")
    print(f"Precision@5: {p_at_k:.2f}")
    print(f"Recall@5: {r_at_k:.2f}")
    print(f"HitRate@5: {hit_at_k:.2f}")
    print(f"MRR@5: {mrr:.2f}")
    print(f"Semantic Similarity: {sem_sim:.2f}")
    print(f"Hallucination Rate: {hall_rate:.2%}")
    print(f"Avg Latency: {avg_lat:.2f} sec")
    print(f"Avg Context Used: {avg_ctx:.1f} chunks")
    print(f"RAG Efficiency (Prec/Lat): {efficiency:.3f}")


if __name__ == "__main__":
    main()
