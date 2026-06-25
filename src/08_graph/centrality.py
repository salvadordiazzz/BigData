"""Stage 8b - Centrality metrics on the co-review graph.

Computes four complementary node-level scores:
    - degree                 : number of distinct co-reviewed neighbours
    - weighted_degree        : sum of edge weights (co-reviewer mass)
    - pagerank               : weighted PageRank (alpha = 0.85)
    - betweenness            : sample-approximated betweenness centrality
                               (full O(V*E) is intractable at this scale)

Inputs  : src/08_graph/outputs/data/graph.gpickle
          src/06_clustering/outputs/data/kmeans_labels.parquet
Outputs : src/08_graph/outputs/data/centrality.parquet
          src/08_graph/outputs/plots/centrality_correlation.png
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

GRAPH_DIR   = Path("src/08_graph/outputs/data")
CLUSTER_DIR = Path("src/06_clustering/outputs/data")
PLOT_DIR    = Path("src/08_graph/outputs/plots")

PAGERANK_ALPHA   = 0.85
BETWEEN_SAMPLE   = 500   # sample size for approximate betweenness
RS               = 42


def correlation_plot(df: pd.DataFrame) -> None:
    metrics = ["degree", "weighted_degree", "pagerank", "betweenness"]
    corr = df[metrics].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(metrics))); ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_yticks(range(len(metrics))); ax.set_yticklabels(metrics)
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if abs(corr.values[i, j]) > 0.5 else "black", fontsize=10)
    ax.set_title("Spearman rank correlation between centrality metrics")
    plt.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout(); fig.savefig(PLOT_DIR / "centrality_correlation.png"); plt.close(fig)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print("Stage 8b - Centrality")

    with open(GRAPH_DIR / "graph.gpickle", "rb") as f:
        G: nx.Graph = pickle.load(f)
    print(f"  graph loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print("  computing degree + weighted degree...")
    degree          = dict(G.degree())
    weighted_degree = dict(G.degree(weight="weight"))

    print(f"  computing weighted PageRank (alpha={PAGERANK_ALPHA})...")
    pagerank = nx.pagerank(G, alpha=PAGERANK_ALPHA, weight="weight")

    print(f"  computing approximate betweenness (k={BETWEEN_SAMPLE} sample)...")
    betweenness = nx.betweenness_centrality(
        G, k=BETWEEN_SAMPLE, normalized=True, weight=None, seed=RS)

    # Assemble per-node table with metadata
    clusters = pd.read_parquet(CLUSTER_DIR / "kmeans_labels.parquet")[
        ["business_id", "cluster_kmeans"]]
    rows = []
    for n in G.nodes():
        attrs = G.nodes[n]
        rows.append({
            "business_id":       n,
            "business_name":     attrs.get("business_name"),
            "primary_category":  attrs.get("primary_category"),
            "degree":            degree[n],
            "weighted_degree":   weighted_degree[n],
            "pagerank":          pagerank[n],
            "betweenness":       betweenness[n],
        })
    df = pd.DataFrame(rows).merge(clusters, on="business_id", how="left")
    df.to_parquet(GRAPH_DIR / "centrality.parquet", index=False)

    print("\n  top 10 by PageRank:")
    print(df.nlargest(10, "pagerank")[
        ["business_name", "primary_category", "cluster_kmeans",
         "degree", "pagerank"]].to_string(index=False))

    print("\n  top 10 by betweenness:")
    print(df.nlargest(10, "betweenness")[
        ["business_name", "primary_category", "cluster_kmeans",
         "degree", "betweenness"]].to_string(index=False))

    correlation_plot(df)
    print(f"\n  outputs -> {GRAPH_DIR}")


if __name__ == "__main__":
    main()
