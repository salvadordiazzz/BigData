"""Stage 8c - Community detection.

Runs the Louvain algorithm on the co-review graph and characterises the
top communities by size, dominant category, and overlap with the PC3
clustering segments.

Inputs  : src/08_graph/outputs/data/graph.gpickle
          src/08_graph/outputs/data/centrality.parquet
Outputs : src/08_graph/outputs/data/community_labels.parquet
          src/08_graph/outputs/data/community_profiles.csv
          src/08_graph/outputs/plots/community_sizes.png
"""
from __future__ import annotations

import pickle
from pathlib import Path

import community as community_louvain   # python-louvain package
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

GRAPH_DIR = Path("src/08_graph/outputs/data")
PLOT_DIR  = Path("src/08_graph/outputs/plots")
RS = 42
TOP_N_COMMUNITIES = 10


def community_size_plot(sizes: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sizes.head(20).plot(kind="bar", ax=ax, color="steelblue")
    ax.set_xlabel("Community id"); ax.set_ylabel("Number of businesses")
    ax.set_title("Louvain communities - size of top 20")
    plt.setp(ax.get_xticklabels(), rotation=0)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "community_sizes.png"); plt.close(fig)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print("Stage 8c - Community detection (Louvain)")

    with open(GRAPH_DIR / "graph.gpickle", "rb") as f:
        G: nx.Graph = pickle.load(f)
    centrality = pd.read_parquet(GRAPH_DIR / "centrality.parquet")
    print(f"  graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print("  running Louvain (modularity optimisation)...")
    partition = community_louvain.best_partition(G, weight="weight", random_state=RS)
    modularity = community_louvain.modularity(partition, G, weight="weight")
    print(f"  modularity = {modularity:.4f}")

    labels = pd.DataFrame([
        {"business_id": n, "community": cid} for n, cid in partition.items()
    ])
    full = centrality.merge(labels, on="business_id", how="left")
    full.to_parquet(GRAPH_DIR / "community_labels.parquet", index=False)

    sizes = full["community"].value_counts().sort_values(ascending=False)
    print(f"  total communities: {len(sizes)}")
    print(f"  top 5 community sizes: {list(sizes.head(5).values)}")

    # Profile top communities
    rows = []
    for cid in sizes.head(TOP_N_COMMUNITIES).index:
        sub = full[full["community"] == cid]
        top_cats = sub["primary_category"].value_counts().head(3)
        top_cluster = sub["cluster_kmeans"].value_counts().head(1)
        rows.append({
            "community":           int(cid),
            "n_businesses":        len(sub),
            "mean_pagerank":       round(float(sub["pagerank"].mean()), 6),
            "mean_degree":         round(float(sub["degree"].mean()), 2),
            "dominant_cluster":    int(top_cluster.index[0]) if len(top_cluster) else -1,
            "cluster_dominance":   round(top_cluster.values[0] / len(sub), 3) if len(top_cluster) else 0,
            "top_categories":      "; ".join(f"{c} ({n})" for c, n in top_cats.items()),
        })
    profile = pd.DataFrame(rows)
    profile.to_csv(GRAPH_DIR / "community_profiles.csv", index=False)
    print("\n  top community profiles:")
    print(profile.to_string(index=False))

    # Cross-tab of graph community vs PC3 K-Means cluster
    crosstab = pd.crosstab(full["community"], full["cluster_kmeans"])
    top_communities = sizes.head(TOP_N_COMMUNITIES).index
    print("\n  community x PC3-cluster crosstab (top communities only):")
    print(crosstab.loc[top_communities].to_string())

    community_size_plot(sizes)
    print(f"\n  outputs -> {GRAPH_DIR}")


if __name__ == "__main__":
    main()
