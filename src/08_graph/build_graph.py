"""Stage 8a - Build the business-business co-review graph.

Construction:
    - Bipartite source: user-business train interactions from PC4 split.
    - Project to a business-business graph where two businesses share an
      edge iff at least MIN_SHARED_USERS distinct users reviewed both.
    - Edge weight = number of shared reviewing users (raw co-occurrence).
    - Undirected graph (co-review is symmetric).

Nodes:        11,442 businesses (those appearing in train).
Edges:        sparse co-review pairs surviving the threshold.

A higher MIN_SHARED_USERS keeps only meaningful connections and prevents
the projection from collapsing into a dense hairball. Tested values:
    1 -> ~7M edges, untractable for visualisation and Louvain.
    3 -> ~600k edges, computationally fine, semantically tight.
    5 -> ~300k edges, conservative.

Inputs  : src/07_recommendation/outputs/data/train.parquet
          src/04_features/outputs/business_index.parquet
Outputs : src/08_graph/outputs/data/graph.gpickle
          src/08_graph/outputs/data/graph_stats.json
          src/08_graph/outputs/data/edge_list.parquet
          src/08_graph/outputs/plots/degree_distribution.png
          src/08_graph/outputs/plots/edge_weight_distribution.png
"""
from __future__ import annotations

import json
import pickle
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

REC_DIR  = Path("src/07_recommendation/outputs/data")
FEAT_DIR = Path("src/04_features/outputs")
OUT_DIR  = Path("src/08_graph/outputs/data")
PLOT_DIR = Path("src/08_graph/outputs/plots")

MIN_SHARED_USERS = 3
POSITIVE_STAR    = 4
MAX_BIZ_PER_USER = 200  # cap super-users to avoid quadratic blow-up


def build_edges(train: pd.DataFrame) -> Counter:
    """Project bipartite -> business-business via shared positive reviewers."""
    train = train[train["review_stars"] >= POSITIVE_STAR]
    user_biz = train.groupby("user_id")["business_id"].apply(list)

    edges: Counter = Counter()
    for user, businesses in user_biz.items():
        if len(businesses) < 2:
            continue
        if len(businesses) > MAX_BIZ_PER_USER:
            businesses = businesses[:MAX_BIZ_PER_USER]
        for a, b in combinations(sorted(set(businesses)), 2):
            edges[(a, b)] += 1
    return edges


def filter_and_build_graph(edges: Counter, min_w: int) -> nx.Graph:
    G = nx.Graph()
    kept = 0
    for (a, b), w in edges.items():
        if w >= min_w:
            G.add_edge(a, b, weight=w)
            kept += 1
    return G


def degree_plot(G: nx.Graph) -> None:
    degrees = [d for _, d in G.degree()]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(degrees, bins=80, color="steelblue", edgecolor="white")
    ax.set_yscale("log")
    ax.set_xlabel("Node degree")
    ax.set_ylabel("Frequency (log)")
    ax.set_title(f"Business-business co-review graph - degree distribution (n={G.number_of_nodes():,})")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(PLOT_DIR / "degree_distribution.png"); plt.close(fig)


def weight_plot(G: nx.Graph) -> None:
    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(weights, bins=50, color="seagreen", edgecolor="white")
    ax.set_yscale("log")
    ax.set_xlabel("Edge weight (shared positive reviewers)")
    ax.set_ylabel("Frequency (log)")
    ax.set_title(f"Edge weight distribution (m={G.number_of_edges():,})")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(PLOT_DIR / "edge_weight_distribution.png"); plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True); PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print("Stage 8a - Build business-business co-review graph")

    train = pd.read_parquet(REC_DIR / "train.parquet")
    index = pd.read_parquet(FEAT_DIR / "business_index.parquet")
    print(f"  train reviews: {len(train):,}")

    print(f"  projecting bipartite -> business-business (positive ratings >= {POSITIVE_STAR})...")
    raw_edges = build_edges(train)
    print(f"  raw projected edges (any weight): {len(raw_edges):,}")

    print(f"  filtering edges with weight >= {MIN_SHARED_USERS}...")
    G = filter_and_build_graph(raw_edges, MIN_SHARED_USERS)
    print(f"  filtered graph: {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges")

    # Add business metadata as node attributes
    bid_to_name = dict(zip(index["business_id"], index["business_name"]))
    bid_to_cat  = dict(zip(index["business_id"], index["primary_category"]))
    for n in G.nodes():
        G.nodes[n]["business_name"]     = bid_to_name.get(n, "?")
        G.nodes[n]["primary_category"]  = bid_to_cat.get(n, "?")

    # Connected components
    components = list(nx.connected_components(G))
    components.sort(key=len, reverse=True)
    largest = components[0] if components else set()
    print(f"  connected components: {len(components)}  "
          f"(largest = {len(largest):,} nodes = {len(largest)/max(1,G.number_of_nodes()):.1%})")

    degrees = dict(G.degree())
    weighted_degrees = dict(G.degree(weight="weight"))

    stats = {
        "n_nodes":              G.number_of_nodes(),
        "n_edges":              G.number_of_edges(),
        "density":              round(nx.density(G), 5),
        "n_components":         len(components),
        "largest_cc_size":      len(largest),
        "largest_cc_fraction":  round(len(largest)/max(1, G.number_of_nodes()), 4),
        "mean_degree":          round(float(np.mean(list(degrees.values()))), 2) if degrees else 0,
        "median_degree":        int(np.median(list(degrees.values()))) if degrees else 0,
        "max_degree":           int(np.max(list(degrees.values()))) if degrees else 0,
        "mean_edge_weight":     round(float(np.mean([d["weight"] for _,_,d in G.edges(data=True)])), 2),
        "max_edge_weight":      int(np.max([d["weight"] for _,_,d in G.edges(data=True)])),
        "min_shared_users":     MIN_SHARED_USERS,
        "positive_star_threshold": POSITIVE_STAR,
        "max_biz_per_user":     MAX_BIZ_PER_USER,
    }
    with open(OUT_DIR / "graph_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  graph stats: density={stats['density']}  "
          f"mean_degree={stats['mean_degree']}  max_degree={stats['max_degree']}")

    # Persist graph (pickle) and edge list (parquet for portability)
    with open(OUT_DIR / "graph.gpickle", "wb") as f:
        pickle.dump(G, f)
    edge_records = [{"source": u, "target": v, "weight": d["weight"]}
                    for u, v, d in G.edges(data=True)]
    pd.DataFrame(edge_records).to_parquet(OUT_DIR / "edge_list.parquet", index=False)

    degree_plot(G); weight_plot(G)
    print(f"  outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
