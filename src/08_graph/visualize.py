"""Stage 8e - Graph visualisations.

Two figures:
    1. Top-200 nodes by weighted PageRank, force-directed layout, coloured
       by the PC3 K-Means cluster. Shows how mainstream popular venues
       form the dense core.
    2. The 'hidden-gem community' (Louvain community where Cluster 1
       dominates) as its own ego layout, coloured by primary category.

Inputs  : src/08_graph/outputs/data/graph.gpickle
          src/08_graph/outputs/data/centrality.parquet
          src/08_graph/outputs/data/community_labels.parquet
Outputs : src/08_graph/outputs/plots/graph_top200_pagerank.png
          src/08_graph/outputs/plots/hidden_gem_community.png
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

GRAPH_DIR = Path("src/08_graph/outputs/data")
PLOT_DIR  = Path("src/08_graph/outputs/plots")

TOP_N         = 200
LAYOUT_ITERS  = 60
RS            = 42

CLUSTER_COLORS = {
    0: "#1F4E91",   # mainstream
    1: "#2E7D32",   # hidden gems
    2: "#888888",   # declining
    3: "#B71C1C",   # low quality
    4: "#FFB300",   # outlier micro
}
CLUSTER_LABELS = {
    0: "Mainstream",
    1: "Hidden gems",
    2: "Declining",
    3: "Low quality",
    4: "Outlier",
}


def top_pagerank_plot(G: nx.Graph, centrality: pd.DataFrame) -> None:
    top = centrality.nlargest(TOP_N, "pagerank")
    sub_nodes = set(top["business_id"])
    H = G.subgraph(sub_nodes).copy()
    print(f"  top-{TOP_N} PageRank subgraph: {H.number_of_nodes()} nodes, "
          f"{H.number_of_edges()} edges")

    pos = nx.spring_layout(H, k=0.15, iterations=LAYOUT_ITERS, seed=RS, weight="weight")

    bid_to_cluster = dict(zip(centrality["business_id"], centrality["cluster_kmeans"]))
    bid_to_pr      = dict(zip(centrality["business_id"], centrality["pagerank"]))

    fig, ax = plt.subplots(figsize=(12, 9))
    nx.draw_networkx_edges(H, pos, alpha=0.06, edge_color="grey", ax=ax)

    drawn = set()
    for cid, colour in CLUSTER_COLORS.items():
        nodes = [n for n in H.nodes() if bid_to_cluster.get(n) == cid]
        if not nodes:
            continue
        sizes = [3000 * bid_to_pr.get(n, 0) for n in nodes]
        nx.draw_networkx_nodes(H, pos, nodelist=nodes,
                               node_color=colour, node_size=sizes,
                               alpha=0.8, linewidths=0.5, edgecolors="white",
                               label=CLUSTER_LABELS[cid], ax=ax)
        drawn.update(nodes)

    # Annotate top-5 PageRank labels
    top5 = centrality.nlargest(5, "pagerank")
    for _, row in top5.iterrows():
        if row["business_id"] in pos:
            x, y = pos[row["business_id"]]
            ax.annotate(row["business_name"], (x, y),
                        fontsize=8, ha="center", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="white", alpha=0.7,
                                  edgecolor="none"))

    ax.set_title(f"Top-{TOP_N} businesses by weighted PageRank "
                 f"(node size proportional to PageRank, colour = PC3 cluster)")
    ax.legend(loc="upper left", scatterpoints=1, fontsize=10)
    ax.set_axis_off()
    fig.tight_layout(); fig.savefig(PLOT_DIR / "graph_top200_pagerank.png", dpi=130)
    plt.close(fig)


def hidden_gem_community_plot(G: nx.Graph, centrality: pd.DataFrame,
                              communities: pd.DataFrame) -> None:
    # Find the community where Cluster 1 dominates
    by_comm = communities.groupby("community").apply(
        lambda d: (d["cluster_kmeans"] == 1).mean(), include_groups=False
    ).sort_values(ascending=False)
    target_comm = int(by_comm.index[0])
    print(f"  hidden-gem community: id={target_comm}  "
          f"hidden-gem fraction={by_comm.iloc[0]:.2%}")

    sub_ids = communities.loc[communities["community"] == target_comm, "business_id"]
    H = G.subgraph(set(sub_ids)).copy()

    # Sample for plotting clarity if too large
    if H.number_of_nodes() > 300:
        # Keep top 300 by within-community degree
        deg_in_comm = sorted(H.degree(), key=lambda x: x[1], reverse=True)[:300]
        H = H.subgraph([n for n, _ in deg_in_comm]).copy()
    print(f"  visualising subgraph: {H.number_of_nodes()} nodes, "
          f"{H.number_of_edges()} edges")

    pos = nx.spring_layout(H, k=0.18, iterations=LAYOUT_ITERS, seed=RS, weight="weight")

    bid_to_cat = dict(zip(centrality["business_id"], centrality["primary_category"]))
    top_cats = (pd.Series([bid_to_cat.get(n, "?") for n in H.nodes()])
                .value_counts().head(8).index.tolist())
    cmap = plt.cm.tab10.colors  # type: ignore[attr-defined]
    cat_colour = {c: cmap[i] for i, c in enumerate(top_cats)}

    fig, ax = plt.subplots(figsize=(12, 9))
    nx.draw_networkx_edges(H, pos, alpha=0.08, edge_color="grey", ax=ax)

    for cat in top_cats + ["Other"]:
        nodes = [n for n in H.nodes()
                 if (bid_to_cat.get(n, "?") == cat
                     or (cat == "Other" and bid_to_cat.get(n, "?") not in top_cats))]
        if not nodes:
            continue
        colour = cat_colour.get(cat, "#cccccc")
        nx.draw_networkx_nodes(H, pos, nodelist=nodes,
                               node_color=[colour], node_size=60,
                               alpha=0.85, linewidths=0.5, edgecolors="white",
                               label=cat, ax=ax)

    ax.set_title(f"Hidden-gem Louvain community (id={target_comm}) "
                 f"- node colour = primary category")
    ax.legend(loc="upper left", scatterpoints=1, fontsize=9, ncol=2)
    ax.set_axis_off()
    fig.tight_layout(); fig.savefig(PLOT_DIR / "hidden_gem_community.png", dpi=130)
    plt.close(fig)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print("Stage 8e - Graph visualisations")

    with open(GRAPH_DIR / "graph.gpickle", "rb") as f:
        G: nx.Graph = pickle.load(f)
    centrality  = pd.read_parquet(GRAPH_DIR / "centrality.parquet")
    communities = pd.read_parquet(GRAPH_DIR / "community_labels.parquet")
    print(f"  graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    top_pagerank_plot(G, centrality)
    hidden_gem_community_plot(G, centrality, communities)
    print(f"  outputs -> {PLOT_DIR}")


if __name__ == "__main__":
    main()
