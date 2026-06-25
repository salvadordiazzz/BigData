"""Stage 8d - Compare graph-based ranking against popularity and the
recommendation system's final architecture.

Three rankings are compared, all restricted to the hidden-gem candidate pool
from PC4 (3,572 businesses present in train):

    R_graph  : businesses ordered by weighted PageRank
    R_pop    : businesses ordered by the PC4 popularity-quality score
    R_router : businesses ordered by mean score from the confidence-weighted
               blend router (predictions_fallback.parquet) - the final
               recommendation architecture after the PC4 Improvements
               addendum (see deliverables/docs/Big Data TF - PC4
               Improvements.docx). The router beats plain tuned ALS on every
               user-activity tier, so it - not the original baseline ALS -
               is the correct comparison point for graph analytics.

Comparison:
    - Spearman / Kendall rank correlations between the three orderings.
    - Top-50 overlap between rankings (Jaccard).
    - Per-cluster representation in each top-50.

Inputs  : src/08_graph/outputs/data/centrality.parquet
          src/07_recommendation/outputs/data/candidates.parquet
          src/07_recommendation/outputs/data/predictions_fallback.parquet
          src/07_recommendation/outputs/data/train.parquet
Outputs : src/08_graph/outputs/data/ranking_comparison.csv
          src/08_graph/outputs/data/ranking_topk_overlap.csv
          src/08_graph/outputs/plots/ranking_scatter.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

GRAPH_DIR = Path("src/08_graph/outputs/data")
REC_DIR   = Path("src/07_recommendation/outputs/data")
PLOT_DIR  = Path("src/08_graph/outputs/plots")

TOP_K = 50


def build_router_ranking(router_path: Path, cand_ids: set) -> pd.Series:
    """Mean router score across users, restricted to the candidate pool."""
    router = pd.read_parquet(router_path)
    router = router[router["business_id"].isin(cand_ids)]
    return router.groupby("business_id")["score"].mean()


def build_pop_ranking(train: pd.DataFrame, cand_ids: set) -> pd.Series:
    sub = train[train["business_id"].isin(cand_ids)]
    agg = sub.groupby("business_id").agg(
        m=("review_stars", "mean"), n=("review_id", "count"))
    return (agg["m"] * np.log1p(agg["n"])).rename("pop_score")


def scatter_plot(df: pd.DataFrame) -> None:
    pairs = [("pagerank", "pop_score", "PageRank vs Popularity"),
             ("pagerank", "router_score", "PageRank vs Router"),
             ("pop_score", "router_score", "Popularity vs Router")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (x, y, title) in zip(axes, pairs):
        ax.scatter(df[x].rank(), df[y].rank(), s=6, alpha=0.4, c="steelblue")
        ax.set_xlabel(f"{x} rank"); ax.set_ylabel(f"{y} rank")
        ax.set_title(title); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(PLOT_DIR / "ranking_scatter.png"); plt.close(fig)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print("Stage 8d - Compare graph vs popularity vs router rankings")

    centrality = pd.read_parquet(GRAPH_DIR / "centrality.parquet")
    candidates = pd.read_parquet(REC_DIR / "candidates.parquet")
    train      = pd.read_parquet(REC_DIR / "train.parquet")
    cand_ids   = set(candidates["business_id"])

    # Graph-based ranking restricted to candidate pool
    graph_in_pool = centrality[centrality["business_id"].isin(cand_ids)].copy()
    print(f"  candidate pool: {len(cand_ids):,}  in graph: {len(graph_in_pool):,}")

    pop    = build_pop_ranking(train, cand_ids)
    router = build_router_ranking(REC_DIR / "predictions_fallback.parquet", cand_ids)

    df = (graph_in_pool[["business_id", "business_name", "primary_category",
                         "cluster_kmeans", "pagerank", "betweenness", "degree"]]
          .merge(pop.rename("pop_score").reset_index(), on="business_id", how="inner")
          .merge(router.rename("router_score").reset_index(), on="business_id", how="inner"))
    print(f"  jointly ranked: {len(df):,} businesses")

    rho_pp, _   = spearmanr(df["pagerank"], df["pop_score"])
    rho_pr, _   = spearmanr(df["pagerank"], df["router_score"])
    rho_popr, _ = spearmanr(df["pop_score"], df["router_score"])
    tau_pp,  _  = kendalltau(df["pagerank"], df["pop_score"])
    tau_pr,  _  = kendalltau(df["pagerank"], df["router_score"])

    summary = pd.DataFrame([
        {"pair": "PageRank vs Popularity", "spearman": round(rho_pp, 4),   "kendall": round(tau_pp, 4)},
        {"pair": "PageRank vs Router",     "spearman": round(rho_pr, 4),   "kendall": round(tau_pr, 4)},
        {"pair": "Popularity vs Router",   "spearman": round(rho_popr, 4), "kendall": None},
    ])
    summary.to_csv(GRAPH_DIR / "ranking_comparison.csv", index=False)
    print("\n  rank correlations:")
    print(summary.to_string(index=False))

    # Top-K overlap (Jaccard) between rankings
    top_graph  = set(df.nlargest(TOP_K, "pagerank")["business_id"])
    top_pop    = set(df.nlargest(TOP_K, "pop_score")["business_id"])
    top_router = set(df.nlargest(TOP_K, "router_score")["business_id"])

    def jacc(a, b):
        return round(len(a & b) / max(1, len(a | b)), 4)

    overlap = pd.DataFrame([
        {"pair": "Top-50 PageRank vs Popularity", "intersection": len(top_graph & top_pop),
         "jaccard": jacc(top_graph, top_pop)},
        {"pair": "Top-50 PageRank vs Router",     "intersection": len(top_graph & top_router),
         "jaccard": jacc(top_graph, top_router)},
        {"pair": "Top-50 Popularity vs Router",   "intersection": len(top_pop & top_router),
         "jaccard": jacc(top_pop, top_router)},
    ])
    overlap.to_csv(GRAPH_DIR / "ranking_topk_overlap.csv", index=False)
    print("\n  top-50 overlaps:")
    print(overlap.to_string(index=False))

    # Per-cluster representation in each top-50
    def cluster_breakdown(name, biz_set):
        sub = df[df["business_id"].isin(biz_set)]
        return dict(sub["cluster_kmeans"].value_counts().sort_index())

    print("\n  cluster representation in top-50 (each ranking):")
    print(f"   PageRank top-50:   {cluster_breakdown('graph', top_graph)}")
    print(f"   Popularity top-50: {cluster_breakdown('pop', top_pop)}")
    print(f"   Router top-50:     {cluster_breakdown('router', top_router)}")

    scatter_plot(df)
    print(f"\n  outputs -> {GRAPH_DIR}")


if __name__ == "__main__":
    main()
