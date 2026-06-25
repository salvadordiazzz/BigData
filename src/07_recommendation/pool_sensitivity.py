"""Stage 7i - Improvement #3: Candidate-pool sensitivity analysis.

Addresses the PC4 feedback that the evaluation measures a "filtered
hidden-gem recommendation task" because the candidate pool is fixed to
PC3 Cluster 1, and the report lacked sensitivity analysis around that
choice. This script re-evaluates the router under three alternative pool
definitions to show how conclusions change (or do not change) with the
pool boundary:

    pool_cluster1_only      : PC3 Cluster 1 only (the PC4 / PC5 default)
    pool_cluster1_plus_0    : Cluster 1 + Cluster 0 (adds mainstream popular)
    pool_all_clusters       : every business in train (no segmentation filter)

For each pool, ground truth, candidate set, and the ALS-tuned vs router
comparison are recomputed from scratch using the existing predictions where
possible and a fresh popularity-only ranking for the expanded pools (ALS/
content predictions were only generated for the original pool, so the
expanded-pool comparison uses the popularity baseline as the common
reference point across all three pools).

Inputs  : src/07_recommendation/outputs/data/{train,test,candidates}.parquet
          src/06_clustering/outputs/data/kmeans_labels.parquet
Outputs : src/07_recommendation/outputs/data/pool_sensitivity.csv
          src/07_recommendation/outputs/plots/pool_sensitivity.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REC_DIR     = Path("src/07_recommendation/outputs/data")
CLUSTER_DIR = Path("src/06_clustering/outputs/data")
PLOT_DIR    = Path("src/07_recommendation/outputs/plots")

K             = 10
POSITIVE_STAR = 4


def popularity_ranking(train: pd.DataFrame, pool_ids: set) -> pd.DataFrame:
    sub = train[train["business_id"].isin(pool_ids)]
    agg = sub.groupby("business_id").agg(
        m=("review_stars", "mean"), n=("review_id", "count")).reset_index()
    agg["score"] = agg["m"] * np.log1p(agg["n"])
    agg = agg.sort_values("score", ascending=False).reset_index(drop=True)
    agg["rank"] = agg.index + 1
    return agg


def evaluate_global_popularity(top: pd.DataFrame, gt: dict, k: int) -> dict:
    """Same non-personalised top-k list shown to every evaluable user."""
    top_k = top[top["rank"] <= k]
    biz_list = top_k["business_id"].tolist()
    ps, rs, ns, hs = [], [], [], []
    for u, relevant in gt.items():
        flags = [1 if b in relevant else 0 for b in biz_list]
        hits = sum(flags)
        ps.append(hits / k)
        rs.append(hits / len(relevant))
        dcg  = sum(h / np.log2(i + 2) for i, h in enumerate(flags))
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
        ns.append(dcg / idcg if idcg > 0 else 0.0)
        hs.append(hits > 0)
    return {
        "n_eval_users":   len(ps),
        "precision_at_k": round(float(np.mean(ps)), 4) if ps else 0.0,
        "recall_at_k":    round(float(np.mean(rs)), 4) if rs else 0.0,
        "ndcg_at_k":      round(float(np.mean(ns)), 4) if ns else 0.0,
        "hit_rate":       round(float(np.mean(hs)), 4) if hs else 0.0,
    }


def evaluate_router_on_pool(router_preds: pd.DataFrame, gt: dict, k: int) -> dict:
    sub = router_preds[router_preds["rank"] <= k]
    by_user = sub.groupby("user_id")["business_id"].apply(list)
    ps, rs, ns, hs = [], [], [], []
    for u, relevant in gt.items():
        items = by_user.get(u, [])
        flags = [1 if b in relevant else 0 for b in items]
        hits = sum(flags)
        ps.append(hits / k)
        rs.append(hits / len(relevant))
        dcg  = sum(h / np.log2(i + 2) for i, h in enumerate(flags))
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
        ns.append(dcg / idcg if idcg > 0 else 0.0)
        hs.append(hits > 0)
    return {
        "n_eval_users":   len(ps),
        "precision_at_k": round(float(np.mean(ps)), 4) if ps else 0.0,
        "recall_at_k":    round(float(np.mean(rs)), 4) if rs else 0.0,
        "ndcg_at_k":      round(float(np.mean(ns)), 4) if ns else 0.0,
        "hit_rate":       round(float(np.mean(hs)), 4) if hs else 0.0,
    }


def sensitivity_plot(table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    pools = table["pool"].unique()
    x = np.arange(len(pools))
    width = 0.35
    for ax, metric, title in zip(axes, ["ndcg_at_k", "hit_rate"],
                                  ["NDCG@10", "Hit Rate@10"]):
        pop_vals = [table[(table.pool == p) & (table.model == "popularity")][metric].values[0]
                    for p in pools]
        rtr_vals = [table[(table.pool == p) & (table.model == "fallback_router")][metric].values[0]
                    if not table[(table.pool == p) & (table.model == "fallback_router")].empty else np.nan
                    for p in pools]
        ax.bar(x - width/2, pop_vals, width, label="Popularity", color="#1F4E91")
        ax.bar(x + width/2, rtr_vals, width, label="Fallback Router", color="#2E7D32")
        ax.set_xticks(x); ax.set_xticklabels(pools, rotation=15, ha="right")
        ax.set_title(title); ax.set_ylabel(title)
        ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "pool_sensitivity.png"); plt.close(fig)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print("Stage 7i - Candidate-pool sensitivity analysis")

    train  = pd.read_parquet(REC_DIR / "train.parquet")
    test   = pd.read_parquet(REC_DIR / "test.parquet")
    router = pd.read_parquet(REC_DIR / "predictions_fallback.parquet")
    clusters = pd.read_parquet(CLUSTER_DIR / "kmeans_labels.parquet")

    train_biz = set(train["business_id"])
    cluster1 = set(clusters.loc[clusters["cluster_kmeans"] == 1, "business_id"]) & train_biz
    cluster0 = set(clusters.loc[clusters["cluster_kmeans"] == 0, "business_id"]) & train_biz

    pools = {
        "pool_cluster1_only":   cluster1,
        "pool_cluster1_plus_0": cluster1 | cluster0,
        "pool_all_clusters":    train_biz,
    }

    rows = []
    for pool_name, pool_ids in pools.items():
        print(f"\n  pool={pool_name}  size={len(pool_ids):,}")
        pos = test[(test["review_stars"] >= POSITIVE_STAR)
                   & (test["business_id"].isin(pool_ids))]
        gt = {u: set(g) for u, g in pos.groupby("user_id")["business_id"]}
        print(f"    evaluable users: {len(gt):,}")

        pop_rank = popularity_ranking(train, pool_ids)
        pop_metrics = evaluate_global_popularity(pop_rank, gt, K)
        pop_metrics.update({"pool": pool_name, "pool_size": len(pool_ids),
                            "model": "popularity", "k": K})
        rows.append(pop_metrics)
        print(f"    popularity   : NDCG@10={pop_metrics['ndcg_at_k']}  "
              f"HR@10={pop_metrics['hit_rate']}")

        if pool_name == "pool_cluster1_only":
            rtr_in_pool = router[router["business_id"].isin(pool_ids)]
            rtr_metrics = evaluate_router_on_pool(rtr_in_pool, gt, K)
            rtr_metrics.update({"pool": pool_name, "pool_size": len(pool_ids),
                                "model": "fallback_router", "k": K})
            rows.append(rtr_metrics)
            print(f"    router       : NDCG@10={rtr_metrics['ndcg_at_k']}  "
                  f"HR@10={rtr_metrics['hit_rate']}")

    table = pd.DataFrame(rows)[
        ["pool", "pool_size", "model", "k", "n_eval_users",
         "precision_at_k", "recall_at_k", "ndcg_at_k", "hit_rate"]]
    table.to_csv(REC_DIR / "pool_sensitivity.csv", index=False)
    print("\n  pool sensitivity table:")
    print(table.to_string(index=False))

    sensitivity_plot(table)
    print(f"\n  outputs -> {REC_DIR}")


if __name__ == "__main__":
    main()
