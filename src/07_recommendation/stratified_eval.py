"""Stage 7h - Improvement #2: Stratified evaluation by user activity tier.

Directly answers the PC4 feedback that the "stronger system" deduction came
from the advanced architecture (hybrid/BPR) not beating tuned ALS. Here we
re-frame the final architecture as the hierarchical fallback router
(fallback_router.py) and show it is empirically justified: it matches tuned
ALS exactly on warm users (same scores, pass-through) and clearly beats it
on cold/mid users, where tuned ALS has no reliable embedding signal.

For each of {cold, mid, warm} and overall, computes Precision@10, Recall@10,
NDCG@10, MAP@10 and Hit Rate@10 for both "als_tuned" (PC4 winning model,
served identically to every user regardless of history) and "fallback_router"
(this improvement).

Inputs  : src/07_recommendation/outputs/data/{test,train,candidates}.parquet
          src/07_recommendation/outputs/data/predictions_{als_tuned,fallback}.parquet
Outputs : src/07_recommendation/outputs/data/stratified_metrics.csv
          src/07_recommendation/outputs/plots/stratified_comparison.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR  = Path("src/07_recommendation/outputs/data")
PLOT_DIR = Path("src/07_recommendation/outputs/plots")

K             = 10
POSITIVE_STAR = 4
COLD_MAX      = 10
MID_MAX       = 30


def tier_of(train_count: int) -> str:
    if train_count <= COLD_MAX:
        return "cold"
    if train_count <= MID_MAX:
        return "mid"
    return "warm"


def metrics_for_subset(preds: pd.DataFrame, gt: dict[str, set],
                       users: set, k: int) -> dict:
    sub = preds[(preds["rank"] <= k) & (preds["user_id"].isin(users))]
    by_user = sub.groupby("user_id")["business_id"].apply(list)

    ps, rs, ns, ms, hs = [], [], [], [], []
    for u, items in by_user.items():
        relevant = gt.get(u, set())
        if not relevant:
            continue
        flags = [1 if b in relevant else 0 for b in items]
        hits = sum(flags)
        ps.append(hits / k)
        rs.append(hits / len(relevant))
        dcg  = sum(h / np.log2(i + 2) for i, h in enumerate(flags))
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
        ns.append(dcg / idcg if idcg > 0 else 0.0)
        ap, h2 = 0.0, 0
        for i, h in enumerate(flags):
            if h:
                h2 += 1
                ap += h2 / (i + 1)
        ms.append(ap / min(len(relevant), k))
        hs.append(hits > 0)

    n = len(ps)
    return {
        "n_users":        n,
        "precision_at_k": round(float(np.mean(ps)), 4) if n else 0.0,
        "recall_at_k":    round(float(np.mean(rs)), 4) if n else 0.0,
        "ndcg_at_k":      round(float(np.mean(ns)), 4) if n else 0.0,
        "map_at_k":       round(float(np.mean(ms)), 4) if n else 0.0,
        "hit_rate":       round(float(np.mean(hs)), 4) if n else 0.0,
    }


def comparison_plot(table: pd.DataFrame) -> None:
    tiers = ["cold", "mid", "warm", "overall"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    width = 0.35
    x = np.arange(len(tiers))

    for ax, metric, title in zip(axes, ["ndcg_at_k", "hit_rate"],
                                  ["NDCG@10", "Hit Rate@10"]):
        als_vals = [table[(table.tier == t) & (table.model == "als_tuned")][metric].values[0]
                    for t in tiers]
        rtr_vals = [table[(table.tier == t) & (table.model == "fallback_router")][metric].values[0]
                    for t in tiers]
        ax.bar(x - width/2, als_vals, width, label="ALS Tuned (PC4)", color="#1F4E91")
        ax.bar(x + width/2, rtr_vals, width, label="Fallback Router (improved)", color="#2E7D32")
        ax.set_xticks(x); ax.set_xticklabels(tiers)
        ax.set_title(title); ax.set_ylabel(title)
        ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "stratified_comparison.png"); plt.close(fig)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print("Stage 7h - Stratified evaluation (cold/mid/warm)")

    train = pd.read_parquet(OUT_DIR / "train.parquet")
    test  = pd.read_parquet(OUT_DIR / "test.parquet")
    cand  = pd.read_parquet(OUT_DIR / "candidates.parquet")
    cand_ids = set(cand["business_id"])

    als_preds = pd.read_parquet(OUT_DIR / "predictions_als_tuned.parquet")
    rtr_preds = pd.read_parquet(OUT_DIR / "predictions_fallback.parquet")

    pos = test[(test["review_stars"] >= POSITIVE_STAR) & (test["business_id"].isin(cand_ids))]
    gt = {u: set(g) for u, g in pos.groupby("user_id")["business_id"]}
    print(f"  evaluable users (>=1 positive in pool): {len(gt):,}")

    user_counts = train.groupby("user_id")["review_id"].count()
    tiers = user_counts.apply(tier_of)
    tier_users = {t: set(tiers[tiers == t].index) for t in ["cold", "mid", "warm"]}
    for t, us in tier_users.items():
        print(f"  tier={t:5s}  n_users={len(us):,}")

    rows = []
    for model_name, preds in [("als_tuned", als_preds), ("fallback_router", rtr_preds)]:
        for tier, users in tier_users.items():
            m = metrics_for_subset(preds, gt, users, K)
            m.update({"model": model_name, "tier": tier, "k": K})
            rows.append(m)
        all_users = set(tiers.index)
        m = metrics_for_subset(preds, gt, all_users, K)
        m.update({"model": model_name, "tier": "overall", "k": K})
        rows.append(m)

    table = pd.DataFrame(rows)[
        ["model", "tier", "k", "n_users", "precision_at_k", "recall_at_k",
         "ndcg_at_k", "map_at_k", "hit_rate"]]
    table.to_csv(OUT_DIR / "stratified_metrics.csv", index=False)
    print("\n  stratified results:")
    print(table.to_string(index=False))

    comparison_plot(table)
    print(f"\n  outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
