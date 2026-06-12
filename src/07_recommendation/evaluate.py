"""Stage 7e - Offline evaluation.

For every model and every k in {5, 10, 20}, computes Precision@k, Recall@k,
NDCG@k, plus catalog coverage and recommendation novelty. Ground truth is the
set of test reviews with rating >= 4 (positive interactions).

Inputs  : src/07_recommendation/outputs/data/test.parquet
          src/07_recommendation/outputs/data/predictions_*.parquet
          src/07_recommendation/outputs/data/candidates.parquet
          src/07_recommendation/outputs/data/train.parquet
Outputs : src/07_recommendation/outputs/data/metrics.csv
          src/07_recommendation/outputs/plots/metrics_bar.png
          src/07_recommendation/outputs/plots/score_distribution.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR  = Path("src/07_recommendation/outputs/data")
PLOT_DIR = Path("src/07_recommendation/outputs/plots")

KS                = [5, 10, 20]
POSITIVE_STAR     = 4
MODELS            = ["popularity", "content", "als"]


def metrics_for_model(preds: pd.DataFrame, gt: dict[str, set],
                      train_pop: dict[str, int],
                      k: int) -> dict:
    p_at_k: list[float] = []
    r_at_k: list[float] = []
    ndcg:   list[float] = []
    novelty: list[float] = []
    rec_items: set[str] = set()
    n_users_hit = 0

    log_total = np.log(max(1, len(train_pop) + 1))

    preds_k = preds[preds["rank"] <= k]
    by_user = preds_k.groupby("user_id")["business_id"]

    for user, items in by_user:
        relevant = gt.get(user, set())
        if not relevant:
            continue
        items_list = items.tolist()
        rec_items.update(items_list)
        hits = [1 if b in relevant else 0 for b in items_list]
        hit_count = sum(hits)

        p_at_k.append(hit_count / k)
        r_at_k.append(hit_count / len(relevant))

        # NDCG
        dcg  = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
        ndcg.append(dcg / idcg if idcg > 0 else 0.0)

        # Novelty: mean -log(pop / total) of recommended items
        nov = np.mean([
            -np.log((train_pop.get(b, 0) + 1)) + log_total
            for b in items_list
        ])
        novelty.append(float(nov))
        if hit_count > 0:
            n_users_hit += 1

    n_eval = len(p_at_k)
    return {
        "k": k,
        "users_evaluated": n_eval,
        "users_with_hit":  n_users_hit,
        "hit_rate":        round(n_users_hit / n_eval, 4) if n_eval else 0,
        "precision_at_k":  round(float(np.mean(p_at_k)), 4) if p_at_k else 0,
        "recall_at_k":     round(float(np.mean(r_at_k)), 4) if r_at_k else 0,
        "ndcg_at_k":       round(float(np.mean(ndcg)),   4) if ndcg else 0,
        "novelty_at_k":    round(float(np.mean(novelty)), 4) if novelty else 0,
        "coverage":        round(len(rec_items) / max(1, len(set(preds["business_id"]))), 4),
    }


def metrics_bar_plot(table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric, title in zip(axes,
                                  ["precision_at_k", "recall_at_k", "ndcg_at_k"],
                                  ["Precision@k", "Recall@k", "NDCG@k"]):
        sub = table.pivot(index="k", columns="model", values=metric)
        sub.plot(kind="bar", ax=ax, rot=0,
                 color=["#1F4E91", "#2E7D32", "#B71C1C"])
        ax.set_title(title); ax.set_xlabel("k"); ax.set_ylabel(title)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "metrics_bar.png"); plt.close(fig)


def score_distribution_plot(predictions: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"popularity": "#1F4E91", "content": "#2E7D32", "als": "#B71C1C"}
    for model, preds in predictions.items():
        s = preds.loc[preds["rank"] == 1, "score"].astype(float)
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)
        ax.hist(s, bins=40, alpha=0.5, label=f"{model} (top-1)",
                color=colors.get(model, "grey"))
    ax.set_xlabel("Top-1 score (normalised per model)")
    ax.set_ylabel("Number of users")
    ax.set_title("Recommendation confidence distribution per model")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(PLOT_DIR / "score_distribution.png"); plt.close(fig)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print("Stage 7e - Evaluation")

    test  = pd.read_parquet(OUT_DIR / "test.parquet")
    train = pd.read_parquet(OUT_DIR / "train.parquet")
    cand  = pd.read_parquet(OUT_DIR / "candidates.parquet")
    cand_ids = set(cand["business_id"])

    # Ground truth: positive test reviews restricted to the candidate pool
    pos = test[(test["review_stars"] >= POSITIVE_STAR)
               & (test["business_id"].isin(cand_ids))]
    gt: dict[str, set] = {u: set(g) for u, g in
                          pos.groupby("user_id")["business_id"]}
    print(f"  positive ground-truth pairs (rating >= {POSITIVE_STAR}, in pool): {len(pos):,}")
    print(f"  evaluation users with >=1 positive test pair: {len(gt):,}")

    train_pop = train.groupby("business_id").size().to_dict()

    predictions = {m: pd.read_parquet(OUT_DIR / f"predictions_{m}.parquet")
                   for m in MODELS}

    rows: list[dict] = []
    for model, preds in predictions.items():
        print(f"\n  evaluating {model}...")
        for k in KS:
            m = metrics_for_model(preds, gt, train_pop, k)
            m["model"] = model
            rows.append(m)
            print(f"    k={k:>2}: P={m['precision_at_k']:.4f}  R={m['recall_at_k']:.4f}  "
                  f"NDCG={m['ndcg_at_k']:.4f}  hit={m['hit_rate']:.4f}  "
                  f"cov={m['coverage']:.4f}")

    table = pd.DataFrame(rows)[[
        "model", "k", "users_evaluated", "users_with_hit", "hit_rate",
        "precision_at_k", "recall_at_k", "ndcg_at_k", "novelty_at_k", "coverage"]]
    table.to_csv(OUT_DIR / "metrics.csv", index=False)

    metrics_bar_plot(table)
    score_distribution_plot(predictions)
    print(f"\n  outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
