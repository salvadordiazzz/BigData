"""Stage 7f - Error analysis.

Per model, characterises:
  - "Strong" users:   P@10 >= 0.20 (multiple correct hits in top-10)
  - "Failure" users:  P@10 == 0 and >=1 positive test pair existed
Identifies what distinguishes them (train history size, mean rating,
catalog overlap).

Also produces qualitative examples: for the ALS model, lists 3 strong
cases (user, recommended business names, test ground truth) and 3 failures.

Inputs  : predictions_*.parquet, test.parquet, train.parquet,
          candidates.parquet, business_features.parquet
Outputs : src/07_recommendation/outputs/data/error_summary.csv
          src/07_recommendation/outputs/data/error_examples.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR  = Path("src/07_recommendation/outputs/data")
FEAT_DIR = Path("src/04_features/outputs")
K        = 10
POSITIVE_STAR = 4
MODELS   = ["popularity", "content", "als"]


def user_p_at_k(preds: pd.DataFrame, gt: dict[str, set],
                k: int) -> pd.DataFrame:
    sub = preds[preds["rank"] <= k]
    by_user = sub.groupby("user_id")["business_id"].apply(list)
    rows = []
    for u, items in by_user.items():
        relevant = gt.get(u, set())
        if not relevant:
            continue
        hits = sum(1 for b in items if b in relevant)
        rows.append({"user_id": u, "p_at_k": hits / k,
                     "n_relevant": len(relevant), "n_hits": hits})
    return pd.DataFrame(rows)


def summarise_segment(seg: pd.DataFrame, stats: pd.DataFrame, label: str) -> dict:
    merged = seg.merge(stats, on="user_id", how="left")
    return {
        "segment":           label,
        "n_users":           len(seg),
        "mean_p_at_k":       round(float(seg["p_at_k"].mean()), 4),
        "mean_n_relevant":   round(float(seg["n_relevant"].mean()), 2),
        "mean_train_history": round(float(merged["train_count"].mean()), 2),
        "mean_train_rating":  round(float(merged["mean_train_rating"].mean()), 3),
    }


def main() -> None:
    print("Stage 7f - Error analysis")

    train  = pd.read_parquet(OUT_DIR / "train.parquet")
    test   = pd.read_parquet(OUT_DIR / "test.parquet")
    cand   = pd.read_parquet(OUT_DIR / "candidates.parquet")
    feats  = pd.read_parquet(FEAT_DIR / "business_features.parquet")[
        ["business_id", "business_name", "primary_category"]]
    cand_ids = set(cand["business_id"])

    pos = test[(test["review_stars"] >= POSITIVE_STAR)
               & (test["business_id"].isin(cand_ids))]
    gt = {u: set(g) for u, g in pos.groupby("user_id")["business_id"]}

    user_stats = train.groupby("user_id").agg(
        train_count=("review_id", "count"),
        mean_train_rating=("review_stars", "mean")).reset_index()

    summary_rows: list[dict] = []
    examples_rows: list[dict] = []

    for model in MODELS:
        preds = pd.read_parquet(OUT_DIR / f"predictions_{model}.parquet")
        per_user = user_p_at_k(preds, gt, K)
        strong   = per_user[per_user["p_at_k"] >= 0.20]
        failure  = per_user[per_user["p_at_k"] == 0.0]

        print(f"\n  {model}: {len(per_user):,} evaluable users  "
              f"strong={len(strong):,}  failures={len(failure):,}")

        summary_rows.append({"model": model,
                             **summarise_segment(strong,  user_stats, "strong")})
        summary_rows.append({"model": model,
                             **summarise_segment(failure, user_stats, "failure")})

        # Qualitative examples (only for ALS - the strongest model)
        if model != "als":
            continue
        for segment_df, segment_name in [(strong, "strong"), (failure, "failure")]:
            sample = segment_df.sort_values("p_at_k", ascending=(segment_name == "failure"))
            sample = sample.head(3)
            for _, row in sample.iterrows():
                u = row["user_id"]
                recs = preds[(preds["user_id"] == u) & (preds["rank"] <= K)]\
                    .merge(feats, on="business_id", how="left")
                rec_names = "; ".join(recs["business_name"].head(5).fillna("?").tolist())
                gt_biz = gt.get(u, set())
                gt_names = "; ".join(
                    feats[feats["business_id"].isin(gt_biz)]["business_name"]
                    .head(5).fillna("?").tolist())
                examples_rows.append({
                    "model": model, "segment": segment_name,
                    "user_id": u, "p_at_k": row["p_at_k"],
                    "n_relevant": row["n_relevant"],
                    "top5_recommendations": rec_names,
                    "test_positive_examples": gt_names,
                })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "error_summary.csv", index=False)
    print("\n  error summary:")
    print(summary.to_string(index=False))

    pd.DataFrame(examples_rows).to_csv(OUT_DIR / "error_examples.csv", index=False)
    print(f"\n  outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
