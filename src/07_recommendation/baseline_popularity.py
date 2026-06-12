"""Stage 7b - Popularity baseline.

Non-personalised ranking: for every user, recommend the same top-K hidden-gem
candidates ranked by a popularity-quality score computed from the train data:

    score(b) = mean_train_rating(b) * log1p(train_count(b))

Inputs  : src/07_recommendation/outputs/data/train.parquet
          src/07_recommendation/outputs/data/candidates.parquet
Outputs : src/07_recommendation/outputs/data/predictions_popularity.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("src/07_recommendation/outputs/data")
TOP_K   = 50


def main() -> None:
    print("Stage 7b - Popularity baseline")

    train      = pd.read_parquet(OUT_DIR / "train.parquet")
    candidates = pd.read_parquet(OUT_DIR / "candidates.parquet")
    cand_ids   = set(candidates["business_id"])

    cand_train = train[train["business_id"].isin(cand_ids)]
    score = (cand_train.groupby("business_id")
             .agg(mean_rating=("review_stars", "mean"),
                  n=("review_id", "count"))
             .reset_index())
    score["score"] = score["mean_rating"] * np.log1p(score["n"])
    score = score.sort_values("score", ascending=False).reset_index(drop=True)

    top_k = score.head(TOP_K).reset_index(drop=True)
    top_k["rank"] = top_k.index + 1
    print(f"  top-5 popularity-quality candidates:")
    print(top_k.head().to_string(index=False))

    # Same top-K for every evaluation user
    train_users = train["user_id"].unique()
    base = top_k[["business_id", "rank", "score"]]
    preds = pd.MultiIndex.from_product([train_users, base["rank"]],
                                       names=["user_id", "rank"]).to_frame(index=False)
    preds = preds.merge(base, on="rank", how="left")
    preds["model"] = "popularity"
    preds = preds[["user_id", "business_id", "rank", "score", "model"]]

    # Cast to small dtypes to keep file small
    preds["rank"]  = preds["rank"].astype("int16")
    preds["score"] = preds["score"].astype("float32")
    preds.to_parquet(OUT_DIR / "predictions_popularity.parquet", index=False)
    print(f"  predictions: {len(preds):,} rows ({len(train_users):,} users x {TOP_K})")
    print(f"  outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
