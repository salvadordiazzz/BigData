"""Stage 7a - Temporal train / validation / test split.

Per-user temporal hold-out with three non-overlapping partitions:

    train  ≈ 70 %  (reviews 0  … n*0.70 - 1  per user, by date)
    val    ≈ 10 %  (reviews n*0.70 … n*0.80 - 1 per user, by date)
    test   ≈ 20 %  (reviews n*0.80 … n-1       per user, by date)

All boundaries use integer floor so there is never overlap between splits.
Only users with >= MIN_USER_REVIEWS reviews are kept. Only businesses that
appear in train are kept in val/test (no cold-start items).

val.parquet is used exclusively for hyperparameter tuning (tune_als.py).
test.parquet is the held-out evaluation set, never touched during tuning.

Inputs  : src/03_join/outputs/reviews_enriched_v1_part_*.parquet
          src/06_clustering/outputs/data/kmeans_labels.parquet
Outputs : src/07_recommendation/outputs/data/train.parquet
          src/07_recommendation/outputs/data/val.parquet
          src/07_recommendation/outputs/data/test.parquet
          src/07_recommendation/outputs/data/candidates.parquet
          src/07_recommendation/outputs/data/split_stats.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

JOIN_DIR    = Path("src/03_join/outputs")
CLUSTER_DIR = Path("src/06_clustering/outputs/data")
OUT_DIR     = Path("src/07_recommendation/outputs/data")

MIN_USER_REVIEWS   = 5
VAL_FRACTION       = 0.10
TEST_FRACTION      = 0.20
HIDDEN_GEM_CLUSTER = 1


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Stage 7a - Temporal split (train / val / test)")

    parts = sorted(JOIN_DIR.glob("reviews_enriched_v1_part_*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df = df[["review_id", "user_id", "business_id", "review_stars", "date"]]
    df["date"] = pd.to_datetime(df["date"])
    print(f"  reviews loaded: {len(df):,}")

    counts = df.groupby("user_id").size()
    keep_users = counts[counts >= MIN_USER_REVIEWS].index
    df = df[df["user_id"].isin(keep_users)]
    print(f"  users with >= {MIN_USER_REVIEWS} reviews: {len(keep_users):,}  "
          f"({len(df):,} reviews retained)")

    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)
    df["rk"] = df.groupby("user_id").cumcount()
    df["n"]  = df.groupby("user_id")["rk"].transform("count")

    # Compute integer split boundaries per review (floor division)
    val_start  = (df["n"] * (1 - TEST_FRACTION - VAL_FRACTION)).astype(int)
    test_start = (df["n"] * (1 - TEST_FRACTION)).astype(int)

    df["is_test"] = df["rk"] >= test_start
    df["is_val"]  = (df["rk"] >= val_start) & ~df["is_test"]

    drop_cols = ["rk", "n", "is_test", "is_val"]
    train = df[~df["is_test"] & ~df["is_val"]].drop(columns=drop_cols).reset_index(drop=True)
    val   = df[ df["is_val"]                ].drop(columns=drop_cols).reset_index(drop=True)
    test  = df[ df["is_test"]               ].drop(columns=drop_cols).reset_index(drop=True)

    # Keep only businesses that appear in train (no cold-start items)
    train_biz = set(train["business_id"])
    val  = val [val ["business_id"].isin(train_biz)].reset_index(drop=True)
    test = test[test["business_id"].isin(train_biz)].reset_index(drop=True)

    eval_users_test = set(test["user_id"]) & set(train["user_id"])
    eval_users_val  = set(val["user_id"])  & set(train["user_id"])

    train.to_parquet(OUT_DIR / "train.parquet", index=False)
    val.to_parquet  (OUT_DIR / "val.parquet",   index=False)
    test.to_parquet (OUT_DIR / "test.parquet",  index=False)

    # Candidate pool: Cluster 1 (hidden gems)
    clusters = pd.read_parquet(CLUSTER_DIR / "kmeans_labels.parquet")
    candidates = clusters[clusters["cluster_kmeans"] == HIDDEN_GEM_CLUSTER][
        ["business_id", "business_name", "primary_category"]
    ].reset_index(drop=True)
    candidates = candidates[candidates["business_id"].isin(train_biz)].reset_index(drop=True)
    candidates.to_parquet(OUT_DIR / "candidates.parquet", index=False)

    stats = {
        "total_reviews":         int(len(df)),
        "train_reviews":         int(len(train)),
        "val_reviews":           int(len(val)),
        "test_reviews":          int(len(test)),
        "train_users":           int(train["user_id"].nunique()),
        "val_eval_users":        int(len(eval_users_val)),
        "test_eval_users":       int(len(eval_users_test)),
        "train_businesses":      int(train["business_id"].nunique()),
        "candidate_pool_size":   int(len(candidates)),
        "min_user_reviews":      MIN_USER_REVIEWS,
        "val_fraction":          VAL_FRACTION,
        "test_fraction":         TEST_FRACTION,
        "hidden_gem_cluster":    HIDDEN_GEM_CLUSTER,
    }
    with open(OUT_DIR / "split_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  train : {len(train):,} reviews / {stats['train_users']:,} users / "
          f"{stats['train_businesses']:,} businesses")
    print(f"  val   : {len(val):,} reviews / {stats['val_eval_users']:,} eval-users")
    print(f"  test  : {len(test):,} reviews / {stats['test_eval_users']:,} eval-users")
    print(f"  candidate pool (Cluster {HIDDEN_GEM_CLUSTER}): {len(candidates):,}")
    print(f"  outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
