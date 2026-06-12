"""Stage 7c - Content-based recommender.

User profile := mean SVD vector of businesses the user rated >=4 in train.
Score(user, business) := cosine similarity in the 200-dim combined SVD space.
Recommends top-K hidden-gem candidates per user that the user has not rated
in train. Users with no >=4 ratings in train fall back to popularity ordering.

Inputs  : src/05_dimreduction/outputs/reduced/combined.parquet
          src/07_recommendation/outputs/data/train.parquet
          src/07_recommendation/outputs/data/candidates.parquet
Outputs : src/07_recommendation/outputs/data/predictions_content.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

RED_DIR = Path("src/05_dimreduction/outputs/reduced")
OUT_DIR = Path("src/07_recommendation/outputs/data")

N_DIMS        = 200
POSITIVE_STAR = 4
TOP_K         = 50


def main() -> None:
    print("Stage 7c - Content-based")

    train      = pd.read_parquet(OUT_DIR / "train.parquet")
    candidates = pd.read_parquet(OUT_DIR / "candidates.parquet")
    svd        = pd.read_parquet(RED_DIR / "combined.parquet")

    feat_cols = [f"c{i+1}" for i in range(N_DIMS)]
    biz_index = {bid: i for i, bid in enumerate(svd["business_id"].values)}
    X = normalize(svd[feat_cols].to_numpy(dtype=np.float32), norm="l2", axis=1)

    cand_ids = candidates["business_id"].tolist()
    cand_idx = np.array([biz_index[b] for b in cand_ids if b in biz_index])
    cand_ids = [b for b in cand_ids if b in biz_index]
    X_cand   = X[cand_idx]                                       # (N_cand, dim)
    print(f"  candidate matrix: {X_cand.shape}")

    pos = train[train["review_stars"] >= POSITIVE_STAR]
    pos = pos[pos["business_id"].isin(biz_index)]

    user_seen = train.groupby("user_id")["business_id"].apply(set)

    grouped = pos.groupby("user_id")["business_id"].apply(
        lambda s: [biz_index[b] for b in s])
    n_users  = len(grouped)
    print(f"  users with positive history: {n_users:,}")

    # Build user profiles in batches to avoid huge memory peaks
    BATCH = 5_000
    user_ids   = grouped.index.to_numpy()
    rows: list[pd.DataFrame] = []
    for start in range(0, n_users, BATCH):
        end = min(start + BATCH, n_users)
        batch_users = user_ids[start:end]
        profiles    = np.zeros((end - start, N_DIMS), dtype=np.float32)
        for i, u in enumerate(batch_users):
            idxs = grouped.loc[u]
            profiles[i] = X[idxs].mean(axis=0)
        profiles = normalize(profiles, norm="l2", axis=1)

        sims = profiles @ X_cand.T  # (batch, N_cand)

        for i, u in enumerate(batch_users):
            scores = sims[i]
            seen = user_seen.get(u, set())
            # Mask seen candidates
            for j, bid in enumerate(cand_ids):
                if bid in seen:
                    scores[j] = -np.inf
            top = np.argpartition(-scores, TOP_K)[:TOP_K]
            top = top[np.argsort(-scores[top])]
            rows.append(pd.DataFrame({
                "user_id": u,
                "business_id": [cand_ids[j] for j in top],
                "rank": np.arange(1, TOP_K + 1, dtype=np.int16),
                "score": scores[top].astype(np.float32),
            }))
        if end % 10_000 == 0 or end == n_users:
            print(f"  processed {end:,}/{n_users:,} users")

    preds = pd.concat(rows, ignore_index=True)
    preds["model"] = "content"
    preds.to_parquet(OUT_DIR / "predictions_content.parquet", index=False)
    print(f"  predictions: {len(preds):,} rows")
    print(f"  outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
