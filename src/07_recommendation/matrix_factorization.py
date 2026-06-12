"""Stage 7d - Collaborative filtering via Alternating Least Squares.

Builds a sparse user-business confidence matrix from train ratings and fits
implicit ALS (Hu, Koren, Volinsky 2008). Confidence is derived from the rating:

    confidence(u, b) = 1 + alpha * rating(u, b)

Recommends top-K hidden-gem candidates per user, excluding businesses the user
already rated in train.

Inputs  : src/07_recommendation/outputs/data/train.parquet
          src/07_recommendation/outputs/data/candidates.parquet
Outputs : src/07_recommendation/outputs/data/predictions_als.parquet
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

OUT_DIR = Path("src/07_recommendation/outputs/data")

FACTORS     = 64
REG         = 0.05
ITERS       = 15
ALPHA       = 40.0
TOP_K       = 50
RS          = 42


def main() -> None:
    print("Stage 7d - ALS collaborative filtering")

    train      = pd.read_parquet(OUT_DIR / "train.parquet")
    candidates = pd.read_parquet(OUT_DIR / "candidates.parquet")

    users = train["user_id"].unique()
    bizs  = train["business_id"].unique()
    u_idx = {u: i for i, u in enumerate(users)}
    b_idx = {b: i for i, b in enumerate(bizs)}
    n_users, n_bizs = len(users), len(bizs)
    print(f"  matrix: {n_users:,} users x {n_bizs:,} businesses  "
          f"(density {len(train)/(n_users*n_bizs):.5%})")

    rows = train["user_id"].map(u_idx).to_numpy()
    cols = train["business_id"].map(b_idx).to_numpy()
    data = (1.0 + ALPHA * train["review_stars"].to_numpy(dtype=np.float32))
    M = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_bizs))

    print(f"  fitting ALS (factors={FACTORS}, iters={ITERS})...")
    als = AlternatingLeastSquares(
        factors=FACTORS, regularization=REG, iterations=ITERS,
        random_state=RS, use_gpu=False)
    als.fit(M)

    cand_mask = np.zeros(n_bizs, dtype=bool)
    for b in candidates["business_id"]:
        if b in b_idx:
            cand_mask[b_idx[b]] = True
    cand_indices = np.where(cand_mask)[0]
    print(f"  candidate indices in matrix: {cand_mask.sum():,}")

    # Recommend in batches; implicit returns (n_recs,) indices and scores
    BATCH = 2_000
    inv_b = bizs  # array - inv_b[idx] -> business_id

    rows_out: list[pd.DataFrame] = []
    for start in range(0, n_users, BATCH):
        end = min(start + BATCH, n_users)
        batch = np.arange(start, end)
        ids, scores = als.recommend(
            batch, M[batch], N=TOP_K, items=cand_indices,
            filter_already_liked_items=True)
        for i, u in enumerate(users[batch]):
            rows_out.append(pd.DataFrame({
                "user_id": u,
                "business_id": inv_b[ids[i]],
                "rank": np.arange(1, TOP_K + 1, dtype=np.int16),
                "score": scores[i].astype(np.float32),
            }))
        if end % 10_000 == 0 or end == n_users:
            print(f"  recommended for {end:,}/{n_users:,} users")

    preds = pd.concat(rows_out, ignore_index=True)
    preds["model"] = "als"
    preds.to_parquet(OUT_DIR / "predictions_als.parquet", index=False)
    print(f"  predictions: {len(preds):,} rows")
    print(f"  outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
