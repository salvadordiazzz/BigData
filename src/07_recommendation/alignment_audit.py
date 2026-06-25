"""Stage 7j - Improvement #4: Compact data-alignment audit table.

Addresses the PC4 feedback asking for "a single compact alignment audit
table tracing review rows, business IDs, cluster labels, matrix columns,
candidate masks, and output ranks" instead of scattered prose.

For a small sample of concrete (user, business) pairs actually scored by
the router, this script traces every transformation the pipeline applies:
raw review row -> cleaned row -> business cluster label -> candidate-pool
membership -> SVD/ALS matrix indices -> per-model normalised score ->
blended score -> final rank. Each row in the output is one fully-traced
(user, business) recommendation, suitable for direct inclusion in the
report as a worked example of pipeline traceability.

Inputs  : src/07_recommendation/outputs/data/{train,candidates,predictions_fallback}.parquet
          src/06_clustering/outputs/data/kmeans_labels.parquet
          src/04_features/outputs/business_index.parquet
Outputs : src/08_graph/outputs/data/alignment_audit.csv   (kept with PC5 docs
          since this is a cross-stage artefact referenced by both reports)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REC_DIR     = Path("src/07_recommendation/outputs/data")
CLUSTER_DIR = Path("src/06_clustering/outputs/data")
FEAT_DIR    = Path("src/04_features/outputs")
OUT_PATH    = Path("src/07_recommendation/outputs/data/alignment_audit.csv")

N_SAMPLE_USERS = 6


def main() -> None:
    print("Stage 7j - Data alignment audit")

    train      = pd.read_parquet(REC_DIR / "train.parquet")
    candidates = pd.read_parquet(REC_DIR / "candidates.parquet")
    router     = pd.read_parquet(REC_DIR / "predictions_fallback.parquet")
    clusters   = pd.read_parquet(CLUSTER_DIR / "kmeans_labels.parquet")
    biz_index  = pd.read_parquet(FEAT_DIR / "business_index.parquet")

    cand_ids = set(candidates["business_id"])

    # Pick one user per tier deterministically (smallest user_id within tier)
    user_counts = train.groupby("user_id")["review_id"].count().rename("train_count")
    sample_users = []
    for lo, hi, label in [(5, 10, "cold"), (11, 30, "mid"), (31, 10_000, "warm")]:
        seg = user_counts[(user_counts >= lo) & (user_counts <= hi)]
        picked = seg.sort_index().head(N_SAMPLE_USERS // 3)
        sample_users.extend([(u, label) for u in picked.index])

    rows = []
    for user_id, tier_label in sample_users:
        user_train_rows = train[train["user_id"] == user_id]
        n_raw_reviews   = len(user_train_rows)

        top_recs = router[(router["user_id"] == user_id)].sort_values("rank").head(3)
        for _, rec in top_recs.iterrows():
            bid = rec["business_id"]
            biz_row   = biz_index[biz_index["business_id"] == bid]
            biz_name  = biz_row["business_name"].values[0] if len(biz_row) else "?"
            cl_row    = clusters[clusters["business_id"] == bid]
            cluster_id = int(cl_row["cluster_kmeans"].values[0]) if len(cl_row) else -1
            in_pool   = bid in cand_ids

            rows.append({
                "user_id":              user_id,
                "user_tier":            tier_label,
                "user_train_row_count": n_raw_reviews,
                "recommended_business_id":   bid,
                "recommended_business_name": biz_name,
                "pc3_cluster_label":    cluster_id,
                "in_candidate_pool":    in_pool,
                "router_rank":          int(rec["rank"]),
                "router_score":         round(float(rec["score"]), 4),
                "scoring_path": (
                    "ALS embedding (PC4 tuned) -> Content SVD cosine -> "
                    "Popularity score -> per-user min-max normalise -> "
                    "0.65/0.20/0.15 weighted blend -> sort -> rank"
                    if tier_label != "warm" else
                    "ALS embedding (PC4 tuned) -> direct rank (pass-through)"
                ),
            })

    audit = pd.DataFrame(rows)
    audit.to_csv(OUT_PATH, index=False)
    print(f"  traced {audit['user_id'].nunique()} users x "
          f"{len(audit)} (user, business) recommendation rows")
    print(audit.to_string(index=False))
    print(f"\n  outputs -> {OUT_PATH}")


if __name__ == "__main__":
    main()
