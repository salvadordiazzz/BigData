"""Evaluate the best ALS config from the sweep against the held-out test set.

Workflow:
  1. Load best hyperparameters from outputs/als_sweep.csv (highest val NDCG@10).
  2. Retrain ALS on train + val combined (≈ 80 % of data) using best params.
  3. Generate top-50 predictions for every user on the candidate pool.
  4. Evaluate predictions on test.parquet with the same metrics as evaluate.py
     (P@k, R@k, MAP@k, NDCG@k, HR@k for k in {5, 10, 20}).
  5. Load existing baseline ALS metrics from metrics.csv for comparison.
  6. Print comparison table and save metrics_tuned.csv.

Note: retraining on train+val (80 %) makes the comparison with the original
      baseline fair — both models see the same amount of data, the only
      difference is the hyperparameters.

Inputs  : outputs/data/train.parquet
          outputs/data/val.parquet
          outputs/data/test.parquet
          outputs/data/candidates.parquet
          outputs/als_sweep.csv
          outputs/data/metrics.csv          (existing baseline, optional)
Outputs : outputs/data/predictions_als_tuned.parquet
          outputs/data/metrics_tuned.csv
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

BASE      = Path("src/07_recommendation")
DATA      = BASE / "outputs/data"
SWEEP_CSV = BASE / "outputs/als_sweep.csv"

TOP_K         = 50
POSITIVE_STAR = 4
RS            = 42
BATCH         = 2_000
KS            = [5, 10, 20]

# Baseline metrics provided at project start (ALS with default config on 80% train)
BASELINE = {
    10: {"precision_at_k": 0.0089, "recall_at_k": 0.0529,
         "ndcg_at_k": 0.0309, "map_at_k": 0.0201, "hit_rate": 0.0842},
}


# ── Metric helpers ─────────────────────────────────────────────────────────

def _compute_metrics(preds_by_user: dict, gt: dict, k: int) -> dict:
    ps, rs, ns, ms, hs = [], [], [], [], []

    for user, relevant in gt.items():
        recs  = preds_by_user.get(user, [])[:k]
        flags = [1 if b in relevant else 0 for b in recs]
        hits  = sum(flags)

        ps.append(hits / k)
        rs.append(hits / len(relevant) if relevant else 0.0)

        dcg  = sum(h / np.log2(i + 2) for i, h in enumerate(flags))
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
        ns.append(dcg / idcg if idcg > 0 else 0.0)

        ap, h2 = 0.0, 0
        for i, h in enumerate(flags):
            if h:
                h2 += 1
                ap += h2 / (i + 1)
        ms.append(ap / min(len(relevant), k) if relevant else 0.0)

        hs.append(hits > 0)

    n = len(ps)
    return {
        "k":              k,
        "users_evaluated": n,
        "users_with_hit": int(sum(hs)),
        "precision_at_k": round(float(np.mean(ps)), 4) if ps else 0.0,
        "recall_at_k":    round(float(np.mean(rs)), 4) if rs else 0.0,
        "ndcg_at_k":      round(float(np.mean(ns)), 4) if ns else 0.0,
        "map_at_k":       round(float(np.mean(ms)), 4) if ms else 0.0,
        "hit_rate":       round(float(np.mean(hs)), 4) if hs else 0.0,
    }


def _delta(tuned: float, base: float) -> str:
    if base == 0:
        return "n/a"
    d   = tuned - base
    pct = d / base * 100
    s   = "+" if d >= 0 else ""
    return f"{s}{d:.4f}  ({s}{pct:.1f} %)"


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("Best ALS — retrain on train+val, evaluate on test")
    print("=" * 70)

    # ── Load best config ───────────────────────────────────────────────────
    if not SWEEP_CSV.exists():
        raise FileNotFoundError(
            f"{SWEEP_CSV} not found — run tune_als.py first"
        )
    sweep = pd.read_csv(SWEEP_CSV).sort_values("ndcg_at_10", ascending=False)
    best  = sweep.iloc[0]

    FACTORS = int(best.factors)
    ALPHA   = float(best.alpha)
    ITERS   = int(best.iterations)
    REG     = float(best.regularization)

    print(f"\nBest config from sweep ({len(sweep)} configs evaluated):")
    print(f"  factors={FACTORS}  alpha={ALPHA}  iterations={ITERS}  regularization={REG}")
    print(f"  Validation NDCG@10 = {best.ndcg_at_10:.4f}  "
          f"(baseline default config val NDCG not available — trained only once)")

    # ── Load data ──────────────────────────────────────────────────────────
    print("\nLoading data...")
    train  = pd.read_parquet(DATA / "train.parquet")
    val    = pd.read_parquet(DATA / "val.parquet")
    test   = pd.read_parquet(DATA / "test.parquet")
    cands  = pd.read_parquet(DATA / "candidates.parquet")

    # Combine train + val for final training (restores 80 % of data)
    train_full = pd.concat([train, val], ignore_index=True)
    print(f"  Retrain set : {len(train_full):,} reviews "
          f"({len(train):,} train + {len(val):,} val)")
    print(f"  Test set    : {len(test):,} reviews")

    # ── Build sparse matrix ────────────────────────────────────────────────
    users = train_full["user_id"].unique()
    bizs  = train_full["business_id"].unique()
    u_idx = {u: i for i, u in enumerate(users)}
    b_idx = {b: i for i, b in enumerate(bizs)}

    rows_m = train_full["user_id"].map(u_idx).to_numpy()
    cols_m = train_full["business_id"].map(b_idx).to_numpy()
    data_m = (1.0 + ALPHA * train_full["review_stars"].to_numpy(dtype=np.float32))
    M = sp.csr_matrix(
        (data_m, (rows_m, cols_m)),
        shape=(len(users), len(bizs)),
    )
    print(f"  Matrix      : {len(users):,} × {len(bizs):,}  "
          f"(density {len(train_full)/(len(users)*len(bizs)):.5%})")

    # ── Train ALS ──────────────────────────────────────────────────────────
    print(f"\nTraining ALS (factors={FACTORS}, alpha={ALPHA}, "
          f"iterations={ITERS}, reg={REG})...")
    als = AlternatingLeastSquares(
        factors=FACTORS,
        regularization=REG,
        iterations=ITERS,
        random_state=RS,
        use_gpu=False,
    )
    als.fit(M)
    print("  Training complete.")

    # ── Candidate indices ──────────────────────────────────────────────────
    cand_biz  = set(cands["business_id"])
    cand_mask = np.zeros(len(bizs), dtype=bool)
    for b in cand_biz:
        if b in b_idx:
            cand_mask[b_idx[b]] = True
    cand_indices = np.where(cand_mask)[0]
    print(f"  Candidates in matrix: {cand_mask.sum():,}")

    # ── Generate predictions ───────────────────────────────────────────────
    print("\nGenerating predictions...")
    inv_b    = bizs
    rows_out = []
    for start in range(0, len(users), BATCH):
        end   = min(start + BATCH, len(users))
        batch = np.arange(start, end)
        ids, scores = als.recommend(
            batch, M[batch],
            N=TOP_K,
            items=cand_indices,
            filter_already_liked_items=True,
        )
        for i, u in enumerate(users[batch]):
            rows_out.append(pd.DataFrame({
                "user_id":     u,
                "business_id": inv_b[ids[i]],
                "rank":        np.arange(1, TOP_K + 1, dtype=np.int16),
                "score":       scores[i].astype(np.float32),
            }))
        if end % 10_000 == 0 or end == len(users):
            print(f"  recommended: {end:,}/{len(users):,} users")

    preds = pd.concat(rows_out, ignore_index=True)
    preds["model"] = "als_tuned"
    preds.to_parquet(DATA / "predictions_als_tuned.parquet", index=False)
    print(f"  Saved: predictions_als_tuned.parquet  ({len(preds):,} rows)")

    # Build user → ranked list mapping for metric computation
    preds_by_user: dict[str, list] = {}
    for row in preds.sort_values("rank").itertuples(index=False):
        preds_by_user.setdefault(row.user_id, []).append(row.business_id)

    # ── Build test ground truth ────────────────────────────────────────────
    cand_ids = set(cands["business_id"])
    pos_test = test[
        (test["review_stars"] >= POSITIVE_STAR) &
        (test["business_id"].isin(cand_ids))
    ]
    gt: dict[str, set] = {
        u: set(g)
        for u, g in pos_test.groupby("user_id")["business_id"]
    }
    print(f"\n  Test eval users: {len(gt):,}")

    # ── Compute metrics ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS ON TEST SET")
    print("=" * 70)

    # Load existing baseline from metrics.csv if available
    baseline_csv: dict[int, dict] = {}
    metrics_path = DATA / "metrics.csv"
    if metrics_path.exists():
        mcsv = pd.read_csv(metrics_path)
        als_rows = mcsv[mcsv["model"] == "als"]
        for _, row in als_rows.iterrows():
            k = int(row["k"])
            baseline_csv[k] = {
                "precision_at_k": float(row["precision_at_k"]),
                "recall_at_k":    float(row["recall_at_k"]),
                "ndcg_at_k":      float(row["ndcg_at_k"]),
                "hit_rate":       float(row["hit_rate"]),
                "map_at_k":       None,  # not in evaluate.py output
            }

    tuned_rows = []
    for k in KS:
        m = _compute_metrics(preds_by_user, gt, k)
        m["model"] = "als_tuned"
        tuned_rows.append(m)

        # Resolve baseline: prefer loaded CSV; fall back to BASELINE constant
        base = baseline_csv.get(k) or BASELINE.get(k, {})

        print(f"\n  k = {k}")
        print(f"  {'Metric':<14} {'Tuned ALS':>12} {'Baseline ALS':>14} {'Delta':>20}")
        print(f"  {'-'*14} {'-'*12} {'-'*14} {'-'*20}")
        for metric_key, label in [
            ("precision_at_k", "Precision"),
            ("recall_at_k",    "Recall"),
            ("ndcg_at_k",      "NDCG"),
            ("map_at_k",       "MAP"),
            ("hit_rate",       "Hit Rate"),
        ]:
            tuned_val = m[metric_key]
            base_val  = base.get(metric_key)
            delta_str = _delta(tuned_val, base_val) if base_val else "n/a"
            base_str  = f"{base_val:.4f}" if base_val else "n/a"
            print(f"  {label:<14} {tuned_val:>12.4f} {base_str:>14} {delta_str:>20}")

    # ── Save ───────────────────────────────────────────────────────────────
    tuned_df = pd.DataFrame(tuned_rows)
    tuned_df.to_csv(DATA / "metrics_tuned.csv", index=False)
    print(f"\n  Saved: metrics_tuned.csv")

    # ── Summary line ───────────────────────────────────────────────────────
    m10 = next(m for m in tuned_rows if m["k"] == 10)
    b10 = baseline_csv.get(10) or BASELINE.get(10, {})
    if b10:
        ndcg_delta = m10["ndcg_at_k"] - b10["ndcg_at_k"]
        ndcg_pct   = ndcg_delta / b10["ndcg_at_k"] * 100 if b10["ndcg_at_k"] else 0
        sign = "+" if ndcg_delta >= 0 else ""
        print(f"\nSUMMARY — NDCG@10: baseline={b10['ndcg_at_k']:.4f}  "
              f"tuned={m10['ndcg_at_k']:.4f}  "
              f"delta={sign}{ndcg_delta:.4f} ({sign}{ndcg_pct:.1f} %)")
    else:
        print(f"\nSUMMARY — Tuned NDCG@10 = {m10['ndcg_at_k']:.4f}")


if __name__ == "__main__":
    main()
