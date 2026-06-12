"""Phase 6 — Hybrid experiment: feature-augmented ALS + BPR comparison.

LightFM is incompatible with Python 3.14 (setup.py uses __LIGHTFM_SETUP__
attribute that was removed in Python 3.14, no pre-built wheel available).
This script implements the same scientific hypothesis via two alternatives:

MODEL A — Feature-Augmented ALS  (closest to LightFM WARP + item features)
    Augments the user-item confidence matrix with user-category affinity
    columns derived from train interactions.  ALS learns a shared latent space
    where items and categories are embedded together.  Only real candidate items
    are recommended at inference time.  No new feature engineering — uses the
    same primary_category from kmeans_labels.parquet already in the pipeline.

MODEL B — BPR  (Bayesian Personalized Ranking, from implicit library)
    Optimises a ranking objective (AUC) instead of ALS's weighted reconstruction.
    Same interaction data, different loss function.  Tests whether the
    optimisation objective matters for this dataset.

Both models retrain on train + val (80 %) and are evaluated on the same
test.parquet with the same metrics as evaluate_best_als.py, so comparisons
are direct.

Inputs  : outputs/data/train.parquet
          outputs/data/val.parquet
          outputs/data/test.parquet
          outputs/data/candidates.parquet
          src/06_clustering/outputs/data/kmeans_labels.parquet
          outputs/als_sweep.csv  (best alpha from tuning)
          outputs/data/metrics_tuned.csv  (tuned ALS reference)
Outputs : outputs/data/predictions_hybrid_als.parquet
          outputs/data/predictions_bpr.parquet
          outputs/data/metrics_hybrid.csv
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

BASE        = Path("src/07_recommendation")
DATA        = BASE / "outputs/data"
CLUSTER_DIR = Path("src/06_clustering/outputs/data")
SWEEP_CSV   = BASE / "outputs/als_sweep.csv"

TOP_K         = 50
POSITIVE_STAR = 4
RS            = 42
BATCH         = 2_000
KS            = [5, 10, 20]

# Category augmentation weight — scales affinity columns relative to item columns
CAT_WEIGHT = 0.5


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
        "k":               k,
        "users_evaluated": n,
        "users_with_hit":  int(sum(hs)),
        "precision_at_k":  round(float(np.mean(ps)), 4) if ps else 0.0,
        "recall_at_k":     round(float(np.mean(rs)), 4) if rs else 0.0,
        "ndcg_at_k":       round(float(np.mean(ns)), 4) if ns else 0.0,
        "map_at_k":        round(float(np.mean(ms)), 4) if ms else 0.0,
        "hit_rate":        round(float(np.mean(hs)), 4) if hs else 0.0,
    }


def _preds_by_user(preds: pd.DataFrame) -> dict:
    out: dict[str, list] = {}
    for row in preds.sort_values("rank").itertuples(index=False):
        out.setdefault(row.user_id, []).append(row.business_id)
    return out


def _recommend_batched(model, M, users, cand_indices, inv_b, top_k, batch=BATCH):
    """Generic batch recommendation for implicit models."""
    rows_out = []
    n = len(users)
    for start in range(0, n, batch):
        end       = min(start + batch, n)
        batch_idx = np.arange(start, end)
        ids, scores = model.recommend(
            batch_idx, M[batch_idx],
            N=top_k,
            items=cand_indices,
            filter_already_liked_items=True,
        )
        for i, u in enumerate(users[batch_idx]):
            rows_out.append(pd.DataFrame({
                "user_id":     u,
                "business_id": inv_b[ids[i]],
                "rank":        np.arange(1, top_k + 1, dtype=np.int16),
                "score":       scores[i].astype(np.float32),
            }))
        if end % 10_000 == 0 or end == n:
            print(f"    recommended: {end:,}/{n:,}")
    return pd.concat(rows_out, ignore_index=True)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("Phase 6 — Hybrid ALS (category-augmented) + BPR")
    print("=" * 70)
    print("Note: LightFM not available on Python 3.14 — using equivalent methods.")
    print()

    # ── Load data ──────────────────────────────────────────────────────────
    train  = pd.read_parquet(DATA / "train.parquet")
    val    = pd.read_parquet(DATA / "val.parquet")
    test   = pd.read_parquet(DATA / "test.parquet")
    cands  = pd.read_parquet(DATA / "candidates.parquet")
    labels = pd.read_parquet(CLUSTER_DIR / "kmeans_labels.parquet")

    # Best alpha from sweep (use same confidence weighting)
    alpha = 20.0
    if SWEEP_CSV.exists():
        best = pd.read_csv(SWEEP_CSV).sort_values("ndcg_at_10", ascending=False).iloc[0]
        alpha = float(best["alpha"])
    print(f"  Using alpha={alpha} from best ALS config")

    # Best ALS factors/reg/iter for augmented model
    FACTORS = 64
    REG     = 0.10
    ITERS   = 50
    if SWEEP_CSV.exists():
        best = pd.read_csv(SWEEP_CSV).sort_values("ndcg_at_10", ascending=False).iloc[0]
        FACTORS = int(best["factors"])
        REG     = float(best["regularization"])
        ITERS   = int(best["iterations"])

    train_full = pd.concat([train, val], ignore_index=True)
    print(f"  Retrain set: {len(train_full):,} reviews "
          f"({len(train):,} train + {len(val):,} val)")

    # ── Index mappings ─────────────────────────────────────────────────────
    users = train_full["user_id"].unique()
    bizs  = train_full["business_id"].unique()
    u_idx = {u: i for i, u in enumerate(users)}
    b_idx = {b: i for i, b in enumerate(bizs)}
    n_u, n_b = len(users), len(bizs)

    # ── Category encoding ──────────────────────────────────────────────────
    labels = labels[labels["business_id"].isin(b_idx)].copy()
    labels["primary_category"] = labels["primary_category"].fillna("Unknown")
    le = LabelEncoder()
    labels["cat_int"] = le.fit_transform(labels["primary_category"])
    biz_to_cat = dict(zip(labels["business_id"], labels["cat_int"]))
    n_cats = len(le.classes_)
    print(f"  Categories: {n_cats} unique primary_category values")

    # ── Build base confidence matrix ───────────────────────────────────────
    rows_m = train_full["user_id"].map(u_idx).to_numpy()
    cols_m = train_full["business_id"].map(b_idx).to_numpy()
    data_m = (1.0 + alpha * train_full["review_stars"].to_numpy(dtype=np.float32))
    M_base = sp.csr_matrix(
        (data_m, (rows_m, cols_m)), shape=(n_u, n_b)
    )
    print(f"  Base matrix: {n_u:,} × {n_b:,}  "
          f"(density {len(train_full)/(n_u*n_b):.5%})")

    # ── Candidate indices ──────────────────────────────────────────────────
    cand_biz  = set(cands["business_id"])
    cand_mask = np.zeros(n_b, dtype=bool)
    for b in cand_biz:
        if b in b_idx:
            cand_mask[b_idx[b]] = True
    cand_indices = np.where(cand_mask)[0]
    print(f"  Candidates in matrix: {cand_mask.sum():,}")

    # ── Test ground truth ──────────────────────────────────────────────────
    cand_ids = set(cands["business_id"])
    pos_test = test[
        (test["review_stars"] >= POSITIVE_STAR) &
        (test["business_id"].isin(cand_ids))
    ]
    gt: dict[str, set] = {
        u: set(g)
        for u, g in pos_test.groupby("user_id")["business_id"]
    }
    print(f"  Test eval users: {len(gt):,}")

    # Load tuned ALS reference metrics
    ref: dict[int, dict] = {}
    tuned_csv = DATA / "metrics_tuned.csv"
    if tuned_csv.exists():
        tm = pd.read_csv(tuned_csv)
        for _, row in tm.iterrows():
            ref[int(row["k"])] = row.to_dict()

    all_rows: list[dict] = []

    # ==================================================================
    # MODEL A — Feature-Augmented ALS
    # ==================================================================
    print("\n" + "─" * 70)
    print("MODEL A: Feature-Augmented ALS")
    print(f"  Augments {n_u:,}×{n_b:,} matrix with {n_cats} category columns")
    print(f"  Category weight factor: {CAT_WEIGHT}")
    print(f"  Config: factors={FACTORS}, alpha={alpha}, iter={ITERS}, reg={REG}")

    # Build user-category affinity: for each (user, category) sum confidences
    train_full["cat_int"] = train_full["business_id"].map(biz_to_cat)
    cat_df = train_full.dropna(subset=["cat_int"]).copy()
    cat_df["cat_int"] = cat_df["cat_int"].astype(int)

    uc_rows = cat_df["user_id"].map(u_idx).to_numpy()
    uc_cols = cat_df["cat_int"].to_numpy()
    uc_data = (1.0 + alpha * cat_df["review_stars"].to_numpy(dtype=np.float32))
    M_cat   = sp.csr_matrix(
        (uc_data * CAT_WEIGHT, (uc_rows, uc_cols)),
        shape=(n_u, n_cats),
    )
    # Augmented matrix: [M_base | M_cat]  shape: n_u × (n_b + n_cats)
    M_aug = sp.hstack([M_base, M_cat], format="csr")
    print(f"  Augmented matrix: {M_aug.shape}  "
          f"(density {M_aug.nnz / (n_u * (n_b + n_cats)):.5%})")

    als_aug = AlternatingLeastSquares(
        factors=FACTORS, regularization=REG,
        iterations=ITERS, random_state=RS, use_gpu=False,
    )
    print("  Training...")
    als_aug.fit(M_aug, show_progress=True)

    # Recommend from real item columns only (indices 0..n_b-1)
    # Build a view of M_aug with only the item columns for filter_already_liked_items
    # (so users' item interactions are correctly masked)
    M_items_only = M_aug[:, :n_b]   # user × n_b submatrix for the filter
    # For recommend(), user_items must match the full model's item space
    # We pass M_aug as user_items so already-liked masking covers both items+cats
    # but items= restricts output to real candidate indices
    print("  Generating predictions...")
    preds_hybrid = _recommend_batched(
        als_aug, M_aug, users, cand_indices, bizs, TOP_K
    )
    preds_hybrid["model"] = "hybrid_als"
    preds_hybrid.to_parquet(DATA / "predictions_hybrid_als.parquet", index=False)
    print(f"  Saved: predictions_hybrid_als.parquet  ({len(preds_hybrid):,} rows)")

    pbyu_hybrid = _preds_by_user(preds_hybrid)
    print(f"\n  Results (test set):")
    for k in KS:
        m = _compute_metrics(pbyu_hybrid, gt, k)
        m["model"] = "hybrid_als"
        all_rows.append(m)
        ref_k = ref.get(k, {})
        als_ndcg = ref_k.get("ndcg_at_k", "n/a")
        als_str  = f"{als_ndcg:.4f}" if isinstance(als_ndcg, float) else "n/a"
        delta_str = ""
        if isinstance(als_ndcg, float):
            d = m["ndcg_at_k"] - als_ndcg
            delta_str = f"  ({'+' if d>=0 else ''}{d:.4f} vs ALS tuned)"
        print(f"    k={k:2d}: P={m['precision_at_k']:.4f}  R={m['recall_at_k']:.4f}  "
              f"NDCG={m['ndcg_at_k']:.4f}  HR={m['hit_rate']:.4f}"
              f"  [ALS tuned NDCG={als_str}]{delta_str}")

    # ==================================================================
    # MODEL B — BPR
    # ==================================================================
    print("\n" + "─" * 70)
    print("MODEL B: Bayesian Personalized Ranking (BPR)")
    print(f"  Optimises AUC ranking loss on same interaction matrix")
    print(f"  Config: factors={FACTORS}, iter={ITERS}")

    bpr = BayesianPersonalizedRanking(
        factors=FACTORS,
        iterations=ITERS,
        random_state=RS,
        use_gpu=False,
    )
    print("  Training...")
    bpr.fit(M_base, show_progress=True)

    print("  Generating predictions...")
    preds_bpr = _recommend_batched(
        bpr, M_base, users, cand_indices, bizs, TOP_K
    )
    preds_bpr["model"] = "bpr"
    preds_bpr.to_parquet(DATA / "predictions_bpr.parquet", index=False)
    print(f"  Saved: predictions_bpr.parquet  ({len(preds_bpr):,} rows)")

    pbyu_bpr = _preds_by_user(preds_bpr)
    print(f"\n  Results (test set):")
    for k in KS:
        m = _compute_metrics(pbyu_bpr, gt, k)
        m["model"] = "bpr"
        all_rows.append(m)
        ref_k = ref.get(k, {})
        als_ndcg = ref_k.get("ndcg_at_k", "n/a")
        als_str  = f"{als_ndcg:.4f}" if isinstance(als_ndcg, float) else "n/a"
        delta_str = ""
        if isinstance(als_ndcg, float):
            d = m["ndcg_at_k"] - als_ndcg
            delta_str = f"  ({'+' if d>=0 else ''}{d:.4f} vs ALS tuned)"
        print(f"    k={k:2d}: P={m['precision_at_k']:.4f}  R={m['recall_at_k']:.4f}  "
              f"NDCG={m['ndcg_at_k']:.4f}  HR={m['hit_rate']:.4f}"
              f"  [ALS tuned NDCG={als_str}]{delta_str}")

    # ==================================================================
    # Final comparison table
    # ==================================================================
    print("\n" + "=" * 70)
    print("FINAL COMPARISON — k=10 — test set")
    print("=" * 70)

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(DATA / "metrics_hybrid.csv", index=False)

    # Build comparison at k=10
    models = {
        "ALS baseline":    {"ndcg_at_k": 0.0309, "precision_at_k": 0.0089,
                            "recall_at_k": 0.0529, "map_at_k": 0.0201, "hit_rate": 0.0842},
        "ALS tuned":       ref.get(10, {}),
        "Hybrid ALS":      next((r for r in all_rows if r["model"]=="hybrid_als" and r["k"]==10), {}),
        "BPR":             next((r for r in all_rows if r["model"]=="bpr"         and r["k"]==10), {}),
    }

    print(f"\n  {'Model':<18} {'P@10':>8} {'R@10':>8} {'NDCG@10':>9} {'MAP@10':>9} {'HR@10':>8}")
    print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*8}")
    for name, m in models.items():
        p  = m.get("precision_at_k", float("nan"))
        r  = m.get("recall_at_k",    float("nan"))
        nd = m.get("ndcg_at_k",      float("nan"))
        mp = m.get("map_at_k",       float("nan"))
        hr = m.get("hit_rate",        float("nan"))
        fmt = lambda v: f"{v:.4f}" if not (isinstance(v, float) and np.isnan(v)) else "  n/a"
        print(f"  {name:<18} {fmt(p):>8} {fmt(r):>8} {fmt(nd):>9} {fmt(mp):>9} {fmt(hr):>8}")

    print(f"\n  Saved: metrics_hybrid.csv")


if __name__ == "__main__":
    main()
