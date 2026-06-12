"""ALS hyperparameter sweep — optimises NDCG@10 on the temporal validation set.

Grid (144 total configurations):
    factors      : 64, 128, 256
    alpha        : 20, 40, 80, 160
    iterations   : 15, 30, 50
    regularization: 0.01, 0.05, 0.10

Evaluation target  : NDCG@10 on val.parquet
Saves              : outputs/als_sweep.csv  (incremental — safe to interrupt/resume)
Convergence report : printed at end; shows NDCG@10 vs iterations per (factors, alpha, reg)

Phase 4 note — Early stopping in implicit ALS:
    AlternatingLeastSquares.fit() does not expose per-iteration callbacks and
    re-initialises factors on every call, so true online early stopping is not
    possible without forking the library.  We simulate convergence detection by
    training separate models at iterations=[15, 30, 50] and measuring NDCG
    saturation.  A plateau (delta < CONVERGENCE_THRESHOLD between two iteration
    levels) identifies the minimum sufficient iteration count.

Inputs  : src/07_recommendation/outputs/data/train.parquet
          src/07_recommendation/outputs/data/val.parquet
          src/07_recommendation/outputs/data/candidates.parquet
Outputs : src/07_recommendation/outputs/als_sweep.csv
"""
from __future__ import annotations

import csv
import itertools
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

BASE      = Path("src/07_recommendation")
DATA      = BASE / "outputs/data"
OUT_DIR   = BASE / "outputs"
SWEEP_CSV = OUT_DIR / "als_sweep.csv"

# ── Hyperparameter grid ────────────────────────────────────────────────────
FACTORS_GRID  = [64, 128, 256]
ALPHA_GRID    = [20, 40, 80, 160]
ITERS_GRID    = [15, 30, 50]
REG_GRID      = [0.01, 0.05, 0.10]

K                   = 10
POSITIVE_STAR       = 4
RS                  = 42
EVAL_BATCH          = 2_000
CONVERGENCE_THRESH  = 0.0005   # NDCG delta below which we call convergence


# ── Metric helpers ─────────────────────────────────────────────────────────

def _precision(recs: list, relevant: set, k: int) -> float:
    return sum(1 for r in recs[:k] if r in relevant) / k


def _recall(recs: list, relevant: set, k: int) -> float:
    h = sum(1 for r in recs[:k] if r in relevant)
    return h / len(relevant) if relevant else 0.0


def _ndcg(recs: list, relevant: set, k: int) -> float:
    dcg  = sum(1.0 / np.log2(i + 2) for i, r in enumerate(recs[:k]) if r in relevant)
    n    = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n))
    return dcg / idcg if idcg > 0 else 0.0


def _map_at_k(recs: list, relevant: set, k: int) -> float:
    hits, acc = 0, 0.0
    for i, r in enumerate(recs[:k]):
        if r in relevant:
            hits += 1
            acc  += hits / (i + 1)
    return acc / min(len(relevant), k) if relevant else 0.0


# ── Batch evaluation ───────────────────────────────────────────────────────

def evaluate(
    model: AlternatingLeastSquares,
    M: sp.csr_matrix,
    val_gt: dict,        # user_id -> set of item indices
    cand_idx: np.ndarray,
    user2idx: dict,
    k: int = K,
) -> dict:
    """Evaluate ALS on validation ground truth using batched recommend()."""
    valid = [
        (uid_str, user2idx[uid_str])
        for uid_str in val_gt
        if uid_str in user2idx and user2idx[uid_str] < M.shape[0]
    ]
    if not valid:
        return {"precision": 0.0, "recall": 0.0, "ndcg": 0.0,
                "map": 0.0, "hit_rate": 0.0, "n_users": 0}

    ps, rs, ns, ms, hs = [], [], [], [], []

    for start in range(0, len(valid), EVAL_BATCH):
        batch   = valid[start : start + EVAL_BATCH]
        u_strs  = [b[0] for b in batch]
        u_ints  = np.array([b[1] for b in batch], dtype=np.int32)

        item_ids_batch, _ = model.recommend(
            u_ints, M[u_ints],
            N=k,
            filter_already_liked_items=True,
            items=cand_idx,
        )

        for i, uid_str in enumerate(u_strs):
            recs     = list(item_ids_batch[i])
            relevant = val_gt[uid_str]
            ps.append(_precision(recs, relevant, k))
            rs.append(_recall(recs, relevant, k))
            ns.append(_ndcg(recs, relevant, k))
            ms.append(_map_at_k(recs, relevant, k))
            hs.append(any(r in relevant for r in recs))

    n = len(ps)
    return {
        "precision": float(np.mean(ps)) if ps else 0.0,
        "recall":    float(np.mean(rs)) if rs else 0.0,
        "ndcg":      float(np.mean(ns)) if ns else 0.0,
        "map":       float(np.mean(ms)) if ms else 0.0,
        "hit_rate":  float(np.mean(hs)) if hs else 0.0,
        "n_users":   n,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("ALS Hyperparameter Sweep")
    print("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading data...")
    train      = pd.read_parquet(DATA / "train.parquet")
    val        = pd.read_parquet(DATA / "val.parquet")
    candidates = pd.read_parquet(DATA / "candidates.parquet")

    users    = train["user_id"].unique()
    bizs     = train["business_id"].unique()
    user2idx = {u: i for i, u in enumerate(users)}
    biz2idx  = {b: i for i, b in enumerate(bizs)}

    cand_set = set(candidates["business_id"])
    cand_idx = np.array(
        [biz2idx[b] for b in cand_set if b in biz2idx], dtype=np.int32
    )

    print(f"  Train  : {len(train):,} reviews | {len(users):,} users | {len(bizs):,} businesses")
    print(f"  Val    : {len(val):,} reviews")
    print(f"  Matrix : {len(users):,} x {len(bizs):,}  "
          f"(density {len(train)/(len(users)*len(bizs)):.5%})")
    print(f"  Candidates in matrix: {len(cand_idx):,}")

    # Build validation ground truth (item indices)
    val_pos = val[
        (val["review_stars"] >= POSITIVE_STAR) &
        (val["business_id"].isin(cand_set))
    ]
    val_gt: dict[str, set] = {}
    for uid, grp in val_pos.groupby("user_id"):
        idx_set = {biz2idx[b] for b in grp["business_id"] if b in biz2idx}
        if idx_set:
            val_gt[uid] = idx_set
    print(f"  Val eval users (>=1 positive candidate): {len(val_gt):,}")

    # ── Resume support ─────────────────────────────────────────────────────
    done: set[tuple] = set()
    if SWEEP_CSV.exists():
        prev = pd.read_csv(SWEEP_CSV)
        for _, row in prev.iterrows():
            done.add((int(row.factors), float(row.alpha),
                      int(row.iterations), float(row.regularization)))
        print(f"  Resume: {len(done)} configs already in {SWEEP_CSV.name}")

    # ── Build grid (sorted fastest → slowest) ─────────────────────────────
    configs = sorted(
        itertools.product(FACTORS_GRID, ALPHA_GRID, ITERS_GRID, REG_GRID),
        key=lambda x: (x[0], x[2]),   # factors asc, then iterations asc
    )
    total = len(configs)
    remaining = total - len(done)
    print(f"\nGrid: {len(FACTORS_GRID)} factors × {len(ALPHA_GRID)} alpha × "
          f"{len(ITERS_GRID)} iters × {len(REG_GRID)} reg = {total} configs "
          f"({remaining} remaining)")
    print("-" * 70)

    write_header = not SWEEP_CSV.exists()
    t_start = time.time()
    completed = 0

    for i, (factors, alpha, iterations, reg) in enumerate(configs, 1):
        key = (factors, float(alpha), iterations, float(reg))
        if key in done:
            print(f"  [{i:3d}/{total}] SKIP  f={factors:3d} a={alpha:3.0f} "
                  f"i={iterations:2d} r={reg:.2f}")
            continue

        print(f"  [{i:3d}/{total}] f={factors:3d}  a={alpha:3.0f}  "
              f"i={iterations:2d}  r={reg:.2f}", end="  ", flush=True)

        t0 = time.time()

        # Build confidence matrix (rebuilt per alpha value)
        rows_m = train["user_id"].map(user2idx).to_numpy()
        cols_m = train["business_id"].map(biz2idx).to_numpy()
        data_m = (1.0 + alpha * train["review_stars"].to_numpy(dtype=np.float32))
        M = sp.csr_matrix(
            (data_m, (rows_m, cols_m)),
            shape=(len(users), len(bizs)),
        )

        als = AlternatingLeastSquares(
            factors=factors,
            regularization=reg,
            iterations=iterations,
            random_state=RS,
            use_gpu=False,
        )
        als.fit(M, show_progress=False)

        metrics = evaluate(als, M, val_gt, cand_idx, user2idx, k=K)
        elapsed = time.time() - t0
        completed += 1

        row = {
            "factors":        factors,
            "alpha":          alpha,
            "iterations":     iterations,
            "regularization": reg,
            "ndcg_at_10":     round(metrics["ndcg"],      6),
            "precision_at_10": round(metrics["precision"], 6),
            "recall_at_10":   round(metrics["recall"],    6),
            "map_at_10":      round(metrics["map"],       6),
            "hit_rate_at_10": round(metrics["hit_rate"],  6),
            "n_eval_users":   metrics["n_users"],
            "runtime_s":      round(elapsed, 1),
        }

        # Incremental write — safe to interrupt
        with open(SWEEP_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
                write_header = False
            w.writerow(row)

        # Progress estimate
        elapsed_total = time.time() - t_start
        rate = elapsed_total / completed if completed else elapsed
        eta_s = rate * (remaining - completed)
        eta_m = eta_s / 60

        print(
            f"NDCG@10={metrics['ndcg']:.4f}  "
            f"P@10={metrics['precision']:.4f}  "
            f"HR={metrics['hit_rate']:.4f}  "
            f"({elapsed:.0f}s | ETA~{eta_m:.0f}m)"
        )

    # ── Final ranking ──────────────────────────────────────────────────────
    if not SWEEP_CSV.exists():
        print("No results yet — run the sweep first.")
        return

    results = pd.read_csv(SWEEP_CSV).sort_values("ndcg_at_10", ascending=False)
    n_done  = len(results)

    print(f"\n{'=' * 70}")
    print(f"SWEEP COMPLETE — {n_done}/{total} configs evaluated")
    print(f"\nTOP 15 CONFIGURATIONS (by NDCG@10 on validation set):")
    print("-" * 100)
    cols = ["factors", "alpha", "iterations", "regularization",
            "ndcg_at_10", "precision_at_10", "recall_at_10",
            "map_at_10", "hit_rate_at_10", "runtime_s"]
    print(results[cols].head(15).to_string(index=False))

    best = results.iloc[0]
    print(f"\nBEST CONFIG (val NDCG@10 = {best.ndcg_at_10:.4f}):")
    print(f"  factors={int(best.factors)}, alpha={best.alpha}, "
          f"iterations={int(best.iterations)}, regularization={best.regularization}")
    print(f"  P@10={best.precision_at_10:.4f}  R@10={best.recall_at_10:.4f}  "
          f"MAP@10={best.map_at_10:.4f}  HR@10={best.hit_rate_at_10:.4%}")

    # ── Convergence analysis ───────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("CONVERGENCE ANALYSIS  (NDCG@10 vs iterations for evaluated configs)")
    print(f"  Plateau threshold: delta < {CONVERGENCE_THRESH}")
    print(f"{'─' * 70}")

    # Restrict to configs where all 3 iteration levels are available
    iter_cols = ITERS_GRID
    for factors in FACTORS_GRID:
        for alpha in ALPHA_GRID:
            for reg in REG_GRID:
                sub = results[
                    (results["factors"] == factors) &
                    (results["alpha"] == alpha) &
                    (results["regularization"] == reg)
                ].sort_values("iterations")
                if len(sub) < 2:
                    continue
                iters  = sub["iterations"].tolist()
                ndcgs  = sub["ndcg_at_10"].tolist()
                best_i = int(sub.loc[sub["ndcg_at_10"].idxmax(), "iterations"])
                # Detect plateau: improvement < threshold
                plateaus = []
                for j in range(1, len(ndcgs)):
                    delta = ndcgs[j] - ndcgs[j - 1]
                    if abs(delta) < CONVERGENCE_THRESH:
                        plateaus.append(iters[j - 1])
                plateau_str = (
                    f"converges at iter≈{plateaus[0]}" if plateaus else "still improving"
                )
                print(
                    f"  f={factors:3d} a={alpha:3.0f} r={reg:.2f}  "
                    f"iters={iters} -> NDCG={[round(v,4) for v in ndcgs]}  "
                    f"best_iter={best_i}  [{plateau_str}]"
                )

    print(f"\nResults saved to: {SWEEP_CSV}")


if __name__ == "__main__":
    main()
