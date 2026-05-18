"""Stage 6b' - DBSCAN tuning experiment.

Re-runs the DBSCAN parameter sweep across multiple SVD dimensionalities,
mirroring the K-Means tuning ablation (kmeans_tuned.py). The motivating
question: K-Means improved sharply when we dropped from 50 to 10 SVD
components - does DBSCAN benefit from the same change?

Grid: n_dims in {10, 20, 30, 50} x eps in {0.1..2.5} x min_samples in {5, 10, 20}.

Inputs  : src/05_dimreduction/outputs/reduced/combined.parquet
          src/04_features/outputs/business_index.parquet
Outputs : src/06_clustering/outputs/data/dbscan_tuning.csv
          src/06_clustering/outputs/plots/dbscan_tuning_heatmap.png
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score, silhouette_score

RED_DIR  = Path("src/05_dimreduction/outputs/reduced")
FEAT_DIR = Path("src/04_features/outputs")
OUT_DIR  = Path("src/06_clustering/outputs")
PLOT_DIR = OUT_DIR / "plots"
DATA_DIR = OUT_DIR / "data"

RS              = 42
N_DIMS_GRID     = [10, 20, 30, 50]
EPS_GRID        = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
MIN_SAMPS_GRID  = [5, 10, 20]
SIL_SAMPLE      = 5_000


def load_features() -> np.ndarray:
    red = pd.read_parquet(RED_DIR / "combined.parquet")
    cols = [f"c{i+1}" for i in range(200)]
    return red[cols].to_numpy(dtype=np.float32)


def silhouette_sampled(X: np.ndarray, labels: np.ndarray, rng) -> float:
    mask = labels != -1
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        return float("nan")
    idx_all = np.where(mask)[0]
    idx = rng.choice(idx_all, size=min(SIL_SAMPLE, len(idx_all)), replace=False)
    return float(silhouette_score(X[idx], labels[idx]))


def sweep_at_dims(X: np.ndarray, n_dims: int) -> list[dict]:
    rng = np.random.default_rng(RS)
    rows: list[dict] = []
    Xn = X[:, :n_dims]
    for ms in MIN_SAMPS_GRID:
        for eps in EPS_GRID:
            t0 = time.time()
            db = DBSCAN(eps=eps, min_samples=ms, n_jobs=-1).fit(Xn)
            runtime = time.time() - t0
            labels = db.labels_
            n_clusters = int((labels >= 0).any() and labels[labels >= 0].max() + 1)
            n_noise    = int((labels == -1).sum())
            noise_pct  = n_noise / len(labels)
            non_noise  = labels != -1
            sil = silhouette_sampled(Xn, labels, rng)
            dbi = (davies_bouldin_score(Xn[non_noise], labels[non_noise])
                   if non_noise.sum() > 1 and len(set(labels[non_noise])) > 1
                   else float("nan"))
            rows.append({
                "n_dims": n_dims, "eps": eps, "min_samples": ms,
                "n_clusters": n_clusters, "n_noise": n_noise,
                "noise_pct": round(noise_pct, 4),
                "silhouette": round(sil, 4) if not np.isnan(sil) else None,
                "davies_bouldin": round(dbi, 4) if not np.isnan(dbi) else None,
                "runtime_s": round(runtime, 2),
            })
            print(f"  n={n_dims:>2}  eps={eps:>4}  ms={ms:>3}  "
                  f"clusters={n_clusters:>3}  noise={noise_pct:>5.1%}  "
                  f"sil={sil:.4f}  t={runtime:.1f}s")
    return rows


def heatmap_plot(grid: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    for ax, n_dims in zip(axes, N_DIMS_GRID):
        sub = grid[grid["n_dims"] == n_dims]
        piv = sub.pivot(index="min_samples", columns="eps", values="silhouette")
        vals = piv.values.astype(float)
        im = ax.imshow(vals, cmap="viridis", aspect="auto", vmin=-0.5, vmax=0.7)
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index)));   ax.set_yticklabels(piv.index)
        ax.set_xlabel("eps"); ax.set_ylabel("min_samples")
        ax.set_title(f"n_dims = {n_dims}")
        for r in range(vals.shape[0]):
            for c in range(vals.shape[1]):
                v = vals[r, c]
                txt = f"{v:.2f}" if not np.isnan(v) else "-"
                ax.text(c, r, txt, ha="center", va="center",
                        color="white" if (not np.isnan(v) and v < 0.3) else "black",
                        fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.6)
    fig.suptitle("DBSCAN silhouette - tuning grid (n_dims x eps x min_samples)",
                 fontsize=14)
    fig.tight_layout(); fig.savefig(PLOT_DIR / "dbscan_tuning_heatmap.png"); plt.close(fig)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True); DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Stage 6b' - DBSCAN tuning")

    X = load_features()
    print(f"  full SVD shape: {X.shape}\n")

    all_rows: list[dict] = []
    for n_dims in N_DIMS_GRID:
        print(f"\n--- n_dims = {n_dims} ---")
        all_rows.extend(sweep_at_dims(X, n_dims))

    grid = pd.DataFrame(all_rows)
    grid.to_csv(DATA_DIR / "dbscan_tuning.csv", index=False)
    heatmap_plot(grid)

    # Summary: best useful configuration per n_dims (>=3 clusters, noise<50 %, sil defined)
    print("\n=== Best useful configurations per n_dims ===")
    print("(useful = >=3 clusters AND noise<50% AND silhouette defined)")
    for n_dims in N_DIMS_GRID:
        sub = grid[(grid["n_dims"] == n_dims)
                   & (grid["n_clusters"] >= 3)
                   & (grid["noise_pct"] < 0.50)
                   & (grid["silhouette"].notna())]
        if sub.empty:
            print(f"  n_dims={n_dims}: NO USEFUL CONFIGURATION found")
        else:
            best = sub.loc[sub["silhouette"].idxmax()]
            print(f"  n_dims={n_dims}: eps={best['eps']} ms={int(best['min_samples'])} "
                  f"clusters={int(best['n_clusters'])} noise={best['noise_pct']:.1%} "
                  f"sil={best['silhouette']:.4f}")

    print(f"\n  outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
