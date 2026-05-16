"""Stage 6a' - K-Means tuning experiments.

Systematic ablation to address the low silhouette (0.154) of the baseline:
    A. baseline                              -> reference
    B. standardize SVD output before K-Means -> equalises component scales
    C. B + remove far-from-centroid outliers -> robustness to extremes
    D. B + sweep over n_dims in {10, 20, 30, 50}
       to find the dimensionality sweet spot

The script reports silhouette for every (n_dims, std, k, outliers) combination,
picks the winning configuration by max silhouette, and rewrites the canonical
kmeans_labels.parquet + kmeans_sweep.csv + plots so that downstream steps
(validate.py, profile.py) consume the improved result.

Inputs  : src/05_dimreduction/outputs/reduced/combined.parquet
          src/04_features/outputs/business_index.parquet
Outputs : src/06_clustering/outputs/data/kmeans_tuning.csv
          src/06_clustering/outputs/data/kmeans_sweep.csv           (overwritten)
          src/06_clustering/outputs/data/kmeans_labels.parquet      (overwritten)
          src/06_clustering/outputs/plots/kmeans_elbow.png          (overwritten)
          src/06_clustering/outputs/plots/kmeans_silhouette.png     (overwritten)
          src/06_clustering/outputs/plots/kmeans_tuning_heatmap.png
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

RED_DIR  = Path("src/05_dimreduction/outputs/reduced")
FEAT_DIR = Path("src/04_features/outputs")
OUT_DIR  = Path("src/06_clustering/outputs")
PLOT_DIR = OUT_DIR / "plots"
DATA_DIR = OUT_DIR / "data"

RS              = 42
N_DIMS_GRID     = [10, 20, 30, 50]
K_GRID          = [3, 5, 8, 10, 12, 15]
SIL_SAMPLE      = 5_000
OUTLIER_PCT     = 0.01     # remove top 1 % farthest-from-centroid points
N_AVAILABLE_DIM = 200


def load_features() -> tuple[np.ndarray, pd.DataFrame]:
    red = pd.read_parquet(RED_DIR / "combined.parquet")
    index = pd.read_parquet(FEAT_DIR / "business_index.parquet")
    red = red.merge(index, on="business_id", how="left")
    cols = [f"c{i+1}" for i in range(N_AVAILABLE_DIM)]
    X = red[cols].to_numpy(dtype=np.float32)
    return X, red[["business_id", "business_name", "primary_category"]]


def silhouette_sampled(X: np.ndarray, labels: np.ndarray, rng) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    idx = rng.choice(X.shape[0], size=min(SIL_SAMPLE, X.shape[0]), replace=False)
    return float(silhouette_score(X[idx], labels[idx]))


def run_kmeans(X: np.ndarray, k: int) -> tuple[KMeans, np.ndarray, float]:
    t0 = time.time()
    km = KMeans(n_clusters=k, n_init=10, random_state=RS).fit(X)
    return km, km.labels_, time.time() - t0


def remove_outliers(X: np.ndarray, labels: np.ndarray, centers: np.ndarray,
                    pct: float) -> np.ndarray:
    dists = np.linalg.norm(X - centers[labels], axis=1)
    threshold = np.quantile(dists, 1 - pct)
    return dists < threshold


def tuning_grid(X_full: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(RS)
    rows = []
    for n_dims in N_DIMS_GRID:
        X_raw = X_full[:, :n_dims]
        X_std = StandardScaler().fit_transform(X_raw)
        for std_flag, X in [("raw", X_raw), ("std", X_std)]:
            for k in K_GRID:
                _, labels, runtime = run_kmeans(X, k)
                sil = silhouette_sampled(X, labels, rng)
                db  = davies_bouldin_score(X, labels) if len(set(labels)) > 1 else float("nan")
                rows.append({
                    "n_dims": n_dims, "scaling": std_flag, "outliers": "kept",
                    "k": k, "silhouette": round(sil, 4),
                    "davies_bouldin": round(db, 4), "runtime_s": round(runtime, 2),
                })
                print(f"  n={n_dims:>2} {std_flag:>3} keep  k={k:>2}  sil={sil:.4f}  DB={db:.4f}")
    return pd.DataFrame(rows)


def best_outlier_removed(X_full: np.ndarray, best_cfg: dict) -> dict:
    """Apply the winning (n_dims, scaling) and re-run with outlier removal."""
    rng = np.random.default_rng(RS)
    n_dims  = int(best_cfg["n_dims"])
    scaling = best_cfg["scaling"]
    k       = int(best_cfg["k"])

    X = X_full[:, :n_dims]
    if scaling == "std":
        X = StandardScaler().fit_transform(X)

    km, labels, _ = run_kmeans(X, k)
    keep = remove_outliers(X, labels, km.cluster_centers_, OUTLIER_PCT)
    X_in = X[keep]
    km2, labels2, runtime = run_kmeans(X_in, k)
    sil = silhouette_sampled(X_in, labels2, rng)
    db  = davies_bouldin_score(X_in, labels2)
    print(f"  outlier-removed: kept={keep.sum():,}/{len(keep):,}  "
          f"sil={sil:.4f}  DB={db:.4f}")
    return {
        "n_dims": n_dims, "scaling": scaling, "outliers": "removed_1pct",
        "k": k, "silhouette": round(sil, 4),
        "davies_bouldin": round(db, 4), "runtime_s": round(runtime, 2),
    }


def heatmap_plot(grid: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, scaling, title in zip(axes, ["raw", "std"],
                                  ["Raw SVD", "Standardised SVD"]):
        sub = grid[grid["scaling"] == scaling]
        piv = sub.pivot(index="n_dims", columns="k", values="silhouette")
        im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index)));   ax.set_yticklabels(piv.index)
        ax.set_xlabel("k"); ax.set_ylabel("n_dims")
        ax.set_title(f"{title} - silhouette")
        for r in range(piv.shape[0]):
            for c in range(piv.shape[1]):
                ax.text(c, r, f"{piv.values[r, c]:.3f}",
                        ha="center", va="center", color="white", fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout(); fig.savefig(PLOT_DIR / "kmeans_tuning_heatmap.png"); plt.close(fig)


def write_canonical_sweep_and_plots(X_best: np.ndarray, best_k: int) -> pd.DataFrame:
    """Re-emit the canonical kmeans_sweep.csv, elbow + silhouette plots
    using the chosen (n_dims, scaling) input."""
    rng = np.random.default_rng(RS)
    K_FULL = [3, 5, 8, 10, 12, 15, 20, 25, 30]
    rows = []
    for k in K_FULL:
        km, labels, runtime = run_kmeans(X_best, k)
        sil = silhouette_sampled(X_best, labels, rng)
        db  = davies_bouldin_score(X_best, labels)
        rows.append({"k": k, "inertia": round(float(km.inertia_), 2),
                     "silhouette": round(sil, 4),
                     "davies_bouldin": round(db, 4),
                     "runtime_s": round(runtime, 2)})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(DATA_DIR / "kmeans_sweep.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sweep["k"], sweep["inertia"], "o-", lw=2, color="steelblue")
    ax.set_xlabel("Number of clusters (k)"); ax.set_ylabel("Inertia")
    ax.set_title("K-Means - elbow plot (tuned input)")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(PLOT_DIR / "kmeans_elbow.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sweep["k"], sweep["silhouette"], "o-", lw=2, color="seagreen")
    ax.axvline(best_k, color="red", ls="--", lw=1, label=f"chosen k = {best_k}")
    ax.set_xlabel("k"); ax.set_ylabel("Silhouette")
    ax.set_title("K-Means - silhouette vs. k (tuned input)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(PLOT_DIR / "kmeans_silhouette.png"); plt.close(fig)
    return sweep


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True); DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Stage 6a' - K-Means tuning")

    X_full, index = load_features()
    print(f"  full SVD shape: {X_full.shape}")

    print("\n  ablation grid (n_dims x scaling x k):")
    grid = tuning_grid(X_full)
    heatmap_plot(grid)

    best = grid.loc[grid["silhouette"].idxmax()].to_dict()
    print(f"\n  best in-grid config: {best}")

    or_row = best_outlier_removed(X_full, best)
    grid_with_or = pd.concat([grid, pd.DataFrame([or_row])], ignore_index=True)
    grid_with_or.to_csv(DATA_DIR / "kmeans_tuning.csv", index=False)

    if or_row["silhouette"] > best["silhouette"]:
        winner = or_row
        keep_outliers = False
        print(f"  outlier-removal improved silhouette: "
              f"{best['silhouette']} -> {or_row['silhouette']}  (winning)")
    else:
        winner = best
        keep_outliers = True
        print(f"  outlier-removal did NOT improve: kept={best['silhouette']}, "
              f"removed={or_row['silhouette']}  (keeping all points)")

    # Build the winning feature matrix
    n_dims  = int(winner["n_dims"])
    scaling = winner["scaling"]
    k       = int(winner["k"])

    X = X_full[:, :n_dims]
    if scaling == "std":
        X = StandardScaler().fit_transform(X)

    keep_mask = np.ones(X.shape[0], dtype=bool)
    if not keep_outliers:
        km_tmp, lab_tmp, _ = run_kmeans(X, k)
        keep_mask = remove_outliers(X, lab_tmp, km_tmp.cluster_centers_, OUTLIER_PCT)

    # Re-emit canonical artefacts on the winning input (full points if keep_outliers)
    X_eval = X[keep_mask]
    sweep = write_canonical_sweep_and_plots(X_eval, k)
    print("\n  canonical sweep on tuned input:")
    print(sweep.to_string(index=False))

    # Final labels file: keeps all 11,671 businesses; outliers (if removed) get -1
    km_final = KMeans(n_clusters=k, n_init=10, random_state=RS).fit(X_eval)
    final_labels = np.full(X.shape[0], -1, dtype=int)
    final_labels[keep_mask] = km_final.labels_

    index.assign(cluster_kmeans=final_labels).to_parquet(
        DATA_DIR / "kmeans_labels.parquet", index=False)

    print(f"\n  WINNER: n_dims={n_dims} scaling={scaling} k={k} "
          f"outliers={'removed_1pct' if not keep_outliers else 'kept'} "
          f"silhouette={winner['silhouette']}")
    print(f"  outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
