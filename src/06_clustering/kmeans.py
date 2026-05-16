"""Stage 6a — K-Means clustering with parameter sweep.

Sweeps k in [3, 5, 8, 10, 12, 15, 20, 25, 30] on the top-50 SVD components
of the combined matrix, records inertia + silhouette + DB index per k, and
saves cluster labels for the chosen k.

Inputs  : src/05_dimreduction/outputs/reduced/combined.parquet
          src/04_features/outputs/business_index.parquet
Outputs : src/06_clustering/outputs/data/kmeans_sweep.csv
          src/06_clustering/outputs/data/kmeans_labels.parquet
          src/06_clustering/outputs/plots/kmeans_elbow.png
          src/06_clustering/outputs/plots/kmeans_silhouette.png
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

RED_DIR  = Path("src/05_dimreduction/outputs/reduced")
FEAT_DIR = Path("src/04_features/outputs")
OUT_DIR  = Path("src/06_clustering/outputs")
PLOT_DIR = OUT_DIR / "plots"
DATA_DIR = OUT_DIR / "data"

RS         = 42
N_DIMS     = 50
K_VALUES   = [3, 5, 8, 10, 12, 15, 20, 25, 30]
SIL_SAMPLE = 5_000
CHOSEN_K_RULE = "best_silhouette"


def load_features() -> tuple[np.ndarray, pd.DataFrame]:
    red = pd.read_parquet(RED_DIR / "combined.parquet")
    index = pd.read_parquet(FEAT_DIR / "business_index.parquet")
    red = red.merge(index, on="business_id", how="left")
    X = red[[f"c{i+1}" for i in range(N_DIMS)]].to_numpy(dtype=np.float32)
    return X, red[["business_id", "business_name", "primary_category"]]


def sweep(X: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(RS)
    sample_idx = rng.choice(X.shape[0], size=min(SIL_SAMPLE, X.shape[0]), replace=False)
    Xs = X[sample_idx]

    rows = []
    for k in K_VALUES:
        t0 = time.time()
        km = KMeans(n_clusters=k, n_init=10, random_state=RS).fit(X)
        runtime = time.time() - t0
        labels = km.labels_
        sil = silhouette_score(Xs, labels[sample_idx]) if k > 1 else float("nan")
        db  = davies_bouldin_score(X, labels) if k > 1 else float("nan")
        rows.append({"k": k, "inertia": round(float(km.inertia_), 2),
                     "silhouette": round(float(sil), 4),
                     "davies_bouldin": round(float(db), 4),
                     "runtime_s": round(runtime, 2)})
        print(f"  k={k:>3}: inertia={km.inertia_:>10.0f}  sil={sil:.4f}  DB={db:.4f}  t={runtime:.1f}s")
    return pd.DataFrame(rows)


def elbow_plot(sweep_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sweep_df["k"], sweep_df["inertia"], "o-", lw=2, color="steelblue")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (within-cluster SSE)")
    ax.set_title("K-Means — elbow plot")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(PLOT_DIR / "kmeans_elbow.png"); plt.close(fig)


def silhouette_plot(sweep_df: pd.DataFrame, chosen_k: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sweep_df["k"], sweep_df["silhouette"], "o-", lw=2, color="seagreen", label="silhouette")
    ax.axvline(chosen_k, color="red", ls="--", lw=1, label=f"chosen k = {chosen_k}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score (higher = better)")
    ax.set_title("K-Means — silhouette vs. k")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(PLOT_DIR / "kmeans_silhouette.png"); plt.close(fig)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True); DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Stage 6a — K-Means sweep")

    X, index = load_features()
    print(f"  input: {X.shape}")

    sweep_df = sweep(X)
    sweep_df.to_csv(DATA_DIR / "kmeans_sweep.csv", index=False)

    chosen_k = int(sweep_df.loc[sweep_df["silhouette"].idxmax(), "k"])
    print(f"  chosen k (max silhouette) = {chosen_k}")

    km = KMeans(n_clusters=chosen_k, n_init=10, random_state=RS).fit(X)
    index.assign(cluster_kmeans=km.labels_).to_parquet(
        DATA_DIR / "kmeans_labels.parquet", index=False)

    elbow_plot(sweep_df)
    silhouette_plot(sweep_df, chosen_k)
    print(f"  outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
