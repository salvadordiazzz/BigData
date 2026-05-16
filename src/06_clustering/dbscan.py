"""Stage 6b — DBSCAN clustering with parameter sweep.

Sweeps eps x min_samples on the top-50 SVD components of the combined matrix,
records cluster count, noise fraction, silhouette and DB index per config,
and saves labels for the chosen configuration.

Inputs  : src/05_dimreduction/outputs/reduced/combined.parquet
          src/04_features/outputs/business_index.parquet
Outputs : src/06_clustering/outputs/data/dbscan_sweep.csv
          src/06_clustering/outputs/data/dbscan_labels.parquet
          src/06_clustering/outputs/plots/dbscan_kdistance.png
          src/06_clustering/outputs/plots/dbscan_noise_vs_eps.png
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

RED_DIR  = Path("src/05_dimreduction/outputs/reduced")
FEAT_DIR = Path("src/04_features/outputs")
OUT_DIR  = Path("src/06_clustering/outputs")
PLOT_DIR = OUT_DIR / "plots"
DATA_DIR = OUT_DIR / "data"

RS         = 42
N_DIMS     = 50
EPS_VALUES = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
MIN_SAMPS  = [5, 10, 20]
SIL_SAMPLE = 5_000


def load_features() -> tuple[np.ndarray, pd.DataFrame]:
    red = pd.read_parquet(RED_DIR / "combined.parquet")
    index = pd.read_parquet(FEAT_DIR / "business_index.parquet")
    red = red.merge(index, on="business_id", how="left")
    X = red[[f"c{i+1}" for i in range(N_DIMS)]].to_numpy(dtype=np.float32)
    return X, red[["business_id", "business_name", "primary_category"]]


def k_distance_plot(X: np.ndarray, k: int = 10) -> None:
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    d, _ = nn.kneighbors(X)
    dk = np.sort(d[:, -1])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(len(dk)), dk, color="steelblue")
    ax.set_xlabel("Points (sorted)")
    ax.set_ylabel(f"Distance to {k}-th nearest neighbour")
    ax.set_title(f"DBSCAN — k-distance plot (k={k}); knee suggests eps")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(PLOT_DIR / "dbscan_kdistance.png"); plt.close(fig)


def sweep(X: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(RS)
    sample_idx = rng.choice(X.shape[0], size=min(SIL_SAMPLE, X.shape[0]), replace=False)
    Xs = X[sample_idx]

    rows = []
    for ms in MIN_SAMPS:
        for eps in EPS_VALUES:
            t0 = time.time()
            db = DBSCAN(eps=eps, min_samples=ms, n_jobs=-1).fit(X)
            runtime = time.time() - t0
            labels = db.labels_
            n_clusters = int((labels >= 0).any() and labels[labels >= 0].max() + 1)
            n_noise    = int((labels == -1).sum())
            noise_pct  = n_noise / len(labels)
            non_noise  = labels != -1
            if n_clusters >= 2 and non_noise.sum() > 1:
                ls = labels[sample_idx]
                mask = ls != -1
                sil = silhouette_score(Xs[mask], ls[mask]) if mask.sum() > 1 and len(set(ls[mask])) > 1 else float("nan")
                dbi = davies_bouldin_score(X[non_noise], labels[non_noise])
            else:
                sil, dbi = float("nan"), float("nan")
            rows.append({
                "eps": eps, "min_samples": ms, "n_clusters": n_clusters,
                "n_noise": n_noise, "noise_pct": round(noise_pct, 4),
                "silhouette": round(float(sil), 4) if not np.isnan(sil) else None,
                "davies_bouldin": round(float(dbi), 4) if not np.isnan(dbi) else None,
                "runtime_s": round(runtime, 2),
            })
            print(f"  eps={eps:>4}  ms={ms:>3}  clusters={n_clusters:>3}  "
                  f"noise={noise_pct:>5.1%}  sil={sil:.4f}  t={runtime:.1f}s")
    return pd.DataFrame(rows)


def noise_plot(sweep_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for ms in sorted(sweep_df["min_samples"].unique()):
        sub = sweep_df[sweep_df["min_samples"] == ms].sort_values("eps")
        ax.plot(sub["eps"], sub["noise_pct"], "o-", lw=2, label=f"min_samples={ms}")
    ax.set_xlabel("eps")
    ax.set_ylabel("Noise fraction")
    ax.set_title("DBSCAN — noise fraction vs. eps")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(PLOT_DIR / "dbscan_noise_vs_eps.png"); plt.close(fig)


def choose_config(sweep_df: pd.DataFrame) -> tuple[float, int]:
    valid = sweep_df[(sweep_df["n_clusters"] >= 2)
                     & (sweep_df["noise_pct"] < 0.50)
                     & (sweep_df["silhouette"].notna())]
    if valid.empty:
        valid = sweep_df[sweep_df["silhouette"].notna()]
    if valid.empty:
        return EPS_VALUES[len(EPS_VALUES) // 2], MIN_SAMPS[0]
    best = valid.loc[valid["silhouette"].idxmax()]
    return float(best["eps"]), int(best["min_samples"])


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True); DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Stage 6b — DBSCAN sweep")

    X, index = load_features()
    print(f"  input: {X.shape}")

    k_distance_plot(X, k=10)

    sweep_df = sweep(X)
    sweep_df.to_csv(DATA_DIR / "dbscan_sweep.csv", index=False)
    noise_plot(sweep_df)

    eps, ms = choose_config(sweep_df)
    print(f"  chosen config: eps={eps}, min_samples={ms}")

    db = DBSCAN(eps=eps, min_samples=ms, n_jobs=-1).fit(X)
    index.assign(cluster_dbscan=db.labels_).to_parquet(
        DATA_DIR / "dbscan_labels.parquet", index=False)
    print(f"  outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
