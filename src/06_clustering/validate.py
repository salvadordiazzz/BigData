"""Stage 6c — Clustering validation table.

Merges the K-Means and DBSCAN sweep results into a single comparison table
and plots a side-by-side silhouette comparison for the chosen configurations.

Inputs  : src/06_clustering/outputs/data/kmeans_sweep.csv
          src/06_clustering/outputs/data/dbscan_sweep.csv
Outputs : src/06_clustering/outputs/validation_table.csv
          src/06_clustering/outputs/plots/validation_summary.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR  = Path("src/06_clustering/outputs")
DATA_DIR = OUT_DIR / "data"
PLOT_DIR = OUT_DIR / "plots"


def main() -> None:
    print("Stage 6c — Validation table")

    km = pd.read_csv(DATA_DIR / "kmeans_sweep.csv")
    db = pd.read_csv(DATA_DIR / "dbscan_sweep.csv")

    km_rows = km.assign(method="KMeans",
                        params=km["k"].apply(lambda k: f"k={k}"),
                        n_clusters=km["k"],
                        noise_pct=0.0)[
        ["method", "params", "n_clusters", "noise_pct",
         "silhouette", "davies_bouldin", "inertia", "runtime_s"]]

    db_rows = db.assign(method="DBSCAN",
                        params=db.apply(lambda r: f"eps={r['eps']}, ms={int(r['min_samples'])}", axis=1),
                        inertia=None)[
        ["method", "params", "n_clusters", "noise_pct",
         "silhouette", "davies_bouldin", "inertia", "runtime_s"]]

    table = pd.concat([km_rows, db_rows], ignore_index=True)
    table.to_csv(OUT_DIR / "validation_table.csv", index=False)
    print(table.to_string(index=False))

    # Best-of-each comparison plot
    km_best = km.loc[km["silhouette"].idxmax()]
    db_valid = db[(db["n_clusters"] >= 2) & (db["noise_pct"] < 0.5)
                  & (db["silhouette"].notna())]
    db_best = db_valid.loc[db_valid["silhouette"].idxmax()] if not db_valid.empty else \
              db.loc[db["silhouette"].fillna(-1).idxmax()]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(km["k"], km["silhouette"], "o-", lw=2, color="steelblue", label="K-Means")
    ax.axvline(int(km_best["k"]), color="red", ls="--",
               label=f"best k = {int(km_best['k'])}  (sil={km_best['silhouette']:.3f})")
    ax.set_xlabel("k"); ax.set_ylabel("Silhouette")
    ax.set_title("K-Means silhouette vs. k"); ax.grid(alpha=0.3); ax.legend()

    ax = axes[1]
    for ms in sorted(db["min_samples"].unique()):
        sub = db[db["min_samples"] == ms].sort_values("eps")
        ax.plot(sub["eps"], sub["silhouette"], "o-", lw=2, label=f"min_samples={ms}")
    ax.set_xlabel("eps"); ax.set_ylabel("Silhouette")
    ax.set_title("DBSCAN silhouette vs. eps"); ax.grid(alpha=0.3); ax.legend()

    fig.tight_layout(); fig.savefig(PLOT_DIR / "validation_summary.png"); plt.close(fig)

    print(f"\n  best K-Means: k={int(km_best['k'])}  sil={km_best['silhouette']:.4f}")
    print(f"  best DBSCAN : eps={db_best['eps']} ms={int(db_best['min_samples'])} "
          f"clusters={int(db_best['n_clusters'])} sil={db_best['silhouette']:.4f}")
    print(f"  outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
