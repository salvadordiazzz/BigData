"""Stage 6d — Cluster profile and failure analysis.

Joins K-Means labels with business_features.parquet and the t-SNE coordinates,
then characterises each cluster (size, mean ratings, top categories) and
plots the t-SNE 2D map coloured by cluster.

Inputs  : src/06_clustering/outputs/data/kmeans_labels.parquet
          src/06_clustering/outputs/data/dbscan_labels.parquet
          src/04_features/outputs/business_features.parquet
          src/05_dimreduction/outputs/reduced/tsne_2d.parquet
Outputs : src/06_clustering/outputs/cluster_profiles.csv
          src/06_clustering/outputs/failure_analysis.csv
          src/06_clustering/outputs/plots/tsne_clusters.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FEAT_DIR = Path("src/04_features/outputs")
RED_DIR  = Path("src/05_dimreduction/outputs/reduced")
OUT_DIR  = Path("src/06_clustering/outputs")
DATA_DIR = OUT_DIR / "data"
PLOT_DIR = OUT_DIR / "plots"

NUMERIC_PROFILE_COLS = [
    "business_stars", "log_review_count", "mean_review_stars",
    "std_review_stars", "mean_useful", "days_since_last",
    "reviews_per_year_recent",
]


def top_categories(df: pd.DataFrame, top_n: int = 3) -> str:
    counts = df["primary_category"].value_counts().head(top_n)
    return "; ".join(f"{c} ({n})" for c, n in counts.items())


def profile(features: pd.DataFrame, label_col: str) -> pd.DataFrame:
    rows = []
    for cid, sub in features.groupby(label_col):
        row = {"cluster": int(cid), "n_businesses": len(sub)}
        for c in NUMERIC_PROFILE_COLS:
            if c in sub.columns:
                row[f"mean_{c}"] = round(float(sub[c].mean()), 3)
        row["top_categories"] = top_categories(sub)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


def tsne_cluster_plot(tsne: pd.DataFrame, labels: pd.Series, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.cm.tab20.colors  # type: ignore[attr-defined]
    uniq = sorted(labels.unique())
    for i, cid in enumerate(uniq):
        m = labels == cid
        color = "lightgrey" if cid == -1 else cmap[i % len(cmap)]
        name  = "noise" if cid == -1 else f"cluster {cid}"
        ax.scatter(tsne.loc[m, "tsne_1"], tsne.loc[m, "tsne_2"],
                   c=[color], label=name, s=6, alpha=0.6, linewidths=0)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.set_title(title)
    ax.legend(markerscale=3, fontsize=8, ncol=2); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def failure_analysis(features: pd.DataFrame, km_profile: pd.DataFrame,
                     db_labels: pd.DataFrame) -> pd.DataFrame:
    notes: list[dict] = []

    sizes = km_profile["n_businesses"]
    max_share = sizes.max() / sizes.sum()
    if max_share > 0.50:
        notes.append({"issue": "K-Means imbalance",
                      "value": f"largest cluster holds {max_share:.1%} of businesses",
                      "interpretation":
                          "One cluster absorbs a heterogeneous tail (likely 'Restaurants'/'Food'); "
                          "k may be too small or text dimension overshadows category structure."})

    small = km_profile[km_profile["n_businesses"] < 50]
    if len(small) > 0:
        notes.append({"issue": "K-Means tiny clusters",
                      "value": f"{len(small)} cluster(s) with <50 businesses",
                      "interpretation":
                          "Possible micro-niches (e.g., very specific cuisines) or boundary points; "
                          "useful for discovery but unstable to small parameter changes."})

    noise_pct = (db_labels["cluster_dbscan"] == -1).mean()
    notes.append({"issue": "DBSCAN noise",
                  "value": f"{noise_pct:.1%} of points labelled as noise",
                  "interpretation":
                      "DBSCAN struggles in ~50-dim SVD space: distances are dense and the eps band "
                      "is narrow. High noise means many businesses are not in any density region, "
                      "i.e., the feature space is locally sparse for this domain."})

    db_clusters = sorted(set(db_labels["cluster_dbscan"]) - {-1})
    if len(db_clusters) <= 3:
        notes.append({"issue": "DBSCAN under-segments",
                      "value": f"only {len(db_clusters)} dense cluster(s) found",
                      "interpretation":
                          "Density-based methods are not the right fit here; convex K-Means partitions "
                          "the SVD space more meaningfully than density connectivity."})

    return pd.DataFrame(notes)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print("Stage 6d — Cluster profiles + failure analysis")

    features = pd.read_parquet(FEAT_DIR / "business_features.parquet")
    km_labels = pd.read_parquet(DATA_DIR / "kmeans_labels.parquet")
    db_labels = pd.read_parquet(DATA_DIR / "dbscan_labels.parquet")
    tsne = pd.read_parquet(RED_DIR / "tsne_2d.parquet")

    features = features.merge(
        km_labels[["business_id", "cluster_kmeans"]], on="business_id", how="left")
    features = features.merge(
        db_labels[["business_id", "cluster_dbscan"]], on="business_id", how="left")

    km_profile = profile(features, "cluster_kmeans")
    km_profile.insert(0, "method", "KMeans")
    db_profile = profile(features, "cluster_dbscan")
    db_profile.insert(0, "method", "DBSCAN")
    full = pd.concat([km_profile, db_profile], ignore_index=True)
    full.to_csv(OUT_DIR / "cluster_profiles.csv", index=False)

    print("\n  K-Means profile (first rows):")
    print(km_profile.head().to_string(index=False))

    fa = failure_analysis(features, km_profile, db_labels)
    fa.to_csv(OUT_DIR / "failure_analysis.csv", index=False)
    print("\n  failure-analysis notes:")
    for _, r in fa.iterrows():
        print(f"   - {r['issue']}: {r['value']}")

    tsne_id = tsne.merge(
        km_labels[["business_id", "cluster_kmeans"]], on="business_id", how="inner")
    tsne_cluster_plot(tsne_id, tsne_id["cluster_kmeans"],
                      "t-SNE — businesses coloured by K-Means cluster",
                      PLOT_DIR / "tsne_clusters.png")

    tsne_db = tsne.merge(
        db_labels[["business_id", "cluster_dbscan"]], on="business_id", how="inner")
    tsne_cluster_plot(tsne_db, tsne_db["cluster_dbscan"],
                      "t-SNE — businesses coloured by DBSCAN cluster (grey = noise)",
                      PLOT_DIR / "tsne_clusters_dbscan.png")

    print(f"\n  outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
