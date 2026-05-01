"""Stage 5a — Dimensionality Reduction.
Runs PCA (dense), TruncatedSVD (text + combined) and t-SNE.
Inputs  : src/04_features/outputs/
Outputs : src/05_dimreduction/outputs/
"""
import json, time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

FEAT_DIR   = Path("src/04_features/outputs")
OUT_DIR    = Path("src/05_dimreduction/outputs")
PLOT_DIR   = OUT_DIR / "plots"
RED_DIR    = OUT_DIR / "reduced"
RS         = 42


def fit(X, method: str, n: int) -> tuple:
    n = min(n, X.shape[1] - (1 if method == "svd" else 0))
    t0 = time.time()
    if method == "pca":
        m = PCA(n_components=n, random_state=RS)
    else:
        m = TruncatedSVD(n_components=n, random_state=RS)
    Xr = m.fit_transform(X.toarray() if (method == "pca" and sp.issparse(X)) else X)
    return m, Xr, time.time() - t0


def save_parquet(Xr: np.ndarray, ids, name: str) -> None:
    cols = [f"c{i+1}" for i in range(Xr.shape[1])]
    pd.DataFrame(Xr, columns=cols).assign(business_id=ids.values).to_parquet(
        RED_DIR / f"{name}.parquet", index=False)


def scree_plot(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, res in results.items():
        ev = np.cumsum(res["evr"])
        ax.plot(range(1, len(ev) + 1), ev, lw=2, label=label)
    for t in (0.80, 0.90, 0.95):
        ax.axhline(t, color="grey", ls=":", lw=0.8)
        ax.text(1, t + 0.005, f"{int(t*100)}%", fontsize=8, color="grey")
    ax.set_xlabel("Components (k)"); ax.set_ylabel("Cumulative explained variance")
    ax.set_title("Scree plot"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(PLOT_DIR / "scree_all.png"); plt.close(fig)


def tsne_plot(Xr: np.ndarray, index_df: pd.DataFrame) -> None:
    top = index_df["primary_category"].value_counts().head(10).index.tolist()
    labels = index_df["primary_category"].apply(lambda c: c if c in top else "Other")
    cats   = top + ["Other"]
    cmap   = plt.cm.tab20.colors  # type: ignore[attr-defined]
    cdict  = {c: cmap[i % len(cmap)] for i, c in enumerate(cats)}

    X2 = TSNE(n_components=2, perplexity=30, random_state=RS,
               init="pca", learning_rate="auto").fit_transform(Xr[:, :50])
    np.save(RED_DIR / "tsne_2d.npy", X2)
    index_df.assign(tsne_1=X2[:, 0], tsne_2=X2[:, 1]).to_parquet(
        RED_DIR / "tsne_2d.parquet", index=False)

    fig, ax = plt.subplots(figsize=(12, 8))
    for cat in cats:
        m = labels == cat
        ax.scatter(X2[m, 0], X2[m, 1], c=[cdict[cat]], label=cat, s=6, alpha=0.6, linewidths=0)
    ax.set_title("t-SNE — businesses colored by primary category")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=3, fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(PLOT_DIR / "tsne_business.png"); plt.close(fig)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    RED_DIR.mkdir(parents=True, exist_ok=True)
    print("Stage 5a — Dimensionality Reduction")

    mats = {
        "dense (PCA)":    sp.load_npz(FEAT_DIR / "X_business_dense.npz"),
        "text (SVD)":     sp.load_npz(FEAT_DIR / "X_business_text.npz"),
        "combined (SVD)": sp.load_npz(FEAT_DIR / "X_business_combined.npz"),
    }
    index_df = pd.read_parquet(FEAT_DIR / "business_index.parquet")
    ids = pd.Index(index_df["business_id"])

    results, Xr_combined = {}, None
    for label, X in mats.items():
        method = "pca" if "PCA" in label else "svd"
        n      = 63 if method == "pca" else 200
        m, Xr, t = fit(X, method, n)
        results[label] = {"evr": m.explained_variance_ratio_.tolist(),
                          "n_features_in": X.shape[1], "runtime_s": t}
        save_parquet(Xr, ids, label.split()[0])
        if "combined" in label:
            Xr_combined = Xr
        print(f"  {label}: {n} components in {t:.1f}s")

    scree_plot(results)

    print("  t-SNE (~1 min)...")
    t0 = time.time()
    tsne_plot(Xr_combined, index_df)
    print(f"  t-SNE done in {time.time()-t0:.1f}s")

    with open(OUT_DIR / "reduction_meta.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
