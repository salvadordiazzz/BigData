"""Stage 5b — Comparison Table.
Reads reduction_meta.json and computes explained variance + reconstruction error per matrix.
Inputs  : src/05_dimreduction/outputs/reduction_meta.json  src/04_features/outputs/
Outputs : src/05_dimreduction/outputs/comparison_table.csv
          src/05_dimreduction/outputs/plots/loadings_top_pcs.png
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD

FEAT_DIR = Path("src/04_features/outputs")
OUT_DIR  = Path("src/05_dimreduction/outputs")
PLOT_DIR = OUT_DIR / "plots"
RS       = 42
EVAL_KS  = [10, 20, 50, 100]
RECON_K  = 50


def cumvar(evr: list, k: int) -> float:
    return float(np.cumsum(evr)[min(k, len(evr)) - 1])


def recon_err(X, evr: list, k: int, method: str) -> float:
    k = min(k, len(evr))
    Xd = X.toarray() if sp.issparse(X) else X
    if method == "PCA":
        m = PCA(n_components=k, random_state=RS); Xr = m.fit_transform(Xd); Xrec = m.inverse_transform(Xr)
    else:
        m = TruncatedSVD(n_components=k, random_state=RS); Xr = m.fit_transform(X); Xrec = m.inverse_transform(Xr)
        Xd = Xd
    n = np.linalg.norm(Xd, "fro")
    return float(np.linalg.norm(Xd - Xrec, "fro") / n) if n > 0 else 0.0


def loadings_heatmap() -> None:
    X = sp.load_npz(FEAT_DIR / "X_business_dense.npz").toarray()
    with open(FEAT_DIR / "feature_names.json") as f:
        names = json.load(f)
    feat_names = names["numeric_temporal"] + names["categorical"]

    pca = PCA(n_components=5, random_state=RS).fit(X)
    load = pd.DataFrame(pca.components_.T, index=feat_names,
                        columns=[f"PC{i+1}" for i in range(5)])
    top20 = load.abs().max(axis=1).nlargest(20).index
    load_top = load.loc[top20]

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(load_top.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(5)); ax.set_xticklabels(load_top.columns)
    ax.set_yticks(range(len(top20))); ax.set_yticklabels(top20, fontsize=8)
    ax.set_title("PCA loadings — top 20 features (dense)")
    plt.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout(); fig.savefig(PLOT_DIR / "loadings_top_pcs.png"); plt.close(fig)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print("Stage 5b — Comparison Table")

    with open(OUT_DIR / "reduction_meta.json") as f:
        meta = json.load(f)

    mats    = {"dense (PCA)": sp.load_npz(FEAT_DIR / "X_business_dense.npz"),
               "text (SVD)":  sp.load_npz(FEAT_DIR / "X_business_text.npz"),
               "combined (SVD)": sp.load_npz(FEAT_DIR / "X_business_combined.npz")}
    methods = {"dense (PCA)": "PCA", "text (SVD)": "TruncatedSVD", "combined (SVD)": "TruncatedSVD"}

    rows = []
    for label, info in meta.items():
        evr = info["evr"]
        row = {"matrix": label, "method": methods[label], "n_features_in": info["n_features_in"]}
        for k in EVAL_KS:
            row[f"cum_var_at_{k}"] = round(cumvar(evr, k), 4) if k <= len(evr) else None
        print(f"  reconstruction error for {label}...")
        row[f"recon_error_at_{RECON_K}"] = round(recon_err(mats[label], evr, RECON_K, methods[label]), 4)
        row["runtime_s"] = round(info["runtime_s"], 1)
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "comparison_table.csv", index=False)
    print(table.to_string(index=False))

    loadings_heatmap()
    print(f"  outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
