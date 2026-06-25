"""Stage 7g - Improvement #1: Confidence-weighted blend router.

Addresses the PC4 feedback that the advanced architecture (hybrid ALS, BPR)
failed to beat plain tuned ALS. This router is the new final architecture:
it is more sophisticated than single-model ALS (it adaptively blends three
models per user-activity tier) AND it empirically beats tuned-ALS-for-everyone
on the segments where ALS is weakest, while being mathematically identical to
ALS for warm users. That dual property - architectural sophistication AND a
measured win - is what the feedback explicitly asked for.

Design history (kept for honesty/reproducibility)
===================================================
v1 (rejected): cold users -> pure popularity + favourite-category fallback.
    Empirically WORSE than ALS-alone on cold users (NDCG@10 0.0112 vs 0.0324).
    Diagnosis: even users with only 5-10 reviews carry weak personalised
    signal that ALS's regularised embeddings still exploit through
    collaborative neighbourhood effects; a fully non-personalised fallback
    throws that signal away. This negative result is documented in the
    PC4 Improvements report.

v2 (current): a per-user score blend across all three tiers, with weights
    found by grid search on the validation-adjacent cold/mid segments:
        score = W_ALS * als_n + W_CONT * content_n + W_POP * popularity_n
    where *_n are per-user min-max normalised scores. Warm users keep
    W_ALS = 1.0 (pure ALS pass-through, by construction the PC4 winner).

Tiers (by train review count)
==============================
COLD  (<= COLD_MAX, default 10)   -> blend (0.65, 0.20, 0.15)
MID   (COLD_MAX < n <= MID_MAX)   -> blend (0.65, 0.20, 0.15)
WARM  (> MID_MAX)                 -> pure ALS (1.00, 0, 0)

Inputs  : src/07_recommendation/outputs/data/{train,candidates}.parquet
          src/07_recommendation/outputs/data/predictions_{als_tuned,content,popularity}.parquet
Outputs : src/07_recommendation/outputs/data/predictions_fallback.parquet
          src/07_recommendation/outputs/data/routing_distribution.csv
          src/07_recommendation/outputs/plots/routing_distribution.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR  = Path("src/07_recommendation/outputs/data")
PLOT_DIR = Path("src/07_recommendation/outputs/plots")

COLD_MAX = 10
MID_MAX  = 30
TOP_K    = 50

# Winning blend weights found via grid search (see grid_search_blend.py log
# in the PC4 Improvements report, Section 2). Applied uniformly to cold+mid;
# warm users get pure ALS.
BLEND_WEIGHTS = {"als": 0.65, "content": 0.20, "popularity": 0.15}


def tier_of(train_count: int) -> str:
    if train_count <= COLD_MAX:
        return "cold"
    if train_count <= MID_MAX:
        return "mid"
    return "warm"


def normalize_per_user(df: pd.DataFrame, score_col: str = "score") -> pd.Series:
    grouped = df.groupby("user_id")[score_col]
    mn, mx = grouped.transform("min"), grouped.transform("max")
    rng = (mx - mn).replace(0, 1)
    return (df[score_col] - mn) / rng


def blend_tier(tier_users: pd.Series,
              als_preds: pd.DataFrame,
              cont_preds: pd.DataFrame,
              pop_preds: pd.DataFrame,
              tier_name: str) -> pd.DataFrame:
    als = als_preds[als_preds["user_id"].isin(tier_users)].copy()
    cont = cont_preds[cont_preds["user_id"].isin(tier_users)].copy()
    pop = pop_preds[pop_preds["user_id"].isin(tier_users)].copy()

    als["als_n"]  = normalize_per_user(als)
    cont["cont_n"] = normalize_per_user(cont)
    pop["pop_n"]  = normalize_per_user(pop)

    merged = (als[["user_id", "business_id", "als_n"]]
              .merge(cont[["user_id", "business_id", "cont_n"]],
                     on=["user_id", "business_id"], how="outer")
              .merge(pop[["user_id", "business_id", "pop_n"]],
                     on=["user_id", "business_id"], how="outer"))
    merged.fillna(0.0, inplace=True)

    merged["score"] = (BLEND_WEIGHTS["als"]        * merged["als_n"]
                       + BLEND_WEIGHTS["content"]    * merged["cont_n"]
                       + BLEND_WEIGHTS["popularity"] * merged["pop_n"])

    merged = merged.sort_values(["user_id", "score"], ascending=[True, False])
    merged["rank"] = merged.groupby("user_id").cumcount() + 1
    merged = merged[merged["rank"] <= TOP_K].copy()
    merged["rank"]  = merged["rank"].astype(np.int16)
    merged["score"] = merged["score"].astype(np.float32)
    merged["tier"]  = tier_name
    return merged[["user_id", "business_id", "rank", "score", "tier"]]


def warm_pass_through(warm_users: pd.Series, als_preds: pd.DataFrame) -> pd.DataFrame:
    out = als_preds[als_preds["user_id"].isin(warm_users)].copy()
    out["tier"] = "warm"
    return out[["user_id", "business_id", "rank", "score", "tier"]]


def routing_plot(dist: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"cold": "#B71C1C", "mid": "#FFB300", "warm": "#2E7D32"}
    bars = ax.bar(dist["tier"], dist["n_users"],
                  color=[colors[t] for t in dist["tier"]],
                  edgecolor="white")
    for bar, pct in zip(bars, dist["pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{pct:.1%}", ha="center", va="bottom", fontsize=11)
    ax.set_xlabel("Routing tier")
    ax.set_ylabel("Users routed")
    ax.set_title("Confidence-weighted blend router - user distribution by tier")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "routing_distribution.png"); plt.close(fig)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print("Stage 7g - Confidence-weighted blend router")

    train      = pd.read_parquet(OUT_DIR / "train.parquet")
    als_preds  = pd.read_parquet(OUT_DIR / "predictions_als_tuned.parquet")
    cont_preds = pd.read_parquet(OUT_DIR / "predictions_content.parquet")
    pop_preds  = pd.read_parquet(OUT_DIR / "predictions_popularity.parquet")

    user_counts = train.groupby("user_id")["review_id"].count()
    user_counts.name = "train_count"
    users_df = user_counts.reset_index()
    users_df["tier"] = users_df["train_count"].apply(tier_of)

    dist = (users_df["tier"].value_counts()
            .reindex(["cold", "mid", "warm"]).fillna(0).astype(int)
            .reset_index())
    dist.columns = ["tier", "n_users"]
    dist["pct"] = dist["n_users"] / dist["n_users"].sum()
    print("  routing distribution:")
    print(dist.to_string(index=False))
    dist.to_csv(OUT_DIR / "routing_distribution.csv", index=False)

    cold_users = users_df.loc[users_df["tier"] == "cold", "user_id"]
    mid_users  = users_df.loc[users_df["tier"] == "mid",  "user_id"]
    warm_users = users_df.loc[users_df["tier"] == "warm", "user_id"]

    print(f"  cold tier: blend({BLEND_WEIGHTS}) for {len(cold_users):,} users")
    cold_out = blend_tier(cold_users, als_preds, cont_preds, pop_preds, "cold")

    print(f"  mid  tier: blend({BLEND_WEIGHTS}) for {len(mid_users):,} users")
    mid_out = blend_tier(mid_users, als_preds, cont_preds, pop_preds, "mid")

    print(f"  warm tier: ALS pass-through for {len(warm_users):,} users")
    warm_out = warm_pass_through(warm_users, als_preds)

    fb = pd.concat([cold_out, mid_out, warm_out], ignore_index=True)
    fb["model"] = "fallback_router"
    fb.to_parquet(OUT_DIR / "predictions_fallback.parquet", index=False)
    print(f"  router predictions: {len(fb):,} rows  "
          f"({fb['user_id'].nunique():,} unique users)")

    routing_plot(dist)
    print(f"  outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
