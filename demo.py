#!/usr/bin/env python3
"""Hidden-Gem Discovery System -- End-to-End Demo.

Loads the blend-router predictions and graph analytics outputs produced by the
full pipeline, then prints personalised hidden-gem recommendations enriched
with PageRank percentile and Louvain community membership.

Requires the pipeline to have been run through Stages 07 and 08 (see RUNBOOK.md).

Usage
-----
    python demo.py                          # 3 example users (one per tier)
    python demo.py --user <user_id>         # specific user
    python demo.py --user <user_id> -k 5   # top-5 (default 10)
    python demo.py --list-users             # print 20 sample user IDs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
PRED_FILE  = Path("src/07_recommendation/outputs/data/predictions_fallback.parquet")
CAND_FILE  = Path("src/07_recommendation/outputs/data/candidates.parquet")
COMM_FILE  = Path("src/08_graph/outputs/data/community_labels.parquet")
TRAIN_FILE = Path("src/07_recommendation/outputs/data/train.parquet")

# ── constants ─────────────────────────────────────────────────────────────────
HIDDEN_GEM_CLUSTER   = 1
HIDDEN_GEM_COMMUNITY = 3
COLD_MAX   = 10
MID_MAX    = 30
BLEND_STR  = "65% ALS + 20% Content + 15% Pop"

# Example users: one per tier (cold/mid/warm) for the no-arg default view
EXAMPLE_USERS: list[tuple[str, str]] = [
    ("cold", "0ZRTwd5xyGwo4cW1vgQJNg"),   # 7 train reviews, 2/2 hits
    ("mid",  "-4RbxLJlFZlu-KRuUiiGLw"),   # 11 train reviews
    ("warm", "-2cKJFFNJ9XVyWBt62mWvA"),   # 51 train reviews
]

# ── helpers ───────────────────────────────────────────────────────────────────

def _check_files() -> None:
    missing = [f for f in [PRED_FILE, CAND_FILE, COMM_FILE, TRAIN_FILE] if not f.exists()]
    if missing:
        print("ERROR: pipeline outputs not found. Run stages 07-08 first (see RUNBOOK.md).")
        for f in missing:
            print(f"  missing: {f}")
        sys.exit(1)


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    preds = pd.read_parquet(PRED_FILE)
    cand  = pd.read_parquet(CAND_FILE)
    comm  = pd.read_parquet(COMM_FILE)[
        ["business_id", "pagerank", "community", "cluster_kmeans"]
    ]
    train = pd.read_parquet(TRAIN_FILE, columns=["user_id"])
    return preds, cand, comm, train


def _train_count(user_id: str, train: pd.DataFrame) -> int:
    return int((train["user_id"] == user_id).sum())


def _tier_label(n: int) -> str:
    if n <= COLD_MAX:
        return f"COLD  ({n} train reviews)  ->  blend: {BLEND_STR}"
    if n <= MID_MAX:
        return f"MID   ({n} train reviews)  ->  blend: {BLEND_STR}"
    return f"WARM  ({n} train reviews)  ->  pure ALS"


def _pagerank_pct(pr: float | None, all_pr: pd.Series) -> str:
    if pr is None or pd.isna(pr):
        return "no graph"
    pct = int(100 * (all_pr < pr).sum() / len(all_pr))
    return f"p{pct}"


def _community_label(com: float | None) -> str:
    if com is None or pd.isna(com):
        return "no graph"
    c = int(com)
    return f"C{c} (*)" if c == HIDDEN_GEM_COMMUNITY else f"C{c}"


def _cluster_label(cl: float | None) -> str:
    if cl is None or pd.isna(cl):
        return "unknown"
    return "HidGem (*)" if int(cl) == HIDDEN_GEM_CLUSTER else "Mainstream"


# ── rendering ─────────────────────────────────────────────────────────────────

W_NAME = 34
W_CAT  = 18
W_CL   = 12
W_PR   = 10
W_COM  = 10
W_SC   =  6
HDR = (
    f"  # | {'Business':<{W_NAME}} | {'Category':<{W_CAT}}"
    f" | {'Cluster':^{W_CL}} | {'PR%ile':>{W_PR}} | {'Community':>{W_COM}} | Score"
)


def _show_user(
    user_id: str,
    top_k: int,
    preds: pd.DataFrame,
    cand: pd.DataFrame,
    comm: pd.DataFrame,
    train: pd.DataFrame,
    all_pr: pd.Series,
) -> None:
    user_preds = preds[preds["user_id"] == user_id].sort_values("rank").head(top_k)
    if user_preds.empty:
        print(f"  User '{user_id}' not found in test predictions.")
        print("  Try --list-users to see available sample IDs.")
        return

    n_train = _train_count(user_id, train)

    merged = (
        user_preds
        .merge(cand, on="business_id", how="left")
        .merge(comm, on="business_id", how="left")
    )

    print(f"\nUser: {user_id}")
    print(f"Tier: {_tier_label(n_train)}")
    print()
    print(HDR)
    print("-" * len(HDR))

    for _, row in merged.sort_values("rank").iterrows():
        name = str(row.get("business_name", "N/A"))[:W_NAME]
        cat  = str(row.get("primary_category", "N/A"))[:W_CAT]
        cl   = _cluster_label(row.get("cluster_kmeans"))
        pr   = _pagerank_pct(row.get("pagerank"), all_pr)
        com  = _community_label(row.get("community"))
        score = float(row["score"])
        print(
            f"{int(row['rank']):>3} | {name:<{W_NAME}} | {cat:<{W_CAT}}"
            f" | {cl:^{W_CL}} | {pr:>{W_PR}} | {com:>{W_COM}} | {score:.4f}"
        )

    print()
    print("(*) HidGem = Cluster 1 (PC3 K-Means hidden gem segment)")
    print("(*) C3     = Community 3 (Louvain, 55% Cluster-1 businesses)")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Yelp Hidden-Gem Discovery -- personalised recommendations with graph enrichment"
    )
    parser.add_argument("--user", default=None, help="Yelp user_id to query")
    parser.add_argument("-k", "--top", type=int, default=10, help="Top-K results (default 10)")
    parser.add_argument("--list-users", action="store_true",
                        help="Print 20 sample user IDs from the test set and exit")
    args = parser.parse_args()

    _check_files()
    print("Loading pipeline outputs...", end=" ", flush=True)
    preds, cand, comm, train = _load()
    all_pr = comm["pagerank"].dropna()
    print("done.")

    banner = "  Yelp Hidden-Gem Discovery System -- Philadelphia"
    sub    = "  Pipeline: Ingestion > Features > Clustering > Router > Graph"
    width  = max(len(banner), len(sub)) + 4
    print("\n" + "=" * width)
    print(banner)
    print(sub)
    print("=" * width)

    if args.list_users:
        sample_ids = preds["user_id"].unique()[:20]
        print(f"\n20 sample user IDs (out of {preds['user_id'].nunique():,} in test set):")
        print(f"  {'user_id':<40}  tier   train_reviews")
        print("  " + "-" * 60)
        for uid in sample_ids:
            n = _train_count(uid, train)
            t = "cold" if n <= COLD_MAX else ("mid" if n <= MID_MAX else "warm")
            print(f"  {uid:<40}  {t:<6} {n}")
        return

    if args.user:
        _show_user(args.user, args.top, preds, cand, comm, train, all_pr)
    else:
        print(f"\nNo --user given. Showing {len(EXAMPLE_USERS)} example users (one per tier).")
        print("Use --user <id> for a specific user, or --list-users for more sample IDs.\n")
        for tier_hint, uid in EXAMPLE_USERS:
            print("-" * 72)
            print(f"  Example: {tier_hint.upper()} user")
            _show_user(uid, args.top, preds, cand, comm, train, all_pr)


if __name__ == "__main__":
    main()
