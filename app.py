"""Hidden-Gem Discovery System — Streamlit Demo.

Two modes:
  1. Build My Profile: pick categories + discovery style → content-based recs
  2. Explore Real Users: choose a user from the test set → blend-router recs

Run from repo root:
    streamlit run app.py

Requires pipeline stages 04-08 to have been executed (see RUNBOOK.md).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# ── paths ─────────────────────────────────────────────────────────────────────
CAND_FILE  = Path("src/07_recommendation/outputs/data/candidates.parquet")
COMM_FILE  = Path("src/08_graph/outputs/data/community_labels.parquet")
PRED_FILE  = Path("src/07_recommendation/outputs/data/predictions_fallback.parquet")
TRAIN_FILE = Path("src/07_recommendation/outputs/data/train.parquet")
SVD_FILE   = Path("src/05_dimreduction/outputs/reduced/svd_combined.parquet")
BIX_FILE   = Path("src/04_features/outputs/business_index.parquet")

COLD_MAX   = 10
MID_MAX    = 30
HG_CLUSTER = 1
HG_COMM    = 3

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hidden-Gem Discovery | Philadelphia",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.gem-card {
    background: #1a1a2e;
    border: 1px solid #333366;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 10px;
    min-height: 200px;
}
.gem-card h4 { color: #f5a623; margin: 0 0 4px 0; font-size: 15px; }
.gem-card .cat { color: #9999cc; font-size: 12px; margin-bottom: 10px; }
.badge-hg  { background:#e94f37; color:white; border-radius:6px; padding:2px 8px; font-size:11px; }
.badge-c3  { background:#27ae60; color:white; border-radius:6px; padding:2px 8px; font-size:11px; }
.badge-pr  { background:#2980b9; color:white; border-radius:6px; padding:2px 8px; font-size:11px; }
.badge-ms  { background:#555; color:#aaa; border-radius:6px; padding:2px 8px; font-size:11px; }
.score-bar-bg { background:#333; border-radius:4px; height:6px; margin:8px 0 4px; }
.score-bar    { background:#f5a623; border-radius:4px; height:6px; }
.tier-cold { color:#e74c3c; font-weight:bold; }
.tier-mid  { color:#e67e22; font-weight:bold; }
.tier-warm { color:#27ae60; font-weight:bold; }
</style>
""", unsafe_allow_html=True)


# ── data loaders (cached) ─────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading candidate pool...")
def load_pool() -> pd.DataFrame:
    cand = pd.read_parquet(CAND_FILE)
    comm = pd.read_parquet(COMM_FILE)[
        ["business_id", "pagerank", "community", "cluster_kmeans"]
    ]
    pool = cand.merge(comm, on="business_id", how="left")
    return pool


@st.cache_data(show_spinner="Loading SVD embeddings...")
def load_svd() -> tuple[pd.DataFrame, pd.DataFrame]:
    svd = pd.read_parquet(SVD_FILE)
    bix = pd.read_parquet(BIX_FILE)
    return svd, bix


@st.cache_data(show_spinner="Loading predictions...")
def load_predictions() -> pd.DataFrame:
    return pd.read_parquet(PRED_FILE)


@st.cache_data(show_spinner="Loading user history...")
def load_train_counts() -> pd.Series:
    train = pd.read_parquet(TRAIN_FILE, columns=["user_id"])
    return train.groupby("user_id").size()


def _missing_files() -> list[str]:
    return [str(f) for f in [CAND_FILE, COMM_FILE, PRED_FILE, TRAIN_FILE, SVD_FILE]
            if not f.exists()]


def _pr_pct(pr: float | None, all_pr: pd.Series) -> int:
    if pr is None or pd.isna(pr):
        return -1
    return int(100 * (all_pr < pr).sum() / len(all_pr))


def _tier_label(n: int) -> str:
    if n <= COLD_MAX:
        return "cold"
    if n <= MID_MAX:
        return "mid"
    return "warm"


def _blend_label(tier: str) -> str:
    if tier in ("cold", "mid"):
        return "65% ALS + 20% Content + 15% Pop"
    return "100% ALS"


# ── card renderer ─────────────────────────────────────────────────────────────

def _render_cards(df: pd.DataFrame, all_pr: pd.Series, score_col: str = "score") -> None:
    max_score = df[score_col].max() if not df.empty else 1.0
    cols = st.columns(3)
    for idx, (_, row) in enumerate(df.iterrows()):
        col = cols[idx % 3]
        name  = str(row.get("business_name", "N/A"))
        cat   = str(row.get("primary_category", "N/A"))
        cl    = row.get("cluster_kmeans", np.nan)
        pr    = row.get("pagerank", np.nan)
        com   = row.get("community", np.nan)
        score = float(row[score_col])
        rank  = int(row["rank"]) if "rank" in row.index else idx + 1

        is_hg  = (not pd.isna(cl)) and int(cl) == HG_CLUSTER
        is_c3  = (not pd.isna(com)) and int(com) == HG_COMM
        pr_pct = _pr_pct(pr, all_pr)

        hg_badge  = '<span class="badge-hg">💎 Hidden Gem</span>' if is_hg else '<span class="badge-ms">Mainstream</span>'
        c3_badge  = '<span class="badge-c3">✦ Gem Community</span>' if is_c3 else ""
        pr_badge  = f'<span class="badge-pr">PR p{pr_pct}</span>' if pr_pct >= 0 else ""

        bar_pct = int(100 * score / max_score) if max_score > 0 else 0

        with col:
            st.markdown(f"""
<div class="gem-card">
  <div style="color:#888;font-size:11px;margin-bottom:2px">#{rank}</div>
  <h4>{name}</h4>
  <div class="cat">{cat}</div>
  <div style="display:flex;gap:5px;flex-wrap:wrap;margin-bottom:8px">
    {hg_badge} {c3_badge} {pr_badge}
  </div>
  <div class="score-bar-bg"><div class="score-bar" style="width:{bar_pct}%"></div></div>
  <div style="color:#888;font-size:11px">Relevance score: {score:.4f}</div>
</div>
""", unsafe_allow_html=True)


# ── content-based scoring for simulated user ──────────────────────────────────

def _simulate_recs(
    selected_cats: list[str],
    discovery_weight: float,
    top_k: int,
    pool: pd.DataFrame,
    svd: pd.DataFrame,
    bix: pd.DataFrame,
    all_pr: pd.Series,
) -> pd.DataFrame:
    if not selected_cats:
        return pd.DataFrame()

    # Build user profile: mean SVD vector of businesses in selected categories
    matching_biz = bix[bix["primary_category"].isin(selected_cats)]["business_id"]
    profile_svd  = svd[svd["business_id"].isin(matching_biz)]

    feat_cols = [c for c in svd.columns if c.startswith("component_")]

    if profile_svd.empty:
        return pd.DataFrame()

    profile_vec = profile_svd[feat_cols].values.mean(axis=0, keepdims=True)

    # Score all candidates
    cand_svd = svd[svd["business_id"].isin(pool["business_id"])].copy()
    if cand_svd.empty:
        return pd.DataFrame()

    cand_vecs = cand_svd[feat_cols].values
    cos_scores = cosine_similarity(profile_vec, cand_vecs)[0]
    cand_svd = cand_svd[["business_id"]].copy()
    cand_svd["cos_sim"] = cos_scores

    # Merge with pool for metadata
    result = pool.merge(cand_svd, on="business_id", how="inner")

    # PageRank percentile score (0-1)
    result["pr_pct_norm"] = result["pagerank"].apply(
        lambda pr: (all_pr < pr).sum() / len(all_pr) if pd.notna(pr) else 0.0
    )

    # Blend: discovery_weight=0 → pure cosine; discovery_weight=1 → pure PR
    result["score"] = (
        (1 - discovery_weight) * result["cos_sim"]
        + discovery_weight * result["pr_pct_norm"]
    )

    result = result.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
    result["rank"] = result.index + 1
    return result


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 💎 Hidden-Gem Discovery")
    st.markdown("**Philadelphia · Yelp Open Dataset**")
    st.divider()
    st.markdown(
        "This system finds high-quality, under-discovered Philadelphia businesses "
        "using a blend of **collaborative filtering**, **content similarity**, and "
        "**graph analytics**.\n\n"
        "Three independent methods — K-Means clustering, ALS recommendation, and "
        "Louvain community detection — all converge on the same hidden-gem segment."
    )
    st.divider()
    st.markdown(
        "**Pipeline stages**\n"
        "01 Ingestion → 02 Cleaning → 03 Join\n"
        "04 Features → 05 SVD → 06 K-Means\n"
        "07 Blend Router → 08 Graph Analytics"
    )
    st.markdown("📄 [RUNBOOK.md](RUNBOOK.md) · 💻 `demo.py`")


# ── main area ─────────────────────────────────────────────────────────────────

st.markdown("# 💎 Yelp Hidden-Gem Discovery")
st.markdown("*Philadelphia · Final Project Demo*")
st.divider()

missing = _missing_files()
if missing:
    st.error(
        "**Pipeline outputs not found.** Run stages 04-08 first (see RUNBOOK.md).\n\n"
        "Missing:\n" + "\n".join(f"- `{f}`" for f in missing)
    )
    st.stop()

pool   = load_pool()
all_pr = pool["pagerank"].dropna()

tab1, tab2 = st.tabs(["🎭 Build My Profile", "👤 Explore Real Users"])


# ═══ TAB 1: SIMULATE ═══════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Tell us what you enjoy and we'll find your hidden gems")
    st.markdown(
        "We'll build a taste profile from your preferences and use our "
        "**content-based model** (SVD cosine similarity) to find matching hidden gems, "
        "optionally boosted by their graph visibility (PageRank)."
    )

    col_a, col_b = st.columns([2, 1])

    with col_a:
        top_cats = pool["primary_category"].value_counts().head(40).index.tolist()
        selected_cats = st.multiselect(
            "**What types of places do you enjoy?**",
            options=sorted(top_cats),
            default=["Restaurants", "Coffee & Tea"],
            help="Select one or more categories. We'll blend businesses from all selected types.",
        )

    with col_b:
        top_k_sim = st.slider("Number of recommendations", min_value=3, max_value=15, value=9)
        discovery_w = st.slider(
            "Discovery style",
            min_value=0.0, max_value=1.0, value=0.2, step=0.05,
            help="← Pure taste match · Best-known hidden gems →",
            format="%.2f",
        )
        st.caption(
            "**0.00** = ranked purely by how well they match your categories\n\n"
            "**1.00** = ranked by PageRank (best-connected hidden gems)"
        )

    if st.button("💎 Find My Hidden Gems", type="primary", use_container_width=True):
        if not selected_cats:
            st.warning("Please select at least one category.")
        else:
            svd, bix = load_svd()
            with st.spinner("Computing your profile..."):
                results = _simulate_recs(
                    selected_cats, discovery_w, top_k_sim,
                    pool, svd, bix, all_pr,
                )
            if results.empty:
                st.warning("No hidden gems found for the selected categories. Try adding more.")
            else:
                st.divider()
                st.markdown(
                    f"**{len(results)} hidden gems** matched your profile "
                    f"({', '.join(selected_cats)}) — "
                    f"discovery style: **{'taste match' if discovery_w < 0.3 else 'balanced' if discovery_w < 0.7 else 'best-known gems'}**"
                )
                _render_cards(results, all_pr, score_col="score")

                with st.expander("How this works"):
                    st.markdown(
                        "1. **Taste profile**: we average the SVD embeddings of all "
                        "Philadelphia businesses in your selected categories.\n"
                        "2. **Cosine similarity**: we score every Cluster-1 (hidden gem) "
                        "business by how close it is to your profile in the 200-dim SVD space.\n"
                        "3. **Discovery blend**: `score = (1-w) × cosine_sim + w × pagerank_percentile`\n"
                        "   where *w* is your discovery style slider.\n"
                        "4. **Graph enrichment**: each card shows its Louvain community. "
                        "**✦ Gem Community** = Community 3 (55% hidden gems by Louvain detection)."
                    )


# ═══ TAB 2: REAL USERS ════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Explore recommendations for real users from the test set")
    st.markdown(
        "These are actual Yelp users from Philadelphia. Recommendations come from the "
        "**confidence-weighted blend router** trained on their review history."
    )

    counts = load_train_counts()
    preds  = load_predictions()

    col_t, col_u = st.columns([1, 2])

    with col_t:
        tier_filter = st.radio(
            "**User type**",
            options=["Cold (≤ 10 reviews)", "Mid (11–30 reviews)", "Warm (> 30 reviews)"],
            index=0,
            help=(
                "Cold users receive a 65/20/15 ALS+Content+Pop blend.\n"
                "Mid users receive the same blend.\n"
                "Warm users receive pure ALS."
            ),
        )
        tier_key = {"C": "cold", "M": "mid", "W": "warm"}[tier_filter[0]]

    # Filter users
    tier_users_df = preds[preds["tier"] == tier_key].drop_duplicates("user_id")[["user_id"]].copy()
    tier_users_df["n_train"] = tier_users_df["user_id"].map(counts).fillna(0).astype(int)
    tier_users_df = tier_users_df.sort_values("n_train").head(100)

    with col_u:
        user_options = {
            f"{row.user_id}  ({row.n_train} reviews)": row.user_id
            for _, row in tier_users_df.iterrows()
        }
        if not user_options:
            st.warning("No users found for this tier.")
            st.stop()

        selected_label = st.selectbox("**Select a user**", list(user_options.keys()))
        selected_uid   = user_options[selected_label]

    top_k_real = st.slider("Recommendations to show", min_value=3, max_value=15, value=9, key="topk_real")

    user_preds = (
        preds[preds["user_id"] == selected_uid]
        .sort_values("rank")
        .head(top_k_real)
    )
    n_train = counts.get(selected_uid, 0)
    tier    = user_preds["tier"].iloc[0] if not user_preds.empty else "cold"

    tier_css = {"cold": "tier-cold", "mid": "tier-mid", "warm": "tier-warm"}[tier]
    blend    = _blend_label(tier)

    st.divider()
    st.markdown(
        f"**User:** `{selected_uid}`  &nbsp;|&nbsp; "
        f'<span class="{tier_css}">{tier.upper()} — {n_train} training reviews</span>  &nbsp;|&nbsp; '
        f"Routing: **{blend}**",
        unsafe_allow_html=True,
    )

    if user_preds.empty:
        st.warning("No predictions found for this user.")
    else:
        enriched = user_preds.merge(pool, on="business_id", how="left")
        _render_cards(enriched, all_pr, score_col="score")

        with st.expander("How the blend router works"):
            st.markdown(
                f"This user has **{n_train} training reviews**, placing them in the "
                f"**{tier}** tier.\n\n"
                "| Tier | Reviews | Blend |\n"
                "|---|---|---|\n"
                "| Cold | ≤ 10 | 65% ALS + 20% Content + 15% Pop |\n"
                "| Mid  | 11–30 | 65% ALS + 20% Content + 15% Pop |\n"
                "| Warm | > 30 | 100% ALS (pure collaborative filtering) |\n\n"
                "**💡 Why not use pure popularity for cold users?** We tried it (v1). "
                "NDCG@10 dropped from 0.0324 to 0.0112 — even 5-10 reviews carry enough "
                "collaborative signal that removing it hurts. The blend keeps ALS as the "
                "dominant signal but adds content and popularity as stabilizers."
            )

st.divider()
st.caption(
    "Project: Yelp Hidden-Gem Discovery · Philadelphia · Ricardo Rivas, Salvador Diaz, Joaquin Arevalo\n\n"
    "Pipeline: Ingestion → Cleaning → Join → Features → SVD → K-Means → Blend Router → Graph Analytics\n\n"
    "Badges: 💎 Hidden Gem = Cluster 1 (PC3 K-Means) · ✦ Gem Community = Louvain Community 3 · PR = PageRank percentile"
)
