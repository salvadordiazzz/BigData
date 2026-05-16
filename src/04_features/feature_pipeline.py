"""Stage 4 — Feature Engineering.
Builds three per-business feature matrices from the enriched reviews parquet.
Inputs  : data/processed/reviews_enriched_v1_part_*.parquet
Outputs : src/04_features/outputs/
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, normalize

PROCESSED_DIR   = Path("src/03_join/outputs")
OUT_DIR         = Path("src/04_features/outputs")
TOP_N_CATS      = 50
TFIDF_FEATURES  = 5_000


def load() -> pd.DataFrame:
    parts = sorted(PROCESSED_DIR.glob("reviews_enriched_v1_part_*.parquet")) or \
            sorted(Path("data/processed").glob("reviews_enriched_v1_part_*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  {len(df):,} reviews, {df['business_id'].nunique():,} businesses")
    return df


def numeric_temporal(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    ref = df["date"].max()
    agg = df.groupby("business_id").agg(
        business_stars       =("business_stars",      "first"),
        log_review_count     =("business_review_count", lambda x: np.log1p(x.iloc[0])),
        mean_review_stars    =("review_stars",         "mean"),
        std_review_stars     =("review_stars",         "std"),
        mean_useful          =("review_useful",        "mean"),
        mean_funny           =("review_funny",         "mean"),
        mean_cool            =("review_cool",          "mean"),
        mean_user_avg_stars  =("user_average_stars",   "mean"),
        mean_user_rev_count  =("user_review_count",    "mean"),
        first_date           =("date", "min"),
        last_date            =("date", "max"),
        review_count         =("review_id", "count"),
    ).reset_index()

    agg["days_since_first"]       = (ref - agg["first_date"]).dt.days.astype(float)
    agg["days_since_last"]        = (ref - agg["last_date"]).dt.days.astype(float)
    agg["review_span_days"]       = (agg["last_date"] - agg["first_date"]).dt.days.astype(float)
    cutoff = ref - pd.Timedelta(days=365)
    recent = df[df["date"] >= cutoff].groupby("business_id").size().rename("recent_count")
    agg = agg.merge(recent, on="business_id", how="left")
    agg["reviews_per_year_recent"] = agg["recent_count"].fillna(0) / 365.0
    agg["std_review_stars"]        = agg["std_review_stars"].fillna(0.0)

    cols = [
        "business_stars", "log_review_count",
        "mean_review_stars", "std_review_stars",
        "mean_useful", "mean_funny", "mean_cool",
        "mean_user_avg_stars", "mean_user_rev_count",
        "days_since_first", "days_since_last", "review_span_days",
        "reviews_per_year_recent",
    ]
    X = StandardScaler().fit_transform(agg[cols].values).astype(np.float32)
    return agg["business_id"], X, cols


def categorical(df: pd.DataFrame, ids: pd.Index) -> tuple[np.ndarray, list[str]]:
    cats = (df[["business_id", "categories_list"]]
            .drop_duplicates("business_id")
            .set_index("business_id")
            .reindex(ids)["categories_list"]
            .apply(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else []))

    top = pd.Series([c for lst in cats for c in lst]).value_counts().head(TOP_N_CATS).index.tolist()
    mlb = MultiLabelBinarizer(classes=top)
    X = mlb.fit_transform(cats).astype(np.float32)
    return X, list(mlb.classes_)


def text_tfidf(df: pd.DataFrame, ids: pd.Index) -> tuple[sp.csr_matrix, list[str]]:
    docs = (df.groupby("business_id")["text"]
              .apply(lambda t: " ".join(t.dropna().astype(str)))
              .reindex(ids).fillna(""))
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                          max_features=TFIDF_FEATURES, min_df=5, max_df=0.6,
                          sublinear_tf=True, dtype=np.float32)
    return vec.fit_transform(docs), vec.get_feature_names_out().tolist()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Stage 4 — Feature Engineering")

    df = load()

    print("  numeric + temporal...")
    biz_ids, X_num, num_cols = numeric_temporal(df)

    print("  categorical (multi-hot top-50)...")
    X_cat, cat_cols = categorical(df, biz_ids)

    print("  TF-IDF text (may take ~1 min)...")
    X_text, text_cols = text_tfidf(df, biz_ids)

    X_dense    = np.hstack([X_num, X_cat])
    X_combined = sp.hstack([sp.csr_matrix(X_dense), normalize(X_text, norm="l2")], format="csr")

    print(f"  dense={X_dense.shape}  text={X_text.shape}  combined={X_combined.shape}")

    sp.save_npz(OUT_DIR / "X_business_dense.npz",    sp.csr_matrix(X_dense))
    sp.save_npz(OUT_DIR / "X_business_text.npz",     X_text)
    sp.save_npz(OUT_DIR / "X_business_combined.npz", X_combined)

    index = (pd.DataFrame({"business_id": biz_ids.values})
               .merge(df[["business_id", "business_name", "categories_list"]]
                      .drop_duplicates("business_id"), on="business_id", how="left")
               .assign(primary_category=lambda r: r["categories_list"].apply(
                   lambda x: list(x)[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else "Unknown"))
               [["business_id", "business_name", "primary_category"]])
    index.to_parquet(OUT_DIR / "business_index.parquet", index=False)

    feat_cols = num_cols + cat_cols
    pd.concat([index.reset_index(drop=True),
               pd.DataFrame(X_dense, columns=feat_cols)], axis=1) \
      .to_parquet(OUT_DIR / "business_features.parquet", index=False)

    with open(OUT_DIR / "feature_names.json", "w") as f:
        json.dump({"numeric_temporal": num_cols, "categorical": cat_cols,
                   "text_total_features": len(text_cols)}, f, indent=2)

    print(f"  outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
