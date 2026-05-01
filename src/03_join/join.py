"""Stage 3 — Join.
Joins cleaned reviews + business + user into a single analytical table and saves as parquet.
Inputs  : src/02_cleaning/outputs/data/
Outputs : src/03_join/outputs/reviews_enriched_v1_part_*.parquet
"""
from pathlib import Path
import pandas as pd

IN_DIR  = Path("src/02_cleaning/outputs/data")
OUT_DIR = Path("src/03_join/outputs")
ROWS_PER_PART = 100_000


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Stage 3 — Join")

    biz = pd.read_json(IN_DIR / "business_clean.json", lines=True)
    rev = pd.read_json(IN_DIR / "review_clean.json",   lines=True)
    usr = pd.read_json(IN_DIR / "user_clean.json",     lines=True)

    df = (
        rev
        .merge(biz[["business_id", "name", "categories", "stars", "review_count"]],
               on="business_id", how="left", suffixes=("_review", "_business"))
        .merge(usr[["user_id", "review_count", "average_stars"]],
               on="user_id", how="left", suffixes=("_business", "_user"))
    )

    df.rename(columns={
        "review_count_business": "business_review_count",
        "review_count_user":     "user_review_count",
        "stars_review":          "review_stars",
        "useful":                "review_useful",
        "funny":                 "review_funny",
        "cool":                  "review_cool",
        "average_stars":         "user_average_stars",
        "name":                  "business_name",
        "stars_business":        "business_stars",
    }, inplace=True)

    df.dropna(inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.astype({
        "review_id": "string", "user_id": "string", "business_id": "string",
        "review_stars": "float", "business_stars": "float", "user_average_stars": "float",
        "review_useful": "Int64", "review_funny": "Int64", "review_cool": "Int64",
        "business_review_count": "Int64", "user_review_count": "Int64",
        "text": "string", "business_name": "string", "categories": "string",
    })
    df["categories_list"] = df["categories"].str.split(", ")

    n_parts = (len(df) + ROWS_PER_PART - 1) // ROWS_PER_PART
    for i in range(n_parts):
        chunk = df.iloc[i * ROWS_PER_PART : (i + 1) * ROWS_PER_PART]
        chunk.to_parquet(OUT_DIR / f"reviews_enriched_v1_part_{i+1}.parquet",
                         index=False, engine="pyarrow")

    print(f"  {len(df):,} rows -> {n_parts} parquet parts -> {OUT_DIR}")


if __name__ == "__main__":
    main()
