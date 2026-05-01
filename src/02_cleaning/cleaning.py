"""Stage 2 — Cleaning.
Inputs  : src/01_ingestion/outputs/
Outputs : src/02_cleaning/outputs/data/
"""
from pathlib import Path
import pandas as pd

IN_DIR  = Path("src/01_ingestion/outputs")
OUT_DIR = Path("src/02_cleaning/outputs/data")

# SEM = std/sqrt(n) = 1.286/sqrt(n)
# Business: SEM = 1.388/sqrt(n) < 0.5 (Yelp rounding)        -> minimum n = 8 (SEM=0.491)
# User    : SEM < half_IQR_gap(user_average_stars) = 0.680   -> minimum n = 5 (SEM=0.621)
MIN_REVIEWS_BUSINESS = 8
MIN_REVIEWS_USER     = 5


def clean_business() -> set[str]:
    biz = pd.read_json(IN_DIR / "business_philadelphia_pa.json", lines=True)
    biz = biz[biz["review_count"] >= MIN_REVIEWS_BUSINESS]
    biz.to_json(OUT_DIR / "business_clean.json", orient="records", lines=True)
    print(f"  business: {len(biz):,}")
    return set(biz["business_id"])


def clean_reviews(business_ids: set) -> None:
    rev = pd.read_json(IN_DIR / "review_philadelphia_pa.json", lines=True)
    rev = rev[rev["business_id"].isin(business_ids)]
    rev = rev.dropna(subset=["review_id", "user_id", "business_id", "stars"])
    rev.to_json(OUT_DIR / "review_clean.json", orient="records", lines=True)
    print(f"  reviews : {len(rev):,}")


def clean_users() -> None:
    usr = pd.read_json(IN_DIR / "user_philadelphia_pa.json", lines=True)
    usr = usr[usr["review_count"] >= MIN_REVIEWS_USER]
    usr.to_json(OUT_DIR / "user_clean.json", orient="records", lines=True)
    print(f"  users   : {len(usr):,}")


def clean_secondary(business_ids: set) -> None:
    for entity in ("checkin", "tip"):
        df = pd.read_json(IN_DIR / f"{entity}_philadelphia_pa.json", lines=True)
        df = df[df["business_id"].isin(business_ids)]
        df.to_json(OUT_DIR / f"{entity}_clean.json", orient="records", lines=True)
        print(f"  {entity}: {len(df):,}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Stage 2 — Cleaning")
    ids = clean_business()
    clean_reviews(ids)
    clean_users()
    clean_secondary(ids)


if __name__ == "__main__":
    main()
