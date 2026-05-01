"""Stage 1 — Ingestion.
Reads raw Yelp JSONL files and extracts the Philadelphia / PA subset (geographic filter only).
Inputs  : data/data_raw/yelp_academic_dataset_*.json
Outputs : src/01_ingestion/outputs/
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/data_raw")
OUT_DIR = Path("src/01_ingestion/outputs")


def filter_business() -> set[str]:
    biz = pd.read_json(RAW_DIR / "yelp_academic_dataset_business.json", lines=True)
    biz["state_clean"] = biz["state"].str.lower().str.strip()
    biz["city_clean"]  = biz["city"].str.lower().str.strip()
    phl = biz[(biz["state_clean"] == "pa") & biz["city_clean"].str.contains("philadelphia")]
    phl.to_json(OUT_DIR / "business_philadelphia_pa.json", orient="records", lines=True)
    print(f"  business: {len(phl):,}")
    return set(phl["business_id"])


def filter_by_business(filename: str, out_name: str, ids: set[str]) -> None:
    kept = 0
    with (RAW_DIR / filename).open("r", encoding="utf-8") as f_in, \
         (OUT_DIR / out_name).open("w", encoding="utf-8") as f_out:
        for line in f_in:
            rec = json.loads(line)
            if rec["business_id"] in ids:
                f_out.write(json.dumps(rec) + "\n")
                kept += 1
    print(f"  {out_name}: {kept:,}")


def filter_users(review_path: Path) -> None:
    user_ids = set(pd.read_json(review_path, lines=True)["user_id"])
    kept = 0
    with (RAW_DIR / "yelp_academic_dataset_user.json").open("r", encoding="utf-8") as f_in, \
         (OUT_DIR / "user_philadelphia_pa.json").open("w", encoding="utf-8") as f_out:
        for line in f_in:
            u = json.loads(line)
            if u["user_id"] in user_ids:
                f_out.write(json.dumps(u) + "\n")
                kept += 1
    print(f"  users: {kept:,}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Stage 1 — Ingestion")
    ids = filter_business()
    filter_by_business("yelp_academic_dataset_review.json",  "review_philadelphia_pa.json",  ids)
    filter_by_business("yelp_academic_dataset_checkin.json", "checkin_philadelphia_pa.json", ids)
    filter_by_business("yelp_academic_dataset_tip.json",     "tip_philadelphia_pa.json",     ids)
    filter_users(OUT_DIR / "review_philadelphia_pa.json")


if __name__ == "__main__":
    main()
