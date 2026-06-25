"""Stage 7k - Improvement #5: Concrete, report-ready error examples.

Addresses the PC4 feedback that "top-credit error analysis should include
several concrete users, recommended businesses, ground-truth items, ranks,
and interpretation" rather than a pointer to error_examples.csv.

For the fallback router (the new final architecture), selects:
    - 3 strong cases  (>=2 hits in top-10)
    - 3 failure cases (0 hits in top-10 despite >=1 positive in pool)
one per activity tier where possible, and writes a fully narrated table:
business names (not just IDs), exact rank of each hit/miss, and a one-line
interpretation per case.

Inputs  : src/07_recommendation/outputs/data/{train,test,candidates,predictions_fallback}.parquet
          src/04_features/outputs/business_index.parquet
Outputs : src/07_recommendation/outputs/data/detailed_error_examples.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REC_DIR  = Path("src/07_recommendation/outputs/data")
FEAT_DIR = Path("src/04_features/outputs")

K             = 10
POSITIVE_STAR = 4
COLD_MAX      = 10
MID_MAX       = 30


def tier_of(n: int) -> str:
    if n <= COLD_MAX:
        return "cold"
    if n <= MID_MAX:
        return "mid"
    return "warm"


def name_lookup(biz_index: pd.DataFrame) -> dict:
    return dict(zip(biz_index["business_id"], biz_index["business_name"]))


def narrate_case(user_id: str, tier: str, train_count: int,
                 recs: pd.DataFrame, relevant: set, names: dict,
                 case_type: str) -> dict:
    rec_list = recs.sort_values("rank")
    hit_ranks = [int(r["rank"]) for _, r in rec_list.iterrows()
                 if r["business_id"] in relevant]
    rec_names = [f"{names.get(r['business_id'], '?')} (rank {int(r['rank'])})"
                 for _, r in rec_list.head(5).iterrows()]
    gt_names = [names.get(b, "?") for b in list(relevant)[:5]]

    if case_type == "strong":
        interp = (f"{len(hit_ranks)} of the user's {len(relevant)} test favourites "
                  f"appear in the top-10, at ranks {hit_ranks}. With "
                  f"{train_count} train reviews ({tier} tier), the router had "
                  "enough signal to place genuine matches near the top.")
    else:
        interp = (f"None of the user's {len(relevant)} test favourites appear "
                  f"in the top-10 despite {train_count} train reviews ({tier} "
                  "tier). " +
                  ("Likely cause: very thin history limits both ALS and "
                   "content signal; the blend still leans on a weak personal "
                   "profile." if tier in ("cold", "mid") else
                   "Likely cause: the user's true favourites may sit outside "
                   "the hidden-gem candidate pool (PC3 Cluster 1), which "
                   "structurally caps what the router can ever recommend."))

    return {
        "case_type":        case_type,
        "user_id":          user_id,
        "user_tier":        tier,
        "user_train_count": train_count,
        "n_test_positives_in_pool": len(relevant),
        "n_hits_in_top10":  len(hit_ranks),
        "hit_ranks":        str(hit_ranks),
        "top5_recommendations": "; ".join(rec_names),
        "ground_truth_examples": "; ".join(gt_names),
        "interpretation":   interp,
    }


def main() -> None:
    print("Stage 7k - Detailed error examples (fallback router)")

    train  = pd.read_parquet(REC_DIR / "train.parquet")
    test   = pd.read_parquet(REC_DIR / "test.parquet")
    cand   = pd.read_parquet(REC_DIR / "candidates.parquet")
    router = pd.read_parquet(REC_DIR / "predictions_fallback.parquet")
    biz_index = pd.read_parquet(FEAT_DIR / "business_index.parquet")
    names = name_lookup(biz_index)

    cand_ids = set(cand["business_id"])
    pos = test[(test["review_stars"] >= POSITIVE_STAR) & (test["business_id"].isin(cand_ids))]
    gt = {u: set(g) for u, g in pos.groupby("user_id")["business_id"]}

    user_counts = train.groupby("user_id")["review_id"].count()

    sub = router[router["rank"] <= K]
    by_user = sub.groupby("user_id")["business_id"].apply(set)

    scored = []
    for u, relevant in gt.items():
        recs_u = sub[sub["user_id"] == u]
        if recs_u.empty:
            continue
        hits = len(by_user.get(u, set()) & relevant)
        scored.append((u, hits, relevant))

    strong = [s for s in scored if s[1] >= 2]
    failure = [s for s in scored if s[1] == 0]

    rows = []
    seen_tiers_strong: set = set()
    for u, hits, relevant in strong:
        t = tier_of(int(user_counts.get(u, 0)))
        if t in seen_tiers_strong:
            continue
        seen_tiers_strong.add(t)
        recs_u = sub[sub["user_id"] == u]
        rows.append(narrate_case(u, t, int(user_counts.get(u, 0)),
                                 recs_u, relevant, names, "strong"))
        if len(seen_tiers_strong) == 3:
            break

    seen_tiers_fail: set = set()
    for u, hits, relevant in failure:
        t = tier_of(int(user_counts.get(u, 0)))
        if t in seen_tiers_fail:
            continue
        seen_tiers_fail.add(t)
        recs_u = sub[sub["user_id"] == u]
        rows.append(narrate_case(u, t, int(user_counts.get(u, 0)),
                                 recs_u, relevant, names, "failure"))
        if len(seen_tiers_fail) == 3:
            break

    out = pd.DataFrame(rows)
    out.to_csv(REC_DIR / "detailed_error_examples.csv", index=False)
    print(f"  wrote {len(out)} narrated cases ({len(seen_tiers_strong)} strong, "
          f"{len(seen_tiers_fail)} failure, one per tier where available)")
    for _, r in out.iterrows():
        print(f"\n  [{r['case_type'].upper()} / {r['user_tier']}] user={r['user_id']}")
        print(f"    train_count={r['user_train_count']}  hits={r['n_hits_in_top10']}/"
              f"{r['n_test_positives_in_pool']}  ranks={r['hit_ranks']}")
        print(f"    top5: {r['top5_recommendations']}")
        print(f"    interpretation: {r['interpretation']}")

    print(f"\n  outputs -> {REC_DIR / 'detailed_error_examples.csv'}")


if __name__ == "__main__":
    main()
