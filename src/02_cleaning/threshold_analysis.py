"""Stage 2a — Threshold analysis.
Justifies the >=10 review thresholds for both businesses and users via SEM.

  Business: SEM = std/sqrt(n) < 0.5 (Yelp rounding granularity) -> minimum n = 8 (SEM=0.491)
  User    : SEM < half_IQR_gap(user_average_stars) = 0.680   -> minimum n = 5 (SEM=0.621)

Inputs  : src/01_ingestion/outputs/
Outputs : src/02_cleaning/outputs/plots/
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

IN_DIR   = Path("src/01_ingestion/outputs")
PLOT_DIR = Path("src/02_cleaning/outputs/plots")

MIN_REVIEWS_BUSINESS = 8
MIN_REVIEWS_USER     = 5
YELP_ROUNDING        = 0.5


def sem(std: float, n: np.ndarray) -> np.ndarray:
    return std / np.sqrt(n)


def sem_curve(std: float, chosen_n: int, target: float, title: str, path: Path) -> None:
    n = np.arange(1, 51)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(n, sem(std, n), lw=2, label=f"SEM = {std:.2f} / √n")
    ax.axhline(target, color="red",   ls="--", label=f"target SEM = {target}")
    ax.axvline(chosen_n, color="green", ls="--",
               label=f"chosen n = {chosen_n}  (SEM = {sem(std, chosen_n):.2f})")
    ax.set_xlabel("Reviews (n)")
    ax.set_ylabel("Standard Error of the Mean (stars)")
    ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def variability_by_bin(review: pd.DataFrame, std: float) -> None:
    biz = review.groupby("business_id")["stars"].agg(mean="mean", n="count").reset_index()
    bins   = [0, 5, 10, 20, 50, 100, 1000, 100_000]
    labels = ["1-5", "6-10", "11-20", "21-50", "51-100", "101-1000", "1000+"]
    biz["bin"] = pd.cut(biz["n"], bins=bins, labels=labels)
    summary = biz.groupby("bin", observed=True)["mean"].std()

    fig, ax = plt.subplots(figsize=(9, 5))
    summary.plot(kind="bar", ax=ax, color="steelblue")
    ax.axhline(YELP_ROUNDING, color="red", ls="--", label=f"Yelp rounding = {YELP_ROUNDING}")
    ax.set_xlabel("Reviews per business (binned)")
    ax.set_ylabel("Std of business mean rating")
    ax.set_title("Empirical rating variability by sample size")
    ax.legend(); plt.setp(ax.get_xticklabels(), rotation=0)
    fig.tight_layout(); fig.savefig(PLOT_DIR / "business_variability_by_n.png"); plt.close(fig)

    summary_df = biz.groupby("bin", observed=True).agg(
        n_businesses=("business_id", "count"),
        mean_rating=("mean", "mean"),
        std_of_means=("mean", "std"),
    )
    summary_df.to_csv(PLOT_DIR.parent / "threshold_stats_business.csv")


def user_distribution(user: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    user.loc[user["review_count"] <= 50, "review_count"].hist(bins=50, ax=ax, color="steelblue")
    ax.axvline(MIN_REVIEWS_USER, color="green", ls="--", label=f"threshold n = {MIN_REVIEWS_USER}")
    ax.set_xlabel("Reviews per user (≤50 shown)")
    ax.set_ylabel("Users")
    ax.set_title("Distribution of total reviews per user")
    ax.legend()
    fig.tight_layout(); fig.savefig(PLOT_DIR / "user_review_count_distribution.png"); plt.close(fig)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print("Stage 2a — Threshold analysis")

    review = pd.read_json(IN_DIR / "review_philadelphia_pa.json", lines=True)
    user   = pd.read_json(IN_DIR / "user_philadelphia_pa.json",   lines=True)

    std      = float(review["stars"].std())
    user_avg = user["user_average_stars"] if "user_average_stars" in user.columns \
               else pd.read_json(IN_DIR / "user_philadelphia_pa.json", lines=True)["average_stars"]
    half_gap = float((user_avg.quantile(0.75) - user_avg.quantile(0.25)) / 2)

    print(f"  std(review_stars) = {std:.4f}")
    print(f"  half IQR gap (user_average_stars) = {half_gap:.4f}")
    print(f"  SEM at n={MIN_REVIEWS_BUSINESS} (business) = {sem(std, MIN_REVIEWS_BUSINESS):.4f} < {YELP_ROUNDING}?  "
          f"{'YES' if sem(std, MIN_REVIEWS_BUSINESS) < YELP_ROUNDING else 'NO'}")
    print(f"  SEM at n={MIN_REVIEWS_USER} (user)     = {sem(std, MIN_REVIEWS_USER):.4f} < {half_gap:.4f}?  "
          f"{'YES' if sem(std, MIN_REVIEWS_USER) < half_gap else 'NO'}")

    sem_curve(std, MIN_REVIEWS_BUSINESS, YELP_ROUNDING,
              "Business threshold: SEM vs review count",
              PLOT_DIR / "sem_vs_n_business.png")

    sem_curve(std, MIN_REVIEWS_USER, half_gap,
              "User threshold: SEM vs review count",
              PLOT_DIR / "sem_vs_n_user.png")

    variability_by_bin(review, std)
    user_distribution(user)
    print(f"  plots -> {PLOT_DIR}")


if __name__ == "__main__":
    main()
