# Runbook — Yelp Hidden-Gem Discovery System

Step-by-step instructions to reproduce every pipeline output from raw data
to the final demo. All commands run from the **repository root**.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10 + |
| pip packages | see `requirements.txt` |
| Disk space (raw data) | ~10 GB |
| Disk space (outputs) | ~3 GB |

```bash
pip install -r requirements.txt
```

---

## 1 — Raw Data

Download the **Yelp Open Dataset** from
<https://www.yelp.com/dataset> (academic licence, free registration required).

Place the five JSON files under `data/data_raw/`:

```
data/data_raw/
  yelp_academic_dataset_business.json
  yelp_academic_dataset_review.json
  yelp_academic_dataset_user.json
  yelp_academic_dataset_checkin.json
  yelp_academic_dataset_tip.json
```

The dataset user agreement is kept at `data/Dataset_User_Agreement.pdf`.

---

## 2 — Stage 01: Ingestion (Philadelphia filter)

```bash
python src/01_ingestion/ingestion.py
```

**What it does:** reads all five raw JSON files, filters to Philadelphia, PA
businesses only, and writes five `*_philadelphia_pa.json` files.

**Outputs:** `src/01_ingestion/outputs/`

---

## 3 — Stage 02: Cleaning

```bash
python src/02_cleaning/cleaning.py
```

**What it does:** applies statistically-justified thresholds (≥ 10 reviews per
business, ≥ 5 reviews per user) derived from a standard-error vs. N analysis.
Removes duplicates, normalises text fields, writes clean JSON files.

Threshold analysis:

```bash
python src/02_cleaning/threshold_analysis.py   # optional – reproduces plots
```

**Outputs:** `src/02_cleaning/outputs/`

---

## 4 — Stage 03: Join

```bash
python src/03_join/join.py
```

**What it does:** joins `review_clean`, `business_clean`, and `user_clean` into
a single enriched reviews table, partitioned across 9 parquet files (one per
calendar year) for memory efficiency.

**Outputs:** `src/03_join/outputs/reviews_enriched_v1_part_*.parquet`

---

## 5 — Stage 04: Feature Engineering

```bash
python src/04_features/feature_pipeline.py
```

**What it does:** aggregates to one row per business; builds:
- numeric/temporal block (13 features, StandardScaler)
- categorical multi-hot block (top-50 Yelp categories)
- TF-IDF text block (15,000 bigrams, sparse)
- combined sparse matrix (dense + text)

**Outputs:** `src/04_features/outputs/`

---

## 6 — Stage 05: Dimensionality Reduction

```bash
python src/05_dimreduction/reduce.py
python src/05_dimreduction/compare.py   # optional – produces comparison table
```

**What it does:** PCA on dense matrix; TruncatedSVD on text and combined
matrices (up to 200 components); t-SNE on top-50 combined SVD components.

**Outputs:** `src/05_dimreduction/outputs/`

---

## 7 — Stage 06: Clustering

```bash
python src/06_clustering/kmeans_tuned.py   # K-means sweep + final K=5 model
python src/06_clustering/dbscan.py         # DBSCAN parameter sweep
python src/06_clustering/validate.py       # silhouette + elbow plots
python src/06_clustering/profile.py        # cluster profile table
```

**Key result:** K=5 selected; Cluster 1 = hidden gems (high quality, low
visibility). Labels saved to `kmeans_labels.parquet`.

**Outputs:** `src/06_clustering/outputs/`

---

## 8 — Stage 07: Recommendation

Run scripts in the order below. Each depends on the previous stage's outputs.

```bash
# 7a — temporal train / val / test split (70/10/20, per-user, chronological)
python src/07_recommendation/split.py

# 7b — popularity baseline
python src/07_recommendation/baseline_popularity.py

# 7c — content-based (SVD cosine similarity)
python src/07_recommendation/content_based.py

# 7d — ALS base model
python src/07_recommendation/matrix_factorization.py

# 7d — ALS hyperparameter sweep (108 configs on val set, ~15 min)
python src/07_recommendation/tune_als.py

# 7d — evaluate best ALS config on test set
python src/07_recommendation/evaluate_best_als.py

# 7g — confidence-weighted blend router (final architecture)
python src/07_recommendation/fallback_router.py

# evaluation scripts (PC4 Improvements)
python src/07_recommendation/stratified_eval.py
python src/07_recommendation/pool_sensitivity.py
python src/07_recommendation/alignment_audit.py
python src/07_recommendation/detailed_error_examples.py
python src/07_recommendation/evaluate.py
```

**Key result:** Blend router (65% ALS + 20% Content + 15% Pop for cold/mid
users; pure ALS for warm) beats tuned ALS on every user tier.
NDCG@10: 0.0355 → 0.0365 (+2.8 %).

**Outputs:** `src/07_recommendation/outputs/`

---

## 9 — Stage 08: Graph Analytics

```bash
python src/08_graph/build_graph.py     # bipartite co-review projection
python src/08_graph/centrality.py      # degree, PageRank, betweenness
python src/08_graph/communities.py     # Louvain community detection
python src/08_graph/compare_rankings.py  # graph vs popularity vs router
python src/08_graph/visualize.py       # network visualizations
```

**Key result:** 4,913-node, 272,230-edge graph; single connected component.
Community 3 = 55 % Cluster-1 (hidden gems), independently validating the
PC3 K-Means segmentation.

**Outputs:** `src/08_graph/outputs/`

---

## 10 — End-to-End Demo

```bash
python demo.py                          # 3 example users (one per tier)
python demo.py --user <user_id>         # personalized for a specific user
python demo.py --user <user_id> -k 5   # top-5 recommendations
python demo.py --list-users             # print sample user IDs
```

The demo prints personalized hidden-gem recommendations enriched with
PageRank percentile and Louvain community membership for each recommended
business.

---

## 11 — Build Final Reports and Slides

```bash
python deliverables/builders/build_tf_report.py    # Final integrated report
python deliverables/builders/build_tf_slides.py    # Final presentation slides
python deliverables/builders/build_pc5_report.py   # PC5 standalone report
python deliverables/builders/build_pc5_slides.py   # PC5 standalone slides
python deliverables/builders/build_pc4_improvements_report.py
python deliverables/builders/build_pc4_improvements_slides.py
```

Generated `.docx` and `.pptx` files are saved to `deliverables/docs/`
(gitignored; run locally or inspect via the pre-built copies).

---

## Output Artifact Summary

| Directory | Contents |
|---|---|
| `src/01_ingestion/outputs/` | Philadelphia-filtered raw JSON (5 files) |
| `src/02_cleaning/outputs/` | Cleaned JSON + threshold plots |
| `src/03_join/outputs/` | Enriched reviews parquet (9 parts) |
| `src/04_features/outputs/` | Business feature matrices |
| `src/05_dimreduction/outputs/` | Reduced embeddings + scree + t-SNE |
| `src/06_clustering/outputs/` | Cluster labels, validation, profiles |
| `src/07_recommendation/outputs/` | Train/val/test, predictions (all models), metrics |
| `src/08_graph/outputs/` | Graph, centrality, communities, rankings, figures |
| `reports/figures/` | Curated figures (tracked in git) |
| `reports/data/` | Key metric summaries (tracked in git) |
| `deliverables/docs/` | Final report and slides (.docx / .pptx) |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: implicit` | `pip install implicit==0.7.3` |
| `ModuleNotFoundError: community` | `pip install python-louvain` |
| Stage 07 OOM | Reduce `factors` or `iterations` in `tune_als.py` |
| Graph build slow | Increase `MIN_SHARED_USERS` in `build_graph.py` |
| Demo: "outputs not found" | Run stages 07–08 first (see sections 8–9 above) |
