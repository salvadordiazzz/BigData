# Limitations and Future Work

**System:** Yelp Hidden-Gem Discovery — Philadelphia  
**Document purpose:** honest, evidence-based limitations tied to specific project artifacts, plus concrete next steps.

---

## 1. Candidate Pool Coverage

**Limitation:** The hidden-gem candidate pool is fixed to Cluster-1 businesses in the train set (3,572 items). Only 39% of test users have at least one Cluster-1 business in their ground truth, meaning 61% of users are structurally unevaluable regardless of model quality.

**Evidence:** `src/07_recommendation/outputs/data/pool_sensitivity.csv` quantifies this directly — expanding to Cluster 1 + Cluster 0 raises the evaluable user fraction from 39% to 84%.

**Implication:** NDCG@10 and Hit Rate as reported are ceiling-constrained by pool definition, not only by model quality. This means the reported metrics understate the system's practical usefulness for users who do visit non-Cluster-1 venues.

**Next step:** implement dynamic pool expansion per user — for users whose Cluster-1 ground truth is empty, fall back to the top-200 highest-scoring businesses across all clusters.

---

## 2. Graph Coverage of Hidden Gems

**Limitation:** Only 1,145 of 3,572 Cluster-1 businesses (32%) appear in the co-review graph. The remaining 68% are too long-tail to cross the MIN_SHARED_USERS=3 threshold and are therefore invisible to graph-based methods.

**Evidence:** `src/08_graph/outputs/data/ranking_comparison.csv` — comparisons are limited to 1,145 businesses.

**Implication:** graph centrality and community membership cannot serve as universal ranking signals; they work only for the most-reviewed fraction of hidden gems. A purely graph-based discovery approach would miss two-thirds of the target segment.

**Next step:** lower the edge threshold to MIN_SHARED_USERS=2 and measure the graph's sensitivity; alternatively, complement graph metrics with a co-occurrence embedding (node2vec) that generalizes to lower-degree nodes.

---

## 3. Evaluation Metric Sensitivity to Candidate Pool Size

**Limitation:** NDCG@10 and Hit Rate@10 are sensitive to the size of the candidate pool. With 3,572 candidates, a random recommender achieves HR@10 ≈ 10/3,572 ≈ 0.003. Our router achieves 0.0997. The absolute values are small and should not be interpreted as "poor quality" — they reflect the difficulty of the retrieval problem.

**Evidence:** `src/07_recommendation/outputs/data/metrics_tuned.csv` and `stratified_metrics.csv`.

**Implication:** comparisons to systems trained on different candidate pool sizes are not valid without normalization.

**Next step:** report relative uplift over the popularity baseline alongside absolute metrics, and compute Mean Average Precision (MAP) as a rank-quality metric less sensitive to pool size.

---

## 4. Implicit Feedback Assumptions

**Limitation:** ALS treats any review as implicit positive feedback (review_stars ≥ 1 = interaction). The confidence weight formula (confidence = 1 + alpha * stars) upweights high-star reviews but still includes negative experiences as weak positive signal.

**Evidence:** `src/07_recommendation/matrix_factorization.py` — confidence formula is standard Hu et al. (2008), applied uniformly.

**Implication:** for users who frequently leave 1-2 star reviews (dissatisfied reviewers), their ALS embeddings will be shaped partly by businesses they disliked. This is expected to increase noise in cold/mid-tier recommendations.

**Next step:** evaluate a binary confidence model (confidence = 1 + alpha only for stars ≥ 4, else 0) and compare NDCG on the cold tier where this effect is largest.

---

## 5. Temporal Generalization

**Limitation:** the 70/10/20 train/val/test split is chronological per user, but the test partition still includes reviews from before 2022. The system has not been evaluated on a strict future-only holdout (e.g., train on reviews ≤ 2020, test on 2021+).

**Evidence:** `src/07_recommendation/outputs/data/split_stats.json` — split boundaries are user-level quantiles, not global date cutoffs.

**Implication:** if review behavior changed significantly (e.g., post-pandemic dining patterns), the model's performance on new reviews from 2023 onward may be lower than reported.

**Next step:** re-run the evaluation with a global date cutoff (e.g., train ≤ 2019-12-31, test ≥ 2020-01-01) to measure temporal degradation explicitly.

---

## 6. Louvain Community Instability

**Limitation:** Louvain community detection is stochastic and not guaranteed to return identical community assignments across runs (partition optimization can converge to different local maxima).

**Evidence:** `src/08_graph/communities.py` — runs without a fixed random seed.

**Implication:** Community 3's exact membership (1,158 businesses, 55% Cluster 1) may vary slightly between runs. The direction of the finding (one community strongly enriched for hidden gems) is robust, but the exact community index and boundary may shift.

**Next step:** run Louvain 10 times with different random seeds and report the stability of the hidden-gem enrichment across runs (fraction of Cluster-1 in the top-enriched community).

---

## 7. Single-City Generalization

**Limitation:** the entire pipeline is trained and evaluated on Philadelphia data only (11,671 businesses, 36,922 users). Cluster labels, ALS embeddings, and graph structure are specific to this city's venue ecosystem.

**Evidence:** `src/01_ingestion/ingestion.py` — hard-coded to `state = "PA"` and `city = "Philadelphia"`.

**Implication:** the system cannot be deployed to other cities without retraining from scratch. The hidden-gem definition (Cluster 1) is relative to Philadelphia norms, not an absolute quality threshold.

**Next step:** extend ingestion to a second city (e.g., Las Vegas, which has the largest Yelp dataset) and measure whether the K-Means clustering recovers a similar hidden-gem segment independently, which would validate the approach's generalizability.

---

## 8. What This System Does Not Do

- **No real-time personalization:** embeddings are static; a user's profile does not update as they leave new reviews between retraining cycles.
- **No spatial awareness:** recommendations ignore geographic proximity (a user in South Philadelphia could be recommended venues in Northeast Philadelphia without distance penalization).
- **No diversity enforcement:** the router may recommend highly similar businesses (e.g., multiple pizza places) if they are all in Cluster 1 and score similarly for a given user.
- **No cold-start for items:** new businesses that open after the training cut are invisible until the next full pipeline run.

---

## Summary Table

| Limitation | Severity | Next step |
|---|---|---|
| Pool coverage (39% evaluable users) | High | Dynamic pool expansion |
| Graph covers only 32% of hidden gems | Medium | Lower threshold, add node2vec |
| Metrics sensitive to pool size | Medium | Report relative uplift + MAP |
| Implicit feedback includes negatives | Low | Binary confidence ablation |
| No global date cutoff test | Medium | Strict temporal holdout |
| Louvain stochasticity | Low | Multi-seed stability check |
| Single-city scope | High | Extend to second city |
