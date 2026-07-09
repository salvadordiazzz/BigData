# Monitoring and Operationalization Plan

**System:** Yelp Hidden-Gem Discovery — Philadelphia  
**Architecture:** Confidence-weighted blend router (65% ALS + 20% Content + 15% Pop for cold/mid users; pure ALS for warm users) ranked against the Cluster-1 hidden-gem candidate pool.

---

## 1. Serving Assumptions

| Assumption | Implication |
|---|---|
| Candidate pool is Cluster-1 businesses from the train set (3,572 items) | New businesses that open after the last training cut will not appear in recommendations until the next retraining cycle. |
| User activity tier is determined at inference time from train review count | A user who crosses the COLD/MID/WARM threshold between retrains may receive a mismatched blend weight. Tier assignment should be refreshed with each retrain. |
| ALS model is loaded from a serialized `.npz` file (scipy sparse) | Latency is dominated by disk I/O on first load (~1–2s); subsequent calls are in-memory (< 1ms per user). |
| Content-based scores require the 200-dim SVD embedding of each business | If a business's review text changes substantially (renovation, new management), its content score will drift until the next feature rebuild. |
| Graph analytics are computed offline; PageRank and community labels are static lookup tables | Graph-based enrichment (community, centrality) is not updated in real time. Drift is expected when new reviews arrive. |

---

## 2. Data Drift Signals

### 2.1 Review Volume Drift

**What to watch:** monthly review counts per business and per user.

**Trigger:** if the fraction of active users (≥ 5 reviews in a rolling 12-month window) drops below 80% of the training-time baseline, business-side features will degrade. Flag for retraining.

**Evidence from project:** the training set uses a 70/10/20 temporal split; users with fewer than 5 reviews are excluded. Any shift in platform engagement patterns (e.g., post-pandemic recovery or platform changes) will alter this baseline.

### 2.2 Candidate Pool Staleness

**What to watch:** fraction of Cluster-1 businesses in the candidate pool that are still active (not permanently closed) in the current review snapshot.

**Trigger:** if > 10% of the 3,572 candidate businesses have zero reviews in the last 6 months, the pool should be refreshed with a new K-Means run on updated features.

### 2.3 Rating Distribution Shift

**What to watch:** mean and variance of review_stars in the incoming review stream vs. the train-set distribution (train mean: ~3.7).

**Trigger:** a KL divergence > 0.1 between monthly rating distributions and the training distribution indicates that the implicit confidence weights used in ALS need recalibration.

---

## 3. Model Drift Signals

### 3.1 ALS Embedding Drift

**What to watch:** NDCG@10 on a rolling monthly held-out set (new reviews from users present in the train set).

**Baseline to beat:** current test NDCG@10 = 0.0365.  
**Trigger:** two consecutive months below 0.0300 (18% degradation) → retrain ALS.

### 3.2 Blend Weight Validity

**What to watch:** per-tier NDCG on monthly held-out reviews.

**Current weights** (grid-searched on validation set): cold/mid = (0.65, 0.20, 0.15).  
**Trigger:** if cold-tier NDCG degrades > 15% while warm-tier holds, the blend weights are stale and should be re-swept on recent validation data.

### 3.3 Community Structure Drift

**What to watch:** fraction of recommendations from Community 3 (the hidden-gem community identified by Louvain on the co-review graph) vs. total recommendations.

**Expected level:** ~35% of router recommendations overlap with Community 3 businesses (based on our analysis of 1,145 graph-present candidates out of 3,572).  
**Trigger:** drop below 20% → the co-review graph structure may have shifted (new hubs, dissolved communities); rebuild graph and re-run Louvain.

---

## 4. Retraining Schedule

| Component | Frequency | Trigger |
|---|---|---|
| Full ALS retrain | Every 3 months | Quarterly or on drift signal above |
| Feature pipeline (PC2) | Every 3 months | Aligned with ALS retrain |
| K-Means clustering (PC3) | Every 6 months | When pool staleness > 10% |
| Blend weight grid search | Every 3 months | After every ALS retrain |
| Co-review graph + Louvain (PC5) | Every 6 months | When community drift signal fires |

**Retrain order (dependencies):**
1. Ingestion → Cleaning → Join (data pipeline)
2. Feature pipeline → Dimensionality reduction
3. K-Means clustering (if scheduled)
4. Train/val/test split → ALS tuning → Fallback router
5. Graph rebuild → Community detection → Ranking comparison

Full pipeline runtime on a single workstation: approximately 2–4 hours.

---

## 5. Logging Requirements

| Log event | Fields | Destination |
|---|---|---|
| Recommendation request | timestamp, user_id, tier, top_k | Application log (structured JSON) |
| Recommendation served | user_id, business_ids, scores, model_version | Recommendations log |
| User interaction (click/visit) | timestamp, user_id, business_id, position | Interaction log |
| Metric evaluation run | run_date, NDCG@k, HR@k, tier, model_version | Metrics log |
| Data pipeline run | start_time, end_time, stage, row_counts, errors | Pipeline log |

**Minimum retention:** 13 months (to enable year-over-year comparisons).

---

## 6. Failure Modes and Mitigations

| Failure mode | Probability | Impact | Mitigation |
|---|---|---|---|
| ALS model file corrupted / missing | Low | High — all recommendations fail | Store two versioned model snapshots; fall back to popularity ranking |
| Candidate pool empty for user (all Cluster-1 businesses filtered) | Low (< 1% of users in analysis) | Medium — user gets no HidGem recommendations | Fall back to Cluster-0 (mainstream) top businesses by PageRank |
| New user with 0 reviews (true cold start, not in train) | High (every new user) | Low — expected path | Serve static PageRank-ordered list of Community-3 businesses (the hidden-gem community) |
| Content-based score file missing (feature rebuild not run) | Medium | Medium — blend falls back to pure ALS | Detect at startup; log warning; degrade gracefully to pure ALS blend (0.85 ALS, 0.15 Pop) |
| Graph outputs missing (centrality/community not built) | Medium | Low — enrichment step skipped | Demo and router still function; graph columns show "no graph" |
| Staleness in rating data (restaurant closed, changed name) | High (ongoing) | Low per case, cumulative | Pool refresh every 6 months; cross-reference with Yelp business status field |

---

## 7. Scalability Notes

The current implementation runs on a single machine using in-memory pandas and scipy sparse matrices. At 775K reviews and 36K users, memory usage peaks at ~4GB during the ALS sweep.

If the dataset doubles in size (plausible if extending to all Pennsylvania):

- Replace pandas ALS training with a distributed alternative (e.g., Spark MLlib ALS or PyTorch-based MF)
- Shard the predictions parquet by user_id for faster per-user lookups
- Move the graph build to NetworkX + GraphFrames or igraph for larger graphs
- Introduce a feature store (e.g., Feast) to decouple feature computation from serving
