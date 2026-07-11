# BigData — Yelp Hidden-Gem Discovery (Philadelphia)

Semester project pipeline: ingestion → cleaning → join → feature
engineering → dimensionality reduction → clustering → recommendation →
graph analytics. See `src/01_ingestion/` through `src/08_graph/` for the
numbered pipeline stages, each with its own `outputs/`.

## Quick start

```bash
git clone --depth 1 https://github.com/salvadordiazzz/BigData.git
cd BigData
pip install -r requirements.txt
# Download Yelp Open Dataset → data/data_raw/  (see RUNBOOK.md §1)
```

Full step-by-step instructions: **[RUNBOOK.md](RUNBOOK.md)**

## End-to-end demo

After running the full pipeline (stages 01–08 per the runbook):

```bash
# Interactive Streamlit app (recommended)
streamlit run app.py                    # opens at http://localhost:8501

# CLI demo
python demo.py                          # 3 example users (cold / mid / warm)
python demo.py --user <user_id>         # personalised recommendations
python demo.py --user <user_id> -k 5   # top-5
python demo.py --list-users             # print sample user IDs
```

The Streamlit app has two modes:
- **Build My Profile** — pick categories you enjoy → content-based SVD recommendations
- **Explore Real Users** — browse test-set users by tier → blend-router recommendations

## Repository structure

```
BigData/
  src/01_ingestion/ … src/08_graph/   Pipeline stages (code + outputs/)
  data/data_raw/                       Raw Yelp JSON (not tracked, gitignored)
  data/processed/                      Intermediate parquet (gitignored)
  reports/
    figures/                           Curated key figures (tracked)
    data/                              Key metric summaries (tracked)
    monitoring_plan.md                 Monitoring + operationalization plan
    limitations.md                     Limitations and future work
  artifacts/                           Final artifacts directory
  RUNBOOK.md                           Step-by-step pipeline runbook
  demo.py                              CLI end-to-end demo
  requirements.txt
```

## Key results

| Metric | Value |
|---|---|
| Hidden-gem candidate pool | 3,572 Cluster-1 businesses |
| Blend router NDCG@10 | 0.0365 (+2.8% vs tuned ALS) |
| Co-review graph | 4,913 nodes, 272,230 edges, 1 component |
| Hidden-gem community | Community 3: 55% Cluster-1 businesses |
| Router–PageRank Jaccard top-50 | 0.09 (genuine personalisation) |

## Plans and reports

- `reports/monitoring_plan.md` — drift signals, retraining schedule, failure modes
- `reports/limitations.md` — honest, evidence-linked limitations and next steps
- `deliverables/` — final report + slides builders (run locally; gitignored)
