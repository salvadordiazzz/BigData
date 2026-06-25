# BigData — Yelp Hidden-Gem Discovery (Philadelphia)

Semester project pipeline: ingestion → cleaning → join → feature
engineering → dimensionality reduction → clustering → recommendation →
graph analytics. See `src/01_ingestion/` through `src/08_graph/` for the
numbered pipeline stages, each with its own `outputs/`.

## Cloning

This repo's git history is large because of the raw/processed dataset
artefacts that passed through it over the semester. If a full clone is slow
or times out, use a shallow clone:

```bash
git clone --depth 1 https://github.com/salvadordiazzz/BigData.git
```

## Reproducing the pipeline

```bash
pip install -r requirements.txt
python src/01_ingestion/ingestion.py
python src/02_cleaning/cleaning.py
python src/03_join/join.py
python src/04_features/feature_pipeline.py
python src/05_dimreduction/reduce.py
python src/06_clustering/kmeans_tuned.py
python src/06_clustering/dbscan.py
python src/07_recommendation/split.py
# ... see each stage's outputs/ for individual script docstrings
```

Raw Yelp Open Dataset files are not tracked in git (see `.gitignore`) and
must be placed under `data/` before running `ingestion.py`.
