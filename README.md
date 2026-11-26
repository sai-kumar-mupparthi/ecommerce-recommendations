# E-commerce Recommendation System

**Repo name on GitHub:** `ecommerce-recommendations`
**Context:** Junior Machine Learning Engineer @ Wayfair, Boston, MA (May 2020 – Jun 2021)

> Case study of production work. This repo describes the system design and results — it contains no employer code or data.

> ▶ **Includes a runnable demo on synthetic data:** see [Run the demo](#run-the-demo) below.

## What it is
Personalized product recommendation models and supporting analytics for e-commerce, built on large-scale user interaction data.

## What I built
- Recommendation models for personalized product suggestions
- Cleaning and validation pipelines for 5M+ user interaction logs
- Exploratory analysis of purchasing patterns that informed 3 new feature engineering strategies
- Customer behavior and product interaction features for recommendation models
- Classification and regression models supporting analytics across 4 product categories
- Dashboards communicating model insights to 20+ cross-functional stakeholders

## Results
- **+5%** user click-through rate
- **+11%** baseline model accuracy from engineered features
- **-8%** error rates through feature selection and hyperparameter tuning
- **-20%** data extraction time through optimized SQL

## Tech stack
Python · scikit-learn · Pandas · NumPy · PostgreSQL · MLflow · Matplotlib · Docker · SQL

## Skills demonstrated
Recommendation systems · Feature engineering · EDA · Model evaluation · Data visualization

---

## Run the demo

[![tests](https://github.com/sai-kumar-mupparthi/ecommerce-recommendations/actions/workflows/tests.yml/badge.svg)](https://github.com/sai-kumar-mupparthi/ecommerce-recommendations/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

> **About this demo.** The production system above is described as a case study and contains no employer code or data. This repo also includes a small, runnable demo that applies the same approach at laptop scale on **synthetic / sample data**. Its numbers come from that synthetic data and are not the production results.

```bash
pip install -r requirements-demo.txt
python recs_demo.py
python recs_demo.py --user 42
pytest -q
```

**Sample output**

```text
$ python recs_demo.py
model             precision@10         recall@10           ndcg@10  catalog_coverage
popularity               0.027             0.092             0.061             0.052
item_knn                 0.066             0.229             0.175             0.282
```

**What's inside `recs_demo.py`**

- Synthetic implicit feedback (views, add-to-carts, purchases) from 2,000 users with latent taste segments
- Event weighting and log damping into a sparse user-item matrix
- A leave-last-3-out temporal split per user, so the model never sees the future
- Item-item collaborative filtering (cosine similarity, top-50 neighbours) that never re-recommends already-seen items
- Comparison against a popularity baseline on precision@10, recall@10, NDCG@10 and catalogue coverage
