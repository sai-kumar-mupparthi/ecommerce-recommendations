"""Runnable recommender demo: popularity baseline vs. item-item collaborative filtering.

Public demo only: user-item interactions are generated in ``make_interactions``
with latent "taste" segments (e.g. one group shops modern living-room
furniture, another rustic outdoor). No real customer data, no downloads.

Shows the production workflow in miniature:
  * implicit-feedback interaction matrix (views, add-to-carts, purchases)
  * leave-last-out temporal split per user (no future leakage)
  * item-item cosine similarity with weighted events, excluding already-seen items
  * offline metrics: precision@k, recall@k, NDCG@k and catalogue coverage

Usage:
    python recs_demo.py
    python recs_demo.py --user 42
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import sparse

EVENT_WEIGHT = {"view": 1.0, "add_to_cart": 3.0, "purchase": 5.0}


def make_interactions(users: int = 2000, items: int = 400, segments: int = 8, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    item_segment = rng.integers(0, segments, items)
    popularity = rng.zipf(1.6, items).clip(1, 50).astype(float)
    rows = []
    for u in range(users):
        taste = rng.choice(segments, 2, replace=False)  # each user likes two segments
        n = rng.integers(8, 40)
        in_taste = np.isin(item_segment, taste)
        weights = popularity * np.where(in_taste, 12.0, 1.0)
        chosen = rng.choice(items, n, p=weights / weights.sum())
        events = rng.choice(list(EVENT_WEIGHT), n, p=[0.75, 0.17, 0.08])
        for t, (i, e) in enumerate(zip(chosen, events)):
            rows.append((u, int(i), e, t))
    return pd.DataFrame(rows, columns=["user", "item", "event", "ts"])


def split_last_out(df: pd.DataFrame, holdout: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["user", "ts"])
    rank_from_end = df.groupby("user").cumcount(ascending=False)
    return df[rank_from_end >= holdout], df[rank_from_end < holdout]


def interaction_matrix(df: pd.DataFrame, users: int, items: int) -> sparse.csr_matrix:
    w = df["event"].map(EVENT_WEIGHT).to_numpy()
    m = sparse.csr_matrix((w, (df["user"], df["item"])), shape=(users, items))
    m.data = np.log1p(m.data)  # dampen heavy repeat viewers
    return m


class ItemKNN:
    def __init__(self, neighbours: int = 50):
        self.neighbours = neighbours

    def fit(self, m: sparse.csr_matrix) -> "ItemKNN":
        norms = np.sqrt(m.multiply(m).sum(axis=0)).A1 + 1e-9
        sim = (m.T @ m).toarray() / np.outer(norms, norms)
        np.fill_diagonal(sim, 0)
        # keep only top-k neighbours per item (sparser, faster, less noise)
        cutoff = -np.sort(-sim, axis=1)[:, self.neighbours - 1 : self.neighbours]
        self.sim = np.where(sim >= cutoff, sim, 0)
        self.m = m
        return self

    def recommend(self, user: int, k: int = 10) -> np.ndarray:
        scores = self.m[user] @ self.sim
        scores = np.asarray(scores).ravel()
        scores[self.m[user].indices] = -np.inf  # don't recommend what they already interacted with
        return np.argsort(-scores)[:k]


class Popularity:
    def fit(self, m: sparse.csr_matrix) -> "Popularity":
        self.rank = np.argsort(-np.asarray(m.sum(axis=0)).ravel())
        self.m = m
        return self

    def recommend(self, user: int, k: int = 10) -> np.ndarray:
        seen = set(self.m[user].indices)
        return np.array([i for i in self.rank if i not in seen][:k])


def evaluate(model, test: pd.DataFrame, k: int = 10) -> dict[str, float]:
    truth = test.groupby("user")["item"].apply(set)
    p = r = ndcg = 0.0
    shown: set[int] = set()
    for user, relevant in truth.items():
        recs = model.recommend(user, k)
        shown.update(recs.tolist())
        hits = [int(i in relevant) for i in recs]
        p += sum(hits) / k
        r += sum(hits) / len(relevant)
        dcg = sum(h / np.log2(rank + 2) for rank, h in enumerate(hits))
        idcg = sum(1 / np.log2(rank + 2) for rank in range(min(len(relevant), k)))
        ndcg += dcg / idcg
    n = len(truth)
    return {f"precision@{k}": p / n, f"recall@{k}": r / n, f"ndcg@{k}": ndcg / n,
            "catalog_coverage": len(shown) / model.m.shape[1]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", type=int, help="print recommendations for one user")
    args = parser.parse_args()

    users, items = 2000, 400
    df = make_interactions(users, items)
    train, test = split_last_out(df)
    m = interaction_matrix(train, users, items)
    models = {"popularity": Popularity().fit(m), "item_knn": ItemKNN().fit(m)}

    if args.user is not None:
        print("history:", sorted(set(train.loc[train.user == args.user, "item"]))[:15])
        print("item_knn recs:", models["item_knn"].recommend(args.user).tolist())
        return
    print(f"{'model':<12}" + "".join(f"{k:>18}" for k in evaluate(models["popularity"], test)))
    for name, model in models.items():
        print(f"{name:<12}" + "".join(f"{v:>18.3f}" for v in evaluate(model, test).values()))


if __name__ == "__main__":
    main()
