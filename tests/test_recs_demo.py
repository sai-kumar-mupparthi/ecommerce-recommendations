import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recs_demo import ItemKNN, Popularity, evaluate, interaction_matrix, make_interactions, split_last_out  # noqa: E402


def _setup():
    users, items = 600, 200
    train, test = split_last_out(make_interactions(users, items))
    return interaction_matrix(train, users, items), train, test


def test_knn_beats_popularity():
    m, _, test = _setup()
    knn = evaluate(ItemKNN().fit(m), test)
    pop = evaluate(Popularity().fit(m), test)
    assert knn["ndcg@10"] > pop["ndcg@10"]
    assert knn["catalog_coverage"] > pop["catalog_coverage"]


def test_no_already_seen_items():
    m, train, _ = _setup()
    model = ItemKNN().fit(m)
    seen = set(train.loc[train.user == 5, "item"])
    assert not seen & set(model.recommend(5).tolist())
