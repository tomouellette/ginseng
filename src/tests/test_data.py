import numpy as np
import pytest

from anndata import AnnData

from ginseng.data import GinsengDataset


@pytest.fixture
def adata():
    np.random.seed(123)
    X = np.vstack(
        [
            np.random.normal(10.0, 3.0, size=100),
            np.random.normal(10.0, 3.0, size=100),
            np.random.normal(10.0, 3.0, size=100),
            np.random.normal(90.0, 1.0, size=100),
            np.random.normal(90.0, 1.0, size=100),
            np.random.normal(90.0, 1.0, size=100),
        ]
    )
    X[0:3, 0] = 10.0
    X[3:6, 0] = 90.0
    X[0:6, -5:] = np.array([1, 2, 3, 4, 5])

    obs = {"cell_type": 3 * [0] + 3 * [1], "donor": [*range(6)]}
    var = {"gene": np.arange(100).astype(str)}

    return AnnData(X, obs=obs, var=var)


def test_create_dense(tmp_path, adata):
    dataset = GinsengDataset.create(tmp_path / "dense_test", adata, "cell_type")

    assert dataset.n_cells == 6
    assert dataset.n_genes == 100
    assert dataset.n_labels == 2
    assert len(dataset.label_indices) == 2
    assert np.all(np.unique(dataset.labels) == [0, 1])


def test_create_dense_groups(tmp_path, adata):
    dataset = GinsengDataset.create(
        tmp_path / "dense_test", adata, "cell_type", groups="donor"
    )

    assert dataset.n_cells == 6
    assert dataset.n_genes == 100
    assert dataset.n_labels == 2
    assert dataset.n_groups == 6
    assert np.all(np.unique(dataset.labels) == [0, 1])


def test_create_sparse(tmp_path, adata):
    dataset = GinsengDataset.create(tmp_path / "sparse_test", adata, "cell_type")

    assert dataset.n_cells == 6
    assert dataset.n_genes == 100
    assert dataset.n_labels == 2
    assert len(dataset.label_indices) == 2
    assert np.all(np.unique(dataset.labels) == [0, 1])


def test_create_sparse_groups(tmp_path, adata):
    dataset = GinsengDataset.create(
        tmp_path / "dense_test", adata, "cell_type", groups="donor"
    )

    assert dataset.n_cells == 6
    assert dataset.n_genes == 100
    assert dataset.n_labels == 2
    assert dataset.n_groups == 6
    assert len(dataset.label_indices) == 2
    assert np.all(np.unique(dataset.labels) == [0, 1])


def test_make_holdout(tmp_path, adata):
    dataset = GinsengDataset.create(tmp_path / "holdout_test", adata, "cell_type")
    dataset.make_holdout(holdout_fraction=0.5, rng=np.random.default_rng(42))

    assert dataset.has_holdout
    assert len(dataset.train_indices) + len(dataset.holdout_indices) == 6

    holdout_labels = dataset.labels[dataset.holdout_indices]
    assert set(holdout_labels) == {0, 1}


def test_make_holdout_groups(tmp_path, adata):
    labels = adata.obs["cell_type"].to_numpy()

    groups = adata.obs["donor"].to_numpy().astype(str)
    groups[:3] = "A"
    groups[3:] = "B"

    dataset = GinsengDataset.create(
        tmp_path / "holdout_test", adata, labels, groups=groups
    )

    dataset.make_holdout(
        holdout_fraction=0.5, rng=np.random.default_rng(42), group_level=True
    )

    assert dataset.has_holdout
    assert len(dataset.train_indices) + len(dataset.holdout_indices) == 6

    train_labels = dataset.labels[dataset.train_indices]
    holdout_labels = dataset.labels[dataset.holdout_indices]
    assert len(np.unique(train_labels)) == 1
    assert len(np.unique(holdout_labels)) == 1

    groups = adata.obs["donor"].to_numpy().astype(str)
    dataset = GinsengDataset.create(
        tmp_path / "holdout_test", adata, labels, groups=groups
    )

    dataset.make_holdout(
        rng=np.random.default_rng(42),
        group_level=True,
        group_mode="loo",
    )

    holdout_groups = dataset.groups[dataset.holdout_indices]
    assert len(np.unique(holdout_groups)) == 1


def test_iter_batches(tmp_path, adata):
    dataset = GinsengDataset.create(tmp_path / "batch_test", adata, "cell_type")
    dataset.make_holdout(holdout_fraction=0.5, rng=np.random.default_rng(42))

    batches = list(
        dataset.iter_batches(batch_size=2, split="train", rng=np.random.default_rng(1))
    )
    assert all(batch[0].shape[0] <= 2 for batch in batches)

    holdout_batches = list(
        dataset.iter_batches(
            batch_size=2, split="holdout", rng=np.random.default_rng(1)
        )
    )
    assert all(batch[0].shape[0] <= 2 for batch in holdout_batches)


def test_iter_batches_balanced(tmp_path, adata):
    labels = adata.obs["cell_type"].to_numpy()
    labels[:2] = 0
    labels[2:] = 1

    dataset = GinsengDataset.create(tmp_path / "batch_test", adata, labels)
    dataset.make_holdout(holdout_fraction=0.5, rng=np.random.default_rng(42))

    labels = {k: 0 for k in np.unique(dataset.labels)}
    for x, y in dataset.iter_batches(split="train", balance_train=True, batch_size=1):
        for yi in y:
            labels[yi] += 1

    counts = list(labels.values())
    assert all([i == counts[0] for i in counts])
    assert np.sum(counts) == 2


def test_getitem_and_len(tmp_path, adata):
    dataset = GinsengDataset.create(tmp_path / "getitem_test", adata, "cell_type")
    dataset.make_holdout(holdout_fraction=0.5, rng=np.random.default_rng(42))

    assert dataset.__len__("all") == 6
    assert dataset.__len__("train") + dataset.__len__("holdout") == 6

    x0, y0 = dataset.__getitem__(0, split="all")
    assert x0.shape[0] == 100
    assert y0 in [0, 1]

    x_train, y_train = dataset.__getitem__(0, split="train")
    x_holdout, y_holdout = dataset.__getitem__(0, split="holdout")
    assert x_train.shape[0] == 100
    assert x_holdout.shape[0] == 100
