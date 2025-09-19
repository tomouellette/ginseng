import numpy as np
import pytest

from anndata import AnnData
from dataclasses import dataclass, field

from ginseng.annotate import annotate
from ginseng.data import GinsengDataset
from ginseng.train import GinsengTrainerSettings, GinsengTrainer


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


def get_model_state(adata, tmp_path):
    dataset = GinsengDataset.create(tmp_path / "ginseng.zarr", adata, "cell_type")
    settings = GinsengTrainerSettings(holdout_fraction=0.5, batch_size=1)
    _, model_state = GinsengTrainer(dataset, settings=settings, silent=True)
    return model_state


def test_gene_overlap_failure(tmp_path, adata):
    model_state = get_model_state(adata, tmp_path)
    model_state.genes = np.arange(100, 200).astype(str)
    with pytest.raises(ValueError):
        annotate(model_state, adata, gene_key="gene")


def test_gene_overlap_partial(tmp_path, adata):
    model_state = get_model_state(adata, tmp_path)
    model_state.genes = np.arange(50, 150).astype(str)
    with pytest.warns(UserWarning):
        annotate(model_state, adata, gene_key="gene")


def test_gene_key_failure(tmp_path, adata):
    model_state = get_model_state(adata, tmp_path)
    model_state.genes = np.arange(50, 150).astype(str)
    with pytest.raises(ValueError):
        annotate(model_state, adata, gene_key="bad_gene_key")


def test_return_probs(tmp_path, adata):
    model_state = get_model_state(adata, tmp_path)
    model_state.genes = np.arange(0, 100).astype(str)
    probs = annotate(model_state, adata, gene_key="gene", return_probs=True)

    row_sum = probs.to_numpy().sum(axis=1)
    assert np.allclose(row_sum, 1)
    assert np.all(probs.columns == list(model_state.label_values))


def test_return_table(tmp_path, adata):
    model_state = get_model_state(adata, tmp_path)
    model_state.genes = np.arange(0, 100).astype(str)
    preds = annotate(model_state, adata, gene_key="gene", return_table=True)

    assert preds.shape[0] == adata.shape[0]
    assert np.unique(preds) in list(model_state.label_values)


def test_annotate_default(tmp_path, adata):
    model_state = get_model_state(adata, tmp_path)
    model_state.genes = np.arange(0, 100).astype(str)

    annotate(model_state, adata, gene_key="gene")

    assert "ginseng_cell_type" in adata.obs.columns
