import numpy as np
import pytest

from anndata import AnnData

from ginseng.data import GinsengDataset
from ginseng.train import GinsengTrainer, GinsengTrainerSettings

from ginseng.io import (
    read_10x_mtx,
    read_10x_h5,
    read_adata,
    save_ginseng_state,
    load_ginseng_state,
)

DATA: str = "src/tests/data/"

FIXTURES: dict[str, tuple[...]] = {
    "mtx": (DATA + "pbmc_10x_mtx", 1222, 33538),
    "v3_h5": (DATA + "pbmc_10x_v3.h5", 1222, 33538),
    "v4_h5": (DATA + "pbmc_10x_v4.h5", 5722, 38606),
    "h5ad": (DATA + "pbmc_10x_v3.h5ad", 1222, 33538),
}


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

    return AnnData(
        X,
        var={
            "gene": np.arange(100).astype(str),
        },
        obs={"cell_type": 3 * [0] + 3 * [1], "donor": [*range(6)]},
    )


def test_read_10x_mtx():
    path, n_cells, n_genes = FIXTURES["mtx"]
    adata = read_10x_mtx(path)
    assert adata.shape == (n_cells, n_genes)
    assert adata.obs.shape[0] == n_cells
    assert adata.var.shape[0] == n_genes


def test_read_10x_h5_v3():
    path, n_cells, n_genes = FIXTURES["v3_h5"]
    adata = read_10x_h5(path)
    assert adata.shape == (n_cells, n_genes)
    assert adata.obs.shape[0] == n_cells
    assert adata.var.shape[0] == n_genes


def test_read_10x_h5_v4():
    path, n_cells, n_genes = FIXTURES["v3_h5"]
    adata = read_10x_h5(path)
    assert adata.shape == (n_cells, n_genes)
    assert adata.obs.shape[0] == n_cells
    assert adata.var.shape[0] == n_genes


def test_read_adata():
    for fmt, (path, n_cells, n_genes) in FIXTURES.items():
        print(f"Format: {fmt}")
        adata = read_adata(path)
        assert adata.shape == (n_cells, n_genes)
        assert adata.obs.shape[0] == n_cells
        assert adata.var.shape[0] == n_genes

    with pytest.raises(FileNotFoundError):
        adata = read_adata("./")

    with pytest.raises(ValueError):
        adata = read_adata("placeholder.txt")


def test_ginseng_state_roundtrip(tmp_path, adata):
    dataset = GinsengDataset.create(tmp_path / "ginseng.zarr", adata, "cell_type")

    settings = GinsengTrainerSettings(holdout_fraction=0.5, batch_size=1)
    logger, model_state = GinsengTrainer(dataset, settings=settings)

    save_ginseng_state(model_state, tmp_path / "ginseng.state")

    model_state_reloaded = load_ginseng_state(tmp_path / "ginseng.state")

    for v1, v2 in zip(
        model_state.params.values(), model_state_reloaded.params.values()
    ):
        assert np.all(v1["W"] == v2["W"])
        assert np.all(v1["b"] == v2["b"])

    assert np.all(model_state.genes == model_state_reloaded.genes)
    assert np.all(model_state.label_keys == model_state_reloaded.label_keys)
    assert np.all(model_state.label_values == model_state_reloaded.label_values)
    assert model_state.normalize == model_state_reloaded.normalize
    assert model_state.target_sum == model_state_reloaded.target_sum
    assert model_state.training == model_state_reloaded.training
