import numpy as np
import pytest

from anndata import AnnData

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

    return AnnData(
        X,
        var={
            "gene": np.arange(100).astype(str),
        },
        obs={"cell_type": 3 * [0] + 3 * [1], "donor": [*range(6)]},
    )


def test_train(tmp_path, adata):
    dataset = GinsengDataset.create(tmp_path / "ginseng.zarr", adata, "cell_type")

    settings = GinsengTrainerSettings(holdout_fraction=0.5, batch_size=1)
    logger, model_state = GinsengTrainer(dataset, settings=settings)

    assert hasattr(logger, "epoch")
    assert hasattr(logger, "train_loss")
    assert hasattr(logger, "holdout_loss")
    assert hasattr(logger, "holdout_accuracy")

    assert hasattr(model_state, "params")
    assert hasattr(model_state, "genes")
    assert hasattr(model_state, "label_keys")
    assert hasattr(model_state, "label_values")
    assert hasattr(model_state, "normalize")
    assert hasattr(model_state, "target_sum")
    assert hasattr(model_state, "training")

    assert len(logger.epoch) == 10
    assert len(logger.train_loss) == 10
    assert len(logger.holdout_loss) == 10
    assert len(logger.holdout_accuracy) == 10

    assert np.all(dataset.root.attrs["genes"][:] == model_state.genes)


def test_train_augment(tmp_path, adata):
    dataset = GinsengDataset.create(tmp_path / "ginseng.zarr", adata, "cell_type")

    settings = GinsengTrainerSettings(
        holdout_fraction=0.5, batch_size=1, rate=0.1, lam_max=0.1, lower=0, upper=10
    )

    logger, model_state = GinsengTrainer(dataset, settings=settings)

    assert hasattr(logger, "epoch")
    assert hasattr(logger, "train_loss")
    assert hasattr(logger, "holdout_loss")
    assert hasattr(logger, "holdout_accuracy")

    assert hasattr(model_state, "params")
    assert hasattr(model_state, "genes")
    assert hasattr(model_state, "label_keys")
    assert hasattr(model_state, "label_values")
    assert hasattr(model_state, "normalize")
    assert hasattr(model_state, "target_sum")
    assert hasattr(model_state, "training")

    assert len(logger.epoch) == 10
    assert len(logger.train_loss) == 10
    assert len(logger.holdout_loss) == 10
    assert len(logger.holdout_accuracy) == 10

    assert np.all(dataset.root.attrs["genes"][:] == model_state.genes)
