# Copyright (c) 2026, Tom Ouellette
# Licensed under the MIT License

import numpy as np
import pytest
import jax
import anndata as ad

from unittest.mock import patch
from pathlib import Path

from ginseng.data.dataset import GinsengDataset
from ginseng.train.trainer import (
    GinsengClassifierTrainer,
    GinsengClassifierTrainerSettings,
)


@pytest.fixture
def trainer_setup(tmp_path: Path):
    """Create a temporary dataset and trainer instance."""
    # Synthetic data
    rng = np.random.default_rng(42)
    n_cells, n_genes = 200, 50
    X = rng.poisson(lam=5.0, size=(n_cells, n_genes)).astype(np.float32)
    labels = rng.integers(0, 2, size=n_cells)

    adata = ad.AnnData(X=X)
    adata.obs["cell_type"] = labels
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    # Setup Dataset
    ds_path = tmp_path / "test_train.zarr"
    dataset = GinsengDataset.create(path=ds_path, adata=adata, label_key="cell_type")

    # Setup settings
    settings = GinsengClassifierTrainerSettings(
        hidden_dim=16, batch_size=32, holdout_fraction=0.2, seed=123
    )

    return dataset, settings


class TestGinsengClassifierTrainer:
    """Test suite for the GinsengClassifierTrainer."""

    def test_initialization(self, trainer_setup):
        dataset, settings = trainer_setup
        trainer = GinsengClassifierTrainer(dataset, settings)

        assert trainer.model.n_genes == dataset.n_genes
        assert trainer.model.n_classes == len(dataset.label_names)
        assert trainer.opt_state is not None

    def test_fit_updates_parameters(self, trainer_setup):
        dataset, settings = trainer_setup
        trainer = GinsengClassifierTrainer(dataset, settings)

        # Capture initial parameters
        init_params = jax.tree_util.tree_leaves(trainer.model.params)

        # Train for 2 epochs
        trainer.fit(epochs=2, silent=True)

        # Capture new parameters
        new_params = jax.tree_util.tree_leaves(trainer.model.params)

        # Check that at least some parameters changed
        for p1, p2 in zip(init_params, new_params):
            assert not np.array_equal(p1, p2)

    def test_logger_updates(self, trainer_setup):
        dataset, settings = trainer_setup
        trainer = GinsengClassifierTrainer(dataset, settings)

        epochs = 2
        trainer.fit(epochs=epochs, silent=True)

        assert len(trainer.logger.epoch) == epochs
        assert len(trainer.logger.train_loss) == epochs
        assert len(trainer.logger.holdout_accuracy) == epochs
        assert all(isinstance(v, float) for v in trainer.logger.train_loss)

    def test_zero_holdout_fraction(self, trainer_setup):
        """Verify trainer handles cases with no validation set."""
        dataset, settings = trainer_setup
        settings.holdout_fraction = 0.0
        trainer = GinsengClassifierTrainer(dataset, settings)

        trainer.fit(epochs=1, silent=True)

        # Holdout metrics should be NaN as per the implementation logic
        assert np.isnan(trainer.logger.holdout_loss[-1])
        assert np.isnan(trainer.logger.holdout_accuracy[-1])

    def test_stratified_training(self, tmp_path):
        """Verify trainer works with group stratification."""
        rng = np.random.default_rng(42)
        n_cells, n_genes = 100, 20
        X = rng.random((n_cells, n_genes)).astype(np.float32)

        adata = ad.AnnData(X=X)
        adata.obs["label"] = rng.integers(0, 2, size=n_cells)
        # 5 distinct groups
        adata.obs["donor"] = np.repeat(np.arange(5), 20)

        ds = GinsengDataset.create(
            tmp_path / "strat.zarr", adata, "label", group_key="donor"
        )

        # Should take exactly 1 donor
        settings = GinsengClassifierTrainerSettings(
            group_level=True,
            holdout_fraction=0.2,
        )

        trainer = GinsengClassifierTrainer(ds, settings)
        trainer.fit(epochs=1, silent=True)

        # Check that split actually happened
        assert ds.test_idx is not None
        assert len(ds.test_idx) == 20


class TestTrainerAugmentation:
    """Test suite for verifying augmentation integration in the Trainer."""

    def test_augmentation_parameters_passed(self, trainer_setup):
        """
        Verify that augment is called with the correct settings
        defined in GinsengClassifierTrainerSettings.
        """
        dataset, _ = trainer_setup

        # Configure specific augmentation settings
        settings = GinsengClassifierTrainerSettings(
            rate=0.5, lam_max=10.0, lower=5, upper=10, batch_size=32
        )

        trainer = GinsengClassifierTrainer(dataset, settings)

        # We patch the 'augment' function in the trainer's module
        # to see if it receives the settings correctly.
        with patch("ginseng.train.trainer.augment") as mock_augment:
            mock_augment.side_effect = lambda key, x, r, l_max, low, up: x

            # Run one epoch (or just one step if possible, but fit works)
            trainer.fit(epochs=1, silent=True)

            # Get the first call to augment
            # Based on: augment(key, x, rate, lam_max, lower, upper)
            args, kwargs = mock_augment.call_args

            # If you call it as: augment(key, x_jax, self.settings.rate, ...)
            # index 0: key, 1: x, 2: rate, 3: lam_max, 4: lower, 5: upper
            assert args[2] == 0.5  # rate
            assert args[3] == 10.0  # lam_max
            assert args[4] == 5  # lower
            assert args[5] == 10  # upper

    def test_no_augmentation_when_none(self, trainer_setup):
        """Verify that augmentation works with None values (default)."""
        dataset, _ = trainer_setup
        settings = GinsengClassifierTrainerSettings(rate=None, lam_max=None)

        trainer = GinsengClassifierTrainer(dataset, settings)

        # This should run without crashing, passing None to the augment function
        trainer.fit(epochs=1, silent=True)
        assert len(trainer.logger.train_loss) == 1

    def test_augmentation_key_uniqueness(self, trainer_setup):
        """
        Ensure the trainer provides unique JAX keys for
        augmentation across batches.
        """
        dataset, _ = trainer_setup
        trainer = GinsengClassifierTrainer(dataset, GinsengClassifierTrainerSettings())

        with patch("ginseng.train.trainer.augment") as mock_augment:
            mock_augment.side_effect = lambda key, x, r, l_max, low, up: x

            trainer.fit(epochs=1, silent=True)

            # Collect all keys passed to augment
            keys = [call.args[0] for call in mock_augment.call_args_list]

            # Check that keys are different (not identical objects/values)
            for i in range(len(keys) - 1):
                assert not np.array_equal(keys[i], keys[i + 1])
