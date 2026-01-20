# Copyright (c) 2025, Tom Ouellette
# Licensed under the MIT License

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from anndata import AnnData

from ginseng.data.dataset import GinsengDataset

from ginseng.train.trainer import (
    GinsengClassifierTrainer,
    GinsengClassifierTrainerSettings,
)

from ginseng.data.io import (
    read_10x_mtx,
    read_10x_h5,
    read_adata,
    save_model,
    load_model,
)


DATA: str = "src/tests/_fixtures/"

FIXTURES: dict[str, tuple[str, int, int]] = {
    "mtx": (DATA + "pbmc_10x_mtx", 1222, 33538),
    "v3_h5": (DATA + "pbmc_10x_v3.h5", 1222, 33538),
    "v4_h5": (DATA + "pbmc_10x_v4.h5", 5722, 38606),
    "h5ad": (DATA + "pbmc_10x_v3.h5ad", 1222, 33538),
}


@pytest.fixture
def rng_key():
    """Provide a fresh PRNG key for each test."""
    return jax.random.PRNGKey(0)


@pytest.fixture
def adata():
    """Create a simple synthetic AnnData for testing."""
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
        X - X.min(),  # Ensure non-negative
        var={
            "gene": np.arange(100).astype(str),
        },
        obs={"cell_type": 3 * ["A"] + 3 * ["B"], "donor": [*range(6)]},
    )


class TestDataIO:
    """Test reading various single-cell data formats."""

    def test_read_10x_mtx(self):
        """Test reading 10x matrix market format."""
        path, n_cells, n_genes = FIXTURES["mtx"]
        adata = read_10x_mtx(path)
        assert adata.shape == (n_cells, n_genes)
        assert adata.obs.shape[0] == n_cells
        assert adata.var.shape[0] == n_genes

    def test_read_10x_h5_v3(self):
        """Test reading 10x v3 HDF5 format."""
        path, n_cells, n_genes = FIXTURES["v3_h5"]
        adata = read_10x_h5(path)
        assert adata.shape == (n_cells, n_genes)
        assert adata.obs.shape[0] == n_cells
        assert adata.var.shape[0] == n_genes

    def test_read_10x_h5_v4(self):
        """Test reading 10x v4 HDF5 format."""
        path, n_cells, n_genes = FIXTURES["v4_h5"]
        adata = read_10x_h5(path)
        assert adata.shape == (n_cells, n_genes)
        assert adata.obs.shape[0] == n_cells
        assert adata.var.shape[0] == n_genes

    def test_read_adata(self):
        """Test reading all supported formats via unified interface."""
        for fmt, (path, n_cells, n_genes) in FIXTURES.items():
            print(f"Format: {fmt}")
            adata = read_adata(path)
            assert adata.shape == (n_cells, n_genes)
            assert adata.obs.shape[0] == n_cells
            assert adata.var.shape[0] == n_genes

    def test_read_adataerror_handling(self):
        """Test error handling for invalid paths and formats."""
        with pytest.raises(FileNotFoundError):
            read_adata("./")

        with pytest.raises(ValueError):
            read_adata("placeholder.txt")


class TestModelStateIO:
    """Test saving and loading model states."""

    def test_state_roundtrip(self, tmp_path, adata):
        """Test complete save/load roundtrip of model state."""
        # Create dataset
        dataset = GinsengDataset.create(tmp_path / "ginseng.zarr", adata, "cell_type")

        # Train a simple model
        settings = GinsengClassifierTrainerSettings(
            holdout_fraction=0.5,
            batch_size=1,
            hidden_dim=32,
        )

        trainer = GinsengClassifierTrainer(dataset, settings=settings)
        model, _ = trainer.fit(epochs=1, silent=True)

        # Create model state
        from ginseng.model.state import state_from_classifier_trainer

        model_state = state_from_classifier_trainer(trainer)

        # Save
        save_model(model_state, tmp_path / "ginseng_model.h5")

        # Load
        model_state_reloaded = load_model(tmp_path / "ginseng_model.h5")

        # Verify parameters are identical
        for key in model_state.params.keys():
            assert np.allclose(
                model_state.params[key]["W"], model_state_reloaded.params[key]["W"]
            )
            assert np.allclose(
                model_state.params[key]["b"], model_state_reloaded.params[key]["b"]
            )

        # Verify metadata
        assert np.all(model_state.genes == model_state_reloaded.genes)
        assert np.all(model_state.label_keys == model_state_reloaded.label_keys)
        assert np.all(model_state.label_values == model_state_reloaded.label_values)
        assert model_state.normalize == model_state_reloaded.normalize
        assert model_state.target_sum == model_state_reloaded.target_sum
        assert model_state.dropout_rate == model_state_reloaded.dropout_rate
        assert model_state.training == model_state_reloaded.training
        assert model_state.n_genes == model_state_reloaded.n_genes
        assert model_state.n_classes == model_state_reloaded.n_classes
        assert model_state.hidden_dim == model_state_reloaded.hidden_dim

    def test_state_inference_after_load(self, tmp_path, adata):
        """Test that loaded model can perform inference."""
        # Create and train
        dataset = GinsengDataset.create(tmp_path / "ginseng.zarr", adata, "cell_type")
        settings = GinsengClassifierTrainerSettings(
            holdout_fraction=0.0,
            batch_size=2,
            hidden_dim=32,
        )
        trainer = GinsengClassifierTrainer(dataset, settings=settings)
        model, _ = trainer.fit(epochs=2, silent=True)

        # Get predictions from original model
        test_data = jnp.array(adata.X[:3])
        original_preds = model.predict(test_data, training=False)

        # Save and load
        from ginseng.model.state import (
            state_from_classifier_trainer,
            classifier_from_state,
        )

        state = state_from_classifier_trainer(trainer)
        save_model(state, tmp_path / "model.h5")

        loaded_state = load_model(tmp_path / "model.h5")
        loaded_model = classifier_from_state(loaded_state)

        # Get predictions from loaded model
        loaded_preds = loaded_model.predict(test_data, training=False)

        # Should be identical
        assert jnp.allclose(original_preds, loaded_preds, atol=1e-6)

    def test_state_gene_order_preserved(self, tmp_path, adata):
        """Test that gene order is correctly preserved."""
        dataset = GinsengDataset.create(tmp_path / "ginseng.zarr", adata, "cell_type")

        # Get original gene order
        original_genes = dataset.gene_names.copy()

        # Train and save
        settings = GinsengClassifierTrainerSettings(batch_size=2, hidden_dim=16)
        trainer = GinsengClassifierTrainer(dataset, settings=settings)
        trainer.fit(epochs=1, silent=True)

        from ginseng.model.state import state_from_classifier_trainer

        state = state_from_classifier_trainer(trainer)
        save_model(state, tmp_path / "model.h5")

        # Load and verify
        loaded_state = load_model(tmp_path / "model.h5")
        assert len(loaded_state.genes) == len(original_genes)
        assert all(g1 == g2 for g1, g2 in zip(loaded_state.genes, original_genes))

    def test_state_label_mapping(self, tmp_path, adata):
        """Test that label keys and values are correctly mapped."""
        dataset = GinsengDataset.create(tmp_path / "ginseng.zarr", adata, "cell_type")

        # Train and save
        settings = GinsengClassifierTrainerSettings(batch_size=2, hidden_dim=16)
        trainer = GinsengClassifierTrainer(dataset, settings=settings)
        trainer.fit(epochs=1, silent=True)

        from ginseng.model.state import state_from_classifier_trainer

        state = state_from_classifier_trainer(trainer)
        save_model(state, tmp_path / "model.h5")

        # Load and verify
        loaded_state = load_model(tmp_path / "model.h5")

        # Check label keys match original
        expected_labels = sorted(adata.obs["cell_type"].unique())
        assert list(loaded_state.label_keys) == expected_labels

        # Check label values are sequential integers
        assert list(loaded_state.label_values) == list(range(len(expected_labels)))

    def test_state_file_extension_handling(self, tmp_path, adata):
        """Test that .h5 extension is added if missing."""
        dataset = GinsengDataset.create(tmp_path / "ginseng.zarr", adata, "cell_type")

        settings = GinsengClassifierTrainerSettings(batch_size=2, hidden_dim=16)
        trainer = GinsengClassifierTrainer(dataset, settings=settings)
        trainer.fit(epochs=1, silent=True)

        from ginseng.model.state import state_from_classifier_trainer

        state = state_from_classifier_trainer(trainer)

        # Save without .h5 extension
        save_model(state, tmp_path / "model")

        # Should create model.h5
        assert (tmp_path / "model.h5").exists()

        # Should be loadable
        loaded = load_model(tmp_path / "model.h5")
        assert loaded.n_genes == state.n_genes

    def test_load_nonexistent_file(self, tmp_path):
        """Test error handling when loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "nonexistent.h5")
