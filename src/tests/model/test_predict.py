# Copyright (c) 2026, Tom Ouellette
# Licensed under the MIT License

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import anndata as ad
from unittest.mock import patch

from scipy.sparse import csr_matrix
from ginseng.model.predict import classify
from ginseng.model.state import GinsengClassifierState


@pytest.fixture
def mock_model_state():
    """Create a synthetic model state with 5 genes and 3 classes."""
    return GinsengClassifierState(
        params={},  # Params stay empty because we will mock the function call
        genes=np.array(["gene_A", "gene_B", "gene_C", "gene_D", "gene_E"]),
        label_keys=np.array(["Type_1", "Type_2", "Type_3"]),
        label_values=np.array([0, 1, 2]),
        n_genes=5,
        n_classes=3,
        hidden_dim=128,
        normalize=True,
        target_sum=1e4,
        dropout_rate=0.0,
        training=False,
    )


@pytest.fixture
def test_adata():
    """AnnData with a subset of model genes."""
    rng = np.random.default_rng(42)
    var_names = ["gene_C", "gene_A", "gene_E", "gene_unknown"]
    X = rng.random((10, len(var_names))).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = var_names
    adata.obs_names = [f"cell_{i}" for i in range(10)]
    return adata


class TestClassifyFunction:
    """Test the main classify entry point."""

    @patch("ginseng.model.predict.nn_annotate")
    def test_classify_inplace_updates(self, mock_nn_func, test_adata, mock_model_state):
        # Note: nn_annotate is called per batch. We use a side_effect to handle different
        # batch sizes if necessary, but here we just ensure it returns the correct shape.
        def mock_side_effect(params, key, x, **kwargs):
            batch_size = x.shape[0]
            return jnp.array([[0.0, 10.0, 0.0]] * batch_size)

        mock_nn_func.side_effect = mock_side_effect

        with pytest.warns(UserWarning, match="Partial gene overlap detected: 60.00%."):
            classify(mock_model_state, test_adata, batch_size=4, store_probs=True)

        # Check labels
        assert "ginseng_cell_type" in test_adata.obs.columns
        assert test_adata.obs["ginseng_cell_type"].values[0] == "Type_2"

        # Check confidence
        assert "ginseng_confidence" in test_adata.obs.columns
        assert test_adata.obs["ginseng_confidence"].values[0] > 0.9

        # Check probabilities (since store_probs=True)
        assert "ginseng_probs" in test_adata.obsm
        assert test_adata.obsm["ginseng_probs"].shape == (10, 3)

    @patch("ginseng.model.predict.nn_annotate")
    def test_classify_return_table(self, mock_nn_func, test_adata, mock_model_state):
        mock_nn_func.side_effect = lambda p, k, x, **kw: jnp.array(
            [[10.0, 0.0, 0.0]] * x.shape[0]
        )

        with pytest.warns(UserWarning, match="Partial gene overlap detected: 60.00%."):
            df = classify(mock_model_state, test_adata, return_table=True)

        assert isinstance(df, pd.DataFrame)
        assert df["ginseng_cell_type"].iloc[0] == "Type_1"
        assert "ginseng_confidence" in df.columns


@pytest.fixture
def test_adata_with_layers():
    """AnnData with multiple layers and varying data values."""
    rng = np.random.default_rng(42)
    var_names = ["gene_A", "gene_B", "gene_C", "gene_D", "gene_E"]
    n_cells = 10

    # X contains 1.0s
    X = np.ones((n_cells, len(var_names))).astype(np.float32)

    # 'raw' layer contains 2.0s
    raw = (np.ones((n_cells, len(var_names))) * 2.0).astype(np.float32)

    # 'sparse_layer' as a CSR matrix
    sparse = csr_matrix(np.ones((n_cells, len(var_names))) * 3.0)

    adata = ad.AnnData(X=X)
    adata.layers["raw"] = raw
    adata.layers["sparse_layer"] = sparse
    adata.var_names = var_names
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    return adata


class TestClassifyLayers:
    """Tests for the layer-specific logic in classify."""

    @patch("ginseng.model.predict.nn_annotate")
    def test_classify_from_layer(
        self, mock_nn_func, test_adata_with_layers, mock_model_state
    ):
        """Verify that classify pulls data from the requested layer."""

        # We capture the 'x' batch passed to the neural network to check its values
        captured_batches = []

        def mock_side_effect(params, key, x, **kwargs):
            captured_batches.append(np.array(x))
            return jnp.zeros((x.shape[0], 3))

        mock_nn_func.side_effect = mock_side_effect

        # 1. Run inference using 'raw' layer (values are 2.0)
        classify(mock_model_state, test_adata_with_layers, layer="raw", silent=True)
        assert np.all(captured_batches[0] == 2.0)

        # 2. Run inference using default X (values are 1.0)
        captured_batches.clear()
        classify(mock_model_state, test_adata_with_layers, layer=None, silent=True)
        assert np.all(captured_batches[0] == 1.0)

    @patch("ginseng.model.predict.nn_annotate")
    def test_classify_from_sparse_layer(
        self, mock_nn_func, test_adata_with_layers, mock_model_state
    ):
        """Verify that sparse layers are correctly densified."""

        captured_batches = []
        mock_nn_func.side_effect = lambda p, k, x, **kw: (
            captured_batches.append(np.array(x)) or jnp.zeros((x.shape[0], 3))
        )

        classify(
            mock_model_state, test_adata_with_layers, layer="sparse_layer", silent=True
        )

        # Check that the data reached the model as a dense array of 3.0s
        assert isinstance(captured_batches[0], np.ndarray)
        assert np.all(captured_batches[0] == 3.0)

    def test_classify_invalid_layer_raises(
        self, test_adata_with_layers, mock_model_state
    ):
        """Verify KeyError is raised if the layer doesn't exist."""
        with pytest.raises(KeyError, match="Layer 'non_existent' not found"):
            classify(mock_model_state, test_adata_with_layers, layer="non_existent")


class TestGeneMappingAndReordering:
    """Tests for complex gene alignment scenarios."""

    @patch("ginseng.model.predict.nn_annotate")
    def test_gene_reordering_with_layer(self, mock_nn_func, mock_model_state):
        """Verify that layer data is reordered to match model gene order."""
        # Model expects A, B, C, D, E
        # Adata has them out of order: C, A, E, B, D
        rng = np.random.default_rng(42)
        var_names = ["gene_C", "gene_A", "gene_E", "gene_B", "gene_D"]

        # Create distinct values for each gene: A=1, B=2, C=3, D=4, E=5
        data_values = np.array([[3.0, 1.0, 5.0, 2.0, 4.0]])
        adata = ad.AnnData(X=data_values)
        adata.layers["test_layer"] = data_values.copy()
        adata.var_names = var_names

        captured_x = []
        mock_nn_func.side_effect = lambda p, k, x, **kw: (
            captured_x.append(np.array(x)) or jnp.zeros((x.shape[0], 3))
        )

        classify(mock_model_state, adata, layer="test_layer", silent=True)

        # The model expects [A, B, C, D, E], so captured_x should be [1, 2, 3, 4, 5]
        expected_order = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        np.testing.assert_allclose(captured_x[0], expected_order)
