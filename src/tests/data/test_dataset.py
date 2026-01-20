# Copyright (c) 2025, Tom Ouellette
# Licensed under the MIT License

import numpy as np
import pytest
import anndata as ad
from pathlib import Path

from ginseng.data.dataset import GinsengDataset


@pytest.fixture
def tmp_dataset_path(tmp_path: Path) -> Path:
    """Temporary directory for dataset storage."""
    return tmp_path / "ginseng.zarr"


@pytest.fixture
def small_adata() -> ad.AnnData:
    """Small synthetic AnnData object with layers and gene metadata."""
    rng = np.random.default_rng(42)
    n_cells, n_genes = 100, 50

    X = rng.poisson(lam=5.0, size=(n_cells, n_genes)).astype(np.float32)
    counts = rng.poisson(lam=100.0, size=(n_cells, n_genes)).astype(np.float32)

    adata = ad.AnnData(X=X)
    adata.layers["counts"] = counts

    # Metadata
    adata.obs["cell_type"] = rng.integers(0, 4, size=n_cells).astype(str)
    adata.obs["batch"] = rng.integers(0, 3, size=n_cells).astype(str)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    # Boolean mask for hvgs
    hvg_mask = np.arange(n_genes) % 2 == 0
    adata.var["highly_variable"] = hvg_mask

    return adata


@pytest.fixture
def dataset(tmp_dataset_path, small_adata) -> GinsengDataset:
    """Create a GinsengDataset on disk."""
    return GinsengDataset.create(
        path=tmp_dataset_path,
        adata=small_adata,
        label_key="cell_type",
        group_key="batch",
        chunk_size=32,
    )


class TestDatasetCreation:
    """Test dataset creation and metadata."""

    def test_dataset_exists_on_disk(self, tmp_dataset_path, dataset):
        assert tmp_dataset_path.exists()

    def test_shape_metadata(self, dataset, small_adata):
        assert dataset.n_cells == small_adata.n_obs
        assert dataset.n_genes == small_adata.n_vars

    def test_labels_loaded(self, dataset):
        assert dataset.labels.shape == (dataset.n_cells,)
        assert dataset.labels.dtype == np.int32

    def test_groups_optional(self, dataset):
        assert dataset.groups is not None
        assert dataset.groups.shape == (dataset.n_cells,)

    def test_gene_names(self, dataset, small_adata):
        assert dataset.gene_names == list(small_adata.var_names)

    def test_label_names(self, dataset):
        assert len(dataset.label_names) > 0


class TestDatasetSplits:
    """Test train/test splitting logic."""

    def test_make_split_creates_indices(self, dataset):
        dataset.make_split(fraction=0.2)
        assert dataset.train_idx is not None
        assert dataset.test_idx is not None

    def test_split_sizes_approximately_correct(self, dataset):
        dataset.make_split(fraction=0.2)
        n_test = len(dataset.test_idx)
        n_train = len(dataset.train_idx)

        assert n_test + n_train == dataset.n_cells
        assert abs(n_test - 0.2 * dataset.n_cells) <= 1

    def test_fraction_zero_creates_empty_test(self, dataset):
        dataset.make_split(fraction=0.0)
        assert len(dataset.test_idx) == 0
        assert len(dataset.train_idx) == dataset.n_cells
        assert not dataset.has_test

    def test_fraction_one_raises_error(self, dataset):
        """Verify that fraction=1.0 is strictly forbidden."""
        with pytest.raises(ValueError):
            dataset.make_split(fraction=1.0)

    def test_reproducibility_with_seed(self, dataset):
        dataset.make_split(fraction=0.3, seed=123)
        train1 = dataset.train_idx[:]
        test1 = dataset.test_idx[:]

        dataset.make_split(fraction=0.3, seed=123)
        train2 = dataset.train_idx[:]
        test2 = dataset.test_idx[:]

        assert np.array_equal(train1, train2)
        assert np.array_equal(test1, test2)

    def test_stratify_by_group(self, dataset):
        dataset.make_split(fraction=0.5, stratify_group=True)

        train_groups = set(dataset.groups[dataset.train_idx])
        test_groups = set(dataset.groups[dataset.test_idx])

        # No group overlap allowed
        assert train_groups.isdisjoint(test_groups)


class TestDatasetStreaming:
    """Test streaming batches from dataset."""

    def test_stream_train_batches(self, dataset):
        dataset.make_split(fraction=0.2)

        batches = list(dataset.stream(batch_size=16, split="train"))
        assert len(batches) > 0

        for X, y in batches:
            assert X.shape[0] == y.shape[0]
            assert X.shape[1] == dataset.n_genes

    def test_stream_test_batches(self, dataset):
        dataset.make_split(fraction=0.2)

        batches = list(dataset.stream(batch_size=16, split="test"))
        for X, y in batches:
            assert X.shape[0] == y.shape[0]

    def test_stream_all(self, dataset):
        batches = list(dataset.stream(batch_size=25, split="all"))
        total = sum(X.shape[0] for X, _ in batches)
        assert total == dataset.n_cells

    def test_empty_test_stream(self, dataset):
        dataset.make_split(fraction=0.0)
        batches = list(dataset.stream(batch_size=16, split="test"))
        assert batches == []

    def test_shuffle_changes_order(self, dataset):
        dataset.make_split(fraction=0.0)

        # Collect target labels from stream to verify order
        y1 = np.concatenate([y for _, y in dataset.stream(batch_size=10, shuffle=True)])
        y2 = np.concatenate([y for _, y in dataset.stream(batch_size=10, shuffle=True)])

        assert not np.array_equal(y1, y2)


class TestLabelBalancing:
    """Test label balancing during training."""

    def test_balance_labels_equalizes_counts(self, dataset):
        dataset.make_split(fraction=0.0)

        idx = dataset.train_idx[:]
        labels = dataset.labels[idx]
        _, counts_before = np.unique(labels, return_counts=True)

        batches = list(
            dataset.stream(
                batch_size=dataset.n_cells,
                split="train",
                balance_labels=True,
                shuffle=False,
            )
        )

        _, y_bal = batches[0]
        _, counts_after = np.unique(y_bal, return_counts=True)

        assert counts_after.min() == counts_after.max()
        assert counts_after.max() <= counts_before.min()

    def test_balance_only_applies_to_train(self, dataset):
        dataset.make_split(fraction=0.3)

        batches = list(
            dataset.stream(
                batch_size=32,
                split="test",
                balance_labels=True,
            )
        )

        # Should not crash and should not rebalance test set
        for _, y in batches:
            assert len(y) > 0


class TestDatasetLayersAndGenes:
    """Tests for layers and gene selection functionality."""

    def test_create_from_layer(self, tmp_dataset_path, small_adata):
        """Verify data is pulled from a specific layer instead of X."""
        ds = GinsengDataset.create(
            path=tmp_dataset_path,
            adata=small_adata,
            label_key="cell_type",
            layer="counts",
        )

        # Load data via stream to check values
        ds.make_split(fraction=0.0)
        X_batch, _ = next(ds.stream(batch_size=100))

        # Should match small_adata.layers["counts"], not small_adata.X
        assert np.allclose(X_batch, small_adata.layers["counts"])
        assert not np.allclose(X_batch, small_adata.X)

    def test_create_with_gene_mask_string(self, tmp_dataset_path, small_adata):
        """Test gene filtering using a string key for a boolean column in var."""
        ds = GinsengDataset.create(
            path=tmp_dataset_path,
            adata=small_adata,
            label_key="cell_type",
            genes="highly_variable",
        )

        expected_n_genes = small_adata.var["highly_variable"].sum()
        assert ds.n_genes == expected_n_genes
        assert ds.root.attrs["gene_names"] == list(
            small_adata.var_names[small_adata.var["highly_variable"]]
        )

    def test_create_with_gene_list(self, tmp_dataset_path, small_adata):
        """Test gene filtering and REORDERING using a list of gene names."""
        subset_genes = ["gene_10", "gene_2", "gene_45"]

        ds = GinsengDataset.create(
            path=tmp_dataset_path,
            adata=small_adata,
            label_key="cell_type",
            genes=subset_genes,
        )

        assert ds.n_genes == 3
        assert ds.gene_names == subset_genes

        ds.make_split(fraction=0.0)

        X_batch, _ = next(ds.stream(batch_size=1, shuffle=False))

        expected = small_adata[:, subset_genes].X[0:1]
        if hasattr(expected, "toarray"):
            expected = expected.toarray()
        else:
            expected = np.array(expected)

        np.testing.assert_allclose(X_batch, expected, rtol=1e-5)

    def test_invalid_layer_raises_error(self, tmp_dataset_path, small_adata):
        """Verify KeyError when requested layer doesn't exist."""
        with pytest.raises(KeyError, match="Layer 'missing_layer' not found"):
            GinsengDataset.create(
                path=tmp_dataset_path,
                adata=small_adata,
                label_key="cell_type",
                layer="missing_layer",
            )

    def test_invalid_gene_mask_raises_error(self, tmp_dataset_path, small_adata):
        """Verify KeyError when var column doesn't exist."""
        with pytest.raises(KeyError, match="Gene mask 'not_here' not found"):
            GinsengDataset.create(
                path=tmp_dataset_path,
                adata=small_adata,
                label_key="cell_type",
                genes="not_here",
            )

    def test_missing_gene_in_list_raises_error(self, tmp_dataset_path, small_adata):
        """Verify ValueError when a gene in the provided list is missing from the data."""
        with pytest.raises(ValueError, match="genes were not found"):
            GinsengDataset.create(
                path=tmp_dataset_path,
                adata=small_adata,
                label_key="cell_type",
                genes=["gene_0", "non_existent_gene"],
            )

    def test_metadata_layer_record(self, tmp_dataset_path, small_adata):
        """Ensure the layer name used is recorded in Zarr attributes."""
        ds = GinsengDataset.create(
            path=tmp_dataset_path,
            adata=small_adata,
            label_key="cell_type",
            layer="counts",
        )
        assert ds.root.attrs["layer_used"] == "counts"

        # Default should be X
        ds_default = GinsengDataset.create(
            path=tmp_dataset_path / "default", adata=small_adata, label_key="cell_type"
        )
        assert ds_default.root.attrs["layer_used"] == "X"


class TestDatasetEdgeCases:
    """Test edge cases and error handling."""

    def test_stream_without_split_raises(self, dataset):
        with pytest.raises(ValueError):
            next(dataset.stream(batch_size=16, split="train"))

    def test_invalid_fraction_raises(self, dataset):
        with pytest.raises(ValueError):
            dataset.make_split(fraction=-0.1)

        with pytest.raises(ValueError):
            dataset.make_split(fraction=1.1)

    def test_single_cell_batch(self, dataset):
        dataset.make_split(fraction=0.0)
        batches = list(dataset.stream(batch_size=1))

        for X, y in batches:
            assert X.shape == (1, dataset.n_genes)
            assert y.shape == (1,)

    def test_large_batch(self, dataset):
        dataset.make_split(fraction=0.0)
        batches = list(dataset.stream(batch_size=1000))
        assert len(batches) == 1
        X, y = batches[0]
        assert X.shape[0] == dataset.n_cells
        assert y.shape[0] == dataset.n_cells

    def test_tiny_dataset_train_safety(self, tmp_dataset_path):
        """Ensure that splitting a very small dataset still leaves training data."""
        rng = np.random.default_rng(42)
        # Create a dataset with only 2 cells
        adata = ad.AnnData(X=rng.random((2, 10)).astype(np.float32))
        adata.obs["label"] = [0, 1]

        ds = GinsengDataset.create(tmp_dataset_path, adata, label_key="label")

        # Even with a high fraction, it should force at least 1 cell into train
        ds.make_split(fraction=0.99)
        assert len(ds.train_idx) == 1
        assert len(ds.test_idx) == 1
