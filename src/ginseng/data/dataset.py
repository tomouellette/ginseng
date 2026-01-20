# Copyright (c) 2026, Tom Ouellette
# Licensed under the MIT License

import shutil
import anndata as ad
import numpy as np
import zarr

from pathlib import Path
from typing import Iterator, Optional, Literal
from tqdm import tqdm

from .io import read_adata


class GinsengDataset:
    """An on-disk dataset for training single-cell classifiers.

    Attributes
    ----------
    path : Path
        Path to the zarr dataset on disk.
    root : zarr.Group
        The root zarr group object.
    n_cells : int
        Total number of cells (observations) in the dataset.
    n_genes : int
        Total number of genes (variables) in the dataset.
    label_names : list of str
        Human-readable names for the integer labels.
    gene_names : list of str
        Names of the genes stored in the dataset.
    labels : np.ndarray
        Integer labels for every cell in the dataset.
    groups : np.ndarray or None
        Categorical group indices (e.g., batch or donor) if provided.
    train_idx : np.ndarray or None
        Indices of cells assigned to the training split.
    test_idx : np.ndarray or None
        Indices of cells assigned to the test split.
    """

    def __init__(self, path: str | Path):
        """Initialize the GinsengDataset by opening an existing zarr store.

        Parameters
        ----------
        path : str | Path
            Path to the zarr v3 dataset directory.
        """
        self.path = Path(path)
        self.root = zarr.open(str(self.path), mode="r")
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load metadata and small arrays into memory."""
        attrs = self.root.attrs

        self.n_cells, self.n_genes = self.root["X"].shape
        self.label_names = attrs.get("label_names", [])
        self.gene_names = attrs.get("gene_names", [])

        self.labels = self.root["labels"][:]
        self.groups = self.root["groups"][:] if "groups" in self.root else None

        self.train_idx = (
            self.root["split/train"][:] if "split/train" in self.root else None
        )

        self.test_idx = (
            self.root["split/test"][:] if "split/test" in self.root else None
        )

    @classmethod
    def create(
        cls,
        path: str | Path,
        adata: str | Path | ad.AnnData,
        label_key: str,
        layer: Optional[str] = None,
        genes: Optional[str] | list[str] | np.ndarray = None,
        group_key: Optional[str] = None,
        chunk_size: int = 4096,
        overwrite: bool = True,
    ) -> "GinsengDataset":
        """Create a GinsengDataset from an AnnData object or file path.

        Parameters
        ----------
        path : str | Path
            Output path where the zarr dataset directory will be created.
        adata : str | Path | ad.AnnData
            Input data. Can be an AnnData object, a local path to a (.h5ad, .h5,
            or 10x directory), or a URL to a supported file format.
        label_key : str
            The column name in `adata.obs` containing the target labels
            (e.g., cell type).
        layer : str, optional
            The key in `adata.layers` to use for expression counts.
            If None, uses `adata.X` (default : None).
        genes : str | list of str | np.ndarray, optional
            Gene selection/filtering logic.
            - If a string: Assumes it is a column in `adata.var` containing a
              boolean mask (e.g., "highly_variable").
            - If a list or array: A specific set of gene names to keep. This
              will also reorder the output to match the provided list.
            - If None: Keeps all genes (default : None).
        group_key : str, optional
            The column name in `adata.obs` containing grouping metadata, such
            as "batch" or "donor" (default : None).
        chunk_size : int
            Number of rows (cells) per zarr chunk for the expression matrix.
            Larger chunks improve compression but require more RAM during
            streaming (default : 4096).
        overwrite : bool
            Whether to delete the existing directory at `path` if it
            exists (default : True).

        Returns
        -------
        GinsengDataset
            An initialized instance of the dataset pointing to the new zarr store.

        Raises
        ------
        FileNotFoundError
            If the provided adata path does not exist.
        KeyError
            If `label_key`, `group_key`, or `layer` are not found in the data.
        ValueError
            If requested `genes` are not found in the input data.
        """
        path = Path(path)

        if path.exists() and overwrite:
            shutil.rmtree(path)

        # Load data
        if not isinstance(adata, ad.AnnData):
            adata_path = Path(adata) if not str(adata).startswith("http") else adata
            adata = read_adata(adata_path, backed=True)

        # Note: We slice the AnnData object to create a View. This is a lazy operation
        # that does not load the counts into memory if the object is backed.
        if genes is not None:
            if isinstance(genes, str):
                if genes not in adata.var.columns:
                    raise KeyError(f"Gene mask '{genes}' not found in adata.var")
                gene_mask = adata.var[genes].values.astype(bool)
                adata = adata[:, gene_mask]
            else:
                # Check for missing genes before slicing
                missing = [g for g in genes if g not in adata.var_names]
                if missing:
                    raise ValueError(
                        f"The following {len(missing)} genes were not found in the "
                        f"input data: {missing[:5]}..."
                    )
                # Reorders columns to match the 'genes' list exactly
                adata = adata[:, genes]

        # Validation
        if label_key not in adata.obs:
            raise KeyError(f"Label key '{label_key}' not found in adata.obs")

        if layer is not None and layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers")

        root = zarr.open_group(str(path), mode="w")
        n_obs, n_vars = adata.shape

        # Write labels
        unique_labels, label_indices = np.unique(
            adata.obs[label_key], return_inverse=True
        )
        root.create_array("labels", data=label_indices.astype(np.int32))
        root.attrs["label_names"] = unique_labels.tolist()

        # Write groups
        if group_key and group_key in adata.obs:
            unique_groups, group_indices = np.unique(
                adata.obs[group_key], return_inverse=True
            )
            root.create_array("groups", data=group_indices.astype(np.int32))
            root.attrs["group_names"] = unique_groups.tolist()

        # Expression matrix
        z_x = root.create_array(
            "X",
            shape=(n_obs, n_vars),
            dtype="f4",
            chunks=(chunk_size, n_vars),
            compressors=[
                zarr.codecs.BloscCodec(
                    cname="lz4",
                    clevel=5,
                    shuffle=zarr.codecs.BloscShuffle.bitshuffle,
                )
            ],
        )

        # Write data in chunks
        for start in tqdm(range(0, n_obs, chunk_size), desc="[ginseng] Writing zarr"):
            end = min(start + chunk_size, n_obs)

            batch = adata.layers[layer][start:end] if layer else adata.X[start:end]

            # Ensure we write dense arrays to zarr
            z_x[start:end] = batch.toarray() if hasattr(batch, "toarray") else batch

        # Metadata
        root.attrs.update(
            {
                "gene_names": adata.var_names.tolist(),
                "n_cells": n_obs,
                "n_genes": n_vars,
                "layer_used": layer if layer else "X",
            }
        )

        return cls(path)

    def make_split(
        self,
        fraction: float = 0.1,
        stratify_group: bool = False,
        seed: int = 123,
    ) -> None:
        """
        Create train/test splits and store indices on disk.

        Parameters
        ----------
        fraction : float
            The proportion of data (or groups) to include in the test split.
            Must be in the range [0.0, 1.0) (default : 0.1).
        stratify_group : bool
            If True, splits by groups (e.g., donors) rather than individual cells.
            Requires group_key to have been provided during creation (default : False).
        seed : int
            Random seed for reproducibility (default : 123).

        Raises
        ------
        ValueError
            If fraction is outside [0.0, 1.0) or if the resulting training set is empty.
        """
        if not 0.0 <= fraction < 1.0:
            raise ValueError("fraction must be in the range [0.0, 1.0)")

        rng = np.random.default_rng(seed)

        if fraction == 0.0:
            self.has_test = False
            train_idx = np.arange(self.n_cells, dtype=np.int32)
            test_idx = np.array([], dtype=np.int32)
        else:
            self.has_test = True
            if stratify_group and self.groups is not None:
                groups = np.unique(self.groups)
                rng.shuffle(groups)

                # Ensure at least one group goes to test,
                # but at least one group stays in train.
                n_test = int(round(len(groups) * fraction))
                n_test = max(1, min(n_test, len(groups) - 1))

                test_groups = groups[:n_test]
                test_mask = np.isin(self.groups, test_groups)
            else:
                idx = np.arange(self.n_cells)
                rng.shuffle(idx)

                # Ensure at least one cell goes to test,
                # but at least one cell stays in train.
                n_test = int(round(self.n_cells * fraction))
                n_test = max(1, min(n_test, self.n_cells - 1))

                test_mask = np.zeros(self.n_cells, dtype=bool)
                test_mask[idx[:n_test]] = True

            train_idx = np.where(~test_mask)[0].astype(np.int32)
            test_idx = np.where(test_mask)[0].astype(np.int32)

        # Note: Final sanity check to ensure training set is not empty
        if train_idx.size == 0:
            raise ValueError(
                f"Split resulted in an empty training set. "
                f"Check if fraction ({fraction}) is too high for the dataset size."
            )

        # Write to zarr
        root = zarr.open_group(str(self.path), mode="r+")
        split_grp = root.require_group("split")
        split_grp.create_array("train", data=train_idx, overwrite=True)
        split_grp.create_array("test", data=test_idx, overwrite=True)

        self._load_metadata()

    def stream(
        self,
        batch_size: int,
        split: Literal["train", "test", "all"] = "train",
        balance_labels: bool = False,
        shuffle: bool = True,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Stream mini-batches of (expression, label) from the zarr store.

        Parameters
        ----------
        batch_size : int
            Number of cells to yield in each batch.
        split : Literal["train", "test", "all"]
            Which data split to stream from (default : "train").
        balance_labels : bool
            If True, downsamples the training split to match the frequency
            of the least common class (default : False).
        shuffle : bool
            Whether to shuffle the indices before streaming (default : True).

        Yields
        ------
        X : np.ndarray
            Expression matrix batch of shape (batch_size, n_genes).
        y : np.ndarray
            Integer labels batch of shape (batch_size,).

        Raises
        ------
        ValueError
            If the requested split has not been initialized via make_split.
        """
        if split == "all":
            idx = np.arange(self.n_cells)
        else:
            split_arr = self.train_idx if split == "train" else self.test_idx
            if split_arr is None:
                raise ValueError("Split not initialized. Call .make_split first.")
            idx = split_arr[:]

        if idx.size == 0:
            return

        if balance_labels and split == "train":
            labels = self.labels[idx]
            unique, counts = np.unique(labels, return_counts=True)
            min_count = counts.min()

            balanced = []
            for u in unique:
                u_idx = idx[labels == u]
                balanced.append(np.random.choice(u_idx, min_count, replace=False))
            idx = np.concatenate(balanced)

        if shuffle:
            np.random.shuffle(idx)

        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start : start + batch_size]
            batch_idx.sort()

            yield (
                self.root["X"][batch_idx, :],
                self.labels[batch_idx],
            )
