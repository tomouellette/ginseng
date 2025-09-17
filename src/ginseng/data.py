# Copyright (c) 2025, Tom Ouellette
# Licensed under the MIT License

import anndata
import numpy as np
import shutil
import zarr

from anndata import AnnData
from pathlib import Path
from scipy import sparse
from tqdm import tqdm
from typing import Iterator
from zarr.storage import LocalStore


class GinsengDataset:
    """A zarr-based dataset for efficient cell-level access.

    Attributes
    ----------
    path : Path
        Path to ginseng dataset zarr file.
    store : LocalStore
        Initialized local zarr store.
    root : zarr.group
        Zarr group containing structured ginseng dataset.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.store = LocalStore(str(self.path))
        self.root = zarr.open_group(store=self.store, mode="r")
        self._load_metadata()

    @classmethod
    def _create_from_numpy(
        cls,
        path: str | Path,
        X: np.ndarray,
        labels: np.ndarray,
        chunk_size: tuple[int, int] | None = None,
        groups: np.ndarray | None = None,
        genes: np.ndarray = None,
        column_mask: np.ndarray | None = None,
    ):
        """Create cached dataset from a dense numpy array.

        Parameters
        ----------
        path : str | Path
            Path to store the new dataset.
        X : np.ndarray
            Dense count matrix.
        labels : np.ndarray
            Array of labels for each barcode.
        chunk_size : tuple[int, int]
            Chunk size for storing counts.
        groups : np.ndarray
            Group-level information used to stratify holdout data. If provided, no unique
            group will have data in both training and holdout when using `make_holdout`.
        genes : np.ndarray
            Gene names arranged in order used for training.
        column_mask : np.ndarray
            Boolean mask specifying which columns to include.

        Returns
        -------
        GinsengDataset
            Instance of the created dataset.
        """
        path = Path(path)
        if path.exists():
            shutil.rmtree(path)

        store = LocalStore(path)
        root = zarr.open_group(store=store, mode="w")

        if column_mask is not None:
            X = X[:, column_mask]

        zarr.create_array(
            store=store,
            name="X",
            chunks=chunk_size,
            data=X.astype(np.float32),
        )

        unique_labels = np.unique(labels)
        label_map = {k: v for k, v in zip(unique_labels, range(len(unique_labels)))}
        integer_labels = np.array([label_map[k] for k in labels])

        zarr.create_array(
            store=store,
            name="labels",
            data=integer_labels.astype(np.int32),
        )

        if groups is not None:
            unique_groups = np.unique(groups)
            group_map = {k: v for v, k in enumerate(unique_groups)}
            integer_groups = np.array([group_map[g] for g in groups])

            zarr.create_array(
                store=store,
                name="groups",
                data=integer_groups.astype(np.int32),
            )
            root.attrs["group_keys"] = [str(k) for k in group_map.keys()]
            root.attrs["group_values"] = [int(v) for v in group_map.values()]
            root.attrs["n_groups"] = len(unique_groups)
        else:
            root.attrs["group_keys"] = []
            root.attrs["group_values"] = []
            root.attrs["n_groups"] = 0

        zarr.create_group(store=store, path="label_indices")
        for integer_label in np.unique(integer_labels):
            indices = np.where(integer_labels == integer_label)[0].astype(np.int32)
            zarr.create_array(
                store=store,
                name=f"label_indices/{integer_label}",
                chunks=(min(len(indices), 10000),),
                data=indices,
            )

        root.attrs.update(
            {
                "n_cells": X.shape[0],
                "n_genes": X.shape[1],
                "n_labels": len(unique_labels),
                "dtype": str(X.dtype),
                "label_keys": [
                    str(k) if isinstance(k, str) else int(k) for k in label_map.keys()
                ],
                "label_values": [int(v) for v in label_map.values()],
                "genes": [str(i) for i in genes],
            }
        )

        return cls(path)

    @classmethod
    def _create_from_sparse(
        cls,
        path: str | Path,
        X: sparse.spmatrix,
        labels: np.ndarray,
        chunk_size: tuple[int, int] | None = None,
        groups: np.ndarray | None = None,
        genes: np.ndarray = None,
        column_mask: np.ndarray | None = None,
    ):
        """Create cached dataset from a sparse count matrix.

        Parameters
        ----------
        path : str | Path
            Path to store the new dataset.
        X : scipy.sparse.spmatrix
            Sparse count matrix.
        labels : np.ndarray
            Array of labels for each barcode.
        chunk_size : tuple[int, int]
            Chunk size for storing counts.
        groups : np.ndarray
            Group-level information used to stratify holdout data. If provided, no unique
            group will have data in both training and holdout when using `make_holdout`.
        genes : np.ndarray
            Gene names arranged in order used for training.
        column_mask : np.ndarray
            Boolean mask specifying which columns to include.

        Returns
        -------
        GinsengDataset
            Instance of the created dataset.
        """
        path = Path(path)
        if path.exists():
            shutil.rmtree(path)

        store = LocalStore(path)
        root = zarr.open_group(store=store, mode="w")

        n_genes = X.shape[1]
        if column_mask is not None:
            X = X[:, column_mask]
            n_genes = int(column_mask.sum())

        zX = zarr.create_array(
            store=store,
            name="X",
            shape=(X.shape[0], n_genes),
            dtype=np.float32,
            chunks=chunk_size,
        )

        block_size = chunk_size[0]
        for start in tqdm(
            range(0, X.shape[0], block_size), desc="[ginseng] Constructing dataset"
        ):
            end = min(start + block_size, X.shape[0])
            zX[start:end, :] = X[start:end].toarray().astype(np.float32)

        unique_labels = np.unique(labels)
        label_map = {k: v for k, v in zip(unique_labels, range(len(unique_labels)))}
        integer_labels = np.array([label_map[k] for k in labels])

        zarr.create_array(
            store=store,
            name="labels",
            data=integer_labels.astype(np.int32),
        )

        if groups is not None:
            unique_groups = np.unique(groups)
            group_map = {k: v for v, k in enumerate(unique_groups)}
            integer_groups = np.array([group_map[g] for g in groups])

            zarr.create_array(
                store=store,
                name="groups",
                data=integer_groups.astype(np.int32),
            )
            root.attrs["group_keys"] = [str(k) for k in group_map.keys()]
            root.attrs["group_values"] = [int(v) for v in group_map.values()]
            root.attrs["n_groups"] = len(unique_groups)
        else:
            root.attrs["group_keys"] = []
            root.attrs["group_values"] = []
            root.attrs["n_groups"] = 0

        zarr.create_group(store=store, path="label_indices")
        for integer_label in np.unique(integer_labels):
            indices = np.where(integer_labels == integer_label)[0].astype(np.int32)
            zarr.create_array(
                store=store,
                name=f"label_indices/{integer_label}",
                chunks=(min(len(indices), 10000),),
                data=indices,
            )

        root.attrs.update(
            {
                "n_cells": X.shape[0],
                "n_genes": X.shape[1],
                "n_labels": len(unique_labels),
                "dtype": str(X.dtype),
                "label_keys": [
                    str(k) if isinstance(k, str) else int(k) for k in label_map.keys()
                ],
                "label_values": [int(v) for v in label_map.values()],
                "genes": [str(i) for i in genes],
            }
        )

        return cls(path)

    @classmethod
    def create(
        cls,
        path: str | Path,
        adata: AnnData,
        labels: np.ndarray | str,
        chunk_size: int | tuple[int, int] = None,
        groups: np.ndarray | str | None = None,
        gene_key: str | None = None,
        gene_mask: np.ndarray | None = None,
    ):
        """Autodetect array type and create a dataset.

        Parameters
        ----------
        path : str | Path
            Path to store the new dataset.
        adata : AnnData
            Annotated data storing count matrix.
        labels : np.ndarray | str
            Array of labels for each barcode or key in AnnData.obs specifying labels.
        chunk_size : int | tuple[int, int]
            Chunk size for storing counts. An integer specifies row-wise chunk size and
            a tuple of integers specifies row and column-wise chunk size. When an integer
            or no chunk size is provided, the column-wise chunk size is set to number of
            the number of included genes.
        groups : np.ndarray
            Array of group-level information for each barcode or key in AnnData.obs that
            specifies groups. Group-level informatiion is used to stratify holdout data.
        gene_key : str
            If provided, the column where gene names are stored. By default, the index of
            `adata.var` is assumed to store the gene names.
        gene_mask : np.ndarray
            Boolean mask specifying which genes to include.

        Returns
        -------
        GinsengDataset
            Instance of the created  dataset.
        """
        if isinstance(labels, str):
            if labels not in adata.obs.columns:
                raise ValueError("'{labels} not found in `adata.obs`.")

            labels = adata.obs[labels].to_numpy()

        if isinstance(groups, str):
            if groups not in adata.obs.columns:
                raise ValueError("'{groups} not found in `adata.obs`.")

            groups = adata.obs[groups].to_numpy()

        if isinstance(gene_mask, str):
            if gene_mask not in adata.var.columns:
                raise ValueError("'{gene_mask} not found in `adata.var`.")

            gene_mask = adata.var[gene_mask].to_numpy()

        if len(labels) != adata.obs.shape[0]:
            raise ValueError("`labels` must be same length as `adata.obs`")

        if gene_mask is not None:
            if len(gene_mask) != adata.var.shape[0]:
                raise ValueError("`column_mask` must be same length as `adata.var`")

        if groups is not None:
            if len(groups) != adata.obs.shape[0]:
                raise ValueError("`groups` must be same length as `adata.obs`")

        if gene_key is None:
            genes = adata.var.index.to_numpy()
        else:
            genes = adata.var.reset_index()[gene_key]

        if isinstance(gene_mask, np.ndarray):
            genes = genes[gene_mask]

        if chunk_size is None:
            chunk_size = (256, len(genes))

        if sparse.issparse(adata.X) or isinstance(
            adata.X, anndata._core.sparse_dataset._CSRDataset
        ):
            return cls._create_from_sparse(
                path, adata.X, labels, chunk_size, groups, genes, gene_mask
            )
        elif isinstance(adata.X, np.ndarray):
            return cls._create_from_numpy(
                path, adata.X, labels, chunk_size, groups, genes, gene_mask
            )
        else:
            raise ValueError(
                "X must be a np.ndarray, scipy.sparse matrix, or backed anndata CSRDataset."
            )

    def _reopen_readonly(self):
        """Helper to reopen store in read-only mode"""
        self.store = LocalStore(self.path)
        self.root = zarr.open_group(store=self.store, mode="r")

    def _load_metadata(self):
        """Load metadata from zarr store."""
        self.n_cells = self.root.attrs["n_cells"]
        self.n_genes = self.root.attrs["n_genes"]
        self.n_labels = self.root.attrs["n_labels"]
        self.n_groups = self.root.attrs["n_groups"]

        self.labels = self.root["labels"][:]

        if "groups" in self.root:
            self.groups = self.root["groups"][:]

        self.label_indices = {}
        if "label_indices" in self.root:
            for key in self.root["label_indices"].keys():
                self.label_indices[int(key)] = self.root["label_indices"][key][:]

        self.unique_labels = list(self.label_indices.keys())

        self.train_indices = (
            self.root["train_indices"][:] if "train_indices" in self.root else None
        )
        self.holdout_indices = (
            self.root["holdout_indices"][:] if "holdout_indices" in self.root else None
        )

    @property
    def has_holdout(self) -> bool:
        """Whether the dataset has a train/holdout split."""
        return self.train_indices is not None and self.holdout_indices is not None

    def make_holdout(
        self,
        holdout_fraction: float = 0.1,
        rng: None | np.random.Generator = None,
        group_level: bool = False,
        group_mode: str = "fraction",
    ):
        """Create and persist a train/holdout split.

        Parameters
        ----------
        holdout_fraction : float
            Fraction of samples to allocate to holdout. The total number of samples held out
            per label is determined by min(N labels per label) * `holdout_fraction`. When
            group_level is True, then holdout_fraction specifies the fraction of unique
            groups to hold out.
        rng : np.random.Generator, optional
            Random number generator for reproducibility.
        group_level : bool
            If True, split data based on group-level information.
        group_mode : str
            If 'fraction' then N groups x `holdout_fraction` groups will be held out.
            If 'loo', then a single group will be held out (leave one out).
        """
        if rng is None:
            rng = np.random.default_rng()

        if group_mode not in ["fraction", "loo"]:
            raise ValueError("`group_mode` must be 'fraction' or 'loo'.")

        store = LocalStore(self.path)
        root = zarr.open_group(store=self.store, mode="r+")

        if group_level and "groups" in root:
            groups = root["groups"][:]
            unique_groups = np.unique(groups)

            if group_mode == "loo":
                n_holdout_groups = 1
            else:
                n_holdout_groups = max(1, int(len(unique_groups) * holdout_fraction))

            permuted_groups = rng.permutation(unique_groups)

            if n_holdout_groups == 0:
                raise ValueError(
                    "No holdout groups were sampled."
                    f"Please increase `holdout_fraction` ({holdout_fraction})"
                )

            holdout_groups = set(permuted_groups[:n_holdout_groups])
            holdout_indices = np.where(np.isin(groups, list(holdout_groups)))[0]
            train_indices = np.where(~np.isin(groups, list(holdout_groups)))[0]
        else:
            train_indices = []
            holdout_indices = []

            n_holdout = np.inf
            for label, indices in self.label_indices.items():
                n_holdout = min(n_holdout, int(len(indices) * holdout_fraction))

            if n_holdout == 0:
                store.close()
                raise ValueError(
                    "The number of holdout samples per label was set to zero. "
                    f"This means `holdout_fraction` is too small ({holdout_fraction}). "
                    "Please re-run with a larger `holdout_fraction`."
                )

            for label, indices in self.label_indices.items():
                n_holdout = max(1, int(len(indices) * holdout_fraction))
                permuted = rng.permutation(indices)
                holdout_indices.append(permuted[:n_holdout])
                train_indices.append(permuted[n_holdout:])

            train_indices = np.concatenate(train_indices).astype(np.int32)
            holdout_indices = np.concatenate(holdout_indices).astype(np.int32)

        for arr_name in ["train_indices", "holdout_indices"]:
            if arr_name in root:
                del root[arr_name]

        root.create_array("train_indices", data=train_indices.astype(np.int32))
        root.create_array("holdout_indices", data=holdout_indices.astype(np.int32))

        self.train_indices = train_indices
        self.holdout_indices = holdout_indices

        store.close()
        self._reopen_readonly()

    def get_batch(self, indices: np.ndarray) -> np.ndarray:
        """Retrieve a batch of count data for given cell indices.

        Parameters
        ----------
        indices : np.ndarray
            Array of cell indices to retrieve.

        Returns
        -------
        np.ndarray
            Expression matrix subset of shape (len(indices), n_features).
        """
        return self.root["X"][indices, :]

    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
        rng: None | np.random.Generator = None,
        split: str = "all",
        balance_train: bool = True,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Iterate through the dataset in batches.

        Parameters
        ----------
        batch_size : int
            Number of samples per batch.
        shuffle : bool
            Whether to shuffle the dataset before batching.
        rng : np.random.Generator
            Random number generator used if shuffling.
        split : str
            Subset of data to iterate over ("all", "train", or "holdout").
        balance_train : bool
            If True, then force training set to have an equal number of each label.

        Yields
        ------
        tuple[np.ndarray, np.ndarray]
            Batches of expression data and labels.
        """
        if rng is None:
            rng = np.random.default_rng()

        if split == "all":
            indices = np.arange(self.n_cells)
        elif split == "train":
            if not self.has_holdout:
                raise ValueError("No holdout split available.")
            indices = self.train_indices

            if balance_train:
                labels = self.labels[indices]
                unique_labels = np.unique(labels)

                min_count = min(np.sum(labels == lbl) for lbl in unique_labels)

                balanced_indices = []
                for lbl in unique_labels:
                    lbl_indices = indices[labels == lbl]
                    sampled = rng.choice(lbl_indices, size=min_count, replace=False)
                    balanced_indices.append(sampled)

                indices = np.concatenate(balanced_indices)
                rng.shuffle(indices)

        elif split == "holdout":
            if not self.has_holdout:
                raise ValueError("No holdout split available.")
            indices = self.holdout_indices
        else:
            raise ValueError('split must be one of {"all", "train", "holdout"}')

        if shuffle and rng is not None:
            indices = rng.permutation(indices)

        n_batches = (len(indices) + batch_size - 1) // batch_size

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]

            X_batch = self.get_batch(batch_indices)
            y_batch = self.labels[batch_indices]

            yield X_batch, y_batch

    def __len__(self, split: str = "all"):
        """Return the number of samples in the dataset.

        Parameters
        ----------
        split : str
            Subset of data to count ("all", "train", or "holdout").

        Returns
        -------
        int
            Number of samples in the chosen subset.
        """
        if split == "all":
            return self.n_cells
        elif split == "train":
            if not self.has_holdout:
                raise ValueError("No holdout split available.")
            return len(self.train_indices)
        elif split == "holdout":
            if not self.has_holdout:
                raise ValueError("No holdout split available.")
            return len(self.holdout_indices)
        else:
            raise ValueError('split must be one of {"all", "train", "holdout"}')

    def __getitem__(self, idx, split: str = "all"):
        """Retrieve a single sample by index.

        Parameters
        ----------
        idx : int
            Position within the chosen split.
        split : str
            Subset of data to retrieve from ("all", "train", or "holdout").

        Returns
        -------
        tuple of (np.ndarray, int)
            Expression vector and corresponding label.
        """
        if split == "all":
            true_idx = idx
        elif split == "train":
            if not self.has_holdout:
                raise ValueError("No holdout split available.")
            true_idx = self.train_indices[idx]
        elif split == "holdout":
            if not self.has_holdout:
                raise ValueError("No holdout split available.")
            true_idx = self.holdout_indices[idx]
        else:
            raise ValueError('split must be one of {"all", "train", "holdout"}')

        return self.root["X"][true_idx, :], self.labels[true_idx]
