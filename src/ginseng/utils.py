# Copyright (c) 2025, Tom Ouellette
# Licensed under the MIT License

import jax.numpy as jnp
import numpy as np

from anndata import AnnData
from scipy import sparse
from tqdm import tqdm


def iter_sequential(
    adata: AnnData,
    label_key: str,
    batch_size: int = 256,
    gene_order: None | list | np.ndarray = None,
    gene_mask: None | np.ndarray = None,
) -> tuple[jnp.ndarray, jnp.ndarray, np.ndarray]:
    """Iterate sequentially through an AnnData for prediction.

    Parameters
    ----------
    adata : AnnData
        AnnData object with `obs[label_key]`.
    label_key : str
        Column in AnnData `obs` with integer or string cell type labels.
    batch_size : int
        Number of cells per batch.
    gene_order : None | list | np.ndarray
        Columns ordered by name (e.g as in training). Missing values filled with zero.
    gene_mask : np.ndarray
        Only include columns set to True in the mask. Cannot be set if `gene_order` is provided.
    """
    n_cells = adata.n_obs

    if gene_order is not None and gene_mask is not None:
        raise ValueError("Only one of gene_order or gene_mask can be provided")

    if gene_order is not None:
        gene_idx_map = np.array(
            [
                adata.var_names.get_loc(g) if g in adata.var_names else -1
                for g in gene_order
            ],
            dtype=int,
        )
        column_indices = gene_idx_map[gene_idx_map >= 0]
    elif gene_mask is not None:
        column_indices = np.flatnonzero(gene_mask)
    else:
        column_indices = np.arange(adata.n_vars)

    labels = adata.obs[label_key].to_numpy()
    n_batches = (n_cells + batch_size - 1) // batch_size

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n_cells)
        batch_idx = np.arange(start, end)

        X_batch = adata.X[batch_idx, :][:, column_indices]
        if sparse.issparse(X_batch):
            X_batch = X_batch.toarray()

        X_batch = jnp.asarray(X_batch)
        y_batch = jnp.asarray(labels[batch_idx])

        yield X_batch, y_batch, batch_idx


def compute_hvgs(
    adata: AnnData,
    n_top_genes: int = 5000,
    min_mean: float = 0.025,
    max_mean: float = 0.995,
    n_bins: int = 20,
    chunk_size: int = 5000,
    silent: bool = False,
    copy: bool = False,
) -> None | AnnData:
    """Select highly variable genes.

    Parameters
    ----------
    data : AnnData
        AnnData object with gene names stored in `var`.
    n_top_genes : int
        Number of top highly variable genes to select.
    min_mean : float
        Lower bound on mean gene expression for selecting highly variable genes.
    max_mean : float
        Upper bound on mean gene expression for selecting highly variable genes.
    n_bins : int
        Select genes across this many gene expression bins.
    copy : bool
        If True, return a copy of the AnnData.

    Returns
    -------
    None
        Marks highly variable genes in `var` in-place. Or if copy=True, then returns a copy of the AnnData.
    """
    if copy:
        adata = adata.copy()

    n_cells, n_genes = adata.shape

    # Accumulators for gene sums and sums of squares
    gene_sums = np.zeros(n_genes, dtype=np.float64)
    gene_sumsq = np.zeros(n_genes, dtype=np.float64)

    iterator = range(0, n_cells, chunk_size)
    if not silent:
        iterator = tqdm(iterator, desc="[ginseng] Selecting genes", unit=" chunks")

    for start in iterator:
        end = min(start + chunk_size, n_cells)
        X_chunk = adata.X[start:end]
        if sparse.issparse(X_chunk):
            X_chunk = X_chunk.toarray()
        X_chunk = np.asarray(X_chunk)

        gene_sums += X_chunk.sum(axis=0)
        gene_sumsq += np.sum(X_chunk**2, axis=0)

    # Gene means and variances
    gene_means = gene_sums / n_cells
    gene_vars = (gene_sumsq - n_cells * gene_means**2) / (n_cells - 1)
    dispersions = gene_vars / np.maximum(gene_means, 1e-12)

    # Filter by mean expression
    min_mean = np.quantile(gene_means, min_mean)
    max_mean = np.quantile(gene_means, max_mean)
    mask_mean = (gene_means >= min_mean) & (gene_means <= max_mean)
    filtered_disp = np.full_like(dispersions, -np.inf)
    filtered_disp[mask_mean] = dispersions[mask_mean]

    # Normalize dispersions by mean bins
    bin_edges = np.linspace(
        np.min(gene_means[mask_mean]), np.max(gene_means[mask_mean]), n_bins + 1
    )
    norm_disp = np.zeros_like(filtered_disp)
    for i in range(n_bins):
        bin_mask = (gene_means >= bin_edges[i]) & (gene_means < bin_edges[i + 1])
        if np.any(bin_mask):
            median_disp = np.median(filtered_disp[bin_mask])
            if median_disp > 0:
                norm_disp[bin_mask] = filtered_disp[bin_mask] / median_disp

    # Select top HVGs
    top_idx = np.argsort(norm_disp)[-n_top_genes:]
    hvg_mask = np.zeros(n_genes, dtype=bool)
    hvg_mask[top_idx] = True
    adata.var.loc[:, ["ginseng_genes"]] = hvg_mask

    return adata if copy else None
