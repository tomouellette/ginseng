# Copyright (c) 2025, Tom Ouellette
# Licensed under the MIT License

import numpy as np

from anndata import AnnData
from scipy import sparse
from tqdm import tqdm


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
