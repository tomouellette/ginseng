# Copyright (c) 2026, Tom Ouellette
# Licensed under the MIT License

import numpy as np
from anndata import AnnData
from scipy import sparse
from tqdm import tqdm
from typing import Optional


def select_hvgs(
    adata: AnnData,
    n_top_genes: int = 5000,
    layer: Optional[str] = None,
    target_sum: Optional[float] = 1e4,
    min_mean: float = 0.025,
    max_mean: float = 0.995,
    n_bins: int = 20,
    chunk_size: int = 5000,
    silent: bool = False,
    copy: bool = False,
) -> Optional[AnnData]:
    """Select highly variable genes from raw or normalized counts.

    Parameters
    ----------
    adata : AnnData
        AnnData object with gene names stored in `var`.
    n_top_genes : int
        Number of top highly variable genes to select.
    layer : str, optional
        Key in `adata.layers` to use. If None, uses `adata.X`.
    target_sum : float, optional
        If provided, scales each cell to this sum and applies log1p
        transformation (default: 1e4).
    min_mean : float
        Lower quantile bound on mean gene expression.
    max_mean : float
        Upper quantile bound on mean gene expression.
    n_bins : int
        Select genes across this many gene expression bins.
    chunk_size : int
        Number of cells to process in memory at once.
    silent : bool
        If True, suppresses progress bar.
    copy : bool
        If True, returns a copy of the AnnData.

    Returns
    -------
    Optional[AnnData]
        Marks highly variable genes in `var['ginseng_genes']`.
    """
    if copy:
        adata = adata.copy()

    n_cells, n_genes = adata.shape
    gene_sums = np.zeros(n_genes, dtype=np.float64)
    gene_sumsq = np.zeros(n_genes, dtype=np.float64)

    # Choose data source
    data_source = adata.layers[layer] if layer is not None else adata.X

    iterator = range(0, n_cells, chunk_size)
    if not silent:
        iterator = tqdm(iterator, desc="[ginseng] Selecting genes", unit=" chunks")

    for start in iterator:
        end = min(start + chunk_size, n_cells)
        X_chunk = data_source[start:end]

        if sparse.issparse(X_chunk):
            X_chunk = X_chunk.toarray()
        X_chunk = np.asarray(X_chunk, dtype=np.float32)

        # Apply on-the-fly normalization if target_sum is requested
        if target_sum is not None:
            counts_per_cell = np.sum(X_chunk, axis=1, keepdims=True)
            # Avoid division by zero for empty cells
            counts_per_cell = np.where(counts_per_cell == 0, 1, counts_per_cell)
            X_chunk = X_chunk * (target_sum / counts_per_cell)
            X_chunk = np.log1p(X_chunk)

        gene_sums += X_chunk.sum(axis=0)
        gene_sumsq += np.sum(X_chunk**2, axis=0)

    # Calculate Mean and Variance
    # Variance formula: E[X^2] - (E[X])^2
    gene_means = gene_sums / n_cells
    gene_vars = (gene_sumsq - n_cells * gene_means**2) / (n_cells - 1)

    # Standard dispersion using variance-to-mean ratio
    dispersions = gene_vars / np.maximum(gene_means, 1e-12)

    # Filter by mean expression quantiles
    actual_min_mean = np.quantile(gene_means, min_mean)
    actual_max_mean = np.quantile(gene_means, max_mean)
    mask_mean = (gene_means >= actual_min_mean) & (gene_means <= actual_max_mean)

    filtered_disp = np.full_like(dispersions, -np.inf)
    filtered_disp[mask_mean] = dispersions[mask_mean]

    # Normalize dispersions within mean-expression bins
    # This controls for the inherent mean-variance relationship

    bin_edges = np.linspace(
        np.min(gene_means[mask_mean]), np.max(gene_means[mask_mean]), n_bins + 1
    )
    norm_disp = np.zeros_like(filtered_disp)

    for i in range(n_bins):
        bin_mask = (gene_means >= bin_edges[i]) & (gene_means < bin_edges[i + 1])
        if i == n_bins - 1:  # include right edge
            bin_mask |= gene_means == bin_edges[i + 1]

        if np.any(bin_mask):
            median_disp = np.median(filtered_disp[bin_mask])
            if median_disp > 0:
                norm_disp[bin_mask] = filtered_disp[bin_mask] / median_disp

    # Select top HVGs based on normalized dispersion
    top_idx = np.argsort(norm_disp)[-n_top_genes:]
    hvg_mask = np.zeros(n_genes, dtype=bool)
    hvg_mask[top_idx] = True
    adata.var["ginseng_genes"] = hvg_mask

    return adata if copy else None
