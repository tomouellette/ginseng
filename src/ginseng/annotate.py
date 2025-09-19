# Copyright (c) 2025, Tom Ouellette
# Licensed under the MIT License

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from anndata import AnnData
from pathlib import Path
from tqdm import tqdm
from typing import Iterator

from .io import read_adata, load_ginseng_state
from .nn import nn_annotate
from .train import GinsengModelState


def annotate_iter_full(
    adata: AnnData,
    gene_names: list[str],
    gene_key: str | None = None,
    batch_size: int = 512,
) -> Iterator[jnp.array]:
    """Iterator over AnnData returning jax arrays when all genes are present.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    gene_names : list[str]
        Genes ordered as in training.
    gene_key : str | None
        Column in .var where gene names are stored.
    batch_size : int
        Number of cells per batch.

    Yields
    ------
    jax.numpy.ndarray
        Array of shape (batch_size, len(gene_names))
    """
    if gene_key is None:
        all_genes = adata.var_names.to_numpy()
    else:
        all_genes = adata.var[gene_key].values

    gene_to_idx = {gene: idx for idx, gene in enumerate(all_genes)}
    idx_order = np.array([gene_to_idx[gene] for gene in gene_names])

    n_obs = adata.n_obs

    for start in range(0, n_obs, batch_size):
        end = min(start + batch_size, n_obs)

        batch = adata.X[start:end, :]

        if hasattr(batch, "toarray"):
            batch = batch.toarray()

        batch = batch[:, idx_order]

        yield jnp.asarray(batch, dtype=jnp.float32)


def annotate_iter_missing(
    adata: AnnData,
    gene_names: list[str],
    gene_key: str | None = None,
    batch_size: int = 512,
) -> Iterator[jnp.array]:
    """Iterator over AnnData when genes are missing which were used during training.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    gene_names : list[str]
        Genes ordered as in training.Missing genes are filled with zeros.
    gene_key : str | None
        Column in .var where gene names are stored.
    batch_size : int
        Number of barcodes per batch.

    Yields
    ------
    jax.numpy.ndarray
        Array of shape (batch_size, genes)
    """
    if gene_key is None:
        all_genes = np.array(adata.var_names)
    else:
        all_genes = adata.var[gene_key].values

    # Note: If genes are missing, we map requested genes to indices and
    # then dynamically fill an array with available indices during iteration.
    var_name_to_idx = {g: i for i, g in enumerate(all_genes)}
    available_idx = []
    out_positions = []
    for pos, g in enumerate(gene_names):
        if g in var_name_to_idx:
            available_idx.append(var_name_to_idx[g])
            out_positions.append(pos)

    n_obs = adata.n_obs
    n_genes = len(gene_names)

    for start in range(0, n_obs, batch_size):
        end = min(start + batch_size, n_obs)
        batch = adata.X[start:end, :]

        if hasattr(batch, "toarray"):
            batch = batch.toarray()

        out_batch = np.zeros((end - start, n_genes), dtype=np.float32)

        if available_idx:
            out_batch[:, out_positions] = batch[:, available_idx]

        yield jnp.asarray(out_batch)


def annotate(
    model_state: GinsengModelState | str | Path,
    adata: AnnData | str | Path,
    backed: bool = True,
    normalize: bool | None = None,
    target_sum: float | None = None,
    randomness: bool = False,
    copy: bool = False,
    return_attn: bool = False,
    return_table: bool = False,
    seed: int = 123,
) -> None | AnnData | pd.DataFrame:
    """Annotate single-cell sequencing data using a trained ginseng model.

    Parameters
    ----------
    model_state : GinsengModelState | str | Path
        Model state containing parameters, genes, labels, and metadata.
    adata : AnnData | str | Path
        AnnData object or path to count data stored in 10x .h5 format, AnnData .h5ad
        format, or in a 10x matrix market format folder.
    backed : bool
        If True and adata is a str or Path, then read count data in on-disk/backed mode.
        When running in backed mode, memory requirements will be lower at cost of
        longer running times (only relevant for large datasets).
    normalize : bool
        Override normalize in model_state. This may be useful if you do not have access
        to raw counts. Be aware that differences in normalization between training and
        inference will likely change performance.
    target_sum  : bool
        Override target_sum in model_state. As noted in the normalize argument, modifying
        target_sum will likely change model performance.
    randomness : bool
        If True, dropout remains active during inference. This enables techniques like
        Monte Carlo dropout, where repeated stochastic forward passes yield an approximate
        predictive distribution by sampling subnetworks.
    copy : bool
        If True, return a copy of the AnnData object else modify in-place.
    return_attn : bool
        If True, return the predicted gene-level weights associated with each barcode. Note
        that this will be an array of N cells by N genes ( not recommended for large datasets).
    return_table : bool
        If True, a two column table with columns 'barcode' and 'cell_type' is returned from
        the function instead.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    None | AnnData | pd.DataFrame
        By default, AnnData is modified in place. If copy=True, then an AnnData is returned.
        If return_table=True, then a data frame with  barcode and cell type info is returned.
    """
    pass
    # -- TODO --
    # if isinstance(model_state, (str, Path)):
    #     model_state = load_ginseng_state(model_state)

    # if isinstance(adata, (str, Path)):
    #     adata = read_adata(adata, backed)

    # label_map = dict(zip(model_state.label_keys, model_state.label_values))

    # key = jax.random.key(seed)

    # nn_annotate(
    #     model_state.params,
    #     key,
    #     x,
    #     dropout_rate=model_state.dropout_rate,
    #     normalize=normalize if normalize else model_state.normalize,
    #     target_sum=target_sum if target_sum else model_state.target_sum,
    #     return_attn=return_attn,
    #     training=randomness,
    # )
