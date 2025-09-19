# Copyright (c) 2025, Tom Ouellette
# Licensed under the MIT License

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import warnings

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

        yield jnp.asarray(batch, dtype=jnp.float32), start, end


def annotate_iter_partial(
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

        yield jnp.asarray(out_batch), start, end


def annotate(
    model_state: GinsengModelState | str | Path,
    adata: AnnData | str | Path,
    gene_key: str | None = None,
    backed: bool = True,
    normalize: bool | None = None,
    target_sum: float | None = None,
    randomness: bool = False,
    batch_size: int = 256,
    copy: bool = False,
    return_probs: bool = False,
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
    gene_key : str | None
        Column in .var where gene names are stored.
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
    batch_size : int
        Annotate data in chunks of size (batch_size). If your dataset is not large and you
        have sufficient memory, then batch size can be set to total number of rows.
    copy : bool
        If True, return a copy of the AnnData object else modify in-place.
    return_probs : bool
        If True, return a table of cell type softmax probabilities for each barcode.
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
    if return_attn:
        raise NotImplementedError(
            "Returning of gene-level weights is not currently implemented. "
            "Please re-run without `return_attn=True`."
        )

    if isinstance(model_state, (str, Path)):
        model_state = load_ginseng_state(model_state)

    if isinstance(adata, (str, Path)):
        adata = read_adata(adata, backed)

    if not isinstance(batch_size, int) or batch_size < 0:
        raise ValueError("`batch_size` must be a positive non-zero integer.")

    if gene_key:
        if gene_key not in adata.var.columns:
            raise ValueError(f"`gene_key` {gene_key} was not found in `adata.var`")

        is_present = np.isin(model_state.genes, adata.var[gene_key].values)
    else:
        is_present = np.isin(model_state.genes, adata.var_names)

    n_present = is_present.sum()
    all_present = is_present.all()

    if n_present == 0:
        raise ValueError(
            f"None of the {len(model_state.genes)} ginseng model gene names are present "
            "in the supplied `adata`. If `gene_key` was supplied as an argument, please "
            "check if it specifies a column of gene names that overlap fully or partially "
            "with `model_state.genes`."
        )

    if not all_present:
        p_overlap = 100 * (n_present / len(model_state.genes))
        warnings.warn(
            f"Only {p_overlap:.2f}% of genes in `adata.var` overlap with "
            "`model_state.genes`. Be aware that when overlap is too low "
            "model performance may degrade."
        )

    if all_present:
        iterator = annotate_iter_full(
            adata, model_state.genes, gene_key, batch_size=batch_size
        )
    else:
        iterator = annotate_iter_partial(
            adata, model_state.genes, gene_key, batch_size=batch_size
        )

    key = jax.random.key(seed)

    label_map = dict(zip(model_state.label_keys, model_state.label_values))

    if return_probs:
        results = np.zeros((adata.shape[0], len(label_map)))
    else:
        results = np.zeros(adata.shape[0])

    for x, start, end in tqdm(
        iterator, desc="[ginseng] Annotating", total=adata.shape[0]
    ):
        logits = nn_annotate(
            model_state.params,
            key,
            x,
            dropout_rate=model_state.dropout_rate,
            normalize=normalize if normalize else model_state.normalize,
            target_sum=target_sum if target_sum else model_state.target_sum,
            return_attn=return_attn,
            training=randomness,
        )

        if return_probs:
            results[start:end] = np.asarray(jax.nn.softmax(logits))
        else:
            results[start:end] = np.asarray(logits).argmax(axis=1)

    if return_probs:
        probs = pd.DataFrame(results)
        probs.columns = list(label_map.values())
        probs.index = adata.obs_names
        return probs

    results = np.array([label_map[k] for k in results])

    if return_table:
        preds = pd.DataFrame(results)
        preds.columns = ["ginseng_cell_type"]
        preds.index = adata.obs_names
        return preds

    adata.obs["ginseng_cell_type"] = results

    return adata if copy else None
