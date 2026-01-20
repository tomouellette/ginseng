# Copyright (c) 2026, Tom Ouellette
# Licensed under the MIT License

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import warnings
import math

from anndata import AnnData
from pathlib import Path
from tqdm import tqdm
from typing import Union, Iterator, Tuple, Optional

from ginseng.data.io import read_adata, load_model
from .nn import nn_annotate
from .state import GinsengClassifierState


def _get_gene_indices(
    adata: AnnData, gene_names: np.ndarray, gene_key: str | None
) -> Tuple[np.ndarray, np.ndarray]:
    """Map model gene names to indices within the AnnData object.

    Parameters
    ----------
    adata : AnnData
        The single-cell data object.
    gene_names : np.ndarray
        Array of gene names expected by the model in a specific order.
    gene_key : str | None
        Column name in `adata.var` containing gene names. If None, uses `adata.var_names`.

    Returns
    -------
    available_idx : np.ndarray
        The integer indices of genes found in `adata` that match the model's expected genes.
    out_positions : np.ndarray
        The corresponding positions in the model's input vector where these genes belong.
    """
    if gene_key is None:
        all_genes = adata.var_names.to_numpy()
    else:
        all_genes = adata.var[gene_key].values

    gene_to_idx = {gene: idx for idx, gene in enumerate(all_genes)}

    available_idx = []
    out_positions = []
    for pos, gene in enumerate(gene_names):
        if gene in gene_to_idx:
            available_idx.append(gene_to_idx[gene])
            out_positions.append(pos)

    return np.array(available_idx), np.array(out_positions)


def annotate_iter(
    adata: AnnData,
    gene_names: np.ndarray,
    gene_key: str | None = None,
    layer: str | None = None,
    batch_size: int = 512,
) -> Iterator[Tuple[jnp.ndarray, int, int]]:
    """Batch iterator that handles gene reordering and zero-padding for missing genes.

    Parameters
    ----------
    adata : AnnData
        The single-cell data object.
    gene_names : np.ndarray
        Ordered gene names from the trained model.
    gene_key : str | None
        Column in `adata.var` to use for gene matching.
    layer : str | None
        Key in `adata.layers` to use for counts. If None, uses `adata.X`.
    batch_size : int
        Number of cells to process per batch.

    Yields
    ------
    batch_tensor : jax.numpy.ndarray
        A JAX-compatible array of shape (batch_size, n_model_genes).
    start : int
        Starting observation index.
    end : int
        Ending observation index.
    """
    available_idx, out_positions = _get_gene_indices(adata, gene_names, gene_key)
    n_obs = adata.n_obs
    n_model_genes = len(gene_names)
    all_present = len(available_idx) == n_model_genes

    # Define the data source (X or specific layer)
    data_source = adata.layers[layer] if layer is not None else adata.X

    for start in range(0, n_obs, batch_size):
        end = min(start + batch_size, n_obs)
        current_batch_size = end - start

        # Slice the specific rows and columns needed
        batch = data_source[start:end, available_idx]

        if hasattr(batch, "toarray"):
            batch = batch.toarray()

        if all_present:
            yield jnp.asarray(batch, dtype=jnp.float32), start, end
        else:
            # Scatter available genes into a zero-filled array for partial overlap
            out_batch = np.zeros((current_batch_size, n_model_genes), dtype=np.float32)
            out_batch[:, out_positions] = batch
            yield jnp.asarray(out_batch), start, end


def classify(
    model_state: Union[GinsengClassifierState, str, Path],
    adata: Union[AnnData, str, Path],
    gene_key: str | None = None,
    layer: str | None = None,
    backed: bool = True,
    normalize: bool | None = None,
    target_sum: float | None = None,
    randomness: bool = False,
    batch_size: int = 256,
    copy: bool = False,
    store_probs: bool = False,
    return_table: bool = False,
    seed: int = 123,
    silent: bool = False,
) -> Union[None, AnnData, pd.DataFrame]:
    """Annotate single-cell sequencing data using a trained ginseng classifier.

    Parameters
    ----------
    model_state : GinsengClassifierState | str | Path
        A loaded GinsengClassifierState or a path to a saved state file.
    adata : AnnData | str | Path
        AnnData object or path to count data (.h5ad, .h5, or Matrix Market).
    gene_key : str | None
        Column in `.var` containing gene names. If None, uses index.
    layer : str | None
        Key in `adata.layers` to use for counts. If None, uses `adata.X`.
    backed : bool
        If True and `adata` is a path, reads data in backed mode.
    normalize : bool | None
        Override model normalization setting.
    target_sum : float | None
        Override model target sum.
    randomness : bool
        If True, enables dropout during inference.
    batch_size : int
        Number of cells to process in each forward pass.
    copy : bool
        If True, returns a modified copy of the AnnData object.
    store_probs : bool
        If True, stores the full probability matrix in `adata.obsm`.
    return_table : bool
        If True, returns a pandas DataFrame instead of modifying AnnData.
    seed : int
        Random seed.
    silent : bool
        If True, suppresses progress bar and warnings.

    Returns
    -------
    AnnData | pd.DataFrame | None
        Returns a DataFrame if `return_table` is True, a copy of AnnData if
        `copy` is True, otherwise modifies in-place and returns None.
    """
    if isinstance(model_state, (str, Path)):
        model_state = load_model(model_state)

    if isinstance(adata, (str, Path)):
        adata = read_adata(adata, backed)

    if copy:
        adata = adata.copy()

    # Validation
    if layer is not None and layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' not found in the AnnData object.")

    available_idx, _ = _get_gene_indices(adata, model_state.genes, gene_key)
    n_cells = adata.n_obs
    n_classes = len(model_state.label_keys)

    if len(available_idx) == 0:
        raise ValueError("No matching genes found between model and adata.")

    if not silent and len(available_idx) < len(model_state.genes):
        p_overlap = 100 * (len(available_idx) / len(model_state.genes))
        warnings.warn(f"Partial gene overlap detected: {p_overlap:.2f}%.")

    # Pre-allocate probability matrix
    prob_matrix = np.zeros((n_cells, n_classes), dtype=np.float32)
    key = jax.random.key(seed)

    iterator = annotate_iter(adata, model_state.genes, gene_key, layer, batch_size)
    total_batches = math.ceil(n_cells / batch_size)

    # Inference loop
    for x_batch, start, end in tqdm(
        iterator, desc="[ginseng] Classifying", total=total_batches, disable=silent
    ):
        logits = nn_annotate(
            model_state.params,
            key,
            x_batch,
            dropout_rate=model_state.dropout_rate,
            normalize=normalize if normalize is not None else model_state.normalize,
            target_sum=target_sum if target_sum is not None else model_state.target_sum,
            return_attn=False,
            training=randomness,
        )

        # Capture batch probabilities
        prob_matrix[start:end] = np.asarray(jax.nn.softmax(logits))

    # Post-process results
    pred_indices = prob_matrix.argmax(axis=1)
    final_labels = np.array(model_state.label_keys)[pred_indices]
    confidences = prob_matrix.max(axis=1)

    if return_table:
        df = pd.DataFrame(
            prob_matrix, columns=model_state.label_keys, index=adata.obs_names
        )
        df.insert(0, "ginseng_cell_type", final_labels)
        df.insert(1, "ginseng_confidence", confidences)
        return df

    # Store classification data
    adata.obs["ginseng_cell_type"] = pd.Categorical(final_labels)
    adata.obs["ginseng_confidence"] = confidences

    if store_probs:
        adata.obsm["ginseng_probs"] = pd.DataFrame(
            prob_matrix, columns=model_state.label_keys, index=adata.obs_names
        )

    return adata if copy else None
