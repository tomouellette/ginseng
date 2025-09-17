# Copyright (c) 2025, Tom Ouellette
# Licensed under the MIT License

import anndata
import h5py
import jax
import os
import numpy as np
import pandas as pd
import pickle
import scipy.io
import warnings

from anndata import AnnData
from pathlib import Path
from scipy import sparse

from .train import GinsengModelState


def read_10x_mtx(
    path: str, var_names: str = "gene_symbols", make_unique=True
) -> AnnData:
    """Read 10x Genomics mtx format into an AnnData object.

    Parameters
    ----------
    path : str
        Path to directory containing matrix.mtx, barcodes.tsv, and genes.tsv/features.tsv.
    var_names : str
        Select 'gene_symbols' or 'gene_ids' as index for var.
    make_unique : bool
        If True, make var_names unique.

    Returns
    -------
    AnnData
    """
    matrix = scipy.io.mmread(os.path.join(path, "matrix.mtx.gz")).T.tocsr()

    barcodes = pd.read_csv(os.path.join(path, "barcodes.tsv.gz"), header=None)[
        0
    ].tolist()

    features_path = None
    for fname in ["features.tsv.gz", "genes.tsv.gz"]:
        p = os.path.join(path, fname)
        if os.path.exists(p):
            features_path = p
            break
    if features_path is None:
        raise FileNotFoundError("Could not find features.tsv.gz or genes.tsv.gz.")

    features = pd.read_csv(features_path, header=None, sep="\t")

    if features.shape[1] == 3:
        features.columns = ["gene_id", "gene_symbol", "feature_type"]
    else:
        features.columns = ["gene_id", "gene_symbol"]

    if var_names == "gene_symbols":
        var = pd.DataFrame(index=features["gene_symbol"])
    elif var_names == "gene_ids":
        var = pd.DataFrame(index=features["gene_id"])
    else:
        raise ValueError("var_names must be 'gene_symbols' or 'gene_ids'")

    obs = pd.DataFrame(index=barcodes)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        adata = AnnData(X=matrix, obs=obs, var=var)

        if make_unique:
            adata.var_names_make_unique()

    return adata


def read_10x_h5(
    path: str,
    genome: str | None = None,
    var_names: str = "gene_symbols",
    make_unique=True,
) -> AnnData:
    """Read 10x Genomics h5 format into an AnnData object.

    Parameters
    ----------
    path : str
        Path to 10x Genomics .h5 file.
    genome : str, optional
        Genome to extract (if file contains multiple genomes).
        If None, will use the first genome available.
    var_names : str
        Select 'gene_symbols' or 'gene_ids' as index for var.
    make_unique : bool
        If True, make var_names unique.

    Returns
    -------
    AnnData
    """
    with h5py.File(path, "r") as f:
        if "matrix" in f:
            group = f["matrix"]
        else:
            available = list(f.keys())
            if genome is None:
                genome = available[0]
            if genome not in f:
                raise ValueError(f"Genome {genome!r} not found. Available: {available}")
            group = f[genome]

        data = group["data"][:]
        indices = group["indices"][:]
        indptr = group["indptr"][:]
        shape = group["shape"][:]

        matrix = sparse.csc_matrix((data, indices, indptr), shape=shape).T
        barcodes = [b.decode("utf-8") for b in group["barcodes"][:]]
        gene_ids = [g.decode("utf-8") for g in group["features"]["id"][:]]
        gene_names = [g.decode("utf-8") for g in group["features"]["name"][:]]

        if var_names == "gene_symbols":
            var = pd.DataFrame(index=gene_names)
        elif var_names == "gene_ids":
            var = pd.DataFrame(index=gene_ids)
        else:
            raise ValueError("var_names must be 'gene_symbols' or 'gene_ids'")

        obs = pd.DataFrame(index=barcodes)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        adata = AnnData(X=matrix, obs=obs, var=var)

        if make_unique:
            adata.var_names_make_unique()

    return adata


def read_adata(path: str | Path, backed: bool = True) -> AnnData:
    """Read an AnnData object from various supported file formats.

    Parameters
    ----------
    path : str | Path
        Path to the input count data stored in 10x .h5 format, AnnData .h5ad format, or
        in a 10x matrix market format folder.
    backed : bool
        If True and the input is an `.h5ad` file, open the file in backed mode.

    Returns
    -------
    AnnData
        The loaded AnnData object containing gene expression data.
    """
    if isinstance(path, str):
        path = Path(path)

    if path.is_dir():
        required = {
            "matrix": ["matrix.mtx", "matrix.mtx.gz"],
            "barcodes": ["barcodes.tsv", "barcodes.tsv.gz"],
            "features": [
                "features.tsv",
                "features.tsv.gz",
                "genes.tsv",
                "genes.tsv.gz",
            ],
        }

        missing = []
        for key, candidates in required.items():
            if not any((path / fname).exists() for fname in candidates):
                missing.append(f"{key} file ({' or '.join(candidates)})")

        if missing:
            raise FileNotFoundError(
                f"The provided path is a directory. "
                f"However, missing required 10x matrix market files in {path}: "
                ",".join(missing)
                + "."
            )

        return read_10x_mtx(path)

    match path.suffix:
        case ".h5":
            return read_10x_h5(path)
        case ".h5ad":
            mode = "r" if backed else None
            return anndata.read_h5ad(path, backed=mode)
        case _:
            raise ValueError(
                "`path` must be an `.h5`, `.h5ad`, or 10x matrix market directory."
            )


def save_ginseng_state(state: GinsengModelState, filename: str | Path):
    """Save a Ginseng model state to disk.

    Parameters
    ----------
    state : GinsengModelState
        Model state containing parameters, genes, labels, and metadata.
    filename : str | Path
        Path where the model state will be stored.
    """
    with h5py.File(filename, "w") as f:
        # Save numpy arrays directly
        f.create_dataset("genes", data=state.genes, compression="gzip")
        f.create_dataset("label_keys", data=state.label_keys, compression="gzip")
        f.create_dataset("label_values", data=state.label_values, compression="gzip")

        # Save scalar metadata
        f.attrs["normalize"] = state.normalize
        f.attrs["target_sum"] = state.target_sum
        f.attrs["training"] = state.training

        # Flatten PyTree and save each array with numeric index
        flat_params, tree_def = jax.tree_util.tree_flatten(state.params)

        params_group = f.create_group("params")
        for i, param in enumerate(flat_params):
            params_group.create_dataset(
                f"param_{i:03d}", data=np.array(param), compression="gzip"
            )

        # Store the number of parameters for reconstruction
        f.attrs["num_params"] = len(flat_params)

        tree_def_bytes = pickle.dumps(tree_def)
        f.create_dataset(
            "tree_def_pickle", data=np.frombuffer(tree_def_bytes, dtype=np.uint8)
        )


def load_ginseng_state(filename: str | Path) -> GinsengModelState:
    """Load a Ginseng model state.

    Parameters
    ----------
    filename : str | Path
        Path to the HDF5 file containing the saved model state.

    Returns
    -------
    GinsengModelState
        Reconstructed model state.
    """
    with h5py.File(filename, "r") as f:
        # Load arrays
        genes = f["genes"][:]
        label_keys = f["label_keys"][:]
        label_values = f["label_values"][:]

        # Load metadata
        normalize = bool(f.attrs["normalize"])
        target_sum = float(f.attrs["target_sum"])
        training = bool(f.attrs["training"])

        # Load parameters
        num_params = int(f.attrs["num_params"])
        flat_params = []
        for i in range(num_params):
            flat_params.append(f["params"][f"param_{i:03d}"][:])

        # Reconstruct tree structure

        tree_def_bytes = f["tree_def_pickle"][:].tobytes()
        tree_def = pickle.loads(tree_def_bytes)

        # Reconstruct PyTree
        params = jax.tree_util.tree_unflatten(tree_def, flat_params)

        return GinsengModelState(
            params=params,
            genes=genes.astype(str),
            label_keys=label_keys,
            label_values=label_values,
            normalize=normalize,
            target_sum=target_sum,
            training=training,
        )
