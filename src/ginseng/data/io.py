# Copyright (c) 2026, Tom Ouellette
# Licensed under the MIT License

import os
import tempfile
import requests
import warnings
import anndata
import h5py
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy.io

from anndata import AnnData
from pathlib import Path
from scipy import sparse
from urllib.parse import urlparse
from ginseng.model.state import GinsengClassifierState


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
    path_str = str(path)
    is_url = urlparse(path_str).scheme in ("http", "https")

    if is_url:
        if not any(path_str.endswith(ext) for ext in [".h5", ".h5ad"]):
            raise ValueError(
                "When providing a URL, the file must end with .h5 or .h5ad."
            )

        # We download to a temporary file because h5ad/h5 requires local disk access
        suffix = "".join(Path(urlparse(path_str).path).suffixes)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            response = requests.get(path_str, stream=True)
            response.raise_for_status()

            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp.flush()

            # URL-sourced files are loaded into memory (backed=False)
            # because the temp file is cleaned up after this function returns.
            return read_adata(Path(tmp.name), backed=False)

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


def _load_pytree(group: h5py.Group) -> dict:
    """Recursively load a PyTree from an HDF5 group."""
    pytree = {}
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            pytree[key] = _load_pytree(item)
        else:
            # Load as JAX array
            pytree[key] = jnp.array(item[:])
    return pytree


def save_model(state: GinsengClassifierState, path: str | Path):
    """Save a Ginseng model state to a single HDF5 file.

    Parameters
    ----------
    state : GinsengClassifierState
        Complete model state to save.
    path : str | Path
        Path to the output HDF5 file (will add .h5 if not present).

    Example
    -------
    >>> # After training
    >>> state = GinsengClassifierState(
    ...     params=model.params,
    ...     genes=dataset.gene_names,
    ...     label_keys=dataset.label_names,
    ...     label_values=np.arange(len(dataset.label_names)),
    ...     n_genes=dataset.n_genes,
    ...     n_classes=len(dataset.label_names),
    ...     hidden_dim=256,
    ...     normalize=True,
    ...     target_sum=1e4,
    ...     dropout_rate=0.5,
    ...     training=False,
    ... )
    >>> save_model(state, "./models/my_classifier.h5")
    """
    path = Path(path)
    if path.suffix != ".h5":
        path = path.with_suffix(".h5")

    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        # Save metadata arrays
        f.create_dataset(
            "genes",
            data=np.array(state.genes).astype("S"),
            compression="gzip",
        )

        f.create_dataset(
            "label_keys",
            data=np.array(state.label_keys).astype("S"),
            compression="gzip",
        )

        f.create_dataset(
            "label_values",
            data=np.array(state.label_values),
            compression="gzip",
        )

        # Save scalar metadata as attributes
        f.attrs["n_genes"] = state.n_genes
        f.attrs["n_classes"] = state.n_classes
        f.attrs["hidden_dim"] = state.hidden_dim
        f.attrs["normalize"] = state.normalize
        f.attrs["target_sum"] = state.target_sum
        f.attrs["dropout_rate"] = state.dropout_rate
        f.attrs["training"] = state.training

        # Save parameters recursively
        params_group = f.create_group("params")
        _save_pytree(params_group, state.params)


def _save_pytree(group: h5py.Group, pytree: dict):
    """Recursively save a PyTree to an HDF5 group."""
    for key, value in pytree.items():
        if isinstance(value, dict):
            subgroup = group.create_group(key)
            _save_pytree(subgroup, value)
        else:
            # Save JAX array as numpy
            group.create_dataset(
                key,
                data=np.asarray(value),
                compression="gzip",
            )


def load_model(path: str | Path) -> GinsengClassifierState:
    """Load a Ginseng model state from an HDF5 file.

    Parameters
    ----------
    path : str | Path
        Path to the HDF5 file.

    Returns
    -------
    GinsengClassifierState
        Complete model state ready for inference or continued training.

    Example
    -------
    >>> state = load_model("./models/my_classifier.h5")
    >>>
    >>> # Recreate model
    >>> model = GinsengClassifier(
    ...     n_genes=state.n_genes,
    ...     n_classes=state.n_classes,
    ...     hidden_dim=state.hidden_dim,
    ...     dropout_rate=state.dropout_rate,
    ...     normalize=state.normalize,
    ...     target_sum=state.target_sum,
    ... )
    >>> model.params = state.params
    >>>
    >>> # Prepare new data with correct gene order
    >>> new_data_ordered = new_data[state.genes]
    >>> predictions = model.predict(new_data_ordered.values, training=False)
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    with h5py.File(path, "r") as f:
        # Load metadata arrays
        genes = f["genes"][:].astype(str)
        label_keys = f["label_keys"][:].astype(str)
        label_values = f["label_values"][:]

        # Load scalar metadata
        n_genes = int(f.attrs["n_genes"])
        n_classes = int(f.attrs["n_classes"])
        hidden_dim = int(f.attrs["hidden_dim"])
        normalize = bool(f.attrs["normalize"])
        target_sum = float(f.attrs["target_sum"])
        dropout_rate = float(f.attrs["dropout_rate"])
        training = bool(f.attrs["training"])

        # Load parameters
        params = _load_pytree(f["params"])

    return GinsengClassifierState(
        params=params,
        genes=genes,
        label_keys=label_keys,
        label_values=label_values,
        n_genes=n_genes,
        n_classes=n_classes,
        hidden_dim=hidden_dim,
        normalize=normalize,
        target_sum=target_sum,
        dropout_rate=dropout_rate,
        training=training,
    )
