# Copyright (c) 2025, Tom Ouellette
# Licensed under the MIT License

import jax
import jax.numpy as jnp
import numpy as np

from jaxtyping import Array, Float, Int, PyTree


def nn_xavier_uniform(
    key: Array, shape: tuple[int, int]
) -> Float[Array, "in_dim out_dim"]:
    """Initialize weights with Xavier uniform distribution.

    Parameters
    ----------
    key : Array
        PRNG key array for random number generation.
    shape : tuple of int
        Shape of the weight matrix.

    Returns
    -------
    Array
        Initialized weight matrix.
    """
    fan_in, fan_out = shape[0], shape[1]
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


def nn_init_linear(
    key: Array, in_dim: int, out_dim: int
) -> PyTree[Float[Array, "..."]]:
    """Initialize parameters for a linear layer.

    Parameters
    ----------
    key : Array
        PRNG key array for random initialization.
    in_dim : int
        Input dimension.
    out_dim : int
        Output dimension.

    Returns
    -------
    PyTree[Float[Array, "..."]]
        Dictionary with weight matrix `W` and bias vector `b`.
    """
    k1, _ = jax.random.split(key)
    W: Float[Array, "in_dim out_dim"] = nn_xavier_uniform(k1, (in_dim, out_dim))
    b: Float[Array, "out_dim"] = jnp.zeros((out_dim,))
    return {"W": W, "b": b}


def nn_linear(
    params: PyTree[Float[Array, "..."]], x: Float[Array, "batch in_dim"]
) -> Float[Array, "batch out_dim"]:
    """Apply a linear transformation.

    Parameters
    ----------
    params : PyTree[Float[Array, "..."]]
        Layer parameters with `W` and `b`.
    x : Array
        Input tensor.

    Returns
    -------
    Array
        Transformed output tensor.
    """
    return x @ params["W"] + params["b"]


def nn_dropout(
    key: Array, x: Float[Array, "..."], rate: float, training: bool = True
) -> Float[Array, "..."]:
    """Apply dropout to input array.

    Parameters
    ----------
    key : Array
        PRNG key array for dropout mask.
    x : Array
        Input tensor.
    rate : float
        Dropout probability.
    training : bool
        If True, apply dropout. Set to False for determinstic output after training.

    Returns
    -------
    Array
        Tensor with dropout applied.
    """
    if not training or rate == 0.0:
        return x
    keep_prob = 1.0 - rate
    mask: Int[Array, "..."] = jax.random.bernoulli(key, p=keep_prob, shape=x.shape)
    return x * mask / keep_prob


def nn_normalize(
    x: Float[Array, "batch genes"], target_sum: float = 1e4
) -> Float[Array, "batch genes"]:
    """
    Normalize counts per cell and apply log transform.

    Parameters
    ----------
    x : Array
        Count matrix.
    target_sum : float
        Target total count per cell after normalization.

    Returns
    -------
    Array
        Normalized expression matrix.
    """
    sf = jnp.clip(jnp.sum(x, axis=1, keepdims=True), min=1e-8)
    return jnp.log1p(target_sum * (x / sf))


def nn_annotate_init(
    key: Array, n_genes: int, n_classes: int, hidden_dim: int = 256
) -> PyTree[Float[Array, "..."]]:
    """Initialize parameters for the cell type annotation network.

    Parameters
    ----------
    key : Array
        PRNG key array for random initialization.
    n_genes : int
        Number of genes.
    n_classes : int
        Number of classes.
    hidden_dim : int
        Dimension of hidden layers.

    Returns
    -------
    PyTree[Float[Array, "..."]]
        Dictionary of initialized model parameters.
    """
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    return {
        "features1": nn_init_linear(k1, n_genes, n_genes),
        "features2": nn_init_linear(k2, n_genes, n_genes),
        "attn1": nn_init_linear(k3, n_genes, hidden_dim),
        "attn2": nn_init_linear(k4, hidden_dim, n_genes),
        "head": nn_init_linear(k5, n_genes, n_classes),
    }


def nn_annotate(
    params: PyTree[Float[Array, "..."]],
    key: Array,
    x: Float[Array, "batch genes"],
    dropout_rate: float = 0.5,
    normalize: bool = True,
    target_sum: None | float = None,
    return_attn: bool = False,
    training: bool = True,
) -> (
    Float[Array, "batch classes"]
    | tuple[Float[Array, "batch classes"], Float[Array, "batch genes"]]
):
    """Annotate cells using a neural attention-based model.

    Parameters
    ----------
    params : PyTree[Float[Array, "..."]]
        Model parameters from `nn_annotate_init`.
    key : Array
        PRNG key array for random number generation.
    x : Array
        Input gene expression matrix.
    dropout_rate : float
        Dropout probability.
    normalize : bool
        If True, normalize and log-transform counts.
    target_sum : None | float
        Target total count per cell after normalization (defaults to number of genes).
    return_attn : bool
        If True, also return attention weights.
    training : bool
        Whether the model is in training mode.

    Returns
    -------
    Float[Array, ...] | tuple[Float[Array, ...], Float[Array, ...]]
        Logits. If `return_attn=True`, gene-level attention weights are also returned.
    """
    key, sub1, sub2 = jax.random.split(key, 3)

    if normalize:
        if target_sum is None:
            target_sum = x.shape[0]
        x = nn_normalize(x, target_sum)

    x = nn_dropout(sub1, x, dropout_rate, training)
    x = jax.nn.relu(nn_linear(params["features1"], x))
    x = nn_linear(params["features2"], x)

    a = nn_dropout(sub2, x, dropout_rate, training)
    a = jnp.tanh(nn_linear(params["attn1"], a))
    a = nn_linear(params["attn2"], a)
    a = jax.nn.sigmoid(a)

    z = a * x
    logits = nn_linear(params["head"], z)

    if logits.shape[1] == 1:
        logits = jnp.squeeze(logits, axis=1)

    if return_attn:
        return logits, a

    return logits


def nn_annotate_loss(
    params: PyTree[Float[Array, "..."]],
    key: Array,
    x: Float[Array, "batch genes"],
    y: Float[Array, "batch classes"],
    dropout_rate: float = 0.5,
    normalize: bool = True,
    target_sum: None | float = None,
    training: bool = True,
) -> float:
    """Cross-entropy loss for cell type annotation model.

    Parameters
    ----------
    params : PyTree[Float[Array, "..."]]
        Model parameters from `nn_annotate_init`.
    key : Array
        PRNG key array for random number generation.
    x : Array
        Input gene expression matrix.
    y : Array
        True cell type labels.
    dropout_rate : float
        Dropout probability.
    normalize : bool
        If True, normalize and log-transform counts.
    target_sum : None | float
        Target total count per cell after normalization (defaults to number of genes).
    training : bool
        Whether the model is in training mode.

    Returns
    -------
    Float[Array, ...] | tuple[Float[Array, ...], Float[Array, ...]]
        Logits. If `return_attn=True`, gene-level attention weights are also returned.
    """
    logits = nn_annotate(
        params,
        key,
        x,
        normalize=normalize,
        dropout_rate=dropout_rate,
        target_sum=target_sum,
        training=True,
    )

    onehot = jax.nn.one_hot(y, params["head"]["W"].shape[-1])
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(onehot * log_probs, axis=-1))


def nn_annotate_evaluate(
    params: PyTree[Float[Array, "..."]],
    key: Array,
    x: Float[Array, "batch genes"],
    y: Float[Array, "batch classes"],
    dropout_rate: float = 0.5,
    normalize: bool = True,
    target_sum: None | float = None,
):
    """Cross-entropy loss for cell type annotation model.

    Parameters
    ----------
    params : PyTree[Float[Array, "..."]]
        Model parameters from `nn_annotate_init`.
    key : Array
        PRNG key array for random number generation.
    x : Array
        Input gene expression matrix.
    y : Array
        True cell type labels.
    dropout_rate : float
        Dropout probability.
    normalize : bool
        If True, normalize and log-transform counts.
    target_sum : None | float
        Target total count per cell after normalization (defaults to number of genes).

    Returns
    -------
    tuple[float, int, int]
        Loss, total number of labels, and number of correct labels
    """
    key, subkey = jax.random.split(key, 2)
    loss = nn_annotate_loss(
        params,
        subkey,
        x,
        y,
        dropout_rate=dropout_rate,
        normalize=normalize,
        target_sum=target_sum,
        training=False,
    )

    logits = nn_annotate(
        params,
        subkey,
        x,
        dropout_rate=dropout_rate,
        normalize=normalize,
        target_sum=target_sum,
        training=False,
    )

    preds = np.asarray(logits).argmax(axis=1)
    correct = (preds == np.asarray(y)).sum()
    total = len(y)

    return loss, total, correct
