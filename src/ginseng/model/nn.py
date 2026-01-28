# Copyright (c) 2026, Tom Ouellette
# Licensed under the MIT License

import jax
import jax.numpy as jnp

from jaxtyping import Array, Float, PyTree


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
    """Apply dropout to input array with optimized implementation.

    Parameters
    ----------
    key : Array
        PRNG key array for dropout mask.
    x : Array
        Input tensor.
    rate : float
        Dropout probability.
    training : bool
        If True, apply dropout. Set to False for deterministic output after training.

    Returns
    -------
    Array
        Tensor with dropout applied.
    """
    if not training or rate == 0.0:
        return x
    keep_prob = 1.0 - rate
    # Optimized: use float mask and fused multiply-divide
    mask = jax.random.bernoulli(key, p=keep_prob, shape=x.shape).astype(x.dtype)
    scale = 1.0 / keep_prob
    return x * mask * scale
