# Copyright (c) 2026, Tom Ouellette
# Licensed under the MIT License

import jax
import jax.numpy as jnp

from jaxtyping import Array, Int


def augment_mask(
    key: Array, x: Int[Array, "..."], rate: float = 0.1
) -> Int[Array, "..."]:
    """Randomly mask out counts across cells.

    Parameters
    ----------
    key : Array
        PRNG key array for dropout mask.
    x : Array
        Input tensor.
    rate : float
        Dropout probability.

    Returns
    -------
    Array
        Tensor with dropout applied.
    """
    keep_prob = 1.0 - rate
    mask: Int[Array, "..."] = jax.random.bernoulli(key, p=keep_prob, shape=x.shape)
    return x * mask


def augment_background(
    key: Array,
    x: Int[Array, "batch genes"],
    lam_max: float = 0.1,
) -> Int[Array, "batch genes"]:
    """Randomly add Poisson-distributed background noise to counts.

    Parameters
    ----------
    key : Array
        PRNG key array for noise generation.
    x : Array
        Non-normalized count matrix.
    lam_max : float
        Maximum mean of Poisson distribution for sampling added noise.

    Returns
    -------
    Array
        Counts with added Poisson noise.
    """
    lam = jax.random.uniform(key, minval=0.0, maxval=lam_max)
    if lam == 0:
        return x

    noise = jax.random.poisson(key, lam=lam, shape=x.shape)
    return x + noise


def augment_dropgene(
    key: Array,
    x: Int[Array, "batch genes"],
    lower: int = 0,
    upper: int = 2,
) -> Int[Array, "batch genes"]:
    """Randomly zero out entire genes.

    Parameters
    ----------
    key : Array
        PRNG key array for mask generation.
    x : Array
        Count matrix.
    lower : int
        Minimum number of genes to mask out.
    upper : int
        Maximum (exclusive) number of genes to mask out.

    Returns
    -------
    Array
        Filtered counts.
    """
    key_n, key_i = jax.random.split(key)
    n = jax.random.randint(key_n, (), lower, upper)
    i = jax.random.choice(key_i, x.shape[1], shape=(n,), replace=False)
    gene_mask = jnp.ones(x.shape[1], dtype=x.dtype).at[i].set(0)
    return x * gene_mask


def augment(
    key: Array,
    x: Int[Array, "batch genes"],
    rate: float | None = 0.1,
    lam_max: float | None = 0.1,
    lower: int | None = 0,
    upper: int | None = 2,
) -> Int[Array, "batch genes"]:
    """Apply a combination of single-cell RNA relevant augmentations.

    Parameters
    ----------
    key : Array
        PRNG key array for mask generation.
    x : Array
        Count matrix.
    rate : float
        Dropout probability.
    lam_max : float
        Maximum mean of Poisson distribution for sampling added noise.
    lower : int
        Minimum number of genes to mask out.
    upper : int
        Maximum (exclusive) number of genes to mask out.

    Returns
    -------
    Array
        Augmented counts.
    """
    key, k1, k2, k3 = jax.random.split(key, 4)

    if rate:
        x = augment_mask(k1, x, rate)

    if lam_max:
        x = augment_background(k2, x, lam_max)

    if lower and upper:
        x = augment_dropgene(k3, x, lower, upper)

    return x
