# Copyright (c) 2026, Tom Ouellette
# Licensed under the MIT License

import jax
import jax.numpy as jnp

from typing import NamedTuple
from jaxtyping import Array, Float, PyTree


class AdamState(NamedTuple):
    """State for the Adam optimizer.

    Attributes
    ----------
    step : int
        Current optimization step.
    m : PyTree of Array
        Exponential moving average of gradients.
    v : PyTree of Array
        Exponential moving average of squared gradients.
    """

    step: int
    m: PyTree[Float[Array, "..."]]
    v: PyTree[Float[Array, "..."]]


def opt_init_adam(params: PyTree[Float[Array, "..."]]) -> AdamState:
    """Initialize Adam optimizer state.

    Parameters
    ----------
    params : PyTree[Float[Array, ...]]
        Model parameters to be optimized.

    Returns
    -------
    AdamState
        Initial optimizer state with zeroed moments.
    """
    zeros = jax.tree.map(jnp.zeros_like, params)
    return AdamState(step=0, m=zeros, v=zeros)


def opt_adam_update(
    grads: PyTree[Float[Array, "..."]],
    params: PyTree[Float[Array, "..."]],
    state: AdamState,
    lr: float = 0.001,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> tuple[PyTree[Float[Array, "..."]], AdamState]:
    """Perform one Adam optimization step.

    Parameters
    ----------
    grads : PyTree[Float[Array, ...]]
        Gradients of the loss w.r.t. parameters.
    params : PyTree[Float[Array, ...]]
        Current model parameters.
    state : AdamState
        Current optimizer state.
    lr : float
        Learning rate.
    betas : tuple[float, float]
        Exponential decay rates for first and second moment estimates.
    eps : float
        Numerical stability constant.
    weight_decay : float
        L2 regularization factor.

    Returns
    -------
    tuple
        Updated parameters and new optimizer state.
    """
    beta1, beta2 = betas
    step = state.step + 1

    m = jax.tree.map(lambda m, g: beta1 * m + (1 - beta1) * g, state.m, grads)
    v = jax.tree.map(lambda v, g: beta2 * v + (1 - beta2) * (g * g), state.v, grads)

    m_hat = jax.tree.map(lambda m: m / (1 - beta1**step), m)
    v_hat = jax.tree.map(lambda v: v / (1 - beta2**step), v)

    def update_param(p, m_h, v_h, g):
        update = m_h / (jnp.sqrt(v_h) + eps)
        if weight_decay != 0.0:
            update += weight_decay * p
        return p - lr * update

    new_params = jax.tree.map(update_param, params, m_hat, v_hat, grads)
    new_state = AdamState(step=step, m=m, v=v)
    return new_params, new_state
