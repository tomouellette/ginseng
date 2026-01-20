# Copyright (c) 2026, Tom Ouellette
# Licensed under the MIT License

import jax
import jax.numpy as jnp
from functools import partial

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


def nn_normalize(
    x: Float[Array, "batch genes"], target_sum: float = 1e4
) -> Float[Array, "batch genes"]:
    """Normalize counts per cell and apply log transform.

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
        Dimension of hidden layers for attention mechanism.

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


@partial(
    jax.jit,
    static_argnames=(
        "dropout_rate",
        "normalize",
        "target_sum",
        "return_attn",
        "training",
    ),
)
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
    """Annotate cells using instance-based attention neural network.

    This model uses gated attention over genes, similar to attention-based
    multiple instance learning, where each gene is treated as an instance
    and the model learns to weight their importance for classification.

    Parameters
    ----------
    params : PyTree[Float[Array, "..."]]
        Model parameters from `nn_annotate_init`.
    key : Array
        PRNG key array for random number generation.
    x : Array
        Input gene expression matrix (batch_size, n_genes).
    dropout_rate : float
        Dropout probability (default: 0.5).
    normalize : bool
        If True, normalize and log-transform counts (default: True).
    target_sum : None | float
        Target total count per cell after normalization (default: 1e4).
    return_attn : bool
        If True, also return attention weights (default: False).
    training : bool
        Whether the model is in training mode (default: True).

    Returns
    -------
    Float[Array, ...] | tuple[Float[Array, ...], Float[Array, ...]]
        Logits for each class. If `return_attn=True`, gene-level
        attention weights are also returned as second element.
    """
    key, sub1, sub2 = jax.random.split(key, 3)

    if normalize:
        if target_sum is None:
            target_sum = 1e4
        x = nn_normalize(x, target_sum)

    # Feature extraction pathway
    x = nn_dropout(sub1, x, dropout_rate, training)
    x = jax.nn.relu(nn_linear(params["features1"], x))
    x = nn_linear(params["features2"], x)

    # Attention pathway (MIL-style gated attention)
    a = nn_dropout(sub2, x, dropout_rate, training)
    a = jnp.tanh(nn_linear(params["attn1"], a))
    a = nn_linear(params["attn2"], a)
    a = jax.nn.sigmoid(a)

    # Gated features
    z = a * x
    logits = nn_linear(params["head"], z)

    if logits.shape[1] == 1:
        logits = jnp.squeeze(logits, axis=1)

    if return_attn:
        return logits, a

    return logits


@partial(jax.jit, static_argnames=("dropout_rate", "normalize", "target_sum"))
def nn_annotate_loss(
    params: PyTree[Float[Array, "..."]],
    key: Array,
    x: Float[Array, "batch genes"],
    y: Int[Array, "batch"],
    dropout_rate: float = 0.5,
    normalize: bool = True,
    target_sum: None | float = None,
) -> Float[Array, ""]:
    """Cross-entropy loss for cell type annotation model.

    Parameters
    ----------
    params : PyTree[Float[Array, "..."]]
        Model parameters from `nn_annotate_init`.
    key : Array
        PRNG key array for random number generation.
    x : Array
        Input gene expression matrix (batch_size, n_genes).
    y : Array
        True cell type labels as integer class indices (batch_size,).
    dropout_rate : float
        Dropout probability (default: 0.5).
    normalize : bool
        If True, normalize and log-transform counts (default: True).
    target_sum : None | float
        Target total count per cell after normalization.
        Defaults to 1e4 (standard for scRNA-seq).

    Returns
    -------
    Float[Array, ""]
        Scalar cross-entropy loss.
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


@partial(jax.jit, static_argnames=("dropout_rate", "normalize", "target_sum"))
def nn_annotate_evaluate(
    params: PyTree[Float[Array, "..."]],
    key: Array,
    x: Float[Array, "batch genes"],
    y: Int[Array, "batch"],
    dropout_rate: float = 0.5,
    normalize: bool = True,
    target_sum: None | float = None,
) -> tuple[Float[Array, ""], Int[Array, ""], Int[Array, ""]]:
    """Evaluate model performance on a batch.

    Parameters
    ----------
    params : PyTree[Float[Array, "..."]]
        Model parameters from `nn_annotate_init`.
    key : Array
        PRNG key array for random number generation.
    x : Array
        Input gene expression matrix (batch_size, n_genes).
    y : Array
        True cell type labels as integer class indices (batch_size,).
    dropout_rate : float
        Dropout probability (default: 0.5).
    normalize : bool
        If True, normalize and log-transform counts (default: True).
    target_sum : None | float
        Target total count per cell after normalization.
        Defaults to 1e4 (standard for scRNA-seq).

    Returns
    -------
    tuple[Float[Array, ""], Int[Array, ""], Int[Array, ""]]
        Tuple of (loss, total_samples, correct_predictions).
    """
    loss = nn_annotate_loss(
        params,
        key,
        x,
        y,
        dropout_rate=dropout_rate,
        normalize=normalize,
        target_sum=target_sum,
    )

    logits = nn_annotate(
        params,
        key,
        x,
        dropout_rate=dropout_rate,
        normalize=normalize,
        target_sum=target_sum,
        training=False,
    )

    preds = jnp.argmax(logits, axis=1)
    correct = jnp.sum(preds == y)
    total = y.shape[0]

    return loss, total, correct


class GinsengClassifier:
    """Wrapper class for cell type annotation with automatic key management.

    Parameters
    ----------
    n_genes : int
        Number of genes in input data.
    n_classes : int
        Number of cell type classes.
    hidden_dim : int
        Hidden dimension for attention mechanism (default: 256).
    dropout_rate : float
        Dropout rate during training (default: 0.5).
    normalize : bool
        Whether to normalize input data (default: True).
    target_sum : None | float
        Target sum for normalization (default: 1e4).
    seed : int
        Random seed for reproducibility (default: 42).

    Example
    -------
    >>> model = GinsengClassifier(n_genes=2000, n_classes=10)
    >>> # During training (JAX-compatible)
    >>> loss = model.loss(model.params, model._get_key(), x_batch, y_batch)
    >>> # For prediction (Standard usage)
    >>> logits = model.predict(x_test, training=False)
    """

    def __init__(
        self,
        n_genes: int,
        n_classes: int,
        hidden_dim: int = 256,
        dropout_rate: float = 0.5,
        normalize: bool = True,
        target_sum: None | float = None,
        seed: int = 42,
    ):
        self.n_genes = n_genes
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.normalize = normalize
        self.target_sum = target_sum if target_sum is not None else 1e4

        # PRNG state
        self.key = jax.random.key(seed)
        self.key, init_key = jax.random.split(self.key)

        # Parameters
        self.params = nn_annotate_init(init_key, n_genes, n_classes, hidden_dim)

    def _get_key(self) -> Array:
        """Get a new PRNG key and update internal state.

        Returns
        -------
        Array
            A new JAX PRNG key.
        """
        self.key, subkey = jax.random.split(self.key)
        return subkey

    def predict(
        self,
        x: Float[Array, "batch genes"],
        params: PyTree | None = None,
        key: Array | None = None,
        training: bool = False,
        return_attn: bool = False,
    ) -> (
        Float[Array, "batch classes"]
        | tuple[Float[Array, "batch classes"], Float[Array, "batch genes"]]
    ):
        """Generate predictions for input data.

        Parameters
        ----------
        x : Array
            Input gene expression matrix.
        params : PyTree, optional
            Model parameters. If None, uses internal self.params.
        key : Array, optional
            PRNG key. If None, uses a new key from self._get_key().
        training : bool
            Whether to use training mode (with dropout).
        return_attn : bool
            Whether to return attention weights.

        Returns
        -------
        Array or tuple
            Logits, or (logits, attention) if return_attn=True.
        """
        p = params if params is not None else self.params
        k = key if key is not None else self._get_key()

        return nn_annotate(
            p,
            k,
            x,
            dropout_rate=self.dropout_rate,
            normalize=self.normalize,
            target_sum=self.target_sum,
            return_attn=return_attn,
            training=training,
        )

    def loss(
        self,
        params: PyTree,
        key: Array,
        x: Float[Array, "batch genes"],
        y: Int[Array, "batch"],
    ) -> Float[Array, ""]:
        """Compute cross-entropy loss for a batch.

        This method is designed to be pure for JAX transformations.

        Parameters
        ----------
        params : PyTree
            Model parameters to differentiate against.
        key : Array
            PRNG key for dropout randomness.
        x : Array
            Input gene expression matrix.
        y : Array
            True class labels.

        Returns
        -------
        Array
            Scalar loss value.
        """
        return nn_annotate_loss(
            params,
            key,
            x,
            y,
            dropout_rate=self.dropout_rate,
            normalize=self.normalize,
            target_sum=self.target_sum,
        )

    def evaluate(
        self,
        x: Float[Array, "batch genes"],
        y: Int[Array, "batch"],
    ) -> tuple[float, float]:
        """Evaluate model on a batch and return loss and accuracy.

        Uses current internal model parameters.

        Parameters
        ----------
        x : Array
            Input gene expression matrix.
        y : Array
            True class labels.

        Returns
        -------
        tuple[float, float]
            (loss, accuracy) on the batch.
        """
        loss, total, correct = nn_annotate_evaluate(
            self.params,
            self._get_key(),
            x,
            y,
            dropout_rate=self.dropout_rate,
            normalize=self.normalize,
            target_sum=self.target_sum,
        )
        accuracy = float(correct) / float(total)
        return float(loss), accuracy
