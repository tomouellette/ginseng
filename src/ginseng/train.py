# Copyright (c) 2025, Tom Ouellette
# Licensed under the MIT License

import jax
import jax.numpy as jnp
import numpy as np

from dataclasses import dataclass, field
from jaxtyping import PyTree
from pathlib import Path
from tqdm import tqdm

from .augment import augment
from .data import GinsengDataset
from .nn import nn_annotate_init, nn_annotate_loss, nn_annotate_evaluate
from .opt import opt_init_adam, opt_adam_update


@dataclass
class GinsengTrainerSettings:
    """Training configuration settings for `GinsengTrainer`.

    Attributes
    ----------
    rate : float | None
        Probability of randomly masking input counts.
    lam_max : float | None
        Maximum mean of Poisson distribution for randomly adding counts.
    lower : int | None
        Minimum number of genes to randomly mask out.
    upper : int | None
        Maximum number of genes to randomly mask out.
    hidden_dim : int
        Number of hidden dimensions.
    dropout_rate : float
        Dropout probability applied during training.
    batch_size : int
        Number of samples per training batch.
    lr : float
        Learning rate for Adam optimizer.
    betas : tuple[float, float]
        Exponential decay rates for first and second moment estimates for ADam optimizer.
    eps : float
        Numerical stability constant for Adam optimizer.
    weight_decay : float
        L2 regularization factor for Adam optimizer.
    normalize : bool
        Whether to normalize the input count data.
    target_sum : float
        Target sum used for data normalization.
    holdout_fraction : float
        Fraction of the dataset to hold out for validation.
    balance_train : bool
        Whether to balance training samples across classes.
    group_level : bool
        Whether to apply grouping at the group level.
    group_mode : str
        Strategy for handling group balancing ('fraction' or 'loo').
    seed : int
        Random seed for reproducibility.
    """

    # Augmentation
    rate: float | None = None
    lam_max: float | None = None
    lower: int | None = None
    upper: int | None = None

    # Model
    hidden_dim: int = 256
    dropout_rate: float = 0.25

    # Optimization
    batch_size: int = 128
    lr: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01

    # Data
    normalize: bool = True
    target_sum: float = 1e4
    holdout_fraction: float = 0.2
    balance_train: bool = True
    group_level: bool = False
    group_mode: str = "fraction"

    # Randomness
    seed: int = 1


@dataclass
class GinsengLogger:
    """Logger for storing training and validation metrics across epochs.

    Attributes
    ----------
    epoch : list[int]
        List of epoch indices.
    train_loss : list[float]
        Training loss values for each epoch.
    holdout_loss : list[float]
        Holdout loss values for each epoch.
    holdout_accuracy : list[float]
        Holdout accuracy values for each epoch.
    """

    epoch: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    holdout_loss: list[float] = field(default_factory=list)
    holdout_accuracy: list[float] = field(default_factory=list)

    def update(
        self,
        epoch: int,
        train_loss: float,
        holdout_loss: float,
        holdout_accuracy: float,
    ):
        """Update the logger with new training and validation metrics.

        Parameters
        ----------
        epoch : int
            Current epoch index.
        train_loss : float
            Training loss at this epoch.
        holdout_loss : float
            Validation loss at this epoch.
        holdout_accuracy : float
            Validation accuracy at this epoch.
        """
        self.epoch.append(epoch)
        self.train_loss.append(train_loss)
        self.holdout_loss.append(holdout_loss)
        self.holdout_accuracy.append(holdout_accuracy)


@dataclass
class GinsengModelState:
    """State of the Ginseng model, including parameters and metadata.

    Attributes
    ----------
    params : PyTree
        Model parameters.
    genes : np.ndarray
        Gene names used during model training.
    label_keys : np.ndarray
        Keys denoting integer identifier for each label.
    label_values : np.ndarray
        Values denoting original string/integer identifiers for each label.
    normalize : bool
        If True, model was trained on normalized data
    target_sum : float
        Target sum used for normalization.
    training : bool
        If True, model weighted should not be frozen.
    """

    params: PyTree
    genes: np.ndarray
    label_keys: np.ndarray
    label_values: np.ndarray
    normalize: bool
    target_sum: float
    training: bool = False


def GinsengTrainer(
    dataset: GinsengDataset | str,
    settings: GinsengTrainerSettings = GinsengTrainerSettings(),
    epochs: int = 10,
    silent: bool = False,
) -> tuple[GinsengLogger, GinsengModelState]:
    """Train a neural network classiifier on a `GinsengDataset`.

    Parameters
    ----------
    dataset : GinsengDataset | str
        GinsengDataset or path to GinsengDataset
    settings : GinsengTrainerSettings
        Training configuration including model, optimization, and augmentation parameters.
    epochs : int
        Number of training epochs to run.
    silent : bool
        If True, suppresses training progress output.

    Returns
    -------
    logger : GinsengLogger
        Training log with loss and accuracy metrics across epochs.
    state : GinsengModelState
        Final trained model state including parameters and metadata.
    """
    if isinstance(dataset, (str, Path)):
        dataset = GinsengDataset(dataset)

    if not isinstance(dataset, GinsengDataset):
        raise TypeError(
            "`dataset` must be a `GinsengDataset` or path to `GinsengDataset`."
        )

    @jax.jit
    def train_step(params, opt_state, key, x, y):
        loss, grads = jax.value_and_grad(nn_annotate_loss)(
            params,
            key,
            x,
            y,
            dropout_rate=settings.dropout_rate,
            normalize=settings.normalize,
            target_sum=settings.target_sum,
            training=True,
        )

        new_params, new_opt_state = opt_adam_update(
            grads,
            params,
            opt_state,
            lr=settings.lr,
            eps=settings.eps,
            betas=settings.betas,
            weight_decay=settings.weight_decay,
        )
        return new_params, new_opt_state, loss

    rng = np.random.default_rng(settings.seed)

    key = jax.random.key(settings.seed)

    params = nn_annotate_init(
        key,
        n_genes=dataset.n_genes,
        n_classes=dataset.n_labels,
        hidden_dim=settings.hidden_dim,
    )

    opt_state = opt_init_adam(params)

    logger = GinsengLogger()

    dataset.make_holdout(
        holdout_fraction=settings.holdout_fraction,
        rng=rng,
        group_level=settings.group_level,
        group_mode=settings.group_mode,
    )

    for epoch in range(epochs):
        train_iterator = dataset.iter_batches(
            batch_size=settings.batch_size,
            shuffle=True,
            rng=rng,
            split="train",
            balance_train=settings.balance_train,
        )

        losses = []
        with tqdm(
            train_iterator,
            desc=f"[ginseng] Train {epoch+1}/{epochs}",
            unit="step",
            disable=silent,
        ) as progress:
            for step, (x, y) in enumerate(progress):
                key, subkey = jax.random.split(key)

                x = jnp.asarray(x)
                y = jnp.asarray(y)

                x = augment(
                    key,
                    x,
                    settings.rate,
                    settings.lam_max,
                    settings.lower,
                    settings.upper,
                )

                params, opt_state, loss = train_step(params, opt_state, subkey, x, y)

                progress.set_postfix(loss=f"{float(loss):.4f}")
                losses.append(float(loss))

        holdout_iterator = dataset.iter_batches(
            batch_size=settings.batch_size,
            shuffle=False,
            rng=rng,
            split="holdout",
        )

        holdout_losses, holdout_correct, holdout_total = [], 0, 0
        with tqdm(
            holdout_iterator,
            desc=f"[ginseng] Holdout {epoch+1}/{epochs}",
            unit="step",
            disable=silent,
        ) as progress:
            for step, (x, y) in enumerate(progress):
                key, subkey = jax.random.split(key)
                holdout_loss, total, correct = nn_annotate_evaluate(
                    params,
                    subkey,
                    jnp.asarray(x),
                    jnp.asarray(y),
                    dropout_rate=settings.dropout_rate,
                    normalize=settings.normalize,
                    target_sum=settings.target_sum,
                )

                holdout_losses.append(float(holdout_loss))
                holdout_correct += correct
                holdout_total += total

        accuracy = holdout_correct / holdout_total if holdout_total > 0 else 0.0

        logger.update(
            epoch=epoch,
            train_loss=np.mean(losses).item(),
            holdout_loss=np.mean(holdout_losses).item(),
            holdout_accuracy=accuracy,
        )

    model_state = GinsengModelState(
        params,
        genes=dataset.root.attrs["genes"][:],
        label_keys=dataset.root.attrs["label_keys"][:],
        label_values=dataset.root.attrs["label_values"][:],
        normalize=settings.normalize,
        target_sum=settings.target_sum,
        training=False,
    )

    return logger, model_state
