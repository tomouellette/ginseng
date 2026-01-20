# Copyright (c) 2026, Tom Ouellette
# Licensed under the MIT License

import jax
import jax.numpy as jnp
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

from ginseng.data.dataset import GinsengDataset
from ginseng.model.nn import GinsengClassifier
from ginseng.model.state import state_from_classifier_trainer

from .augment import augment
from .logger import GinsengLogger
from .opt import opt_init_adam, opt_adam_update


@dataclass
class GinsengClassifierTrainerSettings:
    """Training configuration settings for `GinsengClassifierTrainer`."""

    # Augmentation
    rate: float | None = None
    lam_max: float | None = None
    lower: int | None = None
    upper: int | None = None

    # Model
    hidden_dim: int = 64
    dropout_rate: float = 0.2

    # Optimization
    batch_size: int = 128
    lr: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-9
    weight_decay: float = 0.01

    # Data
    normalize: bool = True
    target_sum: float = 1e4
    holdout_fraction: float = 0.1
    balance_train: bool = True
    group_level: bool = False
    group_mode: str = "fraction"

    # Randomness
    seed: int = 1


class GinsengClassifierTrainer:
    """
    Trainer class for orchestrating the training of a GinsengClassifier.
    """

    def __init__(
        self,
        dataset: GinsengDataset | str | Path,
        settings: GinsengClassifierTrainerSettings = GinsengClassifierTrainerSettings(),
    ):
        """
        Initialize the trainer with a dataset and configuration.

        Parameters
        ----------
        dataset : GinsengDataset | str | Path
            The dataset to train on.
        settings : GinsengClassifierTrainerSettings
            Training configuration including model and optimization parameters (default : GinsengClassifierTrainerSettings()).
        """
        if isinstance(dataset, (str, Path)):
            self.dataset = GinsengDataset(dataset)
        else:
            self.dataset = dataset

        self.settings = settings
        self.logger = GinsengLogger()
        self.rng = np.random.default_rng(settings.seed)

        # Initialize the Classifier
        self.model = GinsengClassifier(
            n_genes=self.dataset.n_genes,
            n_classes=len(self.dataset.label_names),
            hidden_dim=settings.hidden_dim,
            dropout_rate=settings.dropout_rate,
            normalize=settings.normalize,
            target_sum=settings.target_sum,
            seed=settings.seed,
        )

        # Initialize optimizer state
        self.opt_state = opt_init_adam(self.model.params)

    def _train_step(
        self,
        params: dict,
        opt_state: dict,
        key: jax.random.PRNGKey,
        x: jnp.ndarray,
        y: jnp.ndarray,
    ) -> tuple[dict, dict, jnp.ndarray]:
        """Internal training step to update parameters."""
        loss, grads = jax.value_and_grad(self.model.loss)(params, key, x, y)

        new_params, new_opt_state = opt_adam_update(
            grads,
            params,
            opt_state,
            lr=self.settings.lr,
            eps=self.settings.eps,
            betas=self.settings.betas,
            weight_decay=self.settings.weight_decay,
        )
        return new_params, new_opt_state, loss

    def fit(self, epochs: int = 10, silent: bool = False) -> GinsengClassifier:
        """
        Execute the training loop.

        Parameters
        ----------
        epochs : int
            Number of training epochs to run (default : 10).
        silent : bool
            If True, suppresses training progress output (default : False).

        Returns
        -------
        GinsengClassifier
            The trained classifier with updated parameters.
        """
        # Prepare data splits
        self.dataset.make_split(
            fraction=self.settings.holdout_fraction,
            stratify_group=self.settings.group_level,
            seed=self.settings.seed,
        )

        # Compile the train step
        jit_train_step = jax.jit(self._train_step)

        for epoch in range(epochs):
            # Training phase
            train_loss = self._run_epoch(epoch, epochs, jit_train_step, silent)

            if self.settings.holdout_fraction == 0.0:
                holdout_loss, accuracy = float("nan"), float("nan")
            else:
                holdout_loss, accuracy = self._validate(epoch, epochs, silent)

            # Logging
            self.logger.update(epoch, train_loss, holdout_loss, accuracy)
            self.logger.report(silent=silent)

        return self.model, state_from_classifier_trainer(self)

    def _run_epoch(self, epoch: int, epochs: int, train_step_fn, silent: bool) -> float:
        """Run a single training epoch."""
        train_iter = self.dataset.stream(
            batch_size=self.settings.batch_size,
            split="train",
            balance_labels=self.settings.balance_train,
            shuffle=True,
        )

        epoch_losses = []
        pbar = tqdm(
            train_iter,
            desc=f"[ginseng] Train {epoch + 1}/{epochs}",
            unit=" batch",
            disable=silent,
        )

        for x, y in pbar:
            # Prepare data
            x_jax = jnp.asarray(x)
            y_jax = jnp.asarray(y)

            # Augmentation
            x_aug = augment(
                self.model._get_key(),
                x_jax,
                self.settings.rate,
                self.settings.lam_max,
                self.settings.lower,
                self.settings.upper,
            )

            # Update
            self.model.params, self.opt_state, loss = train_step_fn(
                self.model.params, self.opt_state, self.model._get_key(), x_aug, y_jax
            )

            loss_val = float(loss)
            epoch_losses.append(loss_val)
            pbar.set_postfix(loss=f"{loss_val:.4f}")

        return np.mean(epoch_losses).item()

    def _validate(self, epoch: int, epochs: int, silent: bool) -> tuple[float, float]:
        """Run validation on the holdout split."""
        holdout_iter = self.dataset.stream(
            batch_size=self.settings.batch_size,
            split="test",
            shuffle=False,
        )

        losses, accuracies = [], []
        for x, y in holdout_iter:
            loss, acc = self.model.evaluate(jnp.asarray(x), jnp.asarray(y))
            losses.append(loss)
            accuracies.append(acc)

        return np.mean(losses).item(), np.mean(accuracies).item()
