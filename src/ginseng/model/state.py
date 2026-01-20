# Copyright (c) 2026, Tom Ouellette
# Licensed under the MIT License

import numpy as np

from dataclasses import dataclass
from jaxtyping import PyTree


@dataclass
class GinsengClassifierState:
    """Complete state of a trained Ginseng model.

    All information needed to save, load, and use a trained model.

    Attributes
    ----------
    params : PyTree
        Model parameters (weights and biases).
    genes : np.ndarray
        Gene names in the exact order expected by the model.
    label_keys : np.ndarray
        Label names (e.g., ['T-cell', 'B-cell', 'Macrophage']).
    label_values : np.ndarray
        Integer values corresponding to each label (e.g., [0, 1, 2]).
    n_genes : int
        Number of genes.
    n_classes : int
        Number of classes.
    hidden_dim : int
        Hidden dimension used in attention mechanism.
    normalize : bool
        Whether input data should be normalized.
    target_sum : float
        Target sum for normalization.
    dropout_rate : float
        Dropout rate used during training.
    training : bool
        Whether weights should be frozen (False after training).
    """

    params: PyTree
    genes: np.ndarray
    label_keys: np.ndarray
    label_values: np.ndarray
    n_genes: int
    n_classes: int
    hidden_dim: int
    normalize: bool
    target_sum: float
    dropout_rate: float
    training: bool = False


def classifier_from_state(state: GinsengClassifierState) -> "GinsengClassifier":
    """Create a GinsengClassifier from a loaded state.

    Parameters
    ----------
    state : GinsengClassifierState
        Loaded model state.

    Returns
    -------
    GinsengClassifier
        Model ready for inference.
    """
    from ginseng.model.nn import GinsengClassifier

    model = GinsengClassifier(
        n_genes=state.n_genes,
        n_classes=state.n_classes,
        hidden_dim=state.hidden_dim,
        dropout_rate=state.dropout_rate,
        normalize=state.normalize,
        target_sum=state.target_sum,
    )
    model.params = state.params

    return model


def state_from_classifier_trainer(
    trainer: "GinsengClassifierTrainer",
) -> GinsengClassifierState:
    """Create a GinsengClassifierState from a trainer after .fit().

    Parameters
    ----------
    trainer : GinsengClassifierTrainer
        Trainer instance with a trained model.

    Returns
    -------
    GinsengClassifierState
        Complete model state.

    Example
    -------
    >>> trainer = GinsengClassifierTrainer(dataset, settings)
    >>> trainer.fit(epochs=50)
    >>> state = state_from_trainer(trainer)
    >>> save_model(state, "./models/my_classifier.h5")
    """
    return GinsengClassifierState(
        params=trainer.model.params,
        genes=trainer.dataset.gene_names,
        label_keys=trainer.dataset.label_names,
        label_values=np.arange(len(trainer.dataset.label_names)),
        n_genes=trainer.dataset.n_genes,
        n_classes=len(trainer.dataset.label_names),
        hidden_dim=trainer.settings.hidden_dim,
        normalize=trainer.settings.normalize,
        target_sum=trainer.settings.target_sum,
        dropout_rate=trainer.settings.dropout_rate,
        training=False,
    )
