# Copyright (c) 2026, Tom Ouellette
# Licensed under the MIT License

from dataclasses import dataclass, field


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

    def report(self, silent: bool = False, flush: bool = True) -> None:
        """Print most recent result to standard output.

        Parameters
        ----------
        silent : bool
            If True, suppresses report output.
        flush : bool
            If True, write report to output immediately.

        Returns
        -------
        None
            Output is printed to standard output.
        """
        has_train = len(self.train_loss) > 0
        has_holdout = len(self.holdout_loss) > 0

        msg = f"[ginseng] Epoch {self.epoch[-1] + 1} report | "
        if has_train:
            msg += f" Training loss: {self.train_loss[-1]:.3e} | "

        if has_holdout:
            msg += f"Holdout loss: {self.holdout_loss[-1]:.3e} | "
            msg += f"Holdout accuracy: {self.holdout_accuracy[-1]:.3e} |"

        if has_train or has_holdout:
            print(msg, flush=flush)
