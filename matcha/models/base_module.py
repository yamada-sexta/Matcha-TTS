"""
This is a base module for Matcha-TTS models using plain PyTorch.
"""
import inspect
import os
from abc import ABC
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from matcha import utils
from matcha.utils.utils import plot_tensor

log = utils.get_pylogger(__name__)


class BaseModule(nn.Module, ABC):
    """Base class for Matcha-TTS models without PyTorch Lightning dependency."""

    def __init__(self):
        super().__init__()
        # Store hyperparameters manually (replaces save_hyperparameters)
        self._hparams: Dict[str, Any] = {}

    @property
    def hparams(self) -> Dict[str, Any]:
        """Returns the hyperparameters dictionary."""
        return self._hparams

    def save_hyperparameters(self, *args, ignore=None, logger=True):
        """
        Save hyperparameters to self._hparams.
        This mimics Lightning's save_hyperparameters but stores them in a simple dict.
        """
        frame = inspect.currentframe().f_back
        init_args = {}

        # Get the signature of the __init__ method
        if hasattr(self, "__init__"):
            sig = inspect.signature(self.__init__)
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                if ignore and param_name in ignore:
                    continue
                # Try to get the value from the caller's local variables
                if param_name in frame.f_locals:
                    init_args[param_name] = frame.f_locals[param_name]

        self._hparams.update(init_args)

    def update_data_statistics(self, data_statistics):
        """Update mel normalization statistics."""
        if data_statistics is None:
            data_statistics = {
                "mel_mean": 0.0,
                "mel_std": 1.0,
            }

        self.register_buffer("mel_mean", torch.tensor(data_statistics["mel_mean"]))
        self.register_buffer("mel_std", torch.tensor(data_statistics["mel_std"]))

    def get_losses(self, batch):
        """Compute losses for a batch."""
        x, x_lengths = batch["x"], batch["x_lengths"]
        y, y_lengths = batch["y"], batch["y_lengths"]
        spks = batch["spks"]

        dur_loss, prior_loss, diff_loss, *_ = self(
            x=x,
            x_lengths=x_lengths,
            y=y,
            y_lengths=y_lengths,
            spks=spks,
            out_size=self.out_size,
            durations=batch["durations"],
        )
        return {
            "dur_loss": dur_loss,
            "prior_loss": prior_loss,
            "diff_loss": diff_loss,
        }

    def save_checkpoint(self, filepath: str, optimizer=None, scheduler=None, epoch=0, global_step=0):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "state_dict": self.state_dict(),
            "hyper_parameters": self._hparams,
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        log.info(f"Checkpoint saved to {filepath}")

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, map_location=None, **kwargs):
        """
        Load model from checkpoint file.
        Supports both new format and legacy Lightning format.
        """
        if map_location is None:
            map_location = torch.device("cpu")

        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

        # Handle both new and legacy (Lightning) checkpoint formats
        if "hyper_parameters" in checkpoint:
            hparams = checkpoint["hyper_parameters"]
        elif "hparams" in checkpoint:
            hparams = checkpoint["hparams"]
        elif "datamodule_hyper_parameters" in checkpoint:
            # Legacy Lightning format
            hparams = {}
            if "hyper_parameters" in checkpoint:
                hparams = checkpoint["hyper_parameters"]
        else:
            hparams = {}

        # Override with any provided kwargs
        hparams.update(kwargs)

        # Remove non-model parameters that might have been saved
        hparams.pop("optimizer", None)
        hparams.pop("scheduler", None)

        # Create model instance
        model = cls(**hparams)

        # Load state dict
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Handle Lightning-style state dict keys (may have "model." prefix)
        new_state_dict = {}
        for key, value in state_dict.items():
            # Remove common prefixes from Lightning checkpoints
            new_key = key
            if key.startswith("model."):
                new_key = key[6:]
            new_state_dict[new_key] = value

        # Try to load state dict, allowing for missing/unexpected keys
        try:
            model.load_state_dict(new_state_dict, strict=True)
        except RuntimeError:
            # Fall back to non-strict loading
            model.load_state_dict(new_state_dict, strict=False)
            log.warning("Loaded checkpoint with non-strict state dict matching")

        # Store loaded epoch for potential use
        model.ckpt_loaded_epoch = checkpoint.get("epoch", 0)

        return model

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Hook called when loading checkpoint."""
        self.ckpt_loaded_epoch = checkpoint.get("epoch", 0)
