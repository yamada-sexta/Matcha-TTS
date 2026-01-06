"""
Legacy logging utilities.
These functions were used with PyTorch Lightning and are kept for backwards compatibility.
The plain PyTorch training script handles logging differently.
"""
from typing import Any, Dict

from omegaconf import OmegaConf

from matcha.utils import pylogger

log = pylogger.get_pylogger(__name__)


def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved (legacy function).

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The model.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["model/params/non_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg.get("trainer")

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    log.info(f"Hyperparameters: {hparams}")
