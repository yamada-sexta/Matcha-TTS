"""
Legacy instantiators module.
These functions were used with PyTorch Lightning and are kept for backwards compatibility.
The plain PyTorch training script does not use these.
"""
from typing import Any, List

import hydra
from omegaconf import DictConfig

from matcha.utils import pylogger

log = pylogger.get_pylogger(__name__)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Any]:
    """Instantiates callbacks from config (legacy function).

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: List[Any] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")  # pylint: disable=protected-access
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Any]:
    """Instantiates loggers from config (legacy function).

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    logger: List[Any] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")  # pylint: disable=protected-access
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger
