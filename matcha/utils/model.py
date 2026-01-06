""" from https://github.com/jaywalnut310/glow-tts """

from typing import Optional
from typing import Union

import numpy as np
import numpy.typing as npt
import torch


def sequence_mask(length: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    if max_length is None:
        max_length = int(length.max().item())
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return (x.unsqueeze(0) < length.unsqueeze(1)).to(length.dtype)


def fix_len_compatibility(
    length: Union[int, torch.Tensor], num_downsamplings_in_unet: int = 2
) -> Union[int, torch.Tensor]:
    factor = torch.scalar_tensor(2).pow(num_downsamplings_in_unet)
    length = (length / factor).ceil() * factor
    if not torch.onnx.is_in_onnx_export():
        return int(length.int().item())
    else:
        return length


def convert_pad_shape(pad_shape: list[list[int]]) -> list[int]:
    inverted_shape = pad_shape[::-1]
    flattened = [item for sublist in inverted_shape for item in sublist]
    return flattened


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    device = duration.device

    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)
    path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - torch.nn.functional.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path * mask
    return path


def duration_loss(logw: torch.Tensor, logw_: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    loss = torch.sum((logw - logw_) ** 2) / torch.sum(lengths)
    return loss


def normalize(
    data: torch.Tensor,
    mu: Union[float, int, list[float], torch.Tensor, npt.NDArray[np.floating]],
    std: Union[float, int, list[float], torch.Tensor, npt.NDArray[np.floating]],
) -> torch.Tensor:
    if not isinstance(mu, (float, int)):
        if isinstance(mu, list):
            mu = torch.tensor(mu, dtype=data.dtype, device=data.device)
        elif isinstance(mu, torch.Tensor):
            mu = mu.to(data.device)
        elif isinstance(mu, np.ndarray):
            mu = torch.from_numpy(mu).to(data.device)
        mu = mu.unsqueeze(-1)

    if not isinstance(std, (float, int)):
        if isinstance(std, list):
            std = torch.tensor(std, dtype=data.dtype, device=data.device)
        elif isinstance(std, torch.Tensor):
            std = std.to(data.device)
        elif isinstance(std, np.ndarray):
            std = torch.from_numpy(std).to(data.device)
        std = std.unsqueeze(-1)

    return (data - mu) / std


def denormalize(
    data: torch.Tensor,
    mu: Union[float, list[float], torch.Tensor, npt.NDArray[np.floating]],
    std: Union[float, list[float], torch.Tensor, npt.NDArray[np.floating]],
) -> torch.Tensor:
    if not isinstance(mu, float):
        if isinstance(mu, list):
            mu = torch.tensor(mu, dtype=data.dtype, device=data.device)
        elif isinstance(mu, torch.Tensor):
            mu = mu.to(data.device)
        elif isinstance(mu, np.ndarray):
            mu = torch.from_numpy(mu).to(data.device)
        mu = mu.unsqueeze(-1)

    if not isinstance(std, float):
        if isinstance(std, list):
            std = torch.tensor(std, dtype=data.dtype, device=data.device)
        elif isinstance(std, torch.Tensor):
            std = std.to(data.device)
        elif isinstance(std, np.ndarray):
            std = torch.from_numpy(std).to(data.device)
        std = std.unsqueeze(-1)

    return data * std + mu
