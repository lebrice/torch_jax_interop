import functools
import logging

import jax
import torch

from .types import T


def log_once(logger: logging.Logger, message: str, level: int):
    logger.log(level=level, msg=message, stacklevel=2)


log_once = functools.cache(log_once)


def is_channels_first(shape: torch.Size | tuple[int, ...]) -> bool:
    if len(shape) == 4:
        return is_channels_first(shape[1:])
    if len(shape) != 3:
        return False
    return (
        shape[0] in (1, 3) and shape[1] not in {1, 3} and shape[2] not in {1, 3}
    ) or (shape[0] < min(shape[1], shape[2]))


def is_channels_last(shape: torch.Size | tuple[int, ...]) -> bool:
    if len(shape) == 4:
        return is_channels_last(shape[1:])
    if len(shape) != 3:
        return False
    return (
        shape[2] in (1, 3) and shape[0] not in {1, 3} and shape[1] not in {1, 3}
    ) or (shape[2] < min(shape[0], shape[1]))


def to_channels_last(tensor: T) -> T:
    shape = tuple(tensor.shape)
    assert len(shape) == 3 or len(shape) == 4
    if not is_channels_first(shape):
        return tensor
    if isinstance(tensor, jax.Array):
        if len(shape) == 3:
            return tensor.transpose(1, 2, 0)
        return tensor.transpose(0, 2, 3, 1)
    else:
        if len(shape) == 3:
            return tensor.permute(1, 2, 0)
        return tensor.permute(0, 2, 3, 1)


def to_channels_first(tensor: T) -> T:
    shape = tuple(tensor.shape)
    assert len(shape) == 3 or len(shape) == 4
    if is_channels_first(shape):
        return tensor
    if not is_channels_last(shape):
        return tensor
    if isinstance(tensor, jax.Array):
        if len(shape) == 3:
            # [H, W, C] -> [C, H, W]
            return tensor.transpose(2, 0, 1)
        # [B, H, W, C] -> [B, C, H, W]
        return tensor.transpose(0, 3, 1, 2)
    else:
        if len(shape) == 3:
            # [H, W, C] -> [C, H, W]
            return tensor.permute(2, 0, 1)
        return tensor.permute(0, 3, 1, 2)
