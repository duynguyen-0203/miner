import random
from typing import List, Union

import numpy as np
import torch
from torch import Tensor


def pairwise_cosine_similarity(x: Tensor, y: Tensor) -> Tensor:
    r"""
    Calculates the pairwise cosine similarity matrix

    Args:
        x: tensor of shape ``(batch_size, M, d)``
        y: tensor of shape ``(batch_size, N, d)``

    Returns:
        A tensor of shape ``(batch_size, M, N)``
    """
    x_norm = torch.linalg.norm(x, dim=2, keepdim=True)
    y_norm = torch.linalg.norm(y, dim=2, keepdim=True)
    result = torch.matmul(torch.div(x, x_norm), torch.div(y, y_norm).permute(0, 2, 1))

    return result


def get_device() -> str:
    """
    Return the device available for execution

    Returns:
        ``cpu`` for CPU or ``cuda`` for GPU
    """
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def to_device(batch: dict, device: object) -> dict:
    """
    Convert a batch to the specified device

    Args:
        batch: The batch needs to be converted
        device: GPU or CPU

    Returns:
        A batch after converting
    """
    converted_batch = dict()
    for key in batch.keys():
        converted_batch[key] = batch[key].to(device)

    return converted_batch


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


def set_seed(seed: int):
    """
    Sets the seed for generating random numbers

    Args:
        seed: Seed value

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def padded_stack(tensors: Union[List[Tensor], List[List]], padding: int = 0):
    if type(tensors[0]) == list:
        tensors = [torch.tensor(tensor) for tensor in tensors]
    n_dim = len(list(tensors[0].shape))
    max_shape = [max([tensor.shape[d] for tensor in tensors]) for d in range(n_dim)]
    padded_tensors = []

    for tensor in tensors:
        extended_tensor = extend_tensor(tensor, max_shape, fill=padding)
        padded_tensors.append(extended_tensor)

    return torch.stack(padded_tensors)


def extend_tensor(tensor: Tensor, extended_shape: List[int], fill: int = 0):
    tensor_shape = tensor.shape

    extended_tensor = torch.zeros(extended_shape, dtype=tensor.dtype).to(tensor.device)
    extended_tensor = extended_tensor.fill_(fill)

    if len(tensor_shape) == 1:
        extended_tensor[:tensor_shape[0]] = tensor
    elif len(tensor_shape) == 2:
        extended_tensor[:tensor_shape[0], :tensor_shape[1]] = tensor
    elif len(tensor_shape) == 3:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2]] = tensor
    elif len(tensor_shape) == 4:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2], :tensor_shape[3]] = tensor

    return extended_tensor
