import random
from typing import List, Union

import numpy as np
import torch
from torch import Tensor


def pairwise_cosine_similarity(x: Tensor, y: Tensor, zero_diagonal: bool = False) -> Tensor:
    r"""
    Calculates the pairwise cosine similarity matrix

    Args:
        x: tensor of shape ``(batch_size, M, d)``
        y: tensor of shape ``(batch_size, N, d)``
        zero_diagonal: determines if the diagonal of the distance matrix should be set to zero

    Returns:
        A tensor of shape ``(batch_size, M, N)``
    """
    x_norm = torch.linalg.norm(x, dim=2, keepdim=True)
    y_norm = torch.linalg.norm(y, dim=2, keepdim=True)
    distance = torch.matmul(torch.div(x, x_norm), torch.div(y, y_norm).permute(0, 2, 1))
    if zero_diagonal:
        assert x.shape[1] == y.shape[1]
        mask = torch.eye(x.shape[1]).repeat(x.shape[0], 1, 1).bool().to(distance.device)
        distance.masked_fill_(mask, 0)

    return distance


def load_embed(path: str):

    return None


def get_device() -> str:
    r"""
    Return the device available for execution

    Returns:
        ``cpu`` for CPU or ``cuda`` for GPU
    """
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def to_device(batch: dict, device: object) -> dict:
    r"""
    Convert a batch to the specified device

    Args:
        batch: the batch needs to be converted.
        device: GPU or CPU.

    Returns:
        A batch after converting
    """
    converted_batch = dict()
    for key in batch.keys():
        converted_batch[key] = batch[key].to(device)

    return converted_batch


def convert_arg_line_to_args(arg_line):
    r"""
    Convert a line of arguments into individual arguments

    Args:
        arg_line: a string read from the argument file.

    Returns:
        A list of arguments parsed from ``arg_line``
    """
    arg_line = arg_line.strip()
    if arg_line.startswith('#') or arg_line == '':
        return []
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


def set_seed(seed: int):
    r"""
    Sets the seed for generating random numbers

    Args:
        seed: seed value.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def padded_stack(tensors: Union[List[Tensor], List[List]], padding: int = 0):
    r"""
    Pad a list of variable length Tensors with ``padding``

    Args:
        tensors: list of variable length sequences.
        padding: value for padded elements. Default: 0.

    Returns:
        Padded sequences
    """
    if type(tensors[0]) == list:
        tensors = [torch.tensor(tensor) for tensor in tensors]
    n_dim = len(list(tensors[0].shape))
    max_shape = [max([tensor.shape[d] for tensor in tensors]) for d in range(n_dim)]
    padded_tensors = []

    for tensor in tensors:
        extended_tensor = expand_tensor(tensor, max_shape, fill=padding)
        padded_tensors.append(extended_tensor)

    return torch.stack(padded_tensors)


def expand_tensor(tensor: Tensor, extended_shape: List[int], fill: int = 0):
    r"""
    Expand a tensor to ``extended_shape``

    Args:
        tensor: tensor to expand.
        extended_shape: new shape.
        fill: value for padded elements. Default: 0.

    Returns:
        An expanded tensor
    """
    tensor_shape = tensor.shape

    expanded_tensor = torch.zeros(extended_shape, dtype=tensor.dtype).to(tensor.device)
    expanded_tensor = expanded_tensor.fill_(fill)

    if len(tensor_shape) == 1:
        expanded_tensor[:tensor_shape[0]] = tensor
    elif len(tensor_shape) == 2:
        expanded_tensor[:tensor_shape[0], :tensor_shape[1]] = tensor
    elif len(tensor_shape) == 3:
        expanded_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2]] = tensor
    elif len(tensor_shape) == 4:
        expanded_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2], :tensor_shape[3]] = tensor

    return expanded_tensor
