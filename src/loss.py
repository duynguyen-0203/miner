from abc import ABC, abstractmethod

from torch import Tensor
import torch.nn.functional as torch_f


class AbstractLoss(ABC):
    @abstractmethod
    def compute(self, *args, **kwargs):
        pass


class Loss(AbstractLoss):
    def __init__(self, criterion):
        self._criterion = criterion

    def compute(self, logits: Tensor, labels: Tensor):
        """
        Compute batch loss

        Args:
            logits: tensor of shape ``(batch_size, npratio + 1)``
            labels: a one-hot tensor of shape ``(batch_size, npratio + 1)``

        Returns:
            Loss value
        """
        targets = labels.argmax(dim=1)
        loss = self._criterion(logits, targets)

        return loss

    @staticmethod
    def compute_eval_loss(logits: Tensor, labels: Tensor):
        """
        Compute loss for evaluation phase

        Args:
            logits: tensor of shape ``(batch_size, 1)``
            labels: a binary tensor of shape ``(batch_size, 1)``

        Returns:
            Loss value
        """
        loss = -(torch_f.logsigmoid(logits) * labels).sum()

        return loss.item()
