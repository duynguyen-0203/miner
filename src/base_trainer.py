from abc import ABC, abstractmethod
from datetime import datetime
import logging
import math
import os
import sys

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src import logger_utils, utils
from src.entities import Dataset


class BaseTrainer(ABC):
    def __init__(self, args):
        self.args = args
        self._init_logger()
        self._device = torch.device(utils.get_device())
        if utils.get_device() == 'cuda':
            self._logger.info(f'GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}')
        utils.set_seed(args.seed)

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def _collate_fn(self, batch):
        pass

    def _init_logger(self):
        r"""
        Initialize all loggers

        Returns:
            None
        """
        time = str(datetime.now()).replace(' ', '_').replace(':', '-')[:-7]
        log_formatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s [%(levelname)-5.5s] %(message)s')
        self._logger = logging.getLogger()
        logger_utils.reset_logger(self._logger)

        if self.args.mode == 'train':
            self._path = os.path.join(self.args.train_path, time)
            self._log_path = os.path.join(self._path, 'log')
            os.makedirs(self._path, exist_ok=True)
            os.makedirs(self._log_path, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(self._log_path, 'all.log'))

            # Create csv log
            self._loss_csv = os.path.join(self._log_path, 'loss.csv')
            logger_utils.create_csv(self._loss_csv,
                                    header=['global_step', 'epoch', 'train_loss', 'current_lr'])

            eval_header = []
            if 'loss' in self.args.evaluation_info:
                eval_header.append('loss')
            if 'metrics' in self.args.evaluation_info:
                eval_header.extend(self.args.metrics)

            self._eval_csv = os.path.join(self._log_path, 'eval.csv')
            logger_utils.create_csv(self._eval_csv, header=['global_step'] + eval_header)
            self._eval_epoch = os.path.join(self._log_path, 'epoch.csv')
            logger_utils.create_csv(self._eval_epoch, header=['global_step', 'train_loss'] + eval_header)

            # Create tensorboard log
            os.makedirs(os.path.join(self._path, self.args.tensorboard_path), exist_ok=True)
            self._writer = SummaryWriter(os.path.join(self._path, self.args.tensorboard_path))
        else:
            self._path = os.path.join(self.args.eval_path, time)
            os.makedirs(self._path, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(self._path, 'all.log'))

        file_handler.setFormatter(log_formatter)
        self._logger.addHandler(file_handler)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_formatter)
        self._logger.addHandler(console_handler)
        self._logger.setLevel(logging.INFO)

    def _log_arguments(self):
        r"""
        Log all arguments

        Returns:
            None
        """
        logger_utils.log_json(path=self._path, data=self.args, name='args')

    def _log_dataset(self, train_dataset: Dataset, valid_dataset: Dataset):
        r"""
        Log information about the dataset

        Args:
            train_dataset: training dataset.
            valid_dataset: validation dataset.

        Returns:
            None
        """
        self._logger.info(f'Dataset: {self.args.data_name}')
        self._logger.info(f'Train dataset: {len(train_dataset)} samples')
        self._logger.info(f'Validation dataset: {len(valid_dataset)} samples')

    def _log_train_step(self, scheduler, loss, global_step):
        r"""
        Log each step

        Args:
            scheduler:
            loss:
            global_step:

        Returns:
            None
        """
        lr = scheduler.get_last_lr()
        self._writer.add_scalar('Training loss per iteration', loss, global_step=global_step)
        self._writer.add_scalar('Learning rate per iteration', float(lr[0]), global_step=global_step)

    def _log_train(self, scheduler, loss, global_step, epoch):
        r"""
        Log every ``logging_steps`` steps

        Args:
            scheduler:
            loss:
            global_step:
            epoch:

        Returns:
            None
        """
        lr = scheduler.get_last_lr()
        data = [global_step, epoch, loss, float(lr[0])]
        logger_utils.log_csv(self._loss_csv, data)
        self._logger.info(f'Training loss at global step {global_step}: {loss}')
        self._writer.add_scalar('Training loss per logging step', loss, global_step=global_step)

    def _log_epoch(self, train_loss: float, valid_loss: float, scores: dict, epoch: int):
        r"""
        Log at the end of each epoch

        Args:
            train_loss:
            valid_loss:
            scores:
            epoch:

        Returns:
            None
        """
        self._logger.info(f'------  Epoch {epoch} Summary  ------')
        self._logger.info(f'Training loss: {train_loss}')
        data = [epoch, train_loss]
        self._writer.add_scalar('Training loss per epoch', train_loss, global_step=epoch)
        if 'loss' in self.args.evaluation_info:
            data.append(valid_loss)
            self._logger.info(f'Validation loss: {valid_loss}')
            self._writer.add_scalar('Validation loss per epoch', valid_loss, global_step=epoch)
        if 'metrics' in self.args.evaluation_info:
            for metric, score in scores.items():
                self._logger.info(f'{metric}: {score}')
            data.extend(list(scores.values()))
            self._writer.add_scalar('Validation AUC score per epoch', scores['auc'], global_step=epoch)
        logger_utils.log_csv(self._eval_epoch, data)

    def _log_eval(self, global_step, loss: float = None, scores: dict = None):
        r"""
        Log for the evaluation phase

        Args:
            global_step:
            loss:
            scores:

        Returns:
            None
        """
        self._logger.info(f'------  Evaluation at global step {global_step}  ------')
        data = [global_step]
        if 'loss' in self.args.evaluation_info:
            data.append(loss)
            self._logger.info(f'Loss {loss}')
            self._writer.add_scalar('Validation loss per step', loss, global_step=global_step)
        if 'metrics' in self.args.evaluation_info:
            for metric, score in scores.items():
                self._logger.info(f'{metric}: {score}')
            data.extend(list(scores.values()))
            self._writer.add_scalar('Validation AUC score per step', scores['auc'], global_step=global_step)

        logger_utils.log_csv(self._eval_csv, data)

    def _save_model(self, model: nn.Module, optimizer, scheduler, flag: str):
        r"""
        Save the model

        Args:
            model:
            optimizer:
            scheduler:
            flag:

        Returns:
            None
        """
        save_path = os.path.join(self._path, flag + '.pt')
        saved_point = {'model': model,
                       'optimizer': optimizer,
                       'scheduler': scheduler.state_dict()}
        torch.save(saved_point, save_path)

    def _load_model(self, model_path: str):
        r"""
        Load the pre-trained model

        Args:
            model_path:

        Returns:
            A model
        """
        saved_point = torch.load(model_path, map_location=self._device)

        return saved_point['model']

    def _get_warmup_steps(self, num_training_steps: int):
        r"""
        Get number of steps used for a linear warmup

        Args:
            num_training_steps:

        Returns:
            Number of warmup steps
        """
        warmup_steps = (self.args.warmup_steps if self.args.warmup_steps is not None
                        else math.ceil(num_training_steps * self.args.warmup_ratio))
        return warmup_steps
