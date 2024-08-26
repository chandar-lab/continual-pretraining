from abc import abstractmethod
import logging
import sys
from argparse import ArgumentParser, Namespace
from contextlib import suppress
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import get_dataset

from conf import get_device, warn_once
from conf import persistent_locals

with suppress(ImportError):
    import wandb


class ContinualModel(nn.Module):
    """
    Continual learning model for LLM pretraining.
    """
    NAME: str
    COMPATIBILITY: List[str]
    AVAIL_OPTIMS = ['sgd', 'adam', 'adamw']

    args: Namespace  # The command line arguments
    device: torch.device  # The device to be used for training
    net: nn.Module  # The backbone of the model (defined by the `dataset`)
    loss: nn.Module  # The loss function to be used (defined by the `dataset`)
    opt: optim.Optimizer  # The optimizer to be used for training
    scheduler: optim.lr_scheduler._LRScheduler  # (optional) The scheduler for the optimizer. If defined, it will overwrite the one defined in the `dataset`
    transform: nn.Module  # The transformation to be applied to the input data (for text data)
    task_iteration: int  # Number of iterations in the current task
    epoch_iteration: int  # Number of iterations in the current epoch. Updated if `epoch` is passed to observe
    num_classes: int  # Total number of classes in the dataset
    n_tasks: int  # Total number of tasks in the dataset

    @staticmethod
    def get_parser() -> ArgumentParser:
        """
        Returns the parser of the model.

        Additional model-specific hyper-parameters can be added by overriding this method.

        Returns:
            the parser of the model
        """
        parser = ArgumentParser(description='Base CL model')
        return parser

    @property
    def task_iteration(self):
        """
        Returns the number of iterations in the current task.
        """
        return self._task_iteration

    @property
    def epoch_iteration(self):
        """
        Returns the number of iterations in the current epoch.
        """
        return self._epoch_iteration

    @property
    def current_task(self):
        """
        Returns the index of the current task.
        """
        return self._current_task

    @property
    def n_classes_current_task(self):
        """
        Returns the number of classes in the current task.
        Returns -1 if task has not been initialized yet.
        """
        if hasattr(self, '_n_classes_current_task'):
            return self._n_classes_current_task
        else:
            return -1

    @property
    def n_seen_classes(self):
        """
        Returns the number of classes seen so far.
        Returns -1 if task has not been initialized yet.
        """
        if hasattr(self, '_n_seen_classes'):
            return self._n_seen_classes
        else:
            return -1

    @property
    def n_remaining_classes(self):
        """
        Returns the number of classes remaining to be seen.
        Returns -1 if task has not been initialized yet.
        """
        if hasattr(self, '_n_remaining_classes'):
            return self._n_remaining_classes
        else:
            return -1

    @property
    def n_past_classes(self):
        """
        Returns the number of classes seen up to the past task.
        Returns -1 if task has not been initialized yet.
        """
        if hasattr(self, '_n_past_classes'):
            return self._n_past_classes
        else:
            return -1

    @property
    def cpt(self):
        """
        Alias of `classes_per_task`: returns the raw number of classes per task.
        Warning: return value might be either an integer or a list of integers depending on the dataset.
        """
        return self._cpt

    @property
    def classes_per_task(self):
        """
        Returns the raw number of classes per task.
        Warning: return value might be either an integer or a list of integers depending on the dataset.
        """
        return self._cpt

    @cpt.setter
    def cpt(self, value):
        """
        Sets the number of classes per task.
        """
        warn_once("Setting the number of classes per task is not recommended.")
        self._cpt = value

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                 args: Namespace, transform: nn.Module) -> None:
        super(ContinualModel, self).__init__()
        print("Using {} as backbone".format(backbone.__class__.__name__))
        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.dataset = get_dataset(self.args)
        self.N_CLASSES = self.dataset.N_CLASSES
        self.num_classes = self.N_CLASSES
        self.N_TASKS = self.dataset.N_TASKS
        self.n_tasks = self.N_TASKS
        self.SETTING = self.dataset.SETTING
        self._cpt = self.dataset.N_CLASSES_PER_TASK
        self._current_task = 0

        if self.net is not None:
            self.net = nn.DataParallel(self.net) if torch.cuda.device_count() > 1 else self.net
            self.net.to(self.device)
            self.opt = self.get_optimizer()
        else:
            logging.warning("no default model for this dataset. You will have to specify the optimizer yourself.")
            self.opt = None
        self.device = get_device()

        if not self.NAME or not self.COMPATIBILITY:
            raise NotImplementedError('Please specify the name and the compatibility of the model.')

        if self.args.label_perc != 1 and 'cssl' not in self.COMPATIBILITY:
            logging.info('label_perc is not explicitly supported by this model -> training may break')

    def to(self, device):
        """
        Captures the device to be used for training.
        """
        self.device = device
        return super().to(device)

    def load_buffer(self, buffer):
        """
        Default way to handle load buffer.
        """
        assert buffer.examples.shape[0] == self.args.buffer_size, "Buffer size mismatch. Expected {} got {}".format(
            self.args.buffer_size, buffer.examples.shape[0])
        self.buffer = buffer

    def get_parameters(self):
        """
        Returns the parameters of the model.
        """
        return self.net.parameters()

    def get_optimizer(self) -> optim.Optimizer:
        # check if optimizer is in torch.optim
        supported_optims = {optim_name.lower(): optim_name for optim_name in dir(optim) if optim_name.lower() in self.AVAIL_OPTIMS}
        opt = None
        if self.args.optimizer.lower() in supported_optims:
            if self.args.optimizer.lower() == 'sgd':
                opt = getattr(optim, supported_optims[self.args.optimizer.lower()])(self.get_parameters(), lr=self.args.lr,
                                                                                    weight_decay=self.args.optim_wd,
                                                                                    momentum=self.args.optim_mom,
                                                                                    nesterov=self.args.optim_nesterov == 1)
            elif self.args.optimizer.lower() == 'adam' or self.args.optimizer.lower() == 'adamw':
                opt = getattr(optim, supported_optims[self.args.optimizer.lower()])(self.get_parameters(), lr=self.args.lr,
                                                                                    weight_decay=self.args.optim_wd)

        if opt is None:
            raise ValueError('Unknown optimizer: {}'.format(self.args.optimizer))
        return opt

    def compute_offsets(self, task: int) -> Tuple[int, int]:
        """
        Compute the start and end offset given the task.

        Args:
            task: the task index

        Returns:
            the start and end offset
        """
        return self.dataset.get_offsets(task)

    def get_debug_iters(self):
        """
        Returns the number of iterations to be used for debugging.
        Default: 3
        """
        return 5



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.

        Args:
            x: batch of inputs
            task_label: some models require the task label

        Returns:
            the result of the computation
        """
        return self.net(x)

    def meta_observe(self, *args, **kwargs):
        """
        Wrapper for `observe` method.

        Takes care of dropping unlabeled data if not supported by the model and of logging to wandb if installed.

        Args:
            inputs: batch of inputs
            labels: batch of labels
            not_aug_inputs: batch of inputs without augmentation
            kwargs: some methods could require additional parameters

        Returns:
            the value of the loss function
        """
        if 'epoch' in kwargs and kwargs['epoch'] is not None:
            epoch = kwargs['epoch']
            if self._past_epoch != epoch:
                self._past_epoch = epoch
                self._epoch_iteration = 0

        if 'cssl' not in self.COMPATIBILITY:  # drop unlabeled data if not supported
            labeled_mask = args[1] != -1
            if (~labeled_mask).any():  # if there are any unlabeled samples
                if labeled_mask.sum() == 0:  # if all samples are unlabeled
                    return 0
                args = [arg[labeled_mask] if isinstance(arg, torch.Tensor) and arg.shape[0] == args[0].shape[0] else arg for arg in args]
        if 'wandb' in sys.modules and not self.args.nowand:
            pl = persistent_locals(self.observe)
            ret = pl(*args, **kwargs)
            extra = {}
            if isinstance(ret, dict):
                assert 'loss' in ret, "Loss not found in return dict"
                extra = {k: v for k, v in ret.items() if k != 'loss'}
                ret = ret['loss']
            self.autolog_wandb(pl.locals, extra=extra)
        else:
            ret = self.observe(*args, **kwargs)
            if isinstance(ret, dict):
                assert 'loss' in ret, "Loss not found in return dict"
                ret = ret['loss']
        self._task_iteration += 1
        self._epoch_iteration += 1
        return ret

    def meta_begin_task(self, dataset):
        """
        Wrapper for `begin_task` method.

        Takes care of updating the internal counters.

        Args:
            dataset: the current task's dataset
        """
        self._task_iteration = 0
        self._epoch_iteration = 0
        self._past_epoch = 0
        self._n_classes_current_task = self._cpt if isinstance(self._cpt, int) else self._cpt[self._current_task]
        self._n_past_classes, self._n_seen_classes = self.compute_offsets(self._current_task)
        self._n_remaining_classes = self.N_CLASSES - self._n_seen_classes
        self.begin_task(dataset)

    def meta_end_task(self, dataset):
        """
        Wrapper for `end_task` method.

        Takes care of updating the internal counters.

        Args:
            dataset: the current task's dataset
        """

        self.end_task(dataset)
        self._current_task += 1

    @abstractmethod
    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor, epoch: int = None) -> float:
        """
        Compute a training step over a given batch of examples.

        Args:
            inputs: batch of examples
            labels: ground-truth labels
            kwargs: some methods could require additional parameters

        Returns:
            the value of the loss function
        """
        raise NotImplementedError
