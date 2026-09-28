import torch
import numpy as np

from matplotlib import pyplot as plt


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs):
        self.warmup = warmup_epochs
        self.max_num_iters = total_epochs

        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= (epoch + 1e-6) * 1.0 / (self.warmup + 1e-6)

        return lr_factor


class ExponentialWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        gamma: float,
        min_lr: float,
        **_,
    ):
        self.warmup = warmup_epochs
        self.gamma = gamma
        self.min_lr = min_lr
        self.factor = 1

        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [
            (base_lr - self.min_lr) * lr_factor + self.min_lr
            for base_lr in self.base_lrs
        ]

    def get_lr_factor(self, epoch):
        lr_factor = self.factor
        if epoch <= self.warmup:
            lr_factor *= (epoch + 1e-6) * 1.0 / (self.warmup + 1e-6)
        else:
            self.factor *= self.gamma

        return lr_factor
