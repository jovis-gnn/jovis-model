from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
import numpy as np


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


def get_scheduler(optimizer, scheduler_type, warmup_steps, total_steps):
    if scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, mode="max", verbose=True
        )
    elif scheduler_type == "cosine":
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer, warmup=warmup_steps, max_iters=total_steps
        )
    return scheduler
