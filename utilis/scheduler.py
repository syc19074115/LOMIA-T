# -- coding: utf-8 --
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
import math
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class CosineAnnealingLRWarmup(CosineAnnealingLR):
    def __init__(self, optimizer, T_max, eta_min=1.0e-5, last_epoch=-1, verbose=False,
                 warmup_steps=2, warmup_start_lr=1.0e-5):
        super(CosineAnnealingLRWarmup, self).__init__(optimizer, T_max=T_max,
                                                      eta_min=eta_min,
                                                      last_epoch=last_epoch)
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        if warmup_steps > 0:
            self.base_warup_factors = [
                (base_lr / warmup_start_lr) ** (1.0 / self.warmup_steps)
                for base_lr in self.base_lrs
            ]
            #print(self.base_lrs)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        if hasattr(self, 'warmup_steps'):
            if self.last_epoch < self.warmup_steps:
                return [self.warmup_start_lr * (warmup_factor ** self.last_epoch)
                        for warmup_factor in self.base_warup_factors]
            else:
                return [self.eta_min + (base_lr - self.eta_min) *
                        (1 + math.cos(
                            math.pi * (self.last_epoch - self.warmup_steps) / (self.T_max - self.warmup_steps))) * 0.5
                        for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                    for base_lr in self.base_lrs]


class LRScheduler():
    def __init__(self, init_lr=1.0e-4, lr_decay_epoch=10):
        self.init_lr = init_lr
        self.lr_decay_epoch = lr_decay_epoch

    def __call__(self, optimizer, epoch):
        '''Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs.'''
        lr = self.init_lr * (0.8 ** (epoch // self.lr_decay_epoch))
        # lr = max(lr, 1e-8)
        lr = max(lr, 1e-16)
        if epoch % self.lr_decay_epoch == 0:
            print ('LR is set to {}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        return optimizer