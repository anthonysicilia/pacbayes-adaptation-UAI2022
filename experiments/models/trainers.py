import torch

from .templates import Trainer
from ..utils import lazy_kwarg_init

class PlaceholderScheduler:

    def __init__(self, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass

class StepDecaySGD(Trainer):

    def __init__(self, epochs=150, base_lr=1e-3, head_lr=1e-2,
        alt_lr=1e-3, batch_size=128, momentum=0.9, decay=0.5, 
        step=0.75):
        super().__init__(epochs, base_lr, head_lr, batch_size)
        self.boptim = lazy_kwarg_init(torch.optim.SGD,
            lr=base_lr, momentum=momentum)
        self.hoptim = lazy_kwarg_init(torch.optim.SGD,
            lr=head_lr, momentum=momentum)
        self.aoptim = lazy_kwarg_init(torch.optim.SGD,
            lr=alt_lr, momentum=momentum)
        scheduler = torch.optim.lr_scheduler.StepLR
        if step < 1:
            lr_step = int(step * self.epochs)
        else:
            lr_step = step
        if lr_step < 1:
            self.sched = lazy_kwarg_init(PlaceholderScheduler)
        else:
            self.sched = lazy_kwarg_init(scheduler,
                step_size=lr_step, gamma=decay)
    
    def initialize(self, base_params, head_params, alt_params=None):
        boptim = self.boptim(base_params)
        hoptim = self.hoptim(head_params)
        self.active_optims = [boptim, hoptim]
        if alt_params is not None:
            aoptim = self.aoptim(alt_params)
            self.active_optims += [aoptim]
        self.active_scheds = [self.sched(o) 
            for o in self.active_optims]
    
    def epoch_step(self, *args, **kwargs):
        for s in self.active_scheds:
            s.step()
