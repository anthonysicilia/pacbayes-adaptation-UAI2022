import torch

from collections import OrderedDict

from ..utils import lazy_kwarg_init

class Trainer:

    ERR_MSG = 'Trainer is abstract - implement before use.'

    def __init__(self, epochs, base_lr, head_lr, batch_size, *args, **kwargs):
        self.epochs = epochs
        self.base_lr = base_lr
        self.head_lr = head_lr
        self.batch_size = batch_size
        self.active_optims = []
        self.active_scheds = []
    
    def epoch_iterator(self):
        return range(self.epochs)
    
    def initialize(self, base_params, head_params):
        raise NotImplementedError(self.ERR_MSG)
    
    def zero_grad(self):
        for o in self.active_optims: o.zero_grad()
        
    def batch_step(self):
        for o in self.active_optims: o.step()
    
    def epoch_step(self, *args, **kwargs):
        raise NotImplementedError(self.ERR_MSG)

class ModelBuilder:

    def __init__(self, base, head, trainer):
        self.base = base
        self.head = head
        self.trainer = trainer
        loader = torch.utils.data.DataLoader
        self.dataloader = lazy_kwarg_init(loader,
            batch_size=trainer.batch_size, 
            shuffle=True)
        self.test_dataloader = lazy_kwarg_init(loader,
            batch_size=trainer.batch_size, 
            shuffle=False, 
            drop_last=False)
    
    def __call__(self):
        layers = [
            ('base', self.base()),
            ('head', self.head())
        ]
        return torch.nn.Sequential(OrderedDict(layers))
    
    def loss_fn(self, yhat, y, w=None, **kwargs):
        w = w if w is not None else torch.ones_like(y).float()
        if len(yhat.size()) > 1:
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        else:
            _loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
            loss_fn = lambda yhat, y: _loss_fn(yhat, y.float())
        l = loss_fn(yhat, y)
        return (l * w).sum() / w.sum()

    def errors(self, yhat, y):
        if len(yhat.size()) > 1:
            errors = lambda yhat, y: (yhat.argmax(dim=1)!=y).sum()
        else:
            errors = lambda yhat, y: ((yhat > 0).long()!=y).sum()
        return errors(yhat, y)