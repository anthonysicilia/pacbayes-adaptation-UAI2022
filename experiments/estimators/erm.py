import torch

from math import exp
from tqdm import tqdm

from ..models.stochastic import sample as sample_model
from .base import Estimator as BaseEstimator
from .expectation import Estimator as Mean
from .utils import to_device, sym_diff_classify

class GradientError(Exception):
    # for exploding or vanishing gradients
    pass

class DomainLabeledSet(torch.utils.data.Dataset):

    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.weight_a = max(len(a), len(b)) / len(a)
        self.weight_b = max(len(a), len(b)) / len(b)
    
    def __len__(self):
        return len(self.a) + len(self.b)
    
    def __getitem__(self, index):
        oidx = index
        if index >= len(self.a):
            index = index - len(self.a)
            x, y, *_ = self.b.__getitem__(index)
            return (x, y, oidx, self.weight_b, 1)
        else:
            x, y, *_ = self.a.__getitem__(index)
            return (x, y, oidx, self.weight_a, 0)

class GradReverse(torch.autograd.Function):
    """
    GRL adpated from Matsuura et al. 2020
    (Code) https://github.com/mil-tokyo/dg_mmld/
    (Paper) https://arxiv.org/pdf/1911.07661.pdf
    """
    @staticmethod
    def forward(ctx, x, beta, reverse=True):
        ctx.beta = beta
        ctx.reverse = reverse
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.reverse:
            return (grad_output * -ctx.beta), None, None
        else:
            return (grad_output * ctx.beta), None, None

def grad_reverse(x, beta=1.0, reverse=True):
    return GradReverse.apply(x, beta, reverse)

class Discriminator(torch.nn.Module):

    def __init__(self, model_builder, gamma=10):
        super().__init__()
        # bypasses stochastic !!
        self.c1 = model_builder.head()
        self.c2 = model_builder.head()
        self.gamma = gamma

    def forward(self, z, p):
        # λ = 2 / (1 + exp(−κ · p)) − 1
        beta = 2 / (1 + exp(-self.gamma * p)) - 1
        z = grad_reverse(z, beta)
        yhat1 = self.c1(z)
        yhat2 = self.c2(z)
        return sym_diff_classify(yhat1, yhat2, binary=False,
            baseline=False)

class Estimator(BaseEstimator):

    # terminates early if error is below 0.5%
    # just to speed things up a bit
    TOLERANCE = 5e-3

    def __init__(self, model_builder, dataset, device='cpu',
        verbose=False, catch_weights=False, sample=False, 
        kl_reg=False, ul_target=None, flag_nans=True,
        stop_early=True):
        super().__init__()
        self.mbuilder = model_builder
        if ul_target is not None:
            self.dann = True
            dataset = DomainLabeledSet(dataset, ul_target)
            self.dataset = self.mbuilder.dataloader(dataset)
        else:
            self.dann = False
            self.dataset = self.mbuilder.dataloader(dataset)
        self.catch_weights = catch_weights
        self.device = device
        self.verbose = verbose
        self.kl_reg = kl_reg
        self.sample = sample
        self.flag_nans = flag_nans
        self.stop_early = stop_early

    def _compute(self):
        model = self.mbuilder().to(self.device)
        if self.dann:
            discriminator = Discriminator(self.mbuilder).to(self.device)
        self.mbuilder.trainer.initialize(model.base.parameters(),
            model.head.parameters(), 
            alt_params=discriminator.parameters() 
            if self.dann else None)
        epoch_iterator = self.mbuilder.trainer.epoch_iterator()
        max_epochs = len(epoch_iterator)
        if self.verbose:
            epoch_iterator = tqdm(epoch_iterator)
        isnan = False
        for e in epoch_iterator:
            error = Mean()
            isnan = False
            for tup in to_device(self.dataset, self.device):
                if self.dann:
                    x, y, _, wd, d, *_ = tup
                    y = y[d==0]
                    w = None
                elif self.catch_weights:
                    x, y, _, w, *_ = tup
                else:
                    x, y, _, *_ = tup; w = None
                model.train()
                if self.sample: sample_model(model)
                prog = e / max_epochs
                if self.dann:
                    z = model.base(x)
                    dhat = discriminator(z, prog)
                    yhat = model.head(z[d==0])
                else:
                    yhat = model(x)
                if torch.isnan(yhat).any().item():
                    isnan = True
                    break
                self.mbuilder.trainer.zero_grad()
                h = model if self.kl_reg else None
                prog = prog if self.kl_reg else None
                if y.size(0) > 0:
                    loss = self.mbuilder.loss_fn(yhat, y, w=w, h=h, p=prog)
                    if self.dann:
                        loss += self.mbuilder.loss_fn(dhat, d, w=wd, h=h)
                    loss.backward()
                    self.mbuilder.trainer.batch_step()
                    errors = self.mbuilder.errors(yhat, y)
                    error.update(errors.item(), weight=y.size(0))
            if (not isnan) and self.verbose:
                epoch_iterator.set_postfix(
                    {'error' : error.compute()})
            elif self.verbose:
                epoch_iterator.set_postfix(
                    {'error' : 'nan'})
            if isnan or (error.compute() < self.TOLERANCE and self.stop_early):
                break
            self.mbuilder.trainer.epoch_step()
        if isnan and self.flag_nans:
            raise GradientError()
        else:
            return model.eval()


