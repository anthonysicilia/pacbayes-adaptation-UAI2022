from turtle import forward
import torch

from ..models.templates import ModelBuilder
from .base import Estimator as BaseEstimator
from .expectation import Estimator as Mean
from .erm import Estimator as ModelEstimator
from .utils import to_device, sym_diff_classify


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
            x, *_ = self.b.__getitem__(index)
            return (x, 1, oidx, self.weight_b)
        else:
            x, *_ = self.a.__getitem__(index)
            return (x, 0, oidx, self.weight_a)

class SymmetricDifferenceBase(torch.nn.Module):

    def __init__(self, m1, m2):
        super().__init__()
        self.b1 = m1.base
        self.b2 = m2.base
    
    def forward(self, x):
        return (self.b1(x), self.b2(x))

class SymmetricDifferenceHead(torch.nn.Module):

    def __init__(self, m1, m2):
        super().__init__()
        self.h1 = m1.head
        self.h2 = m2.head
    
    def forward(self, x1, x2):
        return (self.h1(x1), self.h2(x2))

class SymmetricDifferenceHypothesis(torch.nn.Module):

    """
    Returns an (untrained) hypothesis within 
    the Symmetric Difference Hypothesis Class
    := {h xor h | h in H} (= {h neq h | h in H})
    """

    def __init__(self, mbuilder, baseline):
        super().__init__()
        self.m1 = mbuilder()
        self.m2 = mbuilder()
        self.base = SymmetricDifferenceBase(self.m1, self.m2)
        self.head = SymmetricDifferenceHead(self.m1, self.m2)
        self.baseline = baseline

    def forward(self, x):
        (z1, z2) = self.base(x)
        (yhat1, yhat2) = self.head(z1, z2)
        return sym_diff_classify(yhat1, yhat2, binary=False,
            baseline=self.baseline)

class SymmetricDifferenceHypothesisSpace(ModelBuilder):

    def __init__(self, mbuilder, baseline=False):
        super().__init__(mbuilder.base, mbuilder.head, mbuilder.trainer)
        self.underlying = mbuilder
        self.baseline = baseline
    
    def __call__(self):
        return SymmetricDifferenceHypothesis(self.underlying, 
            self.baseline)

class Estimator(BaseEstimator):

    def __init__(self, mbuilder, a, b, device='cpu', 
        verbose=False, baseline=False):
        super().__init__()
        self.mbuilder = SymmetricDifferenceHypothesisSpace( 
            mbuilder, baseline=baseline)
        self.a = a
        self.b = b
        self.device = device
        self.verbose = verbose
    
    def _compute(self):
        x = self._asymmetric_compute(self.a, self.b)
        y = self._asymmetric_compute(self.b, self.a)
        return max(x, y)
    
    def _asymmetric_compute(self, a, b):
        dataset = DomainLabeledSet(a, b)
        model = ModelEstimator(self.mbuilder, dataset,
            device=self.device, verbose=self.verbose,
            catch_weights=True).compute()
        iterator = to_device(self.mbuilder.test_dataloader(dataset),
            self.device)
        prob_ind_a = Mean()
        prob_ind_b = Mean()
        with torch.no_grad():
            model.eval()
            for x, y, *_ in iterator:
                yhat = (model(x) >= 0).long()
                a_ind = yhat[y == 0] == 1
                b_ind = yhat[y == 1] == 1
                prob_ind_a.update(a_ind.sum().item(), 
                    weight=len(a_ind))
                prob_ind_b.update(b_ind.sum().item(), 
                    weight=len(b_ind))
        return abs(prob_ind_a.compute() - prob_ind_b.compute()) 