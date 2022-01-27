import torch

from tqdm import tqdm

from ..models.stochastic import sample as sample_model
from .base import Estimator as BaseEstimator
from .expectation import Estimator as Mean
from .utils import to_device

class Estimator(BaseEstimator):

    def __init__(self, model, mbuilder, dataset,  
        device='cpu', verbose=False, sample=False):
        super().__init__()
        self.model = model.to(device)
        with torch.no_grad():
            if sample: sample_model(self.model)
        self.dataset = mbuilder.test_dataloader(dataset)
        self.verbose = verbose
        self.to_device = lambda x: to_device(x, device)
    
    def _compute(self):
        error = Mean()
        iterator = self.to_device(self.dataset)
        if self.verbose:
            iterator = tqdm(iterator)
        with torch.no_grad():
            self.model.eval()
            for x, y, *_ in iterator:
                yhat = self.model(x).argmax(dim=1)
                errors = (yhat != y).sum().item()
                error.update(errors, weight=y.size(0))
        return error.compute()
