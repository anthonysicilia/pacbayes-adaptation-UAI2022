from tqdm import tqdm

from .base import Estimator as BaseEstimator
from .expectation import Estimator as Mean

class Estimator(BaseEstimator):

    def __init__(self, m, estimator, verbose=False):
        super().__init__()
        self.m = m
        self.estimator = estimator
        self.verbose = verbose
    
    def _compute(self):
        mc_estimate = Mean()
        iterator = range(self.m)
        if self.verbose:
            iterator = tqdm(iterator)
        for _ in iterator:
            # lazy kwarg init
            est = self.estimator()
            mc_estimate.update(est.compute())
        return mc_estimate.compute()