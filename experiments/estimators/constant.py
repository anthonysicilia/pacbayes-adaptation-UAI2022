from .base import Estimator as BaseEstimator

class Estimator(BaseEstimator):

    def __init__(self, value):
        super().__init__()
        self.value = value
    
    def _compute(self):
        return self.value