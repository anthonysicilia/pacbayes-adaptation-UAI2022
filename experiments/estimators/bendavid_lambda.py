import torch

from .base import Estimator as BaseEstimator
from .erm import Estimator as ModelEstimator
from .expectation import Estimator as Mean
from .utils import to_device

class JointSet(torch.utils.data.Dataset):

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.weight_a = (max(len(a), len(b))) / len(a)
        self.weight_b = (max(len(a), len(b))) / len(b)

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

class Estimator(BaseEstimator):

    def __init__(self, model_builder, a, b, base=None,
        device='cpu', verbose=False, return_model=False):
        super().__init__()
        self.a = a
        self.b = b
        self.mbuilder = model_builder
        self.base = base
        self.device = device
        self.verbose = verbose
        self.return_model = return_model
    
    def _compute(self):
        dataset = JointSet(self.a, self.b)
        model = ModelEstimator(self.mbuilder, 
            dataset, device=self.device, verbose=self.verbose, 
            catch_weights=True).compute()
        iterator = to_device(self.mbuilder.test_dataloader(dataset),
            self.device)
        error0 = Mean()
        error1 = Mean()
        with torch.no_grad():
            model.eval()
            for x, y, _, _, z in iterator:
                yhat = model(x).argmax(dim=1)
                errors = (yhat != y)
                errors0 = errors[z==0].sum().item()
                errors1 = errors[z==1].sum().item()
                error0.update(errors0, weight=(z==0).sum().item())
                error1.update(errors1, weight=(z==1).sum().item())
        if self.return_model:
            return error0.compute() + error1.compute(), model
        else:
            return error0.compute() + error1.compute()
