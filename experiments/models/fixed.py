import torch

from .templates import ModelBuilder

class FixedBase(ModelBuilder):

    class EvalOnlyClass(torch.nn.Module):

        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x):
            self.model.eval()
            return self.model(x)

    def __init__(self, fixed_base, head, trainer):
        super().__init__(self._base, head, trainer)
        self._fixed_base = fixed_base
    
    def _freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False
    
    def _base(self):
        base = self.EvalOnlyClass(self._fixed_base)
        self._freeze(base)
        return base