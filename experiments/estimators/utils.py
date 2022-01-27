import torch

from random import Random, shuffle

from .median import Estimator as Median

def to_device(iterator, device):
    for x in iterator:
        arr = []
        for xi in x:
            if type(xi) == dict:
                xi = {k : v.to(device) for k,v in xi.items()}
                arr.append(xi)
            else:
                arr.append(xi.to(device))
        yield tuple(arr)

def lazy_kwarg_init(init, **kwargs):

    class LazyCallable:

        def __init__(self, init, kwargs):
            self.init = init
            self.kwargs = kwargs
        
        def __call__(self, *args):
            return self.init(*args, **kwargs)
    
    return LazyCallable(init, kwargs)

def stack(dataset, max_samples=None):
    if type(dataset[0]) != tuple:
        raise ValueError('If dataset.__getitem__(...) does not'
            ' return tuples, this function may malfunction.'
            ' Please, ensure dataset.__getitem__(...) returns'
            ' a tuple; e.g., return "(x, )" in place of "x".'
            f' The misused type was {type(dataset[0])}.')
    arr = [x.reshape(1, -1) if torch.is_tensor(x)
        else torch.tensor(x.reshape(1, -1)) 
        for x, *_ in dataset]
    if max_samples is not None and len(arr) > max_samples:
        shuffle(arr)
        return torch.cat(arr)[:max_samples]
    else:
        return torch.cat(arr)

def approx_median_distance(a, b, nsamples=100, seed=None):
    pooled = [(x, _) for x,*_ in a] + [(x, _) for x,*_ in b]
    if seed is not None: # use specified seed, don't modify global
        Random(seed).shuffle(pooled)
    else: # use global seed
        shuffle(pooled)
    pooled = stack(pooled)[:nsamples]
    med = Median()
    for i, xi in enumerate(pooled):
        for j, xj in enumerate(pooled):
            if i != j:
                sum_of_squares = ((xi - xj) ** 2).sum()
                dist = torch.sqrt(sum_of_squares).item()
                med.update(dist)
    return med.compute()

def _binary_sym_diff_classify(yhat1, yhat2):
    if len(yhat1.size()) == 2:
        # this is techincally incorrect for multiclass
        return -(yhat1[:, 0] * yhat2[:, 0])
    elif len(yhat1.size()) == 1:
        return -(yhat1 * yhat2)
    else:
        raise NotImplementedError(
            f'Size of output {yhat1.size()}'
            ' not implemented in _binary_sym_diff_classify(...)')

def _multiclass_sym_diff_classify(yhat1, yhat2):
    if len(yhat1.size()) <= 1:
        raise NotImplementedError(
            f'Size of output {yhat1.size()}'
            ' not implemented in _multiclass_sym_diff_classify(...)')
    batch_size = yhat1.size(0)
    num_classes = yhat1.size(1)
    make_pos = torch.nn.Softplus()
    A = torch.einsum('bi,bj->bij', make_pos(yhat1), make_pos(yhat2))
    I = torch.eye(num_classes).bool().repeat(batch_size, 1, 1)
    max_off_diag_A = A[~I].reshape(batch_size, -1).max(dim=1).values
    max_diag_A = A[I].reshape(batch_size, -1).max(dim=1).values
    return max_off_diag_A - max_diag_A

def _baseline_sym_diff_classify(yhat1):
    if len(yhat1.size()) == 2:
        return yhat1[:, 0]
    elif len(yhat1.size()) == 1:
        return yhat1
    else:
        raise NotImplementedError(
            f'Size of output {yhat1.size()}'
            ' not implemented in _binary_sym_diff_classify(...)')

def sym_diff_classify(yhat1, yhat2, binary=True, baseline=False):
    if baseline:
        return _baseline_sym_diff_classify(yhat1)
    else:
        if binary:
            return _binary_sym_diff_classify(yhat1, yhat2)
        else:
            return _multiclass_sym_diff_classify(yhat1, yhat2)
