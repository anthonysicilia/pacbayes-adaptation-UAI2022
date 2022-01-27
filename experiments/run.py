import argparse
import numpy as np
import random
import pandas as pd
import torch

from pathlib import Path

from experiments.models.templates import ModelBuilder
# from tqdm import tqdm

from .datasets.amazon import DATASETS as AMAZON_DATASETS
from .datasets.digits import DATASETS as DIGITS_DATASETS, \
    ROTATION_PAIRS, NOISY_PAIRS, FAKE_PAIRS
from .datasets.discourse import PDTB_DATASETS, GUM_DATASETS
from .datasets.images import PACS_DATASETS, OFFICEHOME_DATASETS
from .datasets.utils import Multisource
from .estimators.bendavid_lambda import Estimator \
    as BenDavidLambdaEstimator
from .estimators.constant import Estimator \
    as Constant
from .estimators.error import Estimator as Error
from .estimators.erm import Estimator as ModelEstimator
from .estimators.dependent_divergence import Estimator \
    as DependentDivergence
from .estimators.independent_divergence import Estimator \
    as IndependentDivergence
from .estimators.monte_carlo import Estimator as MonteCarloEstimator
from .estimators.stoch_disagreement import Estimator \
    as StochDisagreementEstimator
from .estimators.stoch_joint_error import Estimator \
    as StochJointErrorEstimator
from .models.amazon import AmazonBase, AmazonHead
from .models.digits import DigitsBase, DigitsHead
from .models.discourse import DiscourseBase, DiscourseHead
from .models.images import ResNet18Base, ResNet18Head, ResNet50Base, \
    ResNet50Head
from .models.fixed import FixedBase
from .models.stochastic import Stochastic, mean, compute_kl
from .models.trainers import StepDecaySGD
from .utils import lazy_kwarg_init, set_random_seed

MC_SAMPLES = 1000
SIGMA_PRIOR = 0.01
MIN_TRAIN_SAMPLES = 1000

GROUPS = {
    'digits' : {
        'name' : 'digits',
        'datasets' : DIGITS_DATASETS,
        'multisource' : False,
        'make_pairs' : True,
        'model' : (DigitsBase, DigitsHead),
        'trainer' : lazy_kwarg_init(StepDecaySGD)
    },

    'digits_m' : {
        'name' : 'digits_m',
        'datasets' : DIGITS_DATASETS,
        'multisource' : True,
        'make_pairs' : True,
        'model' : (DigitsBase, DigitsHead),
        'trainer' : lazy_kwarg_init(StepDecaySGD)
    },

    'r_digits' : {
        'name' : 'r_digits',
        'datasets' : ROTATION_PAIRS,
        'multisource' : False,
        'make_pairs' : False,
        'model' : (DigitsBase, DigitsHead),
        'trainer' : lazy_kwarg_init(StepDecaySGD)
    },

    'n_digits' : {
        'name' : 'n_digits',
        'datasets' : NOISY_PAIRS,
        'multisource' : False,
        'make_pairs' : False,
        'model' : (DigitsBase, DigitsHead),
        'trainer' : lazy_kwarg_init(StepDecaySGD)
    },

    'f_digits' : {
        'name' : 'f_digits',
        'datasets' : FAKE_PAIRS,
        'multisource' : False,
        'make_pairs' : False,
        'model' : (DigitsBase, DigitsHead),
        'trainer' : lazy_kwarg_init(StepDecaySGD)
    },

    # Q: runs 10x longer after exp 42 ?? 
    # A: photo too small to train
    'pacs' : {
        'name' : 'pacs',
        'datasets' : PACS_DATASETS,
        'multisource' : False,
        'make_pairs' : True,
        'model' : (ResNet18Base, ResNet18Head),
        'trainer' : lazy_kwarg_init(StepDecaySGD)
    },

    'officehome' : {
        'name' : 'officehome',
        'datasets' : OFFICEHOME_DATASETS,
        'multisource' : False,
        'make_pairs' : True,
        'model' : (ResNet50Base, ResNet50Head),
        'trainer' : lazy_kwarg_init(StepDecaySGD)
    },

    'amazon' : {
        'name' : 'amazon',
        'datasets' : AMAZON_DATASETS,
        'multisource' : False,
        'make_pairs' : True,
        'model' : (AmazonBase, AmazonHead),
        'trainer' : lazy_kwarg_init(StepDecaySGD)
    },

    'amazon_m' : {
        'name' : 'amazon_m',
        'datasets' : AMAZON_DATASETS,
        'multisource' : True,
        'make_pairs' : True,
        'model' : (AmazonBase, AmazonHead),
        'trainer' : lazy_kwarg_init(StepDecaySGD)
    },

    'pdtb_sentence' : {
        'name' : 'pdtb_sentence',
        'datasets' : PDTB_DATASETS('sentence'),
        'multisource' : False, 
        'make_pairs' : True,
        'two_sample' : True,
        'model' : (DiscourseBase, DiscourseHead),
        'trainer' : lazy_kwarg_init(StepDecaySGD)
    },

    'pdtb_average' : {
        'name' : 'pdtb_average',
        'datasets' : PDTB_DATASETS('average'),
        'multisource' : False, 
        'make_pairs' : True,
        'model' : (DiscourseBase, DiscourseHead),
        'trainer' : lazy_kwarg_init(StepDecaySGD)
    },

    'pdtb_pooled' : {
        'name' : 'pdtb_pooled',
        'datasets' : PDTB_DATASETS('pooled'),
        'multisource' : False, 
        'make_pairs' : True,
        'two_sample' : True,
        'model' : (DiscourseBase, DiscourseHead),
        'trainer' : lazy_kwarg_init(StepDecaySGD)
    },

    'gum_sentence' : {
        'name' : 'gum_sentence',
        'datasets' : GUM_DATASETS('sentence'),
        'multisource' : True, 
        'make_pairs' : True,
        'two_sample' : True,
        'model' : (DiscourseBase, DiscourseHead),
        'trainer' : lazy_kwarg_init(StepDecaySGD)
    },

    'gum_average' : {
        'name' : 'gum_average',
        'datasets' : GUM_DATASETS('average'),
        'multisource' : True, 
        'make_pairs' : True,
        'two_sample' : True,
        'model' : (DiscourseBase, DiscourseHead),
        'trainer' : lazy_kwarg_init(StepDecaySGD)
    },

    'gum_pooled' : {
        'name' : 'gum_pooled',
        'datasets' : GUM_DATASETS('pooled'),
        'multisource' : True, 
        'make_pairs' : True,
        'two_sample' : True,
        'model' : (DiscourseBase, DiscourseHead),
        'trainer' : lazy_kwarg_init(StepDecaySGD)
    }
}

# most popular seeds in python according to:
# https://blog.semicolonsoftware.de/the-most-popular-random-seeds/
# SEEDS = [0, 1, 100, 1234, 12345]
# use 0, 1, 100 for dataset seeds and 100, ... for exp seeds
# SEEDS = [100 , 1234, 12345]

class SeededEstimator:

    def __init__(self, estimator, seed):
        self.seed = seed
        self.estimator = estimator
    
    def compute(self):
        set_random_seed(self.seed)
        return self.estimator.compute()

def _make_single_source_exps(group, dataset_seed):
    datasets = [
        (f'{desc}-{int(train)}', lazy_kwarg_init(dset,  
            train=train, seed=dataset_seed))
        for desc, dset in group['datasets'] 
        for train in [True, False]]
    b, h = group['model']
    tr = group['trainer']
    exps = [(s, t, (b, h, tr)) 
        for i, s in enumerate(datasets)
        for j, t in enumerate(datasets)
        if i != j]
    return exps

def _make_multisource_exps(group, dataset_seed):
    datasets = [(f'{desc}', lazy_kwarg_init(dset, 
        train=True, seed=dataset_seed))
        for desc, dset in group['datasets']]
    exps = []
    b, h = group['model']
    tr = group['trainer']
    for i in range(len(datasets)):
        target = datasets[i]
        source = [t for j,t in enumerate(datasets) 
            if j != i]
        descs = [t[0] for t in source]
        dsets = [t[1] for t in source]
        mdset = lazy_kwarg_init(Multisource, dsets=dsets)
        source = ('+'.join(descs), mdset)
        exps.append((source, target, (b, h, tr)))
    return exps

def _make_prepackaged_exps(group, dataset_seed):
    exps = []
    b, h = group['model']
    tr = group['trainer']
    for (sname, s, tname, t) in group['datasets']:
        s = lazy_kwarg_init(s, train=True, 
            seed=dataset_seed)
        t = lazy_kwarg_init(t, train=True, 
            seed=dataset_seed)
        source = (sname, s)
        target = (tname, t)
        exps.append((source, target, (b, h, tr)))
    return exps
    
def make_experiments(group, dataset_seed):
    if group['make_pairs']:
        if group['multisource']:
            return _make_multisource_exps(group, dataset_seed)
        else:
            return _make_single_source_exps(group, dataset_seed)
    else:
        return _make_prepackaged_exps(group, dataset_seed)

def disjoint_split(dataset, seed=0, prefix_ratio=0.5):

    random.seed(seed)
    indices = [(i, random.random() <= prefix_ratio) 
        for i in range(len(dataset))]
    prefix = [i for i, b in indices if b]
    bound = [i for i, b in indices if not b]

    class Dataset:

        def __init__(self, a, index_list):
            self.index_list = index_list
            self.a = a
        
        def __len__(self):
            return len(self.index_list)
        
        def __getitem__(self, index):
            data = self.a.__getitem__(self.index_list[index])
            # can't assing to tuple... 
            # NOTE: this might break cacheing
            # data[2] = index
            return data
    
    return Dataset(dataset, prefix), Dataset(dataset, bound)

# Estimate shorthand
# sid: sample independent
# idl: ideal model
# src: source
# trg: target
# sdp: sample dependent
# pri: prior
# pst: posterior
# mid: model-independent
# mdp: model-dependent
# div: divergence
# fb: fixed base
# pm: posterior mean
# dis: disagreement
# jer: joint error
# siz: sample size
# kld: kl divergence

def ben_david_lambda_estimates(source, target, base, head, trainer, seed, 
    verbose=False, device='cpu'):

    src_train, src_test = disjoint_split(source, seed=seed,
        prefix_ratio=0.7)
    trg_train, trg_test = disjoint_split(target, seed=seed,
        prefix_ratio=0.7)
    mbuilder = ModelBuilder(base, head, trainer())
    mestimator = BenDavidLambdaEstimator(mbuilder, src_train, trg_train, 
        return_model=True, device=device, verbose=verbose)
    ideal = SeededEstimator(mestimator, seed).compute()[1]
    estimators = {
        'sid_idl_src_err' : Error(ideal, mbuilder, src_test, 
            verbose=verbose, device=device),
        'sid_idl_trg_err' : Error(ideal, mbuilder, trg_test, 
            verbose=verbose, device=device),
        'sid_idl_src_siz' : Constant(len(src_test)),
        'sid_idl_tar_siz' : Constant(len(trg_test))
    }
    return {k : SeededEstimator(e, seed).compute() 
        for k, e in estimators.items()}

def no_dann_lambda_estimates(source, target, base, head, trainer, seed, 
    verbose=False, device='cpu'):

    ideal_builder = ModelBuilder(base, head, trainer())
    ideal_estimator = BenDavidLambdaEstimator(ideal_builder, 
        source, target, return_model=True, device=device, 
        verbose=verbose)
    ideal = SeededEstimator(ideal_estimator, seed).compute()[1]
    estimators = {
        'sdp_idl_src_err' : Error(ideal, ideal_builder,
            source, verbose=verbose, device=device),
        'sdp_idl_trg_err' : Error(ideal, ideal_builder, 
            target, verbose=verbose, device=device)
    }

    return {k : SeededEstimator(e, seed).compute() 
        for k, e in estimators.items()}

def dann_mc_estimates(dann, mbuilder, source, target, seed, verbose, device):
    estimators = {
        'src_err' : lazy_kwarg_init(Error,
            model=dann, mbuilder=mbuilder, sample=True,
            dataset=source, verbose=verbose, device=device),
        'trg_err' : lazy_kwarg_init(Error, 
            model=dann, mbuilder=mbuilder, sample=True,
            dataset=target, verbose=verbose, device=device),
        'src_dis' : lazy_kwarg_init(StochDisagreementEstimator,
            model=dann, mbuilder=mbuilder, dataset=source, 
            device=device, verbose=verbose, sample=True),
        'src_jer' : lazy_kwarg_init(StochJointErrorEstimator,
            model=dann, mbuilder=mbuilder, dataset=source, 
            device=device, verbose=verbose, sample=True),
        'trg_dis' : lazy_kwarg_init(StochDisagreementEstimator,
            model=dann, mbuilder=mbuilder, dataset=target, 
            device=device, verbose=verbose, sample=True),
        'trg_jer' : lazy_kwarg_init(StochJointErrorEstimator,
            model=dann, mbuilder=mbuilder, dataset=target, 
            device=device, verbose=verbose, sample=True),
    }
    estimators = {k : MonteCarloEstimator(MC_SAMPLES, e) 
        for k, e in estimators.items()}

    return {k : SeededEstimator(e, seed).compute() 
        for k, e in estimators.items()}

def dann_estimates(source, target, base, head, trainer, seed, 
    verbose=False, device='cpu'):

    prior_builder = ModelBuilder(base, head, trainer())
    prior_estimator = ModelEstimator(prior_builder, 
        source, device=device, verbose=verbose)
    prior = SeededEstimator(prior_estimator, seed).compute()
    estimates = {}
    estimators = {}
    estimators[f'pri_src_err'] = Error(prior,
        prior_builder, source, sample=False,
        verbose=verbose, device=device)
    estimators[f'pri_trg_err'] = Error(prior,
        prior_builder, target, sample=False,
        verbose=verbose, device=device)
    estimators['mid_div'] = IndependentDivergence(
        prior_builder, source, target, device=device, 
        verbose=verbose)
    estimators['pri_mdp_div'] = DependentDivergence(prior,
        prior_builder, source, target, device=device,
        verbose=verbose)
    fb_prior_builder = FixedBase(prior.base, head, trainer())
    estimators['fb_pri_mid_div'] = IndependentDivergence(
        fb_prior_builder, source, target, device=device, 
        verbose=verbose)
    estimators['fb_pri_mdp_div'] = DependentDivergence(prior,
        fb_prior_builder, source, target, device=device,
        verbose=verbose)

    # (TIMESAVER) CHECKPOINT FOR TESTS, RM AFTER TESTED
    for k, e in estimators.items():
        estimates[k] = SeededEstimator(e, seed).compute()
    # END CHECKPOINT

    for sigma_prior in [0.01, 0.05, 0.1]:
        for damp in [0.001, 0.01, 0.1, 1]:

            x = f'_{sigma_prior}_{damp}'
            dann_builder = Stochastic(base, head, trainer(), 
                prior=prior, kl_damp=damp, m=len(target), 
                sigma_prior=sigma_prior, device=device)
            dann_estimator = ModelEstimator(dann_builder,
                source, device=device, verbose=verbose, 
                ul_target=target, sample=True, kl_reg=True)
            dann = SeededEstimator(dann_estimator, seed).compute()

            # (TIMESAVER) CHECKPOINT FOR TESTS, RM AFTER TESTED
            mc_estimates = dann_mc_estimates(dann, dann_builder, 
                source, target, seed, verbose, device)
            for k, v in mc_estimates.items():
                estimates[f'{k}{x}'] = v
            # END CHECKPOINT
            
            with torch.no_grad():
                estimates[f'kld{x}'] = compute_kl(dann, device).item()

            with torch.no_grad():
                mean(dann) # deterministic reference

            estimators = {}
            estimators[f'pm_src_err{x}'] = Error(dann,
                dann_builder, source, sample=False,
                verbose=verbose, device=device)
            estimators[f'pm_trg_err{x}'] = Error(dann,
                dann_builder, target, sample=False,
                verbose=verbose, device=device)
            estimators[f'pst_mdp_div{x}'] = DependentDivergence(dann,
                prior_builder, source, target, device=device,
                verbose=verbose) # not stoch builder b/c sup
            fb_post_builder = FixedBase(dann.base, head, 
                trainer()) # not stoch builder b/c sup
            estimators[f'fb_pst_mid_div{x}'] = IndependentDivergence(
                fb_post_builder, source, target, device=device, 
                verbose=verbose)
            estimators[f'fb_pst_mdp_div{x}'] = DependentDivergence(
                dann, fb_post_builder, source, target, 
                device=device, verbose=verbose)
            fb_ideal_estimator = BenDavidLambdaEstimator(
                fb_post_builder, source, target, return_model=True, 
                device=device, verbose=verbose)
            fb_ideal = SeededEstimator(fb_ideal_estimator, 
                seed).compute()[1]
            estimators['fb_sdp_idl_src_err{x}'] = Error(fb_ideal, 
                fb_post_builder, source, verbose=verbose, 
                device=device)
            estimators['fb_sdp_idl_trg_err{x}'] = Error(fb_ideal, 
                fb_post_builder, target, verbose=verbose, 
                device=device)    
            for k, e in estimators.items():
                estimates[k] = SeededEstimator(e, seed).compute()
    
    return estimates

def simple_error_rates(source, target, base, head, trainer, seed, 
    verbose=False, device='cpu'):

    prior_builder = ModelBuilder(base, head, trainer())
    prior_estimator = ModelEstimator(prior_builder, 
        source, device=device, verbose=verbose)
    prior = SeededEstimator(prior_estimator, seed).compute()

    estimators = {}
    estimators[f'pri_src_err'] = Error(prior,
        prior_builder, source, sample=False,
        verbose=verbose, device=device)
    estimators[f'pri_trg_err'] = Error(prior,
        prior_builder, target, sample=False,
        verbose=verbose, device=device)
    
    dann_builder = Stochastic(base, head, trainer(), 
        prior=prior, kl_damp=0, m=len(target), 
        sigma_prior=0.01, device=device)
    dann_estimator = ModelEstimator(dann_builder,
        source, device=device, verbose=verbose, 
        ul_target=target, sample=True, kl_reg=True)
    dann = SeededEstimator(dann_estimator, seed).compute()

    estimators['src_err'] = lazy_kwarg_init(Error,
        model=dann, mbuilder=dann_builder, sample=True,
        dataset=source, verbose=verbose, device=device)
    estimators['src_err'] = MonteCarloEstimator(MC_SAMPLES,
        estimators['src_err'])
    estimators['trg_err'] = lazy_kwarg_init(Error, 
        model=dann, mbuilder=dann_builder, sample=True,
        dataset=target, verbose=verbose, device=device)
    estimators['trg_err'] = MonteCarloEstimator(MC_SAMPLES,
        estimators['trg_err'])
    
    return {k : SeededEstimator(e, seed).compute() 
        for k, e in estimators.items()}

def run_experiment(*args, param_search=False, **kwargs):
    if not param_search:
        a = dann_estimates(*args, **kwargs)
        b = no_dann_lambda_estimates(*args, **kwargs)
        c = ben_david_lambda_estimates(*args, **kwargs)
        estimates = {}
        for d in [a, b, c]:
            for k, v in d.items():
                estimates[k] = v
        return estimates
    else:
        return simple_error_rates(*args, **kwargs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group',
        type=str,
        default='digits',
        help='The group of experiments to run.')
    parser.add_argument('--verbose',
        action='store_true',
        help='Use verbose option for all estimators.')
    # making this an argument to help make parallel
    parser.add_argument('--dataset_seed',
        default=0,
        type=int,
        help='The seed to use for the dataset.')
    parser.add_argument('--experiment_seed',
        default=100,
        type=int,
        help='The seed to use for the experiment.')
    parser.add_argument('--device',
        type=str,
        default='cpu',
        help='The device to use for all experiments.')
    parser.add_argument('--test',
        action='store_true',
        help='Take steps for a shorter run'
        ' (results will be invalid).')
    parser.add_argument('--param_search',
        action='store_true',
        help='Only gets stats for picking params.')
    parser.add_argument('--base_lr',
        type=float,
        default=1e-3,
        help='init. lr for base feature extractor.')
    parser.add_argument('--head_lr',
        type=float,
        default=1e-2,
        help='init. lr for classifier head')
    parser.add_argument('--alt_lr',
        type=float,
        default=1e-2,
        help='init. lr for dann components')
    args = parser.parse_args()
    data = []
    group = GROUPS[args.group]
    print('group:', group['name'])
    exps = make_experiments(group, args.dataset_seed)
    enumber = 1
    print('Num exps:', len(exps))
    # exps = tqdm(make_experiments(group, args.dataset_seed))
    for (sname, s), (tname, t), (b, h, tr) in exps:
        if args.test: # don't train too long
            tr.kwargs['epochs'] = 1
            MC_SAMPLES = 1
        tr.kwargs['base_lr'] = args.base_lr
        tr.kwargs['head_lr'] = args.head_lr
        tr.kwargs['alt_lr'] = args.alt_lr
        s = s(); t = t()
        res = run_experiment(s, t, b, h, tr, 
            args.experiment_seed,
            param_search=args.param_search, 
            verbose=args.verbose, 
            device=args.device)
        res['source'] = sname
        res['target'] = tname
        res['dataset_seed'] = args.dataset_seed
        res['experiment_seed'] = args.experiment_seed
        res['group_num'] = group['name']
        data.append(res)
        # incrementally save results
        g = group['name']
        ds = args.dataset_seed
        es = args.experiment_seed
        if args.test: ext += '-t'
        elif args.param_search: ext += '-p'
        else: ext = ''
        dir_loc = 'out/results'
        Path(dir_loc).mkdir(parents=True, exist_ok=True)
        write_loc = f'{dir_loc}/{g}-{ds}-{es}{ext}.csv'
        pd.DataFrame(data).to_csv(write_loc)
        print('Done', enumber); enumber += 1