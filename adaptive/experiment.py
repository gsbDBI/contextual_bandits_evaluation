"""
This script contains functions to run a contextual bandit experiment.
"""
from adaptive.region import *
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from adaptive.thompson import *


def run_experiment(xs: np.ndarray, ys: np.ndarray, config: dict, batch_sizes) -> dict:
    """
    Run contextual bandit experiment.

    INPUT
        - xs: covariate X_t of shape [T, p]
        - ys: potential outcomes of shape [T, K]
        - config: dictionary of experiment configs
        - batch_sizes: nparray of batch sizes

    OUTPUT
        - pulled arms, observed rewards, assignment probabilities.
    """
    T, K = ys.shape
    _, p = xs.shape
    ws = np.empty(T, dtype=np.int_)
    yobs = np.empty(T)
    probs = np.zeros((T, T, K))
    Probs_t = np.zeros((T, K))
    floor_start = config['floor_start']
    floor_decay = config['floor_decay']

    if config['bandit_model'] == 'TSModel':
        bandit_model = LinTSModel(
            K=K, p=p, floor_start=floor_start, floor_decay=floor_decay)
        draw_model = partial(draw_thompson)
    elif config['bandit_model'] == 'RegionModel':
        bandit_model = RegionModel(
            T=T, K=K, p=p, floor_start=floor_start, 
            floor_decay=floor_decay, prior_sigma2=.1)
    elif config['bandit_model'] == 'kasy-RegionModel':
        bandit_model = RegionModel(
            T=T, K=K, p=p, floor_start=floor_start, 
            floor_decay=floor_decay, prior_sigma2=.1, kasy=True)

    # uniform sampling at the first batch
    batch_size_cumsum = list(np.cumsum(batch_sizes))
    ws[: batch_size_cumsum[0]] = np.arange(batch_size_cumsum[0]) % K
    yobs[: batch_size_cumsum[0]] = ys[np.arange(
        batch_size_cumsum[0]), ws[: batch_size_cumsum[0]]]
    probs[: batch_size_cumsum[0]] = 1/K
    Probs_t[: batch_size_cumsum[0]] = 1/K
    if config['bandit_model'].endswith('RegionModel'):
        bandit_model = update_region(xs[:batch_size_cumsum[0]],
                                     ws[:batch_size_cumsum[0]], 
                                     yobs[:batch_size_cumsum[0]],
                                     Probs_t[:batch_size_cumsum[0]], 
                                     1, bandit_model)
    elif config['bandit_model'] == 'TSModel':
        bandit_model = update_thompson(xs[:batch_size_cumsum[0]],
                                       ws[:batch_size_cumsum[0]],
                                       yobs[:batch_size_cumsum[0]],
                                       bandit_model)

    # adaptive sampling at the subsequent batches
    for idx, (f, l) in enumerate(zip(batch_size_cumsum[:-1], batch_size_cumsum[1:])):
        if config['bandit_model'].endswith('RegionModel'):
            w, p = draw_region(
                xs=xs, model=bandit_model, start=f, end=l)
        else:
            w, p = draw_thompson(
                xs=xs, model=bandit_model, 
                start=f, end=l, current_t=f)  
        yobs[f:l] = ys[np.arange(f, l), w]
        ws[f:l] = w
        probs[f:l, :] = np.stack([p] * (l-f))
        Probs_t[f:l] = p[f:l]
        if config['bandit_model'].endswith('RegionModel'):
            bandit_model = update_region(
                xs[:l], ws[:l], yobs[:l], Probs_t[:l], idx+2, bandit_model)
        elif config['bandit_model'] == 'TSModel':
            bandit_model = update_thompson(
                    xs[f:l], ws[f:l], yobs[f:l],  bandit_model)

    # probs are assignment probabilities e_t(X_s, w) of shape [T, T, K]
    data = dict(yobs=yobs, ws=ws, xs=xs, ys=ys, probs=probs,
                fitted_bandit_model=bandit_model)

    return data
