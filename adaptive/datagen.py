"""
This script includes functions of data generating processes.
"""
import numpy as np
from scipy.stats import multivariate_normal
from adaptive.compute import expand
import warnings
import pandas as pd
import os


def generate_bandit_data(X=None, y=None, noise_std=1.0, signal_strength=1.0):
    """
    Generate covariates and potential outcomes from a classification dataset.

    Returns:
        - data: covariates and potential outcomes
        - mus: arm expected reward over the covariate space
    """
    shuffler = np.random.permutation(len(X))
    xs = X[shuffler]
    ys = y[shuffler]
    T, p = xs.shape
    T = min(T, 20000)
    xs, ys = xs[:T], ys[:T]
    K = len(np.unique(ys))
    muxs = np.array(pd.get_dummies(ys), dtype=float) * signal_strength
    ys = muxs + np.random.normal(scale=noise_std, size=(T, K))
    mus = np.bincount(np.array(y, dtype=int)) / T
    data = dict(xs=xs, ys=ys, muxs=muxs, T=T, p=p, K=K)
    return data, mus


def simple_tree_data(T, K=4, p=3, noise_std=1.0, split=1.676, signal_strength=1.0, seed=None, noise_form='normal'):
    """
    Generate covariates and potential outcomes of a synthetic dataset.

    Splits the covariate space into four regions.
    In each region one of the arms is best on average (see diagram below).

    The larger the 'split' is, the larger the region where arm w=0 is best.
        ie. for more personalization decrease split toward zero.

    Arms w>3 are never best, and covariates x>2 are always noise.

    Default values give optimal/(best_fixed) ratio at 10%.

            ^ x1
            |
        Arm 1 best  |    Arm 3 best
            |       |
    ~~~~~~~~|~(split,split)~~~~~~
            |       |
        Arm 0 best  |    Arm 2 best
    ------(0,0)------------------>x0
            |       |
            |       |
            |       |
            |       |
    Returns:
        - data: covariates and potential outcomes
        - mus: arm expected reward over the covariate space
    """
    assert p >= 2
    assert K >= 4
    assert split >= 0

    rng = np.random.RandomState(seed)
    # Generate experimental data
    xs = rng.normal(size=(T, p))

    r0 = (xs[:, 0] < split) & (xs[:, 1] < split)
    r1 = (xs[:, 0] < split) & (xs[:, 1] > split)
    r2 = (xs[:, 0] > split) & (xs[:, 1] < split)
    r3 = (xs[:, 0] > split) & (xs[:, 1] > split)

    wxs = np.empty((T, K), dtype=int)
    wxs[r0] = np.eye(K)[0]
    wxs[r1] = np.eye(K)[1]
    wxs[r2] = np.eye(K)[2]
    wxs[r3] = np.eye(K)[3]
    muxs = wxs * signal_strength
    if noise_form == 'normal':
        ys = muxs + np.random.normal(scale=noise_std, size=(T, K))
    else:
        ys = muxs + np.random.uniform(-noise_std, noise_std, size=(T, K))

    mvn = multivariate_normal([0, 0], np.eye(2))
    mus = np.zeros((K))
    mus[0] = mvn.cdf([split, split])
    mus[1] = mvn.cdf([split, np.inf]) - mvn.cdf([split, split])
    mus[2] = mvn.cdf([split, np.inf]) - mvn.cdf([split, split])
    mus[3] = mvn.cdf([-split, -split])
    mus = mus * signal_strength

    data = dict(xs=xs, ys=ys, muxs=muxs, wxs=wxs)

    return data, mus
