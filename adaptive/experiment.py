"""
This script contains functions to run contextual bandits and multi-armed bandits.
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

def ts_mab_probs(sum, sum2, neff, prev_t, floor_start=0.005, floor_decay=0.0, num_mc=20):
    """
    Return arm assignment probabilities of Thompson sampling agent with prior N(0, 1) and update the posterior mean and var    iance based on data.

    INPUT:
        - sum: summation of arm rewards of shape K (number of arms)
        - sum2: summation of squard arm rewards of shape K (number of arms)
        - neff: number of observations on each arm, shape K (number of arms)
        - prev_t: t-1
        - floor_start: assignment probability floor starting value
        - floor_decay: assignment probability floor decaying rate
        - num_mc: number of Monte Carlo simulations to calculate poterior probability

    OUTPUT:
        - probs: poterior probability computed by Thompson sampling
    """
    # -------------------------------------------------------
    # agent prior N(0,1)
    K = len(sum)
    Z = np.random.normal(size=(num_mc, K))

    # -------------------------------------------------------
    # estimate empirical mean and variance
    mu = sum / np.maximum(neff, 1)
    var = sum2 / np.maximum(neff, 1) - (sum / np.maximum(neff, 1)) ** 2
    # var = 1

    # -------------------------------------------------------
    # calculate posterior
    # 1/sigma(n)^2 = 1/sigma(0)^2 + n / sigma^2
    # sigma(n)^2 = 1 / (n / sigma^2 + 1/sigma(0)^2)
    posterior_var = 1 / (neff / var + 1 / 1.0)
    # mu(n) = sigma^2 / (n * sigma(0)^2 + sigma^2) * mu(0) + n*sigma(0)^2 /
    ## (n*sigma(0)^2+sigma^2) * mu
    posterior_mean = neff / (var + neff) * mu

    idx = np.argmax(Z * np.sqrt(posterior_var) + posterior_mean, axis=1)
    w_mc = np.array([np.sum(idx == k) for k in range(K)])
    p_mc = w_mc / num_mc

    # -------------------------------------------------------
    # assignment probability floor 1/(t)
    probs = apply_floor(p_mc, amin=floor_start / (prev_t + 1) ** floor_decay)

    return probs

def run_mab_experiment(ys,
                       initial=0,
                       floor_start=0.005,
                       floor_decay=0.0,
                       init_sum=None, init_sum2=None,
                       init_neff=None):
    """
    Run multi-arm bandits experiment.

    INPUT:
        - ys: rewards from environment of shape [T].
        - initial: initial number of samples for each arm to do pure exploration.
        - floor_start: assignment probability floor starting value
        - floor_decay: assignment probability floor decaying rate
        (assignment probability floor = floor start * t ^ {-floor_decay})
        - init_sum: prior summation of rewards of each arm, shape [K]
        - init_sum2: prior summation of squared rewards of each arm, shape [K]
        - init_neff: prior number of observations of each arm, shape [K]

    OUTPUT:
        - a dictionary describing generated samples:
            - arms: indices of pulled arms of shape [T]
            - rewards: rewards of shape [T]
            - ndraws: number of samples on each arm up to time t, shape [T, K]
            - probs: assignment probabilities of shape [T, K]
    """

    T, K = ys.shape
    T0 = initial * K
    arms = np.empty(T, dtype=np.int_)
    rewards = np.empty(T)
    probs = np.empty((T, K))

    # Initialize if at the middle of an experiment
    sum = np.zeros(K) if init_sum is None else init_sum
    sum2 = np.zeros(K) if init_sum2 is None else init_sum2
    neff = np.zeros(K) if init_neff is None else init_neff
    ndraws = np.zeros((T, K))

    for c, t in enumerate(range(T)):

        if t < T0:
            # Run first "batch": deterministically select each arm `initial`
            # times
            p = np.full(K, 1 / K)
            w = t % K
        else:
            p = ts_mab_probs(
                sum, sum2, neff, t, floor_start=floor_start, floor_decay=floor_decay)
            w = np.random.choice(K, p=p)

        # TS with Gaussian prior
        sum[w] += ys[t, w]
        sum2[w] += ys[t, w] ** 2
        neff[w] += 1

        arms[t] = w
        rewards[t] = ys[t, w]
        probs[t] = p
        ndraws[t] = neff

    data = {"arms": arms,
            "rewards": rewards,
            "ndraws": ndraws,
            "probs": probs,
            "future_p": ts_mab_probs(sum, sum2, neff, t, floor_start=floor_start, floor_decay=floor_decay)}
    return data
