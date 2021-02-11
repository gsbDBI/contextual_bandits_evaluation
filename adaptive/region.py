"""
This script contains implementation of a customized Thompson sampling agent, which firstly discretizes the covariate space using PolicyTree and then conduct TS sampling non-contextually within each subspace. 
"""
import numpy as np
from adaptive.compute import *
from adaptive.ridge import *
from adaptive.policy import *
from adaptive.inference import aw_scores
from itertools import product
from copy import deepcopy

_all_ = [
    "RegionModel",
    "update_region",
    "draw_region"
]

def bayesian_update(
    like_mu: np.ndarray,
    like_sigma2: np.ndarray,
    prior_mu: np.ndarray,
    prior_sigma2: np.ndarray,
    n: int):
    """
    Posterior with normal prior, normal likelihood.
    Ref: http://www.ams.sunysb.edu/~zhu/ams570/Bayesian_Normal.pdf (page 4)
    """
    post_sigma2 = 1 / (1 / prior_sigma2 + n / like_sigma2)
    post_mu = post_sigma2 * (prior_mu / prior_sigma2 + n * like_mu / like_sigma2)
    return post_mu, post_sigma2

class RegionModel:
    """
    Divides the covariate space into regions using a tree policy,
    then applies a Thompson Sampling MAB with gaussian prior and likelihood within
    each region.

    Parameters
    ----------
    T, K, p: int
        config parameters

    floor: float
        assignment probability floor

    depth: int
        policy tree depth

    prior_mu, prior_sigma2: float
        [IMPORTANT]
        If prior_sigma2 small, algorithm's belief in zero-effect is stronger, so convergence is slower.
        As prior_sigma2 tends to inf, we get our usual frequentist / uninformative thompson sampling.

    mc: int
        Number of monte-carlo draws to compute thompson probabilities.

    kasy: bool
        Experimental functionality.
        When enabled, uses Kasy and Sautmann (2019)'s 'exploration sampling'
        by applying the transformation: e_{t}(w) ↦ e_{t}(w)*(1 - e_{t}(w)).
        Note: this is applied *before* the applying the floor.
    """

    def __init__(self,
                 T, K, p,
                 floor_start: float = 0.05,
                 floor_decay: float = 0.001,
                 depth: int = 2,
                 prior_mu: float = 0.,
                 prior_sigma2: float = 1.,
                 balanced: bool = False,
                 mc: int = 20,
                 kasy: bool = False):
        self.T = T
        self.K = K
        self.p = p
        self.depth = depth
        self.floor_start = floor_start
        self.floor_decay = floor_decay
        self.mc = mc
        self.balanced = balanced
        self.kasy = kasy

        # Policy that determines regions
        self._policy = None

        # Prior and posterior parameters
        self._prior_mu = np.full(shape=(K, K), fill_value=prior_mu)
        self._prior_sigma2 = np.full(shape=(K, K), fill_value=prior_sigma2)
        self._mu = np.full(shape=(K, K), fill_value=np.nan)
        self._sigma2 = np.full(shape=(K, K), fill_value=np.nan)

        # Thompson sampling probabilities
        self._tsprobs = np.full(shape=(K, K), fill_value=np.nan)


def update_region(xs_t, ws_t, yobs_t, probs_t, current_t, model: RegionModel):
    """
    Returns a new RegionModel with same configurations but
    updated policy, statistics and Thompson Sampling probabilities
    for each region.

    xs, ws, yobs, probs: np.ndarray
        Input arrays.

    [IMPORTANT] The model is retrained rather than 'updated',
    so inputs must range from time zero up to current time.
    Use: update_region(xs[:t], ws[:t], ...).

    # TODO:
        + Verify correctness of balancing method
        + Study overfitting
            (Optimal policy is fitted and 'evaluated' on same data).
    """
    model = deepcopy(model)
    balwts_t = 1 / collect(probs_t, ws_t)

    #current_t = len(xs_t)
    #floor = model.floor_start - (model.floor_start -  model.floor_decay) * current_t / model.T
    floor = model.floor_start /  ( 5 * current_t)** model.floor_decay

    # Compute scores, fit new policy and predict new region
    muhat = ridge_muhat_lfo(xs_t, ws_t, yobs_t, model.K)
    gammahat = aw_scores(yobs_t, ws_t, balwts_t, model.K, muhat)
    model._policy = fit_policytree(xs_t, gammahat, depth=model.depth)
    region = predict_policytree(model._policy, xs_t)

    # Update statistics for each region
    for r, w in product(range(model.K), range(model.K)):
        # Observations in region r that were assigned to arm w
        idx_w = ws_t == w
        idx_r = region == r
        idx_rw = idx_w & idx_r
        n_rw = np.sum(idx_rw)
        # If not enough data, then default to uncond. prob. for each arm
        if n_rw >= 2:
            idx = idx_rw
            n = n_rw
        else:
            idx = idx_w
            n = np.sum(idx_w)

        if model.balanced:
            model._mu[r, w], model._sigma2[r, w] = bayesian_update(
                like_mu=np.mean(gammahat[idx, w]),
                like_sigma2=np.var(gammahat[idx, w]),
                prior_mu=model._prior_mu[r, w],
                prior_sigma2=model._prior_sigma2[r, w],
                n=n)
        else:
            model._mu[r, w], model._sigma2[r, w] = bayesian_update(
                like_mu=np.mean(yobs_t[idx]),
                like_sigma2=np.var(yobs_t[idx]),
                prior_mu=model._prior_mu[r, w],
                prior_sigma2=model._prior_sigma2[r, w],
                n=n)


    # Updates thompson sampling probabilities for each region
    for r in range(model.K):
        mcdraws = [np.random.normal(model._mu[r], np.sqrt(model._sigma2[r])) for _ in range(model.mc)]
        argmax = np.argmax(mcdraws, axis=1)
        freq = np.bincount(argmax, minlength=model.K).astype(float) / model.mc
        freq = apply_floor(freq , floor)
        if model.kasy:
            freq = ks_transform(freq)
        model._tsprobs[r] = freq
    return model


def draw_region(xs=None, model=None, start=None, end=None) -> dict:
    """
    Uses the fitted policy to predict a region,
    then draws from appropriate thompson sampling probabilities.
    ! calculate the probabilities for all xs, but only draw for time t 
    """
    r = predict_policytree(model._policy, xs)
    ps = model._tsprobs[r]
    w = [np.random.choice(model.K, p=ps[t]) for t in range(start, end)] 
    return w, ps


def draw_region_discrete_X(x_space=None, model=None, idx=None) -> dict:
    """
    Uses the fitted policy to predict a region,
    then draws from appropriate thompson sampling probabilities.
    ! calculate the probabilities for all xs, but only draw for time t 
    """
    r = predict_policytree(model._policy, x_space)
    ps = model._tsprobs[r]
    w = [np.random.choice(model.K, p=ps[i]) for i in idx] 
    return w, ps
