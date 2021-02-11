"""
This script contains helper functions to do inference including computing scores and statistics of estimating policy values.
"""
import numpy as np
from adaptive.compute import *
from scipy.stats import norm
import warnings


def sample_mean(rewards, arms, K):
    """
    Compute F_{t} measured sample mean estimator

    INPUT:
        - rewards: observed rewards of shape [T]
        - arms: pulled arms of shape [T]
        - K: number of arms

    Output:
        - estimate: F_{t} measured sample mean estimator of shape [T,K]
    """
    # return F_t measured sample mean
    T = len(arms)
    W = expand(np.ones(T), arms, K)
    Y = expand(rewards, arms, K)
    estimate = np.cumsum(W * Y, 0) / np.maximum(np.cumsum(W, 0), 1)
    return estimate


def aw_scores(yobs, ws, balwts, K, muhat=None):
    """
    Compute AIPW/doubly robust scores. Return IPW scores if muhat is None.
    e[t] and mu[t, w] are functions of the history up to t-1.
    INPUT
        - yobs: observed rewards, shape [T]
        - ws: pulled arms, shape [T]
        - balwts: inverse probability score 1[W_t=w]/e_t(w) of pulling arms, shape [T, K]
        - K: number of arms
        - muhat: plug-in estimator of arm outcomes, shape [T, K]
    OUTPUT
        - scores: AIPW scores, shape [T, K]
    """
    scores = expand(balwts * yobs, ws, K)  # Y[t]*W[t]/e[t] term
    if muhat is not None:  # (1 - W[t]/e[t])*mu[t,w] term
        scores += (1 - expand(balwts, ws, K)) * muhat
    return scores


def aw_estimate(scores, policy, evalwts=None):
    """
    Estimate policy value via non-contextual adaptive weighting.

    INPUT
        - scores: AIPW score, shape [T, K]
        - policy: policy matrix pi(X_t, w), shape [T, K]
        - evalwts: non-contextual adaptive weights h_t, shape [T]
    OUTPUT
        - estimated policy value.
    """
    if evalwts is None:
        evalwts = np.ones((scores.shape[0]))
    return np.sum(evalwts * np.sum(scores * policy, -1)) / np.sum(evalwts)


def aw_var(scores, estimate, policy, evalwts=None):
    """
    Variance of policy value estimator via non-contextual adaptive weighting.

    INPUT
        - scores: AIPW score, shape [T, K]
        - estimate: policy value estimate
        - policy: policy matrix pi(X_t, w), shape [T, K]
        - evalwts: non-contextual adaptive weights h_t, shape [T]
    OUTPUT
        - variance of policy value estimate

    var =  sum[t=0 to T] h[t]^2 * (sum[w] scores[t, w] * policy[t, w] - estimate)^2 
          --------------------------------------------------------------------------
                          (sum[t=0 to T] h[t])^2
    """
    if evalwts is None:
        evalwts = np.ones((scores.shape[0]))
    return np.sum((np.sum(policy * scores, -1) - estimate) ** 2 * evalwts ** 2) / (np.sum(evalwts)**2)


def estimate(w, gammahat, policy, policy_value):
    """
    Return bias and variance of policy evaluation via non-contextual weighting.

    INPUT
        - w: non-contextual weights of shape [T]
        - gammahat: AIPW score of shape [T, K]
        - policy: policy matrix pi(X_t, w), shape [T, K]
        - policy_value: ground truth policy value

    OUTPUT
        - np.array([bias, var])
    """
    estimate = aw_estimate(gammahat, policy, w)
    var = aw_var(gammahat, estimate, policy, w)
    bias = estimate - policy_value
    return np.array([bias, var])


def calculate_continuous_X_statistics(h, gammahat, policy, policy_value):
    """
    Return bias and variance of policy evaluation via contextual weighting.

    INPUT
        - h: adaptive weights h_t(X_s) of size (T, T)
        - gammahat: AIPW score of shape [T, K]
        - policy: policy matrix pi(X_t, w), shape [T, K]
        - policy_value: ground truth policy value

    OUTPUT:
        - np.array([bias, var])
    """
    T, _ = h.shape
    Z = np.sum(h, axis=0)  # size (T)
    gamma_policy = np.sum(gammahat * policy, 1)
    ht_Xt_Z = collect(h, np.arange(T))
    ht_Xt_Z[Z > 1e-6] = ht_Xt_Z[Z > 1e-6] / \
        Z[Z > 1e-6]  # size (T), h_t(X_t) / Z(X_t)
    B = ht_Xt_Z * gamma_policy
    h_Z = np.copy(h)
    h_Z[:, Z > 1e-6] = h_Z[:, Z > 1e-6] / Z[Z > 1e-6]
    V_estimate = np.sum((B - np.sum(h_Z * B, 1)) ** 2)
    Q_estimate = np.sum(B)

    return np.array([Q_estimate-policy_value, V_estimate])


def analyze_by_continuous_X(probs, gammahat, policy, policy_value):
    """
    Generate statistics of estimating policy values with different estimators.

    INPUT
        probs: assignment probability matrix, e_t(X_s, w), shape [T, T, K]
        gammahat: AIPW score of shape [T, K]
        policy: policy matrix pi(X_t, w), shape [T, K]
        policy_value: ground truth policy value.

    OUTPUT
        a dictionary with keys to be evaluation method and values to be (bias, variance) of corresponding estimator.
    """
    T, K = gammahat.shape
    all_condVars = np.sum(policy ** 2 / probs, -1)  # (T, T)
    all_condVars_inverse = np.zeros_like(all_condVars)
    all_condVars_inverse[all_condVars > 1e-6] = 1 / \
        all_condVars[all_condVars > 1e-6]
    expected_condVars = np.mean(all_condVars, 1)
    expected_condVars_inverse = np.mean(all_condVars_inverse, 1)
    #expected_condVars_inverse = np.zeros_like(expected_condVars)
    #expected_condVars_inverse[expected_condVars > 1e-6] = 1 / expected_condVars[expected_condVars > 1e-6]

    return dict(
        uniform=estimate(np.arange(T), gammahat, policy, policy_value),
        propscore_expected=estimate(
            expected_condVars_inverse, gammahat, policy, policy_value),
        propscore_X=calculate_continuous_X_statistics(
            all_condVars_inverse, gammahat, policy, policy_value),
        lvdl_expected=estimate(np.sqrt(expected_condVars_inverse),
                               gammahat, policy, policy_value),
        lvdl_X=calculate_continuous_X_statistics(np.sqrt(
            all_condVars_inverse), gammahat, policy, policy_value),
    )
