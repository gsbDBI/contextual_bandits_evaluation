"""
Ridge regression utility functions.
"""
import numpy as np
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.tree import DecisionTreeRegressor

__all__ = [
    "ridge_init",
    "ridge_update",
    "ridge_muhat_lfo",
    "ridge_muhat_lfo_pai",
    "ridge_muhat_DM",
]


def ridge_init(p, K):
    A = np.empty((K, p + 1, p + 1))
    Ainv = np.empty((K, p + 1, p + 1))
    for k in range(K):
        A[k] = np.eye(p + 1)
        Ainv[k] = np.eye(p + 1)
    b = np.zeros((K, p + 1))
    theta = np.zeros((K, p + 1))
    return A, Ainv, b, theta


def ridge_update(A, b, xt, ytobs):
    xt1 = np.empty(len(xt) + 1)
    xt1[0] = 1.0
    xt1[1:] = xt
    A += np.outer(xt1, xt1)
    b += ytobs * xt1
    Ainv = np.linalg.inv(A)
    theta = Ainv @ b
    return A, Ainv, b, theta


def ridge_muhat_DM(xs, ws, yobs, K):
    if len(xs.shape) == 1:
        xs = np.reshape(xs, [len(xs), 1])
    T, p = xs.shape
    A, Ainv, b, theta = ridge_init(p, K)
    muhat = np.zeros((T, K))
    for t in range(T):
        w = ws[t]
        A[w], Ainv[w], b[w], theta[w] = ridge_update(A[w], b[w], xs[t], yobs[t])
    for t in range(T):
        xt1 = np.empty(p + 1)
        xt1[0] = 1.0
        xt1[1:] = xs[t]
        for w in range(K):
            muhat[t, w] = theta[w] @ xt1
    return muhat


def ridge_muhat_lfo(xs, ws, yobs, K, alpha=1.):
    """
    Return plug-in estimates of arm expected reward.

    INPUT
        xs: observed covariates X_t of shape [T, p]
        ws: pulled arm of shape [T]
        yobs: observed outcome of shape [T]
        K: number of arms
        alpha: ridge regression regularization parameter

    OUTPUT:
        muhat: \muhat_t(X_t, w) of shape [T, K]
    """
    T, p = xs.shape
    A, Ainv, b, theta = ridge_init(p, K)
    muhat = np.zeros((T, K))
    for t in range(T):
        for w in range(K):
            xt1 = np.empty(p + 1)
            xt1[0] = 1.0
            xt1[1:] = xs[t]
            muhat[t, w] = theta[w] @ xt1
            if ws[t] == w:
                A[w], Ainv[w], b[w], theta[w] = ridge_update(A[w], b[w], xs[t], yobs[t])
    return muhat


def ridge_muhat_lfo_pai(xs, ws, yobs, K, batch_sizes, alpha=1.):
    """
    Return plug-in estimates of arm expected reward.

    INPUT
        xs: observed covariates X_t of shape [T, p]
        ws: pulled arm of shape [T]
        yobs: observed outcome of shape [T]
        K: number of arms
        alpha: ridge regression regularization parameter

    OUTPUT:
        muhat: \muhat_t(X_s, w)_{s=1:T} of shape [T, T, K]
    """
    if len(xs.shape) == 1:
        xs = np.reshape(xs, [len(xs), 1])
    T, p = xs.shape
    A, Ainv, b, theta = ridge_init(p, K)
    muhat = np.zeros((T, T, K))

    batch_cumsum = np.cumsum(batch_sizes)
    batch_cumsum = [0] + list(batch_cumsum)

    for l,r in zip(batch_cumsum[:-1], batch_cumsum[1:]):
        for w in range(K):
            # predict from t to T
            xt1 = np.empty((p + 1, T))
            xt1[0] = 1.0
            xt1[1:] = xs.transpose()
            muhat[l:r, :, w] = theta[w] @ xt1 # dim (T)
        for t in range(l, r):
            w = ws[t]
            A[w], Ainv[w], b[w], theta[w] = ridge_update(A[w], b[w], xs[t], yobs[t])
    return muhat



