import numpy as np
from adaptive.compute import *
from sklearn.linear_model import RidgeCV

class LinTSModel:
    """
    Linear Thompson Sampling. Computes a different model for each arm.

    Reference: Agrawal, Goyal (2012) Thompson Sampling for Contextual Bandits with
    Linear Payoffs. (See in particular Section 2.2)
    https://arxiv.org/pdf/1209.3352.pdf

    Parameters
    ----------
    K: int
        Number of arms
    p: int
        Dimension of contexts
    floor_start: float
        Assignment probability floor starting point
    floor_decay: float
        Assignment probability floor decaying rate
        floor = floor_start * t^{-floor_decay}
    """
    
    def __init__(self, K, p, floor_start, floor_decay, num_mc=100):
        self.num_mc = num_mc
        self.K = K
        self.p = p 
        self.floor_start = floor_start
        self.floor_decay = floor_decay
        self.mu = np.zeros((K, p+1))
        self.V = np.zeros((K, p+1, p+1))
        self.X = [[] for _ in range(K)]
        self.y = [[] for _ in range(K)]

def _update_thompson(x, y, w, model):
    model.X[w].extend(x)
    model.y[w].extend(y)
    regr = RidgeCV(alphas=[1e-5, 1e-4, 
        1e-3, 1e-2, 1e-1, 1.0]).fit(model.X[w], model.y[w])
    yhat = regr.predict(model.X[w])
    model.mu[w] = [regr.intercept_, *list(regr.coef_)]
    X = np.concatenate([np.ones((len(model.X[w]), 1)), model.X[w]], axis=1)
    B = np.matmul(X.T, X) + regr.alpha_ * np.eye(model.p+1)
    model.V[w] = np.mean((model.y[w] - yhat) ** 2) * np.linalg.inv(B)
    return model

def update_thompson(xs, ws, ys, model):
    """
    Updates LinTS agent with newly observed data.
    """
    for w in range(model.K):
        model = _update_thompson(xs[ws==w], ys[ws==w], w, model)
    return model

def draw_thompson(xs, model, start, end, current_t):
    """
    Draws arms with a LinTS agent for the observed covariates.
    """
    T, p = xs.shape
    xt = np.concatenate([np.ones((T, 1)), xs], axis=1)
    floor = model.floor_start / current_t ** model.floor_decay
    coeff = np.empty((model.K, model.num_mc, p + 1))
    for w in range(model.K):
        coeff[w] = np.random.multivariate_normal(
                model.mu[w], model.V[w], size=model.num_mc
                )
    draws = np.matmul(coeff, xt.T) # (model.K, model.num_mc, T)
    ps = np.empty((T, model.K))
    for s in range(T):
        ps[s, :] = np.bincount(np.argmax(draws[:, :, s], axis=0), minlength=model.K) / model.num_mc
        ps[s, :] = apply_floor(ps[s, :], floor)
    w = [np.random.choice(model.K, p=ps[t]) for t in range(start, end)]
    return w, ps
