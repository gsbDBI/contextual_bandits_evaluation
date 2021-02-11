"""
PolicyTree utility functions.
"""
import numpy as np
import autograd.numpy as npg

from autograd import elementwise_grad as grad
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from statsmodels.api import add_constant
from scipy.optimize import minimize

numpy2ri.activate()
pt = importr("policytree")

__all__ = ["fit_policytree", "predict_policytree", "fit_policylogit", "predict_policylogit"]


""" policytree """


def fit_policytree(xtrain, gammatrain, depth=2):
    pol = pt.policy_tree(X=xtrain, Gamma=gammatrain, depth=depth)
    return pol


def predict_policytree(pol, xtest):
    w = pt.predict_policy_tree(pol, np.atleast_2d(xtest))
    w = np.array(w, dtype=np.int_) - 1
    return w


""" multinomial logit """


def fit_policylogit(xtrain, gammatrain, reg_mult=0.0):
    pol = fit_mlogit(contexts=add_constant(xtrain), scores_matrix=gammatrain, reg_mult=reg_mult).thetahat
    return pol


def predict_policylogit(pol, xtest):
    w = np.argmax(add_constant(xtest) @ pol, axis=1)
    return w


def logsumexp(a, b=1, axis=1):
    cmax = npg.max(a, axis=axis, keepdims=True)
    s = npg.sum(b * npg.exp(a - cmax), axis=1)
    return npg.log(s) + cmax.flatten()


def l1_penalty(theta):
    return npg.sum(npg.abs(theta))


def fit_mlogit(contexts, scores_matrix, reg_mult=0.0):
    def obj(theta):
        value = average_scores(theta, contexts=contexts, scores_matrix=scores_matrix)
        reg = reg_mult * l1_penalty(theta)
        return value + reg

    obj_grad = grad(obj)
    theta_shape = contexts.shape[1], scores_matrix.shape[1]
    opt = minimize(
        x0=np.random.normal(size=theta_shape).flatten(), fun=obj, jac=obj_grad, method='SLSQP', options={"maxiter": 200}
    )
    opt.thetahat = opt.x.reshape(*theta_shape)
    return opt


def average_scores(theta, contexts, scores_matrix):
    y = np.argmax(scores_matrix, axis=1)
    theta_shape = (contexts.shape[1], scores_matrix.shape[1])
    theta = theta.reshape(*theta_shape)
    Xt = contexts @ theta
    n = len(Xt)
    num = Xt[np.arange(n), y]
    denom = logsumexp(a=Xt)
    diff = num - denom
    value = npg.mean(diff)
    return -value
