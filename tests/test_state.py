from adelie.logger import logger
import logging
logger.setLevel(logging.DEBUG)

import adelie.state as mod
import adelie.matrix as matrix
import numpy as np


def test_state_gaussian_pin_naive():
    n = 1000
    p = 100
    G = 2

    X = matrix.dense(np.random.normal(0, 1, (n, p)), method="naive", n_threads=4)
    groups = np.array([0, 1])
    alpha = 1.0
    penalty = np.random.uniform(0, 1, G)
    screen_set = np.array([0, 1])
    lmda_path = np.array([0.1, 1.0, 0.5])
    rsq = 0.0
    resid = np.random.normal(0, 1, n)
    y_mean = 0
    y_var = 1
    weights = np.random.uniform(1, 2, n)
    weights /= np.sum(weights)
    screen_beta = np.zeros(p)
    screen_is_active = np.zeros(screen_set.shape[0], dtype=bool)
    active_set_size = 0
    active_set = np.empty(G, dtype=int)

    state = mod.gaussian_pin_naive(
        X=X,
        y_mean=y_mean,
        y_var=y_var,
        groups=groups,
        alpha=alpha,
        penalty=penalty,
        weights=weights,
        screen_set=screen_set,
        lmda_path=lmda_path,
        rsq=rsq,
        resid=resid,
        screen_beta=screen_beta,
        screen_is_active=screen_is_active,
        active_set_size=active_set_size,
        active_set=active_set,
    )

    state.check(method="assert")

    assert id(X) == id(state.X)
    assert np.allclose(groups, state.groups)
    assert np.allclose(alpha, state.alpha)
    assert np.allclose(penalty, state.penalty)
    assert np.allclose(screen_set, state.screen_set)
    assert np.allclose(lmda_path, state.lmda_path)
    assert np.allclose(rsq, state.rsq)
    assert np.allclose(screen_beta, state.screen_beta)
    assert np.allclose(screen_is_active, state.screen_is_active)
    assert state.iters == 0


def test_state_gaussian_pin_cov():
    n = 1000
    p = 100
    G = 2

    X = np.random.normal(0, 1, (n, p))
    A = matrix.dense(X.T @ X / n, method="cov", n_threads=4)
    groups = np.array([0, 1])
    alpha = 1.0
    penalty = np.random.uniform(0, 1, G)
    screen_set = np.array([0, 1])
    lmda_path = np.array([0.1, 1.0, 0.5])
    rsq = 0.0
    screen_beta = np.zeros(p)
    screen_grad = X.T @ np.random.normal(0, 1, n)
    screen_is_active = np.zeros(screen_set.shape[0], dtype=bool)
    active_set_size = 0
    active_set = np.empty(G, dtype=int)

    state = mod.gaussian_pin_cov(
        A=A,
        groups=groups,
        alpha=alpha,
        penalty=penalty,
        screen_set=screen_set,
        lmda_path=lmda_path,
        rsq=rsq,
        screen_beta=screen_beta,
        screen_grad=screen_grad,
        screen_is_active=screen_is_active,
        active_set_size=active_set_size,
        active_set=active_set,
    )

    state.check(method="assert")

    assert id(A) == id(state.A)
    assert np.allclose(groups, state.groups)
    assert np.allclose(alpha, state.alpha)
    assert np.allclose(penalty, state.penalty)
    assert np.allclose(screen_set, state.screen_set)
    assert np.allclose(lmda_path, state.lmda_path)
    assert np.allclose(rsq, state.rsq)
    assert np.allclose(screen_beta, state.screen_beta)
    assert np.allclose(screen_grad, state.screen_grad)
    assert np.allclose(screen_is_active, state.screen_is_active)
    assert state.iters == 0


def test_state_gaussian_naive():
    n = 3
    p = 100
    G = 2

    _X = np.random.normal(0, 1, (n, p))
    X = matrix.dense(_X, method="naive", n_threads=4)
    y = np.random.normal(0, 1, n)
    X_means = np.mean(_X, axis=0)
    groups = np.array([0, 1])
    group_sizes = np.array([1, 99])
    y_mean = 0.0
    y_var = 1.0
    alpha = 1.0
    penalty = np.random.uniform(0, 1, G)
    weights = np.random.uniform(1, 2, n)
    weights /= np.sum(weights)
    offsets = np.zeros(n)
    screen_set = np.array([0, 1])
    lmda_path = np.array([0.1, 1.0, 0.5])
    lmda_max = 0.9
    rsq = 0.0
    lmda = 2.0
    grad = np.random.normal(0, 1, p)
    resid = y
    resid_sum = np.sum(resid)
    screen_beta = np.zeros(p)
    screen_is_active = np.zeros(screen_set.shape[0], dtype=bool)
    active_set_size = 0
    active_set = np.empty(G, dtype=int)

    state = mod.gaussian_naive(
        X=X,
        y=y,
        X_means=X_means,
        y_mean=y_mean,
        y_var=y_var,
        resid=resid,
        resid_sum=resid_sum,
        groups=groups,
        group_sizes=group_sizes,
        alpha=alpha,
        penalty=penalty,
        weights=weights,
        offsets=offsets,
        lmda_path=lmda_path,
        lmda_max=lmda_max,
        screen_set=screen_set,
        screen_beta=screen_beta,
        screen_is_active=screen_is_active,
        active_set_size=active_set_size,
        active_set=active_set,
        rsq=rsq,
        lmda=lmda,
        grad=grad,
    )

    assert id(X) == id(state.X)
    assert np.allclose(X_means, state.X_means)
    assert np.allclose(y_mean, state.y_mean)
    assert np.allclose(y_var, state.y_var)
    assert np.allclose(groups, state.groups)
    assert np.allclose(alpha, state.alpha)
    assert np.allclose(penalty, state.penalty)
    assert np.allclose(screen_set, state.screen_set)
    assert np.allclose(lmda_path, state.lmda_path)
    assert np.allclose(lmda_max, state.lmda_max)
    assert np.allclose(rsq, state.rsq)
    assert np.allclose(lmda, state.lmda)
    assert np.allclose(screen_set, state.screen_set)
    assert np.allclose(screen_beta, state.screen_beta)
    assert np.allclose(screen_is_active, state.screen_is_active)
    assert np.allclose(grad, state.grad)
    assert np.allclose(resid, state.resid)
