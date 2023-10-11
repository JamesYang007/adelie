from adelie.logger import logger
import logging
logger.setLevel(logging.DEBUG)

import adelie.state as mod
import adelie.matrix as matrix
import numpy as np


def test_state_pin_naive():
    n = 1000
    p = 100
    G = 2

    X = matrix.pin_naive_dense(np.random.normal(0, 1, (n, p)), n_threads=4)
    groups = np.array([0, 1])
    group_sizes = np.array([1, p-1])
    alpha = 1.0
    penalty = np.random.uniform(0, 1, G)
    strong_set = np.array([0, 1])
    lmda_path = np.array([0.1, 1.0, 0.5])
    rsq = 0.0
    resid = np.random.normal(0, 1, n)
    strong_beta = np.zeros(p)
    strong_is_active = np.zeros(strong_set.shape[0], dtype=bool)

    state = mod.pin_naive(
        X=X,
        groups=groups,
        group_sizes=group_sizes,
        alpha=alpha,
        penalty=penalty,
        strong_set=strong_set,
        lmda_path=lmda_path,
        rsq=rsq,
        resid=resid,
        strong_beta=strong_beta,
        strong_is_active=strong_is_active,
    )

    state.check(method="assert")

    assert id(X._core_mat) == id(state.X)
    assert np.allclose(groups, state.groups)
    assert np.allclose(group_sizes, state.group_sizes)
    assert np.allclose(alpha, state.alpha)
    assert np.allclose(penalty, state.penalty)
    assert np.allclose(strong_set, state.strong_set)
    assert np.allclose(lmda_path, state.lmda_path)
    assert np.allclose(rsq, state.rsq)
    assert np.allclose(strong_beta, state.strong_beta)
    assert np.allclose(strong_is_active, state.strong_is_active)
    assert state.iters == 0


def test_state_pin_cov():
    n = 1000
    p = 100
    G = 2

    X = np.random.normal(0, 1, (n, p))
    A = matrix.pin_cov_dense(X.T @ X / n, n_threads=4)
    groups = np.array([0, 1])
    group_sizes = np.array([1, p-1])
    alpha = 1.0
    penalty = np.random.uniform(0, 1, G)
    strong_set = np.array([0, 1])
    lmda_path = np.array([0.1, 1.0, 0.5])
    rsq = 0.0
    strong_beta = np.zeros(p)
    strong_grad = X.T @ np.random.normal(0, 1, n)
    strong_is_active = np.zeros(strong_set.shape[0], dtype=bool)

    state = mod.pin_cov(
        A=A,
        groups=groups,
        group_sizes=group_sizes,
        alpha=alpha,
        penalty=penalty,
        strong_set=strong_set,
        lmda_path=lmda_path,
        rsq=rsq,
        strong_beta=strong_beta,
        strong_grad=strong_grad,
        strong_is_active=strong_is_active,
    )

    state.check(method="assert")

    assert id(A._core_mat) == id(state.A)
    assert np.allclose(groups, state.groups)
    assert np.allclose(group_sizes, state.group_sizes)
    assert np.allclose(alpha, state.alpha)
    assert np.allclose(penalty, state.penalty)
    assert np.allclose(strong_set, state.strong_set)
    assert np.allclose(lmda_path, state.lmda_path)
    assert np.allclose(rsq, state.rsq)
    assert np.allclose(strong_beta, state.strong_beta)
    assert np.allclose(strong_grad, state.strong_grad)
    assert np.allclose(strong_is_active, state.strong_is_active)
    assert state.iters == 0


def test_state_basil_naive():
    n = 3
    p = 2
    G = 2

    _X = np.random.normal(0, 1, (n, p))
    X = matrix.basil_naive_dense(_X, n_threads=4)
    X_means = np.mean(_X, axis=0)
    groups = np.array([0, 1])
    group_sizes = np.array([1, p-1])
    X_group_norms = np.array([
        np.linalg.norm(_X[:, g:g+gs], ord='fro') 
        for g, gs in zip(groups, group_sizes)
    ])
    y_mean = 0.0
    y_var = 1.0
    setup_edpp = True
    alpha = 1.0
    penalty = np.random.uniform(0, 1, G)
    strong_set = np.array([0, 1])
    lmda_path = np.array([0.1, 1.0, 0.5])
    lmda_max = 0.9
    rsq = 0.0
    lmda = 2.0
    grad = np.random.normal(0, 1, p)
    resid = np.random.normal(0, 1, n)
    edpp_safe_set = strong_set
    edpp_v1_0 = np.zeros(n)
    edpp_resid_0 = np.zeros(n)
    strong_beta = np.zeros(p)
    strong_is_active = np.zeros(strong_set.shape[0], dtype=bool)

    state = mod.basil_naive(
        X=X,
        X_means=X_means,
        X_group_norms=X_group_norms,
        y_mean=y_mean,
        y_var=y_var,
        setup_edpp=setup_edpp,
        resid=resid,
        edpp_safe_set=edpp_safe_set,
        edpp_v1_0=edpp_v1_0,
        edpp_resid_0=edpp_resid_0,
        groups=groups,
        group_sizes=group_sizes,
        alpha=alpha,
        penalty=penalty,
        lmda_path=lmda_path,
        lmda_max=lmda_max,
        strong_set=strong_set,
        strong_beta=strong_beta,
        strong_is_active=strong_is_active,
        rsq=rsq,
        lmda=lmda,
        grad=grad,
    )

    state.check(method="assert")

    assert id(X._core_mat) == id(state.X)
    assert np.allclose(X_means, state.X_means)
    assert np.allclose(X_group_norms, state.X_group_norms)
    assert np.allclose(y_mean, state.y_mean)
    assert np.allclose(y_var, state.y_var)
    assert np.allclose(setup_edpp, state.setup_edpp)
    assert np.allclose(edpp_safe_set, state.edpp_safe_set)
    assert np.allclose(edpp_v1_0, state.edpp_v1_0)
    assert np.allclose(edpp_resid_0, state.edpp_resid_0)
    assert np.allclose(groups, state.groups)
    assert np.allclose(group_sizes, state.group_sizes)
    assert np.allclose(alpha, state.alpha)
    assert np.allclose(penalty, state.penalty)
    assert np.allclose(strong_set, state.strong_set)
    assert np.allclose(lmda_path, state.lmda_path)
    assert np.allclose(lmda_max, state.lmda_max)
    assert np.allclose(rsq, state.rsq)
    assert np.allclose(lmda, state.lmda)
    assert np.allclose(strong_set, state.strong_set)
    assert np.allclose(strong_beta, state.strong_beta)
    assert np.allclose(strong_is_active, state.strong_is_active)
    assert np.allclose(grad, state.grad)
    assert np.allclose(resid, state.resid)
    assert state.iters == 0
