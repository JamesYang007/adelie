from adelie.logger import logger
import logging
logger.setLevel(logging.DEBUG)

import adelie.state as mod
import adelie.matrix as matrix
import numpy as np


def test_pin_naive():
    n = 1000
    p = 100
    G = 2

    X = matrix.dense(np.random.normal(0, 1, (n, p)))
    groups = np.array([0, 1])
    group_sizes = np.array([1, p-1])
    alpha = 1.0
    penalty = np.random.uniform(0, 1, G)
    strong_set = np.array([0, 1])
    lmdas = np.array([0.1, 1.0, 0.5])
    rsq = 0.0
    resid = np.random.normal(0, 1, n)
    strong_beta = np.zeros(p)
    active_set = np.empty(0, dtype=int)

    state = mod.pin_naive(
        X=X,
        groups=groups,
        group_sizes=group_sizes,
        alpha=alpha,
        penalty=penalty,
        strong_set=strong_set,
        lmdas=lmdas,
        rsq=rsq,
        resid=resid,
        strong_beta=strong_beta,
        active_set=active_set,
    )

    state.check(method="assert")

    assert id(X._core_mat) == id(state.X)
    assert np.allclose(groups, state.groups)
    assert np.allclose(group_sizes, state.group_sizes)
    assert np.allclose(alpha, state.alpha)
    assert np.allclose(penalty, state.penalty)
    assert np.allclose(strong_set, state.strong_set)
    assert np.allclose(lmdas, state.lmdas)
    assert np.allclose(rsq, state.rsq)
    assert np.allclose(strong_beta, state.strong_beta)
    assert np.allclose(active_set, state.active_set)
    assert state.iters == 0
