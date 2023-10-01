import adelie.adelie_core as core
import adelie.state as mod
import adelie.matrix as matrix
import numpy as np
import scipy


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
    strong_g1 = np.array([0])
    strong_g2 = np.array([1])
    strong_begins = np.array([0, 1])
    strong_A_diag = np.random.uniform(0, 1, p)
    lmdas = np.array([1.0, 0.5])
    max_cds = 1000
    thr = 1e-8
    cond_0_thresh = 1e-2
    cond_1_thresh = 1e-2
    newton_tol = 1e-10
    newton_max_iters = 1000
    rsq = 0.0
    resid = np.zeros(n)
    strong_beta = np.zeros(p)
    strong_grad = np.zeros(p)
    active_set = np.empty(0, dtype=int)
    active_g1 = np.empty(0, dtype=int)
    active_g2 = np.empty(0, dtype=int)
    active_begins = np.empty(0, dtype=int)
    active_order = np.empty(0, dtype=int)
    is_active = np.zeros(p, dtype=bool)

    state = mod.pin_naive(
        X=X,
        groups=groups,
        group_sizes=group_sizes,
        alpha=alpha,
        penalty=penalty,
        strong_set=strong_set,
        strong_g1=strong_g1,
        strong_g2=strong_g2,
        strong_begins=strong_begins,
        strong_A_diag=strong_A_diag,
        lmdas=lmdas,
        max_cds=max_cds,
        thr=thr,
        cond_0_thresh=cond_0_thresh,
        cond_1_thresh=cond_1_thresh,
        newton_tol=newton_tol,
        newton_max_iters=newton_max_iters,
        rsq=rsq,
        resid=resid,
        strong_beta=strong_beta,
        strong_grad=strong_grad,
        active_set=active_set,
        active_g1=active_g1,
        active_g2=active_g2,
        active_begins=active_begins,
        active_order=active_order,
        is_active=is_active,
    )

    assert isinstance(state._core_state, core.state.PinNaive64)

    assert id(X._core_mat) == id(state.X)
    assert np.allclose(groups, state.groups)
    assert np.allclose(group_sizes, state.group_sizes)
    assert np.allclose(alpha, state.alpha)
    assert np.allclose(penalty, state.penalty)
    assert np.allclose(strong_set, state.strong_set)
    assert np.allclose(strong_g1, state.strong_g1)
    assert np.allclose(strong_g2, state.strong_g2)
    assert np.allclose(strong_begins, state.strong_begins)
    assert np.allclose(strong_A_diag, state.strong_A_diag)
    assert np.allclose(lmdas, state.lmdas)
    assert np.allclose(max_cds, state.max_cds)
    assert np.allclose(thr, state.thr)
    assert np.allclose(cond_0_thresh, state.cond_0_thresh)
    assert np.allclose(cond_1_thresh, state.cond_1_thresh)
    assert np.allclose(newton_tol, state.newton_tol)
    assert np.allclose(newton_max_iters, state.newton_max_iters)
    assert np.allclose(rsq, state.rsq)
    assert np.allclose(strong_beta, state.strong_beta)
    assert np.allclose(strong_grad, state.strong_grad)
    assert np.allclose(active_set, state.active_set)
    assert np.allclose(active_g1, state.active_g1)
    assert np.allclose(active_g2, state.active_g2)
    assert np.allclose(active_begins, state.active_begins)
    assert np.allclose(active_order, state.active_order)
    assert np.allclose(is_active, state.is_active)
    assert isinstance(state.betas, scipy.sparse.csr_matrix)
    assert isinstance(state.rsqs, np.ndarray)
    assert state.n_cds == 0
    assert isinstance(state.time_strong_cd, np.ndarray)
    assert isinstance(state.time_active_cd, np.ndarray)

    assert isinstance(state.resid, np.ndarray)
    assert isinstance(state.resids, np.ndarray)