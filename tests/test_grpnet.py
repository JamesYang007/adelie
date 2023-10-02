from adelie.state import (
    pin_naive,
)
from adelie.grpnet import (
    objective,
    solve_pin,
)
import adelie as ad
import cvxpy as cp
import numpy as np


def solve_cvxpy(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    group_sizes: np.ndarray,
    lmda: float,
    alpha: float,
    penalty: np.ndarray,
    strong_set: np.ndarray,
):
    n, p = X.shape
    beta = cp.Variable(p)
    expr = (
        0.5 * cp.sum_squares(y - X @ beta)
    )
    for g, gs, w in zip(groups, group_sizes, penalty):
        expr += lmda * w * (
            alpha * cp.norm2(beta[g:g+gs]) 
            + 0.5 * (1-alpha) * cp.sum_squares(beta[g:g+gs])
        )
    constraints = [
        beta[groups[i] : groups[i] + group_sizes[i]] == 0
        for i in strong_set
    ]
    prob = cp.Problem(cp.Minimize(expr), constraints)
    prob.solve()
    return beta.value


def run_solve_pin(state, X, y):
    state.check(method="assert")

    state = solve_pin(state)    

    # check beta matches (if not, at least that objective is better)
    betas = state.betas.toarray()
    cvxpy_betas = np.array([
        solve_cvxpy(
            X=X,
            y=y,
            groups=state.groups,
            group_sizes=state.group_sizes,
            lmda=lmda,
            alpha=state.alpha,
            penalty=state.penalty,
            strong_set=state.strong_set,
        )
        for lmda in state.lmdas
    ])
    is_beta_close = np.allclose(betas, cvxpy_betas, atol=1e-6)
    if not is_beta_close:
        my_objs = np.array([
            objective(
                beta,
                X=state.X,
                y=y,
                groups=state.groups,
                group_sizes=state.group_sizes,
                lmda=lmda,
                alpha=state.alpha,
                penalty=state.penalty,
            )
            for beta, lmda in zip(betas, state.lmdas)
        ])
        cvxpy_objs = np.array([
            objective(
                beta,
                X=state.X,
                y=y,
                groups=state.groups,
                group_sizes=state.group_sizes,
                lmda=lmda,
                alpha=state.alpha,
                penalty=state.penalty,
            )
            for beta, lmda in zip(cvxpy_betas, state.lmdas)
        ])
        assert np.all(my_objs <= cvxpy_objs)


def test_solve_pin():
    def _test(n, p, G, S, alpha=1, seed=0):
        np.random.seed(seed)

        # generate raw data
        _X = np.random.normal(0, 1, (n, p))
        beta = np.random.normal(0, 1, p)
        beta[np.random.choice(p, int(0.95 * p), replace=False)] = 0
        y = _X @ beta + np.random.normal(0, 1, n)

        # define groups
        groups = np.concatenate([
            [0],
            np.random.choice(np.arange(1, p), size=G-1, replace=False)
        ])
        groups = np.sort(groups).astype(int)
        group_sizes = np.concatenate([groups, [p]], dtype=int)
        group_sizes = group_sizes[1:] - group_sizes[:-1]

        # NOTE: MUST modify X such that each block has diagonal variance.
        for g, gs in zip(groups, group_sizes):
            block = _X[:, g:g+gs]
            u, d, vh = np.linalg.svd(block, full_matrices=True)
            _X[:, g:g+gs] = block @ vh.T

        X = ad.matrix.dense(_X)
        strong_set = np.random.choice(G, S, replace=False)
        active_set = np.empty(0, dtype=int)
        penalty = np.random.uniform(0, 1, G)
        penalty /= np.sum(penalty)
        lmdas = [10, 5, 1]
        rsq = 0
        resid = np.copy(y)
        strong_beta = np.zeros(np.sum(group_sizes[strong_set]))

        state = pin_naive(
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
        run_solve_pin(state, _X, y)

    _test(10, 4, 2, 2)