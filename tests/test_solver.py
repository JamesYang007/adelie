from adelie.solver import (
    objective,
    solve_pin,
    create_lambdas,
)
import adelie as ad
import cvxpy as cp
import numpy as np

# ========================================================================
# TEST helpers
# ========================================================================

# ========================================================================
# TEST solve_pin
# ========================================================================

def create_test_data(
    n, p, G, S, 
    alpha=1,
    sparsity=0.95,
    seed=0,
):
    np.random.seed(seed)

    # define groups
    groups = np.concatenate([
        [0],
        np.random.choice(np.arange(1, p), size=G-1, replace=False)
    ])
    groups = np.sort(groups).astype(int)
    group_sizes = np.concatenate([groups, [p]], dtype=int)
    group_sizes = group_sizes[1:] - group_sizes[:-1]

    # generate raw data
    X = np.random.normal(0, 1, (n, p))

    # NOTE: MUST modify X such that each block has diagonal variance.
    for g, gs in zip(groups, group_sizes):
        block = X[:, g:g+gs]
        _, _, vh = np.linalg.svd(block, full_matrices=True)
        X[:, g:g+gs] = block @ vh.T

    beta = np.random.normal(0, 1, p)
    beta[np.random.choice(p, int(sparsity * p), replace=False)] = 0
    y = X @ beta + np.random.normal(0, 1, n)
    X /= np.sqrt(n)
    y /= np.sqrt(n)

    strong_set = np.random.choice(G, S, replace=False)
    strong_is_active = np.zeros(strong_set.shape[0], dtype=bool)
    penalty = np.random.uniform(0, 1, G)
    penalty /= np.sum(penalty)
    lmdas = create_lambdas(
        X=X, y=y, groups=groups, group_sizes=group_sizes,
        alpha=alpha, penalty=penalty, 
    )
    rsq = 0
    strong_beta = np.zeros(np.sum(group_sizes[strong_set]))

    return (
        X, 
        y, 
        groups, 
        group_sizes, 
        penalty,
        lmdas,
        strong_set, 
        strong_is_active, 
        rsq,
        strong_beta,
    )


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
    _, p = X.shape
    beta = cp.Variable(p)
    expr = (
        0.5 * cp.sum_squares(y - X @ beta)
    )
    for g, gs, w in zip(groups, group_sizes, penalty):
        expr += lmda * w * (
            alpha * cp.norm(beta[g:g+gs]) 
            + 0.5 * (1-alpha) * cp.sum_squares(beta[g:g+gs])
        )
    constraints = [
        beta[groups[i] : groups[i] + group_sizes[i]] == 0
        for i in range(len(groups))
        if not (i in strong_set)
    ]
    prob = cp.Problem(cp.Minimize(expr), constraints)
    prob.solve()
    return beta.value


def run_solve_pin(state, X, y):
    state.check(method="assert")

    state = solve_pin(state)    

    # get solved lmdas
    n_lmdas = len(state.rsqs)
    lmdas = state.lmdas[:n_lmdas]

    # check beta matches (if not, at least that objective is better)
    betas = state.betas.toarray()[:n_lmdas]
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
        for lmda in lmdas
    ])
    is_beta_close = np.allclose(betas, cvxpy_betas, atol=1e-6)
    if not is_beta_close:
        my_objs = np.array([
            objective(
                beta,
                X=X,
                y=y,
                groups=state.groups,
                group_sizes=state.group_sizes,
                lmda=lmda,
                alpha=state.alpha,
                penalty=state.penalty,
            )
            for beta, lmda in zip(betas, lmdas)
        ])
        cvxpy_objs = np.array([
            objective(
                beta,
                X=X,
                y=y,
                groups=state.groups,
                group_sizes=state.group_sizes,
                lmda=lmda,
                alpha=state.alpha,
                penalty=state.penalty,
            )
            for beta, lmda in zip(cvxpy_betas, lmdas)
        ])
        assert np.all(my_objs <= cvxpy_objs)

    return state


def test_solve_pin_naive():
    def _test(n, p, G, S, alpha=1, sparsity=0.95, seed=0):
        (
            X, 
            y, 
            groups, 
            group_sizes, 
            penalty,
            lmdas,
            strong_set, 
            strong_is_active, 
            rsq,
            strong_beta,
        ) = create_test_data(
            n, p, G, S, alpha, sparsity, seed,
        )
        resid = y
        Xs = [
            ad.matrix.pin_naive_dense(X, n_threads=2)
        ]
        for Xpy in Xs:
            state = ad.state.pin_naive(
                X=Xpy,
                groups=groups,
                group_sizes=group_sizes,
                alpha=alpha,
                penalty=penalty,
                strong_set=strong_set,
                lmdas=lmdas,
                rsq=rsq,
                resid=resid,
                strong_beta=strong_beta,
                strong_is_active=strong_is_active,
            )
            state = run_solve_pin(state, X, y)
            state = ad.state.pin_naive(
                X=Xpy,
                groups=groups,
                group_sizes=group_sizes,
                alpha=alpha,
                penalty=penalty,
                strong_set=strong_set,
                lmdas=[state.lmdas[:len(state.rsqs)][-1] * 0.8],
                rsq=state.rsq,
                resid=state.resid,
                strong_beta=state.strong_beta,
                strong_is_active=state.strong_is_active
            )
            run_solve_pin(state, X, y)

    _test(10, 4, 2, 2)
    _test(10, 100, 10, 2)
    _test(10, 100, 20, 13)
    _test(100, 23, 4, 3)
    _test(100, 100, 50, 20)


def test_solve_pin_cov():
    def _test(n, p, G, S, alpha=1, sparsity=0.95, seed=0):
        (
            X, 
            y, 
            groups, 
            group_sizes, 
            penalty,
            lmdas,
            strong_set, 
            strong_is_active, 
            rsq,
            strong_beta,
        ) = create_test_data(
            n, p, G, S, alpha, sparsity, seed,
        )

        A = X.T @ X
        grad = X.T @ y
        strong_grad = np.concatenate([
            grad[g:g+gs]
            for g, gs in zip(groups[strong_set], group_sizes[strong_set])
        ])

        # list of different types of cov matrices to test
        As = [
            ad.matrix.pin_cov_dense(A, n_threads=3),
            ad.matrix.pin_cov_lazy(X, n_threads=3),
        ]

        for Apy in As:
            state = ad.state.pin_cov(
                A=Apy,
                groups=groups,
                group_sizes=group_sizes,
                alpha=alpha,
                penalty=penalty,
                strong_set=strong_set,
                lmdas=lmdas,
                rsq=rsq,
                strong_beta=strong_beta,
                strong_grad=strong_grad,
                strong_is_active=strong_is_active,
            )
            state = run_solve_pin(state, X, y)
            state = ad.state.pin_cov(
                A=Apy,
                groups=groups,
                group_sizes=group_sizes,
                alpha=alpha,
                penalty=penalty,
                strong_set=strong_set,
                lmdas=[state.lmdas[:len(state.rsqs)][-1] * 0.8],
                rsq=state.rsq,
                strong_beta=state.strong_beta,
                strong_grad=state.strong_grad,
                strong_is_active=state.strong_is_active
            )
            run_solve_pin(state, X, y)

    _test(10, 4, 2, 2)
    _test(10, 100, 10, 2)
    _test(10, 100, 20, 13)
    _test(100, 23, 4, 3)
    _test(100, 100, 50, 20)
