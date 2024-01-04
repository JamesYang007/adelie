from adelie.solver import (
    objective,
    solve_gaussian_pin,
    solve_gaussian,
)
import adelie as ad
import cvxpy as cp
import numpy as np
import os

# ========================================================================
# TEST solve_gaussian_pin
# ========================================================================

def create_test_data_gaussian_pin(
    n, p, G, S, 
    alpha=1,
    sparsity=0.95,
    seed=0,
    min_ratio=1e-2,
    n_lmdas=20,
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
    beta = np.random.normal(0, 1, p)
    beta[np.random.choice(p, int(sparsity * p), replace=False)] = 0
    y = X @ beta + np.random.normal(0, 1, n)
    X /= np.sqrt(n)
    y /= np.sqrt(n)

    screen_set = np.random.choice(G, S, replace=False)
    screen_is_active = np.zeros(screen_set.shape[0], dtype=bool)
    penalty = np.random.uniform(0, 1, G)
    penalty /= np.sum(penalty)
    grad = X.T @ y
    abs_grad = np.array([
        np.linalg.norm(grad[g:g+gs])
        for g, gs in zip(groups, group_sizes)
    ])
    lmda_candidates = abs_grad / (alpha * penalty)
    lmda_max = np.max(lmda_candidates[~np.isinf(lmda_candidates)])
    lmda_path = lmda_max * min_ratio ** (np.arange(n_lmdas) / (n_lmdas-1))
    rsq = 0
    screen_beta = np.zeros(np.sum(group_sizes[screen_set]))

    return (
        X, 
        y, 
        groups, 
        group_sizes, 
        penalty,
        lmda_path,
        screen_set, 
        screen_is_active, 
        rsq,
        screen_beta,
    )


def solve_cvxpy(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    group_sizes: np.ndarray,
    lmda: float,
    alpha: float,
    penalty: np.ndarray,
    weights: np.ndarray,
    screen_set: np.ndarray,
    intercept: bool,
    pin: bool,
):
    _, p = X.shape
    beta = cp.Variable(p)
    beta0 = cp.Variable(1)
    expr = (
        0.5 * cp.sum(cp.multiply(weights, (y - X @ beta - beta0) ** 2))
    )
    for g, gs, w in zip(groups, group_sizes, penalty):
        expr += lmda * w * (
            alpha * cp.norm(beta[g:g+gs]) 
            + 0.5 * (1-alpha) * cp.sum_squares(beta[g:g+gs])
        )
    constraints = [] 
    if pin:
        constraints = [
            beta[groups[i] : groups[i] + group_sizes[i]] == 0
            for i in range(len(groups))
            if not (i in screen_set)
        ]
    if not intercept:
        constraints += [ beta0 == 0 ]
    prob = cp.Problem(cp.Minimize(expr), constraints)
    prob.solve()
    return beta0.value, beta.value


def run_solve_gaussian_pin(state, X, y, weights):
    state.check(method="assert")

    state = solve_gaussian_pin(state)    

    # get solved lmdas
    lmdas = state.lmdas

    # check beta matches (if not, at least that objective is better)
    betas = state.betas.toarray()
    beta0s = state.intercepts
    cvxpy_res = [
        solve_cvxpy(
            X=X,
            y=y,
            groups=state.groups,
            group_sizes=state.group_sizes,
            lmda=lmda,
            alpha=state.alpha,
            penalty=state.penalty,
            weights=weights,
            screen_set=state.screen_set,
            intercept=state.intercept,
            pin=True,
        )
        for lmda in lmdas
    ]
    cvxpy_beta0s = [out[0] for out in cvxpy_res]
    cvxpy_betas = [out[1] for out in cvxpy_res]

    is_beta_close = np.allclose(betas, cvxpy_betas, atol=1e-6)
    if not is_beta_close:
        my_objs = np.array([
            objective(
                beta0,
                beta,
                X=X,
                y=y,
                groups=state.groups,
                group_sizes=state.group_sizes,
                lmda=lmda,
                alpha=state.alpha,
                penalty=state.penalty,
                weights=weights,
            )
            for beta0, beta, lmda in zip(beta0s, betas, lmdas)
        ])
        cvxpy_objs = np.array([
            objective(
                beta0,
                beta,
                X=X,
                y=y,
                groups=state.groups,
                group_sizes=state.group_sizes,
                lmda=lmda,
                alpha=state.alpha,
                penalty=state.penalty,
                weights=weights,
            )
            for beta0, beta, lmda in zip(cvxpy_beta0s, cvxpy_betas, lmdas)
        ])
        assert np.all(my_objs <= cvxpy_objs * (1 + 1e-10))

    return state


def test_solve_gaussian_pin_naive():
    def _test(n, p, G, S, intercept=True, alpha=1, sparsity=0.95, seed=0):
        (
            X, 
            y, 
            groups, 
            group_sizes, 
            penalty,
            lmda_path,
            screen_set, 
            screen_is_active, 
            rsq,
            screen_beta,
        ) = create_test_data_gaussian_pin(
            n, p, G, S, alpha, sparsity, seed,
        )
        np.random.seed(seed)
        weights = np.random.uniform(1, 2, y.shape[0])
        weights = weights / np.sum(weights)
        y_mean = np.sum(y * weights)
        resid = weights * (y - intercept * y_mean)
        y_var = np.sum(resid ** 2 / weights)
        Xs = [
            ad.matrix.dense(X, method="naive", n_threads=2)
        ]
        for Xpy in Xs:
            state = ad.state.gaussian_pin_naive(
                X=Xpy,
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
                intercept=intercept,
                tol=1e-7,
            )
            state = run_solve_gaussian_pin(state, X, y, weights)
            state = ad.state.gaussian_pin_naive(
                X=Xpy,
                y_mean=y_mean,
                y_var=y_var,
                groups=groups,
                alpha=alpha,
                penalty=penalty,
                weights=weights,
                screen_set=screen_set,
                lmda_path=[state.lmdas[-1] * 0.8],
                rsq=state.rsq,
                resid=state.resid,
                screen_beta=state.screen_beta,
                screen_is_active=state.screen_is_active,
                intercept=intercept,
                tol=1e-7,
            )
            run_solve_gaussian_pin(state, X, y, weights)

    _test(10, 4, 2, 2)
    _test(10, 100, 10, 2)
    _test(10, 100, 20, 13)
    _test(100, 23, 4, 3)
    _test(100, 100, 50, 20)


def test_solve_gaussian_pin_cov():
    def _test(n, p, G, S, alpha=1, sparsity=0.95, seed=0):
        (
            X, 
            y, 
            groups, 
            group_sizes, 
            penalty,
            lmda_path,
            screen_set, 
            screen_is_active, 
            rsq,
            screen_beta,
        ) = create_test_data_gaussian_pin(
            n, p, G, S, alpha, sparsity, seed,
        )
        np.random.seed(seed)
        weights = np.random.uniform(1, 2, y.shape[0])
        WsqrtX = np.sqrt(weights)[:, None] * X
        A = WsqrtX.T @ WsqrtX
        grad = X.T @ (y * weights)
        screen_grad = np.concatenate([
            grad[g:g+gs]
            for g, gs in zip(groups[screen_set], group_sizes[screen_set])
        ])

        # list of different types of cov matrices to test
        As = [
            ad.matrix.dense(A, method="cov", n_threads=3),
            ad.matrix.cov_lazy(WsqrtX, n_threads=3),
        ]

        for Apy in As:
            state = ad.state.gaussian_pin_cov(
                A=Apy,
                groups=groups,
                alpha=alpha,
                penalty=penalty,
                screen_set=screen_set,
                lmda_path=lmda_path,
                rsq=rsq,
                screen_beta=screen_beta,
                screen_grad=screen_grad,
                screen_is_active=screen_is_active,
                tol=1e-8,
            )
            state = run_solve_gaussian_pin(state, X, y, weights)
            state = ad.state.gaussian_pin_cov(
                A=Apy,
                groups=groups,
                alpha=alpha,
                penalty=penalty,
                screen_set=screen_set,
                lmda_path=[state.lmdas[-1] * 0.8],
                rsq=state.rsq,
                screen_beta=state.screen_beta,
                screen_grad=state.screen_grad,
                screen_is_active=state.screen_is_active,
                tol=1e-8,
            )
            run_solve_gaussian_pin(state, X, y, weights)

    _test(10, 4, 2, 2)
    _test(10, 100, 10, 2)
    _test(10, 100, 20, 13)
    _test(100, 23, 4, 3)
    _test(100, 100, 50, 20)


# ========================================================================
# TEST solve_gaussian
# ========================================================================


def create_dense(
    n, p, G,
    intercept=True,
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

    penalty = np.random.uniform(0, 1, G)
    penalty[np.random.choice(G, int(0.05 * G), replace=False)] = 0
    penalty /= np.linalg.norm(penalty) / np.sqrt(p)

    # generate raw data
    X = np.random.normal(0, 1, (n, p))
    beta = np.random.normal(0, 1, p)
    beta[np.random.choice(p, int(sparsity * p), replace=False)] = 0
    y = X @ beta + np.random.normal(0, 1, n)
    X /= np.sqrt(n)
    y /= np.sqrt(n)

    weights = np.random.uniform(1, 2, n)
    weights /= np.sum(weights)

    X_means = np.sum(weights[:, None] * X, axis=0)
    X_c = X - intercept * X_means[None]
    y_mean = np.sum(weights * y)
    y_c = y - y_mean * intercept
    y_var = np.sum(weights * y_c ** 2)
    resid = weights * y_c
    screen_set = np.arange(G)[(penalty <= 0) | (alpha <= 0)]
    screen_beta = np.zeros(np.sum(group_sizes[screen_set]))
    screen_is_active = np.zeros(screen_set.shape[0], dtype=bool)
    grad = X_c.T @ resid

    return {
        "X": X, 
        "y": y,
        "X_means": X_means,
        "y_mean": y_mean,
        "y_var": y_var,
        "resid": resid,
        "groups": groups,
        "alpha": alpha,
        "penalty": penalty,
        "weights": weights,
        "screen_set": screen_set,
        "screen_beta": screen_beta,
        "screen_is_active": screen_is_active,
        "rsq": 0,
        "lmda": np.inf,
        "grad": grad,
        "intercept": intercept,
    }


def run_solve_gaussian(state, X, y):
    state.check(y, method="assert")

    state = solve_gaussian(state)    

    state.check(y, method="assert")

    # get solved lmdas
    lmdas = state.lmdas

    # check beta matches (if not, at least that objective is better)
    betas = state.betas.toarray()
    beta0s = state.intercepts
    cvxpy_res = [
        solve_cvxpy(
            X=X,
            y=y,
            groups=state.groups,
            group_sizes=state.group_sizes,
            lmda=lmda,
            alpha=state.alpha,
            penalty=state.penalty,
            weights=state.weights,
            screen_set=state.screen_set,
            intercept=state.intercept,
            pin=False,
        )
        for lmda in lmdas
    ]
    cvxpy_beta0s = [out[0] for out in cvxpy_res]
    cvxpy_betas = [out[1] for out in cvxpy_res]

    is_beta_close = np.allclose(betas, cvxpy_betas, atol=1e-6)
    if not is_beta_close:
        my_objs = np.array([
            objective(
                beta0,
                beta,
                X=X,
                y=y,
                groups=state.groups,
                group_sizes=state.group_sizes,
                lmda=lmda,
                alpha=state.alpha,
                penalty=state.penalty,
                weights=state.weights,
            )
            for beta0, beta, lmda in zip(beta0s, betas, lmdas)
        ])
        cvxpy_objs = np.array([
            objective(
                beta0,
                beta,
                X=X,
                y=y,
                groups=state.groups,
                group_sizes=state.group_sizes,
                lmda=lmda,
                alpha=state.alpha,
                penalty=state.penalty,
                weights=state.weights,
            )
            for beta0, beta, lmda in zip(cvxpy_beta0s, cvxpy_betas, lmdas)
        ])
        assert np.all(my_objs <= cvxpy_objs * (1 + 1e-8))

    return state


def test_solve_gaussian():
    def _test(n, p, G, intercept=True, alpha=1, sparsity=0.95, seed=0):
        test_data = create_dense(
            n, p, G, intercept, alpha, sparsity, seed,
        )
        X, y = test_data["X"], test_data["y"]
        test_data.pop("y")
        Xs = [
            ad.matrix.dense(X, method="naive", n_threads=2)
        ]
        for Xpy in Xs:
            test_data["X"] = Xpy
            state = ad.state.gaussian_naive(
                **test_data,
                tol=1e-11,
            )
            state = run_solve_gaussian(state, X, y)
            state = ad.state.gaussian_naive(
                X=state.X,
                X_means=state.X_means,
                y_mean=state.y_mean,
                y_var=state.y_var,
                resid=state.resid,
                groups=state.groups,
                alpha=state.alpha,
                penalty=state.penalty,
                weights=state.weights,
                screen_set=state.screen_set,
                screen_beta=state.screen_beta,
                screen_is_active=state.screen_is_active,
                rsq=state.rsq,
                lmda=state.lmda,
                grad=state.grad,
                lmda_path=[state.lmdas[-1] * 0.8],
                lmda_max=state.lmda_max,
                intercept=state.intercept,
                tol=1e-11,
            )
            run_solve_gaussian(state, X, y)

    _test(10, 4, 2)
    _test(10, 100, 10)
    _test(10, 100, 20)
    _test(100, 23, 4)
    _test(100, 100, 50)


def test_solve_gaussian_snp_unphased():
    def _test(n, p, intercept=True, alpha=1, sparsity=0.5, seed=0, n_threads=7):
        test_data = ad.data.create_snp_unphased(
            n, 
            p, 
            sparsity=sparsity, 
            seed=seed,
        )
        filenames = [f"/tmp/test_snp_unphased.snpdat"]
        handler = ad.io.snp_unphased(filenames[0])
        handler.write(test_data["X"], n_threads)
        Xs = [
            ad.matrix.snp_unphased(
                filenames=filenames,
                dtype=np.float64,
                n_threads=n_threads,
            )
        ]
        for f in filenames:
            os.remove(f)

        X, y = test_data["X"], test_data["y"]

        weights = np.random.uniform(1, 2, n)
        weights /= np.sum(weights)

        test_data["weights"] = weights
        test_data["alpha"] = alpha
        test_data["X_means"] = np.sum(weights[:, None] * X, axis=0)
        test_data["y_mean"] = np.sum(weights * y)
        X_c = X - intercept * test_data["X_means"][None]
        y_c = y - test_data["y_mean"] * intercept
        test_data["y_var"] = np.sum(weights * y_c ** 2)
        test_data["resid"] = weights * y_c
        test_data["screen_set"] = np.arange(p)[(test_data["penalty"] <= 0) | (alpha <= 0)]
        test_data["screen_beta"] = np.zeros(np.sum(test_data["group_sizes"][test_data["screen_set"]]))
        test_data["screen_is_active"] = np.zeros(test_data["screen_set"].shape[0], dtype=bool)
        test_data["grad"] = X_c.T @ test_data["resid"]
        test_data["rsq"] = 0 
        test_data["lmda"] = np.inf
        test_data["tol"] = 1e-7
        test_data["n_threads"] = n_threads

        test_data.pop("y")
        test_data.pop("group_sizes")

        for Xpy in Xs:
            test_data["X"] = Xpy
            state_special = ad.solver.solve_gaussian(
                ad.state.gaussian_naive(**test_data),
            )
            test_data["X"] = ad.matrix.dense(
                X.astype(np.float64), 
                method="naive", 
                n_threads=n_threads,
            )
            state_dense = ad.solver.solve_gaussian(
                ad.state.gaussian_naive(**test_data),
            )

            assert np.allclose(state_special.lmdas, state_dense.lmdas)
            assert np.allclose(state_special.rsqs, state_dense.rsqs)
            assert np.allclose(state_special.intercepts, state_dense.intercepts)
            assert np.allclose(state_special.betas.toarray(), state_dense.betas.toarray())

    _test(10, 4)
    _test(10, 100)
    _test(10, 100)
    _test(100, 23)
    _test(100, 100)
    _test(100, 10000)


def test_solve_gaussian_snp_phased_ancestry():
    def _test(n, p, A=8, intercept=True, alpha=1, sparsity=0.5, seed=0, n_threads=7):
        test_data = ad.data.create_snp_phased_ancestry(
            n, 
            p, 
            A,
            sparsity=sparsity, 
            seed=seed,
        )
        filenames = [f"/tmp/test_snp_phased_ancestry.snpdat"]
        handler = ad.io.snp_phased_ancestry(filenames[0])
        handler.write(test_data["X"], test_data["ancestries"], A, n_threads)
        Xs = [
            ad.matrix.snp_phased_ancestry(
                filenames=filenames,
                dtype=np.float64,
                n_threads=n_threads,
            )
        ]
        handler.read() 
        for f in filenames:
            os.remove(f)

        X, y = handler.to_dense(n_threads), test_data["y"]

        weights = np.random.uniform(1, 2, n)
        weights /= np.sum(weights)

        test_data["weights"] = weights
        test_data["alpha"] = alpha
        test_data["X_means"] = np.sum(weights[:, None] * X, axis=0)
        test_data["y_mean"] = np.sum(weights * y)
        X_c = X - intercept * test_data["X_means"][None]
        y_c = y - test_data["y_mean"] * intercept
        test_data["y_var"] = np.sum(weights * y_c ** 2)
        test_data["resid"] = weights * y_c
        test_data["screen_set"] = np.arange(p)[(test_data["penalty"] <= 0) | (alpha <= 0)]
        test_data["screen_beta"] = np.zeros(np.sum(test_data["group_sizes"][test_data["screen_set"]]))
        test_data["screen_is_active"] = np.zeros(test_data["screen_set"].shape[0], dtype=bool)
        test_data["grad"] = X_c.T @ test_data["resid"]
        test_data["rsq"] = 0 
        test_data["lmda"] = np.inf
        test_data["tol"] = 1e-7
        test_data["n_threads"] = n_threads

        test_data.pop("y")
        test_data.pop("group_sizes")
        test_data.pop("ancestries")

        for Xpy in Xs:
            test_data["X"] = Xpy
            state_special = ad.solver.solve_gaussian(
                ad.state.gaussian_naive(**test_data),
            )
            test_data["X"] = ad.matrix.dense(
                X.astype(np.float64), 
                method="naive", 
                n_threads=n_threads,
            )
            state_dense = ad.solver.solve_gaussian(
                ad.state.gaussian_naive(**test_data),
            )

            assert np.allclose(state_special.lmdas, state_dense.lmdas)
            assert np.allclose(state_special.rsqs, state_dense.rsqs)
            assert np.allclose(state_special.intercepts, state_dense.intercepts)
            assert np.allclose(state_special.betas.toarray(), state_dense.betas.toarray())

    _test(10, 4)
    _test(10, 100)
    _test(10, 100)
    _test(100, 23)
    _test(100, 100)
    _test(100, 10000)