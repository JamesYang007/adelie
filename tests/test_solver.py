from adelie.solver import (
    _solve,
)
from adelie.diagnostic import (
    objective,
)
import adelie as ad
import cvxpy as cp
import numpy as np
import os

# ========================================================================
# TEST _solve
# ========================================================================

def create_data_gaussian(
    n, p, G, S, 
    alpha=1,
    sparsity=0.95,
    seed=0,
    min_ratio=0.5,
    n_lmdas=20,
    intercept=True,
    pin=False,
    method="naive",
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

    penalty = np.random.uniform(0, 1, G)
    penalty[np.random.choice(G, int(0.05 * G), replace=False)] = 0
    penalty /= np.linalg.norm(penalty) / np.sqrt(p)
    weights = np.random.uniform(1, 2, n)
    weights /= np.sum(weights)

    X_means = np.sum(weights[:, None] * X, axis=0)
    X_c = X - intercept * X_means[None]
    y_mean = np.sum(weights * y)
    y_c = (y - intercept * y_mean)
    y_var = np.sum(weights * y_c ** 2)
    resid = y_c
    grad = X_c.T @ (weights * resid)

    args = {
        "X": X, 
        "y": y,
        "groups": groups,
        "alpha": alpha,
        "penalty": penalty,
        "weights": weights,
        "rsq": 0,
        "intercept": intercept,
    }

    if pin:
        screen_set = np.random.choice(G, S, replace=False)
        screen_is_active = np.zeros(screen_set.shape[0], dtype=bool)
        abs_grad = np.array([
            np.linalg.norm(grad[g:g+gs])
            for g, gs in zip(groups, group_sizes)
        ])
        is_nnz = penalty > 0
        lmda_candidates = abs_grad[is_nnz] / (alpha * penalty[is_nnz])
        lmda_max = np.max(lmda_candidates[~np.isinf(lmda_candidates)])
        lmda_path = lmda_max * min_ratio ** (np.arange(n_lmdas) / (n_lmdas-1))

        args["lmda_path"] = lmda_path

        if method == "cov":
            WsqrtX = np.sqrt(weights)[:, None] * X_c
            A = WsqrtX.T @ WsqrtX
            screen_grad = np.concatenate([
                grad[g:g+gs]
                for g, gs in zip(groups[screen_set], group_sizes[screen_set])
            ])

            args["A"] = A
            args["WsqrtX"] = WsqrtX
            args["screen_grad"] = screen_grad
        else:
            args["y_mean"] = y_mean
            args["resid"] = resid
            args["y_var"] = y_var
    else:
        screen_set = np.arange(G)[(penalty <= 0) | (alpha <= 0)]
        screen_is_active = np.zeros(screen_set.shape[0], dtype=bool)
        resid_sum = np.sum(weights * resid)

        args["X_means"] = X_means
        args["y_mean"] = y_mean
        args["y_var"] = y_var
        args["offsets"] = np.zeros(n)
        args["group_sizes"] = group_sizes
        args["resid"] = resid
        args["resid_sum"] = resid_sum
        args["lmda"] = np.inf
        args["grad"] = grad

    args["screen_set"] = screen_set
    args["screen_is_active"] = screen_is_active
    args["screen_beta"] = np.zeros(np.sum(group_sizes[screen_set]))

    return args


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


def run_solve_gaussian(state, args, pin):
    state.check(method="assert")

    state = _solve(state)    

    state.check(method="assert")

    # get solved lmdas
    lmdas = state.lmdas

    # check beta matches (if not, at least that objective is better)
    X = args["X"]
    y = args["y"]
    intercept = args["intercept"]
    weights = args["weights"]
    betas = state.betas
    intercepts = state.intercepts
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
            intercept=intercept,
            pin=pin,
        )
        for lmda in lmdas
    ]
    cvxpy_intercepts = np.array([out[0][0] for out in cvxpy_res])
    cvxpy_betas = np.array([out[1] for out in cvxpy_res])

    is_intercept_close = np.allclose(intercepts, cvxpy_intercepts, atol=1e-6)
    is_beta_close = np.allclose(betas.toarray(), cvxpy_betas, atol=1e-6)
    if not (is_beta_close and is_intercept_close):
        objective_args = {
            "X": X,
            "glm": ad.glm.gaussian(y=y, weights=weights),
            "lmdas": lmdas,
            "groups": state.groups,
            "alpha": state.alpha,
            "penalty": state.penalty,
        }
        my_objs = objective(
            **objective_args,
            betas=betas,
            intercepts=intercepts,
        )
        cvxpy_objs = objective(
            **objective_args,
            betas=cvxpy_betas,
            intercepts=cvxpy_intercepts,
        )
        assert np.all(my_objs <= cvxpy_objs * (1 + 1e-8))

    return state


def test_solve_gaussian_pin_naive():
    def _test(n, p, G, S, intercept=True, alpha=1, sparsity=0.95, seed=0):
        args = create_data_gaussian(
            n=n, 
            p=p, 
            G=G, 
            S=S, 
            intercept=intercept,
            alpha=alpha, 
            sparsity=sparsity, 
            seed=seed,
            pin=True,
            method="naive",
        )
        Xs = [
            ad.matrix.dense(args["X"], method="naive", n_threads=2)
        ]
        for Xpy in Xs:
            args_c = args.copy()
            args_c["X"] = Xpy
            args_c.pop("y")
            state = ad.state.gaussian_pin_naive(
                **args_c,
                tol=1e-7,
            )
            state = run_solve_gaussian(state, args, pin=True)
            args_c["lmda_path"] = [state.lmdas[-1] * 0.8]
            args_c["rsq"] = state.rsq
            args_c["resid"] = state.resid
            args_c["screen_beta"] = state.screen_beta
            args_c["screen_is_active"] = state.screen_is_active
            state = ad.state.gaussian_pin_naive(
                **args_c,
                tol=1e-7,
            )
            run_solve_gaussian(state, args, pin=True)

    _test(10, 4, 2, 2)
    _test(10, 100, 10, 2)
    _test(10, 100, 20, 13)
    _test(100, 23, 4, 3)
    _test(100, 100, 50, 20)


def test_solve_gaussian_pin_cov():
    def _test(n, p, G, S, alpha=1, sparsity=0.95, seed=0):
        # for simplicity of testing routine, should only work for intercept=False
        args = create_data_gaussian(
            n=n, 
            p=p, 
            G=G, 
            S=S, 
            intercept=False,
            alpha=alpha, 
            sparsity=sparsity, 
            seed=seed,
            pin=True,
            method="cov",
        )

        # list of different types of cov matrices to test
        As = [
            ad.matrix.dense(args["A"], method="cov", n_threads=3),
            ad.matrix.cov_lazy(args["WsqrtX"], n_threads=3),
        ]

        for Apy in As:
            args_c = args.copy()
            args_c.pop("X")
            args_c.pop("y")
            args_c.pop("WsqrtX")
            args_c.pop("weights")
            args_c.pop("intercept")
            args_c["A"] = Apy
            state = ad.state.gaussian_pin_cov(
                **args_c,
                tol=1e-7,
            )
            state = run_solve_gaussian(state, args, pin=True)
            args_c["lmda_path"] = [state.lmdas[-1] * 0.8]
            args_c["rsq"] = state.rsq
            args_c["screen_beta"] = state.screen_beta
            args_c["screen_grad"] = state.screen_grad
            args_c["screen_is_active"] = state.screen_is_active
            state = ad.state.gaussian_pin_cov(
                **args_c,
                tol=1e-7,
            )
            run_solve_gaussian(state, args, pin=True)

    _test(10, 4, 2, 2)
    _test(10, 100, 10, 2)
    _test(10, 100, 20, 13)
    _test(100, 23, 4, 3)
    _test(100, 100, 50, 20)


# ========================================================================
# TEST solve_gaussian
# ========================================================================

def test_solve_gaussian():
    def _test(n, p, G, intercept=True, alpha=1, sparsity=0.95, seed=0):
        args = create_data_gaussian(
            n=n, 
            p=p, 
            G=G, 
            S=None,
            intercept=intercept, 
            alpha=alpha, 
            sparsity=sparsity, 
            seed=seed,
        )
        Xs = [
            ad.matrix.dense(args["X"], method="naive", n_threads=2)
        ]
        for Xpy in Xs:
            args_c = args.copy()
            args_c["X"] = Xpy
            state = ad.state.gaussian_naive(
                **args_c,
                tol=1e-10,
                min_ratio=1e-1,
                lmda_path_size=30,
            )
            state = run_solve_gaussian(state, args, pin=False)
            args_c["resid"] = state.resid
            args_c["resid_sum"] = state.resid_sum
            args_c["screen_set"] = state.screen_set
            args_c["screen_beta"] = state.screen_beta
            args_c["screen_is_active"] = state.screen_is_active
            args_c["rsq"] = state.rsq
            args_c["lmda"] = state.lmda
            args_c["grad"] = state.grad
            args_c["lmda_path"] = [state.lmdas[-1] * 0.8]
            args_c["lmda_max"] = state.lmda_max
            state = ad.state.gaussian_naive(
                **args_c,
                tol=1e-10,
            )
            run_solve_gaussian(state, args, pin=False)

    _test(10, 4, 2)
    _test(10, 100, 10)
    _test(10, 100, 20)
    _test(100, 23, 4)
    _test(100, 100, 50)


def test_solve_gaussian_concatenate():
    def _test(n, ps, G, intercept=True, alpha=1, sparsity=0.5, seed=0, n_threads=7):
        test_datas = [
            ad.data.dense(n=n, p=p, G=G, sparsity=sparsity, seed=seed)
            for p in ps
        ]
        Xs = [
            ad.matrix.concatenate(
                [
                    ad.matrix.dense(data["X"], method="naive", n_threads=n_threads) 
                    for data in test_datas
                ],
                n_threads=n_threads,
            )
        ]
        X = np.concatenate([
            data["X"] for data in test_datas
        ], axis=-1)
        y = np.mean([data["glm"].y for data in test_datas], axis=0)

        groups = np.concatenate([
            begin + data["groups"]
            for begin, data in zip(
                np.cumsum(np.concatenate([[0], ps[:-1]])),
                test_datas,
            )
        ])
        group_sizes = np.concatenate([groups, [X.shape[-1]]], dtype=int)
        group_sizes = group_sizes[1:] - group_sizes[:-1]
        penalty = np.concatenate([data["penalty"] for data in test_datas])
        weights = np.random.uniform(1, 2, n)
        weights /= np.sum(weights)
        offsets = np.zeros(n)
        X_means = np.sum(weights[:, None] * X, axis=0)
        y_mean = np.sum(weights * y)
        X_c = X - intercept * X_means[None]
        y_c = y - y_mean * intercept
        y_var = np.sum(weights * y_c ** 2)
        resid = weights * y_c
        resid_sum = np.sum(resid)
        screen_set = np.arange(len(groups))[(penalty <= 0) | (alpha <= 0)]
        screen_beta = np.zeros(np.sum(group_sizes[screen_set]))
        screen_is_active = np.zeros(screen_set.shape[0], dtype=bool)
        grad = X_c.T @ resid

        test_data = {
            "y": y,
            "X_means": X_means,
            "y_mean": y_mean,
            "y_var": y_var,
            "resid": resid,
            "resid_sum": resid_sum,
            "groups": groups,
            "group_sizes": group_sizes,
            "alpha": alpha,
            "penalty": penalty,
            "weights": weights,
            "offsets": offsets,
            "screen_set": screen_set,
            "screen_beta": screen_beta,
            "screen_is_active": screen_is_active,
            "rsq": 0,
            "lmda": np.inf,
            "grad": grad,
            "n_threads": n_threads,
        }

        for Xpy in Xs:
            test_data["X"] = Xpy
            state_special = ad.solver._solve(
                ad.state.gaussian_naive(**test_data),
            )
            test_data["X"] = ad.matrix.dense(
                X.astype(np.float64), 
                method="naive", 
                n_threads=n_threads,
            )
            state_dense = ad.solver._solve(
                ad.state.gaussian_naive(**test_data)
            )

            assert np.allclose(state_special.lmdas, state_dense.lmdas)
            assert np.allclose(state_special.devs, state_dense.devs)
            assert np.allclose(state_special.intercepts, state_dense.intercepts, atol=1e-3)
            assert np.allclose(state_special.betas.toarray(), state_dense.betas.toarray(), atol=1e-3)

    ps = np.array([4, 3, 20, 10, 252, 71, 1000])
    _test(10, ps[ps >= 2], 2)
    _test(10, ps[ps >= 17], 17)
    _test(100, ps[ps >= 23], 23)
    _test(100, ps[ps >= 2], 2)
    _test(100, ps[ps >= 6], 6)


def test_solve_gaussian_snp_unphased():
    def _test(n, p, intercept=True, alpha=1, sparsity=0.5, seed=0, n_threads=7):
        test_data = ad.data.snp_unphased(n=n, p=p, sparsity=sparsity, seed=seed)
        filename = f"/tmp/test_snp_unphased.snpdat"
        handler = ad.io.snp_unphased(filename)
        handler.write(test_data["X"], n_threads)
        Xs = [
            ad.matrix.snp_unphased(
                filename=filename,
                dtype=np.float64,
                n_threads=n_threads,
            )
        ]
        os.remove(filename)

        X, y = test_data["X"], test_data.pop("glm").y

        weights = np.random.uniform(1, 2, n)
        weights /= np.sum(weights)

        test_data["y"] = y
        test_data["weights"] = weights
        test_data["offsets"] = np.zeros(n)
        test_data["alpha"] = alpha
        test_data["X_means"] = np.sum(weights[:, None] * X, axis=0)
        test_data["y_mean"] = np.sum(weights * y)
        X_c = X - intercept * test_data["X_means"][None]
        y_c = y - test_data["y_mean"] * intercept
        test_data["y_var"] = np.sum(weights * y_c ** 2)
        test_data["resid"] = weights * y_c
        test_data["resid_sum"] = np.sum(test_data["resid"])
        test_data["screen_set"] = np.arange(p)[(test_data["penalty"] <= 0) | (alpha <= 0)]
        test_data["screen_beta"] = np.zeros(np.sum(test_data["group_sizes"][test_data["screen_set"]]))
        test_data["screen_is_active"] = np.zeros(test_data["screen_set"].shape[0], dtype=bool)
        test_data["grad"] = X_c.T @ test_data["resid"]
        test_data["rsq"] = 0 
        test_data["lmda"] = np.inf
        test_data["tol"] = 1e-10
        test_data["n_threads"] = n_threads

        for Xpy in Xs:
            test_data["X"] = Xpy
            state_special = ad.solver._solve(
                ad.state.gaussian_naive(**test_data),
            )
            test_data["X"] = ad.matrix.dense(
                X.astype(np.float64), 
                method="naive", 
                n_threads=n_threads,
            )
            state_dense = ad.solver._solve(
                ad.state.gaussian_naive(**test_data),
            )

            assert np.allclose(state_special.lmdas, state_dense.lmdas)
            assert np.allclose(state_special.devs, state_dense.devs)
            assert np.allclose(state_special.intercepts, state_dense.intercepts, atol=1e-3)
            assert np.allclose(state_special.betas.toarray(), state_dense.betas.toarray(), atol=1e-3)

    _test(10, 4)
    _test(10, 100)
    _test(100, 23)
    _test(100, 100)
    _test(100, 10000)


def test_solve_gaussian_snp_phased_ancestry():
    def _test(n, p, A=8, intercept=True, alpha=1, sparsity=0.5, seed=0, n_threads=7):
        test_data = ad.data.snp_phased_ancestry(n=n, s=p, A=A, sparsity=sparsity, seed=seed)
        filename = "/tmp/test_snp_phased_ancestry.snpdat"
        handler = ad.io.snp_phased_ancestry(filename)
        handler.write(test_data["X"], test_data["ancestries"], A, n_threads)
        Xs = [
            ad.matrix.snp_phased_ancestry(
                filename=filename,
                dtype=np.float64,
                n_threads=n_threads,
            )
        ]
        handler.read() 
        os.remove(filename)

        X, y = handler.to_dense(n_threads), test_data.pop("glm").y

        weights = np.random.uniform(1, 2, n)
        weights /= np.sum(weights)

        test_data["y"] = y
        test_data["weights"] = weights
        test_data["offsets"] = np.zeros(n)
        test_data["alpha"] = alpha
        test_data["X_means"] = np.sum(weights[:, None] * X, axis=0)
        test_data["y_mean"] = np.sum(weights * y)
        X_c = X - intercept * test_data["X_means"][None]
        y_c = y - test_data["y_mean"] * intercept
        test_data["y_var"] = np.sum(weights * y_c ** 2)
        test_data["resid"] = weights * y_c
        test_data["resid_sum"] = np.sum(test_data["resid"])
        test_data["screen_set"] = np.arange(p)[(test_data["penalty"] <= 0) | (alpha <= 0)]
        test_data["screen_beta"] = np.zeros(np.sum(test_data["group_sizes"][test_data["screen_set"]]))
        test_data["screen_is_active"] = np.zeros(test_data["screen_set"].shape[0], dtype=bool)
        test_data["grad"] = X_c.T @ test_data["resid"]
        test_data["rsq"] = 0 
        test_data["lmda"] = np.inf
        test_data["tol"] = 1e-7
        test_data["n_threads"] = n_threads

        test_data.pop("ancestries")

        for Xpy in Xs:
            test_data["X"] = Xpy
            state_special = ad.solver._solve(
                ad.state.gaussian_naive(**test_data),
            )
            test_data["X"] = ad.matrix.dense(
                X.astype(np.float64), 
                method="naive", 
                n_threads=n_threads,
            )
            state_dense = ad.solver._solve(
                ad.state.gaussian_naive(**test_data),
            )

            assert np.allclose(state_special.lmdas, state_dense.lmdas)
            assert np.allclose(state_special.devs, state_dense.devs)
            assert np.allclose(state_special.intercepts, state_dense.intercepts)
            assert np.allclose(state_special.betas.toarray(), state_dense.betas.toarray())

    _test(10, 4)
    _test(10, 100)
    _test(100, 23)
    _test(100, 100)
    _test(100, 10000)