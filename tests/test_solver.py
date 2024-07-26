from adelie.diagnostic import (
    objective,
)
from adelie.solver import (
    _solve,
)
from adelie import adelie_core as core
import adelie as ad
import cvxpy as cp
import numpy as np
import os
import pytest


# ========================================================================
# Helper Classes and Functions
# ========================================================================


class CvxpyGlmGaussian():
    def __init__(self, y, weights):
        self.y = y
        self.weights = weights
        self.is_multi = False

    def loss(self, eta):
        return self.weights @ (
            -cp.multiply(self.y, eta) + 0.5 * cp.square(eta)
        )

    def to_adelie(self):
        return ad.glm.gaussian(self.y, weights=self.weights)


class CvxpyGlmBinomial():
    def __init__(self, y, weights):
        self.y = y
        self.weights = weights
        self.is_multi = False

    def loss(self, eta):
        return self.weights @ (
            -cp.multiply(self.y, eta) + cp.logistic(eta)
        )

    def to_adelie(self):
        return ad.glm.binomial(self.y, weights=self.weights)


class CvxpyGlmPoisson():
    def __init__(self, y, weights):
        self.y = y
        self.weights = weights
        self.is_multi = False

    def loss(self, eta):
        return self.weights @ (
            -cp.multiply(self.y, eta) + cp.exp(eta)
        )

    def to_adelie(self):
        return ad.glm.poisson(self.y, weights=self.weights)


class CvxpyGlmCox():
    def __init__(self, start, stop, status, weights):
        self.start = start
        self.stop = stop
        self.status = status
        self.weights = weights
        self.is_multi = False

        n = start.shape[0]

        self.stop_order = np.argsort(stop)
        self.inv_stop_order = np.argsort(self.stop_order)

        self.weights_sum = np.empty(n)
        core.glm.GlmCox64._nnz_event_ties_sum(
            self.weights[self.stop_order], 
            stop[self.stop_order], 
            status[self.stop_order], 
            weights[self.stop_order], 
            self.weights_sum,
        )
        self.weights_sum = self.weights_sum[self.inv_stop_order]

        self.weights_size = np.empty(n)
        core.glm.GlmCox64._nnz_event_ties_sum(
            np.ones(n),
            stop[self.stop_order], 
            status[self.stop_order], 
            weights[self.stop_order], 
            self.weights_size,
        )
        self.weights_size = self.weights_size[self.inv_stop_order]

        self.weights_mean = np.divide(
            self.weights_sum,
            self.weights_size,
            where=self.weights_size > 0,
        )
        self.weights_mean[self.weights_size == 0] = 0

    def loss(self, eta):
        is_at_risk = (
            (self.start[None] < self.stop[:, None]) &
            (self.stop[None] >= self.stop[:, None])
        )
        # Should be -np.inf but MOSEK cannot understand it
        is_at_risk = np.where(is_at_risk, is_at_risk, -1e4)
        A = cp.log_sum_exp(
            cp.multiply(is_at_risk, (eta + np.log(self.weights))[None]),
            axis=1
        )
        return (
            - (self.weights * self.status) @ eta
            + (self.weights_mean * self.status) @ A
        )

    def to_adelie(self):
        return ad.glm.cox(
            self.start,
            self.stop,
            self.status,
            weights=self.weights,
            tie_method="breslow",
        )


class CvxpyGlmMultiGaussian():
    def __init__(self, y, weights):
        self.y = y
        self.weights = weights
        self.is_multi = True

    def loss(self, eta):
        return self.weights @ cp.sum(
            -cp.multiply(self.y, eta) + 0.5 * cp.square(eta),
            axis=1,
        ) / self.y.shape[-1]

    def to_adelie(self):
        return ad.glm.multigaussian(self.y, weights=self.weights)


class CvxpyGlmMultinomial():
    def __init__(self, y, weights):
        self.y = y
        self.weights = weights
        self.is_multi = True

    def loss(self, eta):
        return self.weights @ (
            - cp.sum(cp.multiply(self.y, eta), axis=1) 
            + cp.log_sum_exp(eta, axis=1)
        ) / self.y.shape[-1]

    def to_adelie(self):
        return ad.glm.multinomial(self.y, weights=self.weights)


def zero_constraint(
    size,
    dtype
):
    core_base = {
        np.float32: ad.constraint.ConstraintBase32,
        np.float64: ad.constraint.ConstraintBase64,
    }[dtype]

    class _zero_constraint(core_base):
        def __init__(self):
            core_base.__init__(self)
            self.size = size
            self.mu = np.zeros(size)

        def solve(self, x, quad, linear, l1, l2, Q, buffer):
            x[...] = 0
            self.mu[...] = linear

        def gradient(self, x, out):
            out[...] = self.mu

        def gradient_static(self, x, mu, out):
            out[...] = mu

        def project(self, x):
            x[...] = 0

        def clear(self):
            self.mu[...] = 0

        def dual(self, indices, values):
            indices[...] = np.arange(self.size)
            values[...] = self.mu

        def duals_nnz(self):
            return self.size

        def duals(self):
            return self.size

        def primals(self):
            return self.size

    return _zero_constraint()


# ========================================================================
# TEST Gaussian
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
    constraint: bool=False,
    dtype =np.float64,
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

    if constraint:
        constraints = [None] * G
        c_order = np.random.choice(G, G // 2, replace=False)
    else:
        constraints = None

    args = {
        "X": X, 
        "y": y,
        "constraints": constraints,
        "groups": groups,
        "alpha": alpha,
        "penalty": penalty,
        "weights": weights,
        "rsq": 0,
        "intercept": intercept,
        "active_set_size": 0,
        "active_set": np.empty(G, dtype=int),
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
    cvxpy_glm,
    groups: np.ndarray,
    lmda: float,
    alpha: float,
    penalty: np.ndarray,
    intercept: bool,
    pin: bool =False,
    screen_set: np.ndarray =None,
    constraints: list =None,
):
    _, p = X.shape
    G = groups.shape[0]

    group_sizes = np.concatenate([groups, [p]], dtype=int)
    group_sizes = group_sizes[1:] - group_sizes[:-1]

    if constraints is None:
        constraints = [None] * G

    if cvxpy_glm.is_multi:
        K = cvxpy_glm.y.shape[-1]

        penalty = penalty[K:] if intercept else penalty
        assert penalty.shape[0] == G

        beta = cp.Variable((p, K))
        beta0 = cp.Variable(K)
        eta = X @ beta + beta0[None]
        expr = cvxpy_glm.loss(eta)
        for g, gs, w in zip(groups, group_sizes, penalty):
            expr += (lmda * w) * (
                alpha * cp.norm(beta[g:g+gs], "fro") 
                + 0.5 * (1-alpha) * cp.sum_squares(beta[g:g+gs])
            )

    else:
        beta = cp.Variable(p)
        beta0 = cp.Variable(1)
        eta = X @ beta + beta0
        expr = cvxpy_glm.loss(eta)
        for g, gs, w in zip(groups, group_sizes, penalty):
            expr += (lmda * w) * (
                alpha * cp.norm(beta[g:g+gs]) 
                + 0.5 * (1-alpha) * cp.sum_squares(beta[g:g+gs])
            )

    constraints = [
        beta[g:g+gs] == 0
        for c, g, gs in zip(constraints, groups, group_sizes)
        if not (c is None)
    ]

    if pin:
        constraints += [
            beta[g:g+gs] == 0
            for i, (g, gs) in enumerate(zip(groups, group_sizes))
            if not (i in screen_set)
        ]

    if not intercept:
        constraints += [ beta0 == 0 ]
    prob = cp.Problem(cp.Minimize(expr), constraints)
    prob.solve()

    if cvxpy_glm.is_multi:
        return beta0.value, beta.value.ravel()
    else:
        return beta0.value[0], beta.value


def check_solutions(
    args,
    state,
    cvxpy_glm,
    pin: bool =False,
    eps: float =1e-8,
):
    # get solved lmdas
    lmdas = state.lmdas

    X = args["X"]
    intercept = args["intercept"]
    groups = args["groups"]
    constraints = args["constraints"]
    cvxpy_res = [
        solve_cvxpy(
            X=X,
            cvxpy_glm=cvxpy_glm,
            groups=groups,
            lmda=lmda,
            alpha=state.alpha,
            penalty=state.penalty,
            intercept=intercept,
            pin=pin,
            constraints=constraints,
            screen_set=state.screen_set,
        )
        for lmda in lmdas
    ]
    cvxpy_intercepts = np.array([out[0] for out in cvxpy_res])
    cvxpy_betas = np.array([out[1] for out in cvxpy_res])

    # check beta matches (if not, at least that objective is better)
    betas = state.betas
    intercepts = state.intercepts

    is_intercept_close = np.allclose(intercepts, cvxpy_intercepts, atol=1e-6)
    is_beta_close = np.allclose(betas.toarray(), cvxpy_betas, atol=1e-6)

    if not (is_beta_close and is_intercept_close):
        objective_args = {
            "X": X,
            "glm": cvxpy_glm.to_adelie(),
            "lmdas": lmdas,
            "groups": groups,
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
        assert np.all(my_objs <= cvxpy_objs * (1 + eps))


# ========================================================================
# TEST Gaussian
# ========================================================================


def run_solve_gaussian(state, args, pin):
    state.check(method="assert")
    state = _solve(state)    
    state.check(method="assert")
    cvxpy_glm = CvxpyGlmGaussian(args["y"], args["weights"])
    check_solutions(args, state, cvxpy_glm, pin)
    return state


@pytest.mark.filterwarnings("ignore: Detected matrix to be C-contiguous.")
@pytest.mark.parametrize("constraint", [False, True])
@pytest.mark.parametrize("n, p, G, S", [
    [10, 4, 2, 2],
    [10, 100, 10, 2],
    [10, 100, 20, 13],
    [100, 23, 4, 3],
    [100, 100, 50, 20],
])
def test_solve_gaussian_pin_naive(
    n, p, G, S, constraint,
    intercept=True, alpha=1, sparsity=0.95, seed=0,
):
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
        constraint=constraint,
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
        args_c["active_set_size"] = state.active_set_size
        args_c["active_set"] = state.active_set
        state = ad.state.gaussian_pin_naive(
            **args_c,
            tol=1e-7,
        )
        run_solve_gaussian(state, args, pin=True)


@pytest.mark.filterwarnings("ignore: Detected matrix to be C-contiguous.")
@pytest.mark.parametrize("constraint", [False, True])
@pytest.mark.parametrize("n, p, G, S", [
    [10, 4, 2, 2],
    [10, 100, 10, 2],
    [10, 100, 20, 13],
    [100, 23, 4, 3],
    [100, 100, 50, 20],
])
def test_solve_gaussian_pin_cov(
    n, p, G, S, constraint,
    alpha=1, sparsity=0.95, seed=0,
):
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
        constraint=constraint,
        method="cov",
    )

    # list of different types of cov matrices to test
    As = [
        ad.matrix.dense(args["A"], method="cov", n_threads=3),
        ad.matrix.lazy_cov(args["WsqrtX"], n_threads=3),
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
        args_c["active_set_size"] = state.active_set_size
        args_c["active_set"] = state.active_set
        state = ad.state.gaussian_pin_cov(
            **args_c,
            tol=1e-7,
        )
        run_solve_gaussian(state, args, pin=True)


@pytest.mark.filterwarnings("ignore: Detected matrix to be C-contiguous.")
@pytest.mark.parametrize("constraint", [False, True])
@pytest.mark.parametrize("n, p, G", [
    [10, 4, 2],
    [10, 100, 10],
    [10, 100, 20],
    [100, 23, 4],
    [100, 100, 50],
])
def test_solve_gaussian(
    n, p, G, constraint, 
    intercept=True, alpha=1, sparsity=0.95, seed=0,
):
    args = create_data_gaussian(
        n=n, 
        p=p, 
        G=G, 
        S=None,
        intercept=intercept, 
        alpha=alpha, 
        sparsity=sparsity, 
        constraint=constraint,
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
        args_c["active_set_size"] = state.active_set_size
        args_c["active_set"] = state.active_set
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


@pytest.mark.parametrize("n, ps, G", [
    [10, [4, 3, 20, 10, 252, 71, 1000], 2],
    [10, [20, 252, 71, 1000], 17],
    [100, [252, 71, 1000], 23],
    [100, [4, 3, 20, 10, 252, 71, 1000], 2],
    [100, [20, 10, 252, 71, 1000], 6],
])
def test_solve_gaussian_concatenate(
    n, ps, G, 
    intercept=True, alpha=1, sparsity=0.5, seed=0, n_threads=2,
):
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
            axis=1,
            n_threads=n_threads,
        )
    ]
    X = np.concatenate([
        data["X"] for data in test_datas
    ], axis=-1)
    y = np.mean([data["glm"].y for data in test_datas], axis=0)

    constraints = None
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
    resid = y_c
    resid_sum = np.sum(weights * resid)
    screen_set = np.arange(len(groups))[(penalty <= 0) | (alpha <= 0)]
    screen_beta = np.zeros(np.sum(group_sizes[screen_set]))
    screen_is_active = np.zeros(screen_set.shape[0], dtype=bool)
    grad = X_c.T @ (weights * resid)

    test_data = {
        "y": y,
        "X_means": X_means,
        "y_mean": y_mean,
        "y_var": y_var,
        "resid": resid,
        "resid_sum": resid_sum,
        "constraints": constraints,
        "groups": groups,
        "group_sizes": group_sizes,
        "alpha": alpha,
        "penalty": penalty,
        "weights": weights,
        "offsets": offsets,
        "screen_set": screen_set,
        "screen_beta": screen_beta,
        "screen_is_active": screen_is_active,
        "active_set_size": 0,
        "active_set": np.empty(groups.shape[0], dtype=int),
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


@pytest.mark.parametrize("n, p", [
    [10, 4],
    [10, 100],
    [100, 23],
    [100, 100],
    [100, 10000],
])
def test_solve_gaussian_snp_unphased(
    n, p, 
    intercept=True, alpha=1, sparsity=0.5, seed=0, n_threads=3,
):
    test_data = ad.data.snp_unphased(n=n, p=p, sparsity=sparsity, seed=seed)
    filename = f"/tmp/test_snp_unphased.snpdat"
    handler = ad.io.snp_unphased(filename)
    handler.write(test_data["X"], impute_method="mean", n_threads=n_threads)
    handler.read()
    impute = handler.impute
    Xs = [
        ad.matrix.snp_unphased(
            io=handler,
            dtype=np.float64,
            n_threads=n_threads,
        )
    ]

    X, y = test_data["X"], test_data.pop("glm").y
    X = np.where(X == -9, impute[None], X)

    weights = np.random.uniform(1, 2, n)
    weights /= np.sum(weights)

    test_data["constraints"] = None
    test_data["y"] = y
    test_data["weights"] = weights
    test_data["offsets"] = np.zeros(n)
    test_data["alpha"] = alpha
    test_data["X_means"] = np.sum(weights[:, None] * X, axis=0)
    test_data["y_mean"] = np.sum(weights * y)
    X_c = X - intercept * test_data["X_means"][None]
    y_c = y - test_data["y_mean"] * intercept
    test_data["y_var"] = np.sum(weights * y_c ** 2)
    test_data["resid"] = y_c
    test_data["resid_sum"] = np.sum(weights * test_data["resid"])
    test_data["screen_set"] = np.arange(p)[(test_data["penalty"] <= 0) | (alpha <= 0)]
    test_data["screen_beta"] = np.zeros(np.sum(test_data["group_sizes"][test_data["screen_set"]]))
    test_data["screen_is_active"] = np.zeros(test_data["screen_set"].shape[0], dtype=bool)
    test_data["active_set_size"] = 0
    test_data["active_set"] = np.empty(p, dtype=int)
    test_data["grad"] = X_c.T @ (weights * test_data["resid"])
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

    os.remove(filename)


@pytest.mark.filterwarnings("ignore: Detected matrix to be C-contiguous.")
@pytest.mark.parametrize("n, p", [
    [10, 4],
    [10, 100],
    [100, 23],
    [100, 100],
    [100, 10000],
])
def test_solve_gaussian_snp_phased_ancestry(
    n, p, 
    A=8, intercept=True, alpha=1, sparsity=0.5, seed=0, n_threads=2,
):
    test_data = ad.data.snp_phased_ancestry(n=n, s=p, A=A, sparsity=sparsity, seed=seed)
    filename = "/tmp/test_snp_phased_ancestry.snpdat"
    handler = ad.io.snp_phased_ancestry(filename)
    handler.write(test_data["X"], test_data["ancestries"], A, n_threads)
    Xs = [
        ad.matrix.snp_phased_ancestry(
            io=handler,
            dtype=np.float64,
            n_threads=n_threads,
        )
    ]
    handler.read() 

    X, y = handler.to_dense(n_threads), test_data.pop("glm").y

    weights = np.random.uniform(1, 2, n)
    weights /= np.sum(weights)

    test_data["constraints"] = None
    test_data["y"] = y
    test_data["weights"] = weights
    test_data["offsets"] = np.zeros(n)
    test_data["alpha"] = alpha
    test_data["X_means"] = np.sum(weights[:, None] * X, axis=0)
    test_data["y_mean"] = np.sum(weights * y)
    X_c = X - intercept * test_data["X_means"][None]
    y_c = y - test_data["y_mean"] * intercept
    test_data["y_var"] = np.sum(weights * y_c ** 2)
    test_data["resid"] = y_c
    test_data["resid_sum"] = np.sum(weights * test_data["resid"])
    test_data["screen_set"] = np.arange(p)[(test_data["penalty"] <= 0) | (alpha <= 0)]
    test_data["screen_beta"] = np.zeros(np.sum(test_data["group_sizes"][test_data["screen_set"]]))
    test_data["screen_is_active"] = np.zeros(test_data["screen_set"].shape[0], dtype=bool)
    test_data["active_set_size"] = 0
    test_data["active_set"] = np.empty(p, dtype=int)
    test_data["grad"] = X_c.T @ (weights * test_data["resid"])
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

    os.remove(filename)


# ==========================================================================================
# TEST grpnet
# ==========================================================================================


@pytest.mark.parametrize("constraint", [False, True])
@pytest.mark.parametrize("glm_type", [
    "gaussian",
    "binomial",
    "poisson",
    "cox",
    "multigaussian",
    "multinomial",
])
@pytest.mark.parametrize("n, p, G", [
    [10, 50, 10],
    [40, 13, 7],
])
def test_grpnet(
    n, p, G, glm_type, constraint,
    intercept=True, adev_tol=0.2,
):
    K = 3 if "multi" in glm_type else 1
    data = ad.data.dense(n, p, p, K=K, glm=glm_type)
    X, glm = data["X"], data["glm"]
    groups = np.concatenate([
        [0],
        np.random.choice(np.arange(1, p), size=G-1, replace=False)
    ])
    groups = np.sort(groups).astype(int)
    group_sizes = np.concatenate([groups, [p]], dtype=int)
    group_sizes = group_sizes[1:] - group_sizes[:-1]

    if glm_type == "cox":
        cvxpy_glm = CvxpyGlmCox(
            glm.start,
            glm.stop,
            glm.status,
            glm.weights,
        )
        glm = cvxpy_glm.to_adelie()
    else:
        cvxpy_glm = {
            "gaussian": CvxpyGlmGaussian,
            "binomial": CvxpyGlmBinomial,
            "poisson": CvxpyGlmPoisson,
            "multigaussian": CvxpyGlmMultiGaussian,
            "multinomial": CvxpyGlmMultinomial,
        }[glm_type](glm.y, glm.weights)

    if constraint:
        constraints = [None] * G
        c_order = np.random.choice(G, G // 2, replace=False)
        for i in c_order:
            size = group_sizes[i] * K
            constraints[i] = zero_constraint(size, dtype=np.float64)
    else:
        constraints = None

    args = {
        "X": X,
        "intercept": intercept,
        "groups": groups,
        "constraints": constraints,
    }
    state = ad.grpnet(
        X=X, 
        glm=glm, 
        constraints=constraints,
        groups=groups,
        intercept=intercept,
        adev_tol=adev_tol,
        irls_tol=1e-9,
        tol=1e-10,
        progress_bar=False,
    )
    check_solutions(args, state, cvxpy_glm, eps=1e-4)