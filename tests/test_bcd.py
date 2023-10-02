import cvxpy as cp
import numpy as np
import adelie.bcd as mod


def test_root_lower_bound():
    def _test(p, sparsity, quad_lower=0, quad_upper=1, seed=0):
        np.random.seed(seed)
        quad = np.random.uniform(quad_lower, quad_upper, p)
        linear = np.random.normal(0, 1, p)

        zero_idx = np.random.choice(np.arange(p), size=int(sparsity * p), replace=False)
        quad[zero_idx] = 0

        l1 = np.random.uniform(0, np.linalg.norm(linear))

        out = mod.root_lower_bound(
            quad=quad, 
            linear=linear, 
            l1=l1,
        )
        f_out = mod.root_function(
            out, 
            quad=quad, 
            linear=linear, 
            l1=l1,
        ) 
        assert f_out >= 0

    ps = [5, 10, 50, 100, 1000, 10000]
    sparsities = [0.01, 0.3, 0.5, 0.7, 0.9, 0.99]

    # reasonable ranged quad
    for sp in sparsities:
        for p in ps:
            _test(p, sparsity=sp, quad_lower=0, quad_upper=1)

    # very small ranged quad
    for sp in sparsities:
        for p in ps:
            _test(p, sparsity=sp, quad_lower=0, quad_upper=1e-16)

    # very wide ranged quad
    for sp in sparsities:
        for p in ps:
            _test(p, sparsity=sp, quad_lower=0, quad_upper=1e8)


def test_root_upper_bound():
    def _test(p, sparsity, quad_lower=0, quad_upper=1, seed=0):
        np.random.seed(seed)
        quad = np.random.uniform(quad_lower, quad_upper, p)
        linear = np.random.normal(0, 1, p)
        l1 = np.random.uniform(0, 1)

        zero_idx = np.random.choice(np.arange(p), size=int(sparsity * p), replace=False)
        quad[zero_idx] = 0
        linear[zero_idx] = 0

        out = mod.root_upper_bound(
            quad=quad, 
            linear=linear, 
            zero_tol=1e-6 * quad_upper,
        )
        f_out = mod.root_function(
            out, 
            quad=quad, 
            linear=linear, 
            l1=l1,
        ) 
        assert f_out <= 0

    ps = [5, 10, 50, 100, 1000, 10000]
    sparsities = [0.01, 0.3, 0.5, 0.7, 0.9, 0.99]

    # reasonable ranged quad
    for sp in sparsities:
        for p in ps:
            _test(p, sparsity=sp, quad_lower=0, quad_upper=1)

    # very small ranged quad
    for sp in sparsities:
        for p in ps:
            _test(p, sparsity=sp, quad_lower=0, quad_upper=1e-16)

    # very wide ranged quad
    for sp in sparsities:
        for p in ps:
            _test(p, sparsity=sp, quad_lower=0, quad_upper=1e8)
            

def test_solve():
    def _solve_cvxpy(quad, linear, l1, l2):
        p = quad.shape[0]
        x = cp.Variable(p)
        expr = (
            0.5 * (quad @ x ** 2) - linear @ x + l1 * cp.norm2(x) + 0.5 * l2 * cp.sum(x ** 2)
        )
        prob = cp.Problem(cp.Minimize(expr))
        prob.solve()
        return x.value

    def _test(
        p,
        sparsity,
        quad_lower, 
        quad_upper,
        l2_upper,
        seed=0,
        tol=1e-16,
        max_iters=10000,
        solvers=["newton_abs"],
    ):
        np.random.seed(seed)
        quad = np.random.uniform(quad_lower, quad_upper, p)
        linear = np.random.normal(0, 1, p)
        l2 = np.random.uniform(0, l2_upper)
        l1 = np.random.uniform(0, 1)

        # zero-out some components of quad
        zero_idx = np.random.choice(np.arange(p), size=int(sparsity * p), replace=False)
        quad[zero_idx] = 0
        # if no l2 regularization, must zero-out same positions of linear
        if l2 <= 0:
            linear[zero_idx] = 0

        # solve using cvxpy
        x_exp = _solve_cvxpy(quad, linear, l1, l2)

        # test each solver
        for solver in solvers:
            print(f"Solver: {solver}")
            x_actual = mod.solve(
                quad=quad,
                linear=linear,
                l1=l1,
                l2=l2,
                tol=tol,
                max_iters=max_iters,
                solver=solver,
            )["beta"]
            is_close = np.allclose(x_actual, x_exp)
            if not is_close:
                f_x_actual = mod.objective(
                    x_actual,
                    quad=quad,
                    linear=linear,
                    l1=l1,
                    l2=l2,
                )
                f_x_exp = mod.objective(
                    x_exp,
                    quad=quad,
                    linear=linear,
                    l1=l1,
                    l2=l2,
                )
                assert f_x_actual <= f_x_exp

    ps = [10, 100, 1000]
    sps = [0.1, 0.5, 0.9]
    solvers = [
        "newton", 
        "newton_brent", 
        "newton_abs", 
    ]

    # non-trivial l2 regularization
    for sp in sps:
        for p in ps:
            _test(p, sp, 0, 1, 1e-2, solvers=solvers)

    # no l2 regularization
    for sp in sps:
        for p in ps:
            _test(p, sp, 0, 1, 0, solvers=solvers)


def test_root():
    def _test(
        p,
        sparsity,
        quad_lower, 
        quad_upper,
        seed=0,
        tol=1e-16,
        max_iters=10000,
        solvers=["newton_abs"],
    ):
        np.random.seed(seed)
        quad = np.random.uniform(quad_lower, quad_upper, p)
        linear = np.random.normal(0, 1, p)
        l1 = np.random.uniform(0, 1)

        # zero-out some components of quad
        zero_idx = np.random.choice(np.arange(p), size=int(sparsity * p), replace=False)
        quad[zero_idx] = 0
        linear[zero_idx] = 0

        # test each solver
        for solver in solvers:
            print(f"Solver: {solver}")
            x_actual = mod.root(
                quad=quad,
                linear=linear,
                l1=l1,
                tol=tol,
                max_iters=max_iters,
                solver=solver,
            )["root"]

            if np.linalg.norm(linear) < l1:
                assert x_actual is None
                continue
            assert not x_actual is None
            f_x_actual = mod.root_function(
                x_actual,
                quad=quad,
                linear=linear,
                l1=l1,
            )
            assert np.allclose(f_x_actual, 0)

    ps = [10, 100, 1000]
    sps = [0.1, 0.5, 0.9]
    solvers = [
        "newton", 
        "newton_brent", 
        "newton_abs", 
    ]

    # non-trivial l2 regularization
    for sp in sps:
        for p in ps:
            _test(p, sp, 0, 1, solvers=solvers)

    # no l2 regularization
    for sp in sps:
        for p in ps:
            _test(p, sp, 0, 1, solvers=solvers)
