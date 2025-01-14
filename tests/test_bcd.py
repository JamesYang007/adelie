import cvxpy as cp
import numpy as np
import adelie.bcd as mod
import pytest


# ========================================================================
# TEST Elastic Net
# ========================================================================


@pytest.mark.parametrize("p", [5, 10, 50, 100, 1000, 10000])
@pytest.mark.parametrize("sparsity", [0.01, 0.3, 0.5, 0.7, 0.9, 0.99])
@pytest.mark.parametrize("quad_lower, quad_upper", [
    # reasonable ranged quad
    [0, 1],
    # very small ranged quad
    [0, 1e-16],
    # very wide ranged quad
    [0, 1e8],
])
def test_elastic_net_root_lower_bound(
    p, sparsity, quad_lower, quad_upper, 
    seed=0,
):
    np.random.seed(seed)
    quad = np.random.uniform(quad_lower, quad_upper, p)
    linear = np.random.normal(0, 1, p)

    zero_idx = np.random.choice(np.arange(p), size=int(sparsity * p), replace=False)
    quad[zero_idx] = 0

    l1 = np.random.uniform(0, np.linalg.norm(linear))

    out = mod.elastic_net.root_lower_bound(
        quad=quad, 
        linear=linear, 
        l1=l1,
    )
    f_out = mod.elastic_net.root_function(
        out, 
        quad=quad, 
        linear=linear, 
        l1=l1,
    ) 
    assert f_out >= 0


@pytest.mark.parametrize("p", [5, 10, 50, 100, 1000, 10000])
@pytest.mark.parametrize("sparsity", [0.01, 0.3, 0.5, 0.7, 0.9, 0.99])
@pytest.mark.parametrize("quad_lower, quad_upper", [
    # reasonable ranged quad
    [0, 1],
    # very small ranged quad
    [0, 1e-16],
    # very wide ranged quad
    [0, 1e8],
])
def test_elastic_net_root_upper_bound(
    p, sparsity, quad_lower, quad_upper, 
    seed=0,
):
    np.random.seed(seed)
    quad = np.random.uniform(quad_lower, quad_upper, p)
    linear = np.random.normal(0, 1, p)
    l1 = np.random.uniform(0, 1) * np.linalg.norm(linear)

    zero_idx = np.random.choice(np.arange(p), size=int(sparsity * p), replace=False)
    quad[zero_idx] = 0
    linear[zero_idx] = 0

    out = mod.elastic_net.root_upper_bound(
        quad=quad, 
        linear=linear, 
        l1=l1,
        zero_tol=0,
    )
    f_out = mod.elastic_net.root_function(
        out, 
        quad=quad, 
        linear=linear, 
        l1=l1,
    ) 
    assert f_out <= 0
            

@pytest.mark.parametrize("p", [10, 100, 1000])
@pytest.mark.parametrize("sparsity", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("solver", ["newton", "newton_brent", "newton_abs"])
@pytest.mark.parametrize("quad_lower, quad_upper, l2_upper", [
    # non-trivial l2 regularization
    [0, 1, 1e-2],
    # no l2 regularization
    [0, 1, 0]
])
def test_elastic_net_solve(
    p, sparsity, solver, quad_lower, quad_upper, l2_upper, 
    seed=0, tol=1e-16, max_iters=10000
):
    def _solve_cvxpy(quad, linear, l1, l2):
        p = quad.shape[0]
        x = cp.Variable(p)
        expr = (
            0.5 * (quad @ x ** 2) - linear @ x + l1 * cp.norm2(x) + 0.5 * l2 * cp.sum(x ** 2)
        )
        prob = cp.Problem(cp.Minimize(expr))
        prob.solve()
        return x.value

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

    x_actual = mod.elastic_net.solve(
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
        f_x_actual = mod.elastic_net.objective(
            x_actual,
            quad=quad,
            linear=linear,
            l1=l1,
            l2=l2,
        )
        f_x_exp = mod.elastic_net.objective(
            x_exp,
            quad=quad,
            linear=linear,
            l1=l1,
            l2=l2,
        )
        assert f_x_actual <= f_x_exp


@pytest.mark.parametrize("p", [10, 100, 1000])
@pytest.mark.parametrize("sparsity", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("solver", ["newton", "newton_brent", "newton_abs"])
@pytest.mark.parametrize("quad_lower, quad_upper", [
    [0, 1],
])
def test_elastic_net_root(
    p, sparsity, solver, quad_lower, quad_upper,
    seed=0, tol=1e-16, max_iters=10000
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
    x_actual = mod.elastic_net.root(
        quad=quad,
        linear=linear,
        l1=l1,
        tol=tol,
        max_iters=max_iters,
        solver=solver,
    )["root"]

    if np.linalg.norm(linear) < l1:
        assert x_actual is None
        return

    assert not x_actual is None
    f_x_actual = mod.elastic_net.root_function(
        x_actual,
        quad=quad,
        linear=linear,
        l1=l1,
    )
    assert np.allclose(f_x_actual, 0)


# ========================================================================
# TEST SGL
# ========================================================================


@pytest.mark.parametrize("seed", [i * 13 for i in range(100)])
def test_sgl_quartic_roots(seed):
    def quartic(x, coeffs):
        X = np.array([x**4, x**3, x**2, x, 1])
        return X.T @ coeffs

    np.random.seed(seed)
    coeffs = np.random.normal(0, 1, 5)
    actual = mod.sgl.quartic_roots(*coeffs)
    quartic_x = np.array([quartic(x, coeffs) for x in actual])
    assert np.allclose(quartic_x, 0, atol=1e-7)


@pytest.mark.parametrize("seed", [i * 13 for i in range(100)])
def test_sgl_root_secular(seed):
    np.random.seed(seed)
    y = np.random.uniform(0, 1)
    m = np.random.uniform(0, 1)
    a = np.random.uniform(0, 1)
    b = np.random.uniform(0, 1)
    h = mod.sgl.root_secular(y, m=m, a=a, b=b)
    assert h >= 0
    assert np.allclose((m + b / np.sqrt(h ** 2 + a)) * h - y, 0)


@pytest.mark.parametrize("n", [1, 10, 100])
@pytest.mark.parametrize("p", [10, 50, 100])
@pytest.mark.parametrize("l1", np.logspace(0, -1, 3))
@pytest.mark.parametrize("l2", np.logspace(0, -1, 3))
def test_sgl_solve(n, p, l1, l2, seed=0, tol=1e-12, max_iters=10000):
    def _solve_cvxpy(quad, linear, l1, l2):
        p = quad.shape[0]
        x = cp.Variable(p)
        expr = (
            0.5 * cp.quad_form(x, quad, assume_PSD=True) - linear @ x + l1 * cp.norm(x, 1) + l2 * cp.norm(x)
        )
        prob = cp.Problem(cp.Minimize(expr))
        prob.solve()
        return x.value

    np.random.seed(seed)
    X = np.random.normal(0, 1, (n, p))
    y = X[:, 0] * np.random.normal(0, 1) + np.random.normal(0, 1, n)
    y_var = 1 if y.size == 1 else np.var(y)
    quad = X.T @ X / n
    linear = X.T @ y / n

    # solve using cvxpy
    x_exp = _solve_cvxpy(quad, linear, l1, l2)

    x_actual = mod.sgl.solve(
        quad=quad,
        linear=linear,
        l1=l1,
        l2=l2,
        tol=tol * y_var,
        max_iters=max_iters,
    )["beta"]
    is_close = np.allclose(x_actual, x_exp)
    if not is_close:
        f_x_actual = mod.sgl.objective(
            x_actual,
            quad=quad,
            linear=linear,
            l1=l1,
            l2=l2,
        )
        f_x_exp = mod.sgl.objective(
            x_exp,
            quad=quad,
            linear=linear,
            l1=l1,
            l2=l2,
        )
        assert (f_x_actual <= f_x_exp) or np.allclose(f_x_actual, f_x_exp)