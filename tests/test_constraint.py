from adelie import constraint
import numpy as np
import pytest


def run_test(
    cnstr,
    cnstr_exp,
    dtype,
    seed: int =0,
):
    atol = 1e-6 if dtype == np.float32 else 5e-7

    # test input/output sizes
    assert cnstr.duals() == cnstr_exp.duals()
    assert cnstr.dual_size == cnstr_exp.dual_size
    assert cnstr.primals() == cnstr_exp.primals()
    assert cnstr.primal_size == cnstr_exp.primal_size

    d, m = cnstr.primal_size, cnstr.dual_size

    # generate data
    np.random.seed(seed)
    quad = np.random.uniform(0, 1, d).astype(dtype)
    linear = np.sqrt(quad) * np.random.normal(0, 1, d).astype(dtype)
    l1 = 0.5 * np.linalg.norm(linear)
    l2 = 0
    Q = np.random.normal(0, 1, (d, d))
    Q, _, _ = np.linalg.svd(Q)
    Q = np.asfortranarray(Q, dtype=dtype)
    buffer = np.empty(cnstr.buffer_size(), dtype=np.uint64)

    # test solve
    x = np.zeros(d, dtype=dtype)
    cnstr.solve(x, quad, linear, l1, l2, Q, buffer)

    # test gradient
    actual = np.empty(d, dtype=dtype)
    cnstr.gradient(x, actual)
    mu = np.zeros(m, dtype=dtype)
    mu_nnz = cnstr.duals_nnz()
    mu_indices = np.empty(mu_nnz, dtype=int)
    mu_values = np.empty(mu_nnz, dtype=dtype)
    cnstr.dual(mu_indices, mu_values)
    mu[mu_indices] = mu_values
    expected = np.empty(d, dtype=dtype)
    cnstr_exp.gradient(x, mu, expected)
    assert np.allclose(actual, expected)

    # KKT first-order condition
    Qx = Q @ x
    grad = np.empty(d, dtype=dtype)
    cnstr_exp.gradient(Qx, mu, grad)
    lagr_grad = (quad + l2) * x - linear + Q.T @ grad
    assert np.allclose(np.maximum(np.linalg.norm(lagr_grad)-l1, 0), 0, atol=atol)

    # KKT primal feasibility
    eval = cnstr_exp.evaluate(Qx)
    assert np.allclose(np.maximum(np.max(eval), 0), 0, atol=atol)

    # test project 
    cnstr.project(Qx)
    eval = cnstr_exp.evaluate(Qx)
    assert np.allclose(np.maximum(np.max(eval), 0), 0, atol=atol)


@pytest.mark.parametrize("d", [1, 2, 5, 10, 20, 50])
@pytest.mark.parametrize("lower", [-1, -1e-14, 0])
@pytest.mark.parametrize("upper", [1, 1e-14, 0])
@pytest.mark.parametrize("method", ["proximal_newton"])
@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("seed", np.arange(10))
def test_box(d, lower, upper, method, dtype, seed):
    np.random.seed(seed)
    lower = np.random.uniform(lower, 0, d)
    upper = np.random.uniform(0, 1, d)
    configs = {
        "proximal_newton": {
            "max_iters": 1000,
            "tol": 1e-14,
            "pinball_tol": 1e-14,
        },
    }[method]
    cnstr = constraint.box(lower, upper, method=method, configs=configs, dtype=dtype)

    class Box:
        def __init__(self):
            self.dual_size = d
            self.primal_size = d
        def evaluate(self, x):
            return np.concatenate([x - upper, lower - x])
        def gradient(self, x, mu, out):
            out[...] = mu
        def duals(self):
            return self.dual_size
        def primals(self):
            return self.primal_size

    cnstr_exp = Box()
    run_test(cnstr, cnstr_exp, dtype, seed)


@pytest.mark.parametrize("m", [1, 2, 5, 10, 20, 50])
@pytest.mark.parametrize("d", [10])
@pytest.mark.parametrize("lower", [-1, -1e-14, 0])
@pytest.mark.parametrize("upper", [1, 1e-14, 0])
@pytest.mark.parametrize("method", ["proximal_newton"])
@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("seed", np.arange(10))
def test_linear(m, d, lower, upper, method, dtype, seed):
    np.random.seed(seed)
    A = np.random.normal(0, 1, (m, d))
    A[0,0] = 0
    lower = np.random.uniform(lower, 0, m)
    upper = np.random.uniform(0, upper, m)
    configs = {
        "proximal_newton": {
            "max_iters": 1000,
            "tol": 1e-16,
            "nnls_tol": 1e-16,
            "pinball_max_iters": 1000000,
            "pinball_tol": 1e-9,
        }
    }[method]
    cnstr = constraint.linear(A, lower, upper, method=method, configs=configs)

    class Linear:
        def __init__(self):
            self.dual_size = m
            self.primal_size = d
        def evaluate(self, x):
            Ax = A @ x
            return np.concatenate([Ax - upper, lower - Ax])
        def gradient(self, x, mu, out):
            out[...] = A.T @ mu
        def duals(self):
            return self.dual_size
        def primals(self):
            return self.primal_size

    cnstr_exp = Linear()
    run_test(cnstr, cnstr_exp, dtype, seed)


@pytest.mark.parametrize("d", [1, 2, 5, 10, 20, 50])
@pytest.mark.parametrize("method", ["proximal_newton"])
@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("seed", np.arange(10))
def test_lower(d, method, dtype, seed):
    np.random.seed(seed)
    b = -np.random.uniform(0, 1, d)
    configs = {
        "proximal_newton": {
            "max_iters": 1000,
            "tol": 1e-14,
            "pinball_tol": 1e-14,
        },
    }[method]
    cnstr = constraint.lower(b, method=method, configs=configs, dtype=dtype)

    class Lower:
        def __init__(self):
            self.dual_size = d
            self.primal_size = d
        def evaluate(self, x):
            return b-x
        def gradient(self, x, mu, out):
            out[...] = -mu
        def duals(self):
            return self.dual_size
        def primals(self):
            return self.primal_size

    cnstr_exp = Lower()
    run_test(cnstr, cnstr_exp, dtype, seed)


@pytest.mark.parametrize("d", [1, 2, 5, 10, 20, 50])
@pytest.mark.parametrize("upper", [1, 1e-14, 0])
@pytest.mark.parametrize("method", ["proximal_newton", "admm"])
@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("seed", np.arange(5))
def test_one_sided(d, upper, method, dtype, seed):
    np.random.seed(seed)
    D = 2 * np.random.binomial(1, 0.5, d) - 1
    b = np.random.uniform(0, upper, d)
    configs = {
        "proximal_newton": {
            "max_iters": 1000,
            "tol": 1e-14,
            "pinball_tol": 1e-14,
        },
        "admm": {
            "max_iters": 4000,
            "tol_abs": 1e-8,
            "tol_rel": 1e-8,
            "rho": 0.2,
        },
    }[method]
    cnstr = constraint.one_sided(D, b, method=method, configs=configs, dtype=dtype)

    class OneSided:
        def __init__(self):
            self.dual_size = d
            self.primal_size = d
        def evaluate(self, x):
            return D * x - b
        def gradient(self, x, mu, out):
            out[...] = D * mu
        def duals(self):
            return self.dual_size
        def primals(self):
            return self.primal_size

    cnstr_exp = OneSided()
    run_test(cnstr, cnstr_exp, dtype, seed)


@pytest.mark.parametrize("d", [1, 2, 5, 10, 20, 50])
@pytest.mark.parametrize("method", ["proximal_newton"])
@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("seed", np.arange(5))
def test_upper(d, method, dtype, seed):
    np.random.seed(seed)
    b = np.random.uniform(0, 1, d)
    configs = {
        "proximal_newton": {
            "max_iters": 1000,
            "tol": 1e-14,
            "pinball_tol": 1e-14,
        },
    }[method]
    cnstr = constraint.upper(b, method=method, configs=configs, dtype=dtype)

    class Upper:
        def __init__(self):
            self.dual_size = d
            self.primal_size = d
        def evaluate(self, x):
            return x - b
        def gradient(self, x, mu, out):
            out[...] = mu
        def duals(self):
            return self.dual_size
        def primals(self):
            return self.primal_size

    cnstr_exp = Upper()
    run_test(cnstr, cnstr_exp, dtype, seed)