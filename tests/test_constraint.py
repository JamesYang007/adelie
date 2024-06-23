from adelie import constraint
import numpy as np
import pytest


def run_test(
    cnstr,
    cnstr_exp,
    dtype,
    seed: int =0,
):
    atol = 1e-6 if dtype == np.float32 else 1e-7

    # test input/output sizes
    assert cnstr.duals() == cnstr_exp.duals()
    assert cnstr.dual_size == cnstr_exp.dual_size
    assert cnstr.primals() == cnstr_exp.primals()
    assert cnstr.primal_size == cnstr_exp.primal_size

    d, m = cnstr.primal_size, cnstr.dual_size

    # test gradient
    x = np.random.normal(0, 1, d).astype(dtype)
    mu = np.random.normal(0, 1, m).astype(dtype)
    actual = np.empty(d, dtype=dtype)
    cnstr.gradient(x, mu, actual)
    expected = np.empty(d, dtype=dtype)
    cnstr_exp.gradient(x, mu, expected)
    assert np.allclose(actual, expected)

    # generate data
    np.random.seed(seed)
    quad = np.random.uniform(0, 1, d).astype(dtype)
    linear = np.sqrt(quad) * np.random.normal(0, 1, d).astype(dtype)
    l1 = 0.5 * np.linalg.norm(linear)
    l2 = 0
    Q = np.random.normal(0, 1, (d, d))
    Q, _, _ = np.linalg.svd(Q)
    Q = np.asfortranarray(Q, dtype=dtype)

    # test solve
    x = np.zeros(d, dtype=dtype)
    mu = np.zeros(m, dtype=dtype)
    cnstr.solve(x, mu, quad, linear, l1, l2, Q)

    # KKT first-order condition
    Qx = Q @ x
    grad = np.empty(d, dtype=dtype)
    cnstr_exp.gradient(Qx, mu, grad)
    lagr_grad = (quad + l2) * x - linear + Q.T @ grad
    assert np.allclose(np.maximum(np.linalg.norm(lagr_grad)-l1, 0), 0, atol=atol)

    # KKT primal feasibility
    eval = cnstr_exp.evaluate(Qx)
    assert np.allclose(np.maximum(np.max(eval), 0), 0, atol=atol)

    # KKT dual feasibility
    assert np.allclose(np.minimum(np.min(mu), 0), 0, atol=atol)

    # KKT slack condition
    slackness = np.mean(mu * eval)
    assert np.allclose(slackness, 0, atol=atol)

    # test project 
    cnstr.project(Qx)
    eval = cnstr_exp.evaluate(Qx)
    assert np.allclose(np.maximum(np.max(eval), 0), 0, atol=atol)


@pytest.mark.parametrize("d", [1, 2, 5, 10, 20, 50])
@pytest.mark.parametrize("method", ["proximal-newton"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("seed", np.arange(10))
def test_lower(d, method, dtype, seed):
    np.random.seed(seed)
    b = np.random.uniform(0, 1, d)
    configs = {
        "proximal-newton": {
            "max_iters": 1000,
            "tol": 1e-14,
            "nnls_tol": 1e-14,
        },
        "admm": {
        },
    }[method]
    cnstr = constraint.lower(b, method=method, configs=configs, dtype=dtype)

    class Lower:
        def __init__(self):
            self.dual_size = d
            self.primal_size = d
        def evaluate(self, x):
            return -x-b
        def gradient(self, x, mu, out):
            out[...] = -mu
        def duals(self):
            return self.dual_size
        def primals(self):
            return self.primal_size

    cnstr_exp = Lower()
    run_test(cnstr, cnstr_exp, dtype, seed)


@pytest.mark.parametrize("d", [1, 2, 5, 10, 20, 50])
@pytest.mark.parametrize("method", ["proximal-newton", "admm"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("seed", np.arange(5))
def test_one_sided(d, method, dtype, seed):
    np.random.seed(seed)
    D = 2 * np.random.binomial(1, 0.5, d) - 1
    b = np.random.uniform(0, 1, d)
    configs = {
        "proximal-newton": {
            "max_iters": 1000,
            "tol": 1e-14,
            "nnls_tol": 1e-14,
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
@pytest.mark.parametrize("method", ["proximal-newton"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("seed", np.arange(5))
def test_upper(d, method, dtype, seed):
    np.random.seed(seed)
    b = np.random.uniform(0, 1, d)
    configs = {
        "proximal-newton": {
            "max_iters": 1000,
            "tol": 1e-14,
            "nnls_tol": 1e-14,
        },
        "admm": {
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