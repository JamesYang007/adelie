import adelie.optimization as opt
import cvxpy as cp
import numpy as np
import pytest


@pytest.mark.parametrize("n, seed", [
    [100, 0],
    [1000, 24],
    [234, 123],
    [2, 239],
])
def test_search_pivot(n, seed):
    np.random.seed(seed)
    x = np.sort(np.random.normal(0, 1, n))
    p = x[np.random.choice(n, 1)]
    t = (p - x) * (x <= p)
    b0 = np.random.normal(0, 1)
    b1 = np.random.normal(0, 1)
    eps = np.random.normal(0, 0.1)
    y = b0 + b1 * t + eps

    _, mses = opt.search_pivot(x, y)

    mses_exp = np.empty(n)
    mses_exp[0] = np.inf
    for j in range(1, n):
        t = (x[j] - x) * (x <= x[j])
        tc = t - np.mean(t)
        yc = y - np.mean(y)
        b1_hat = (yc @ tc) / (tc @ tc)
        mses_exp[j] = -b1_hat ** 2 * (tc @ tc)

    assert np.allclose(mses, mses_exp)


@pytest.mark.parametrize("n", [3, 5, 10, 20])
@pytest.mark.parametrize("seed", np.arange(20))
def test_symmetric_penalty(n, seed):
    def compute_y(ts, x, alpha):
        return np.sum(
            0.5 * (1-alpha) * (x[:, None] - ts[None]) ** 2 + alpha * np.abs(x[:, None] - ts[None]),
            axis=0
        )

    ts = np.linspace(-2, 2, 10000)

    np.random.seed(seed)
    x = np.sort(np.random.uniform(-1, 1, n))
    alpha = np.random.uniform(0, 1)
    t_star = opt.symmetric_penalty(x, alpha)
    ys = compute_y(ts, x, alpha)
    y_star = compute_y(np.array(t_star), x, alpha)
    assert np.all(ys >= y_star)


@pytest.mark.parametrize("d", [3, 5, 10, 20])
@pytest.mark.parametrize("seed", np.arange(20))
def test_nnqp_full(d, seed):
    def run_cvxpy(quad, linear):
        d = quad.shape[0]
        x = cp.Variable(d)
        expr = 0.5 * cp.quad_form(x, quad) - linear @ x
        constraints = [x >= 0]
        prob = cp.Problem(cp.Minimize(expr), constraints)
        prob.solve()
        return x.value

    def objective(x, quad, linear):
        return 0.5 * (x.T @ quad @ x) - linear @ x

    np.random.seed(seed)
    n = 10
    X = np.random.normal(0, 1, (n, d))
    y = np.random.normal(0, 1, n)
    X /= np.sqrt(n)
    y /= np.sqrt(n)
    quad = np.asfortranarray(X.T @ X)
    linear = X.T @ y

    x_cvxpy = run_cvxpy(quad, linear)

    x = np.zeros(d)
    grad = linear.copy()
    state = opt.StateNNQPFull(quad, 1000000, 1e-24, 0, x, grad)
    state.solve()

    # test loss against truth
    loss_actual = objective(x, quad, linear)
    loss_expected = objective(x_cvxpy, quad, linear)
    assert np.allclose(loss_actual, loss_expected)

    # test gradient
    grad_actual = state.grad
    grad_expected = linear - quad @ x
    assert np.allclose(grad_actual, grad_expected)


@pytest.mark.parametrize("d", [3, 5, 10, 20])
@pytest.mark.parametrize("seed", np.arange(20))
def test_nnls(d, seed):
    def run_cvxpy(X, y):
        d = X.shape[1]
        x = cp.Variable(d)
        expr = 0.5 * cp.sum_squares(y - X @ x)
        constraints = [x >= 0]
        prob = cp.Problem(cp.Minimize(expr), constraints)
        prob.solve()
        return x.value

    def objective(x, X, y):
        return 0.5 * np.sum((y - X @ x) ** 2)

    np.random.seed(seed)
    n = 10
    X = np.random.normal(0, 1, (n, d))
    X = np.asfortranarray(X)
    y = np.random.normal(0, 1, n)
    X /= np.sqrt(n)
    y /= np.sqrt(n)

    x_cvxpy = run_cvxpy(X, y)

    x = np.zeros(d)
    X_vars = np.sum(X ** 2, axis=0)
    resid = y.copy()
    loss = 0.5 * np.sum(resid ** 2)
    state = opt.StateNNLS(X, X_vars, 1000000, 1e-24, 0, x, resid, loss)
    state.solve()

    # test loss against truth
    loss_actual = objective(x, X, y)
    loss_expected = objective(x_cvxpy, X, y)
    assert np.allclose(loss_actual, loss_expected)
    loss_actual = state.loss
    assert np.allclose(loss_actual, loss_expected)

    # test residual
    resid_actual = resid
    resid_expected = y - X @ x
    assert np.allclose(resid_actual, resid_expected)