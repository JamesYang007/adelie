import adelie.matrix as matrix
import adelie.optimization as opt
import cvxpy as cp
import numpy as np
import pytest


@pytest.mark.parametrize("d", [3, 5, 10, 20])
@pytest.mark.parametrize("seed", np.arange(20))
def test_pinball_full(d, seed):
    def run_cvxpy(quad, linear, penalty_pos, penalty_neg):
        d = quad.shape[0]
        x = cp.Variable(d)
        expr = 0.5 * cp.quad_form(x, quad) - linear @ x + penalty_pos @ cp.pos(x) + penalty_neg @ cp.neg(x)
        constraints = []
        prob = cp.Problem(cp.Minimize(expr), constraints)
        prob.solve()
        return x.value

    def objective(x, quad, linear, penalty_pos, penalty_neg):
        return 0.5 * (x.T @ quad @ x) - linear @ x + penalty_pos @ np.maximum(x, 0) + penalty_neg @ np.maximum(-x, 0)

    np.random.seed(seed)
    n = 10
    X = np.random.normal(0, 1, (n, d))
    y = np.random.normal(0, 1, n)
    penalty_pos = np.random.uniform(0, 1, d)
    penalty_neg = np.random.uniform(0, 1, d)
    X /= np.sqrt(n)
    y /= np.sqrt(n)
    quad = np.asfortranarray(X.T @ X)
    linear = X.T @ y

    x_cvxpy = run_cvxpy(quad, linear, penalty_pos, penalty_neg)

    x = np.zeros(d)
    grad = linear.copy()
    state = opt.StatePinballFull(quad, penalty_neg, penalty_pos, d, 100000, 1e-24, x, grad)
    state.solve()

    # test loss against truth
    loss_actual = objective(x, quad, linear, penalty_pos, penalty_neg)
    loss_expected = objective(x_cvxpy, quad, linear, penalty_pos, penalty_neg)
    assert np.all(loss_actual <= loss_expected * (1 + np.sign(loss_expected) * 1e-7))

    # test gradient
    grad_actual = state.grad
    grad_expected = linear - quad @ x
    assert np.allclose(grad_actual, grad_expected, atol=1e-7)


@pytest.mark.parametrize("d", [3, 5, 10, 20])
@pytest.mark.parametrize("seed", np.arange(20))
def test_lasso_full(d, seed):
    def run_cvxpy(quad, linear, penalty):
        d = quad.shape[0]
        x = cp.Variable(d)
        expr = 0.5 * cp.quad_form(x, quad) - linear @ x + penalty @ cp.abs(x)
        constraints = []
        prob = cp.Problem(cp.Minimize(expr), constraints)
        prob.solve()
        return x.value

    def objective(x, quad, linear, penalty):
        return 0.5 * (x.T @ quad @ x) - linear @ x + penalty @ np.abs(x)

    np.random.seed(seed)
    n = 10
    X = np.random.normal(0, 1, (n, d))
    y = np.random.normal(0, 1, n)
    penalty = np.random.uniform(0, 1, d)
    X /= np.sqrt(n)
    y /= np.sqrt(n)
    quad = np.asfortranarray(X.T @ X)
    linear = X.T @ y

    x_cvxpy = run_cvxpy(quad, linear, penalty)

    x = np.zeros(d)
    grad = linear.copy()
    state = opt.StateLassoFull(quad, penalty, 1000000, 1e-24, x, grad)
    state.solve()

    # test loss against truth
    loss_actual = objective(x, quad, linear, penalty)
    loss_expected = objective(x_cvxpy, quad, linear, penalty)
    assert np.allclose(loss_actual, loss_expected)

    # test gradient
    grad_actual = state.grad
    grad_expected = linear - quad @ x
    assert np.allclose(grad_actual, grad_expected)


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
    state = opt.StateNNQPFull(quad, 1000000, 1e-24, x, grad)
    state.solve()

    # test loss against truth
    loss_actual = objective(x, quad, linear)
    loss_expected = objective(x_cvxpy, quad, linear)
    assert np.allclose(loss_actual, loss_expected)

    # test gradient
    grad_actual = state.grad
    grad_expected = linear - quad @ x
    assert np.allclose(grad_actual, grad_expected)


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
