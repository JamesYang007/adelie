import adelie.optimization as opt
import cvxpy as cp
import numpy as np


def test_search_pivot():
    def test(
        n=100,
        seed=0,
    ):
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

    test(100, 0)
    test(1000, 24)
    test(234, 123)
    test(2, 239)


def test_symmetric_penalty():
    def compute_y(ts, x, alpha):
        return np.sum(
            0.5 * (1-alpha) * (x[:, None] - ts[None]) ** 2 + alpha * np.abs(x[:, None] - ts[None]),
            axis=0
        )

    ts = np.linspace(-2, 2, 10000)

    def test(
        n=10,
        seed=0,
    ):
        np.random.seed(seed)
        x = np.sort(np.random.uniform(-1, 1, n))
        alpha = np.random.uniform(0, 1)
        t_star = opt.symmetric_penalty(x, alpha)
        ys = compute_y(ts, x, alpha)
        y_star = compute_y(np.array(t_star), x, alpha)
        assert np.all(ys >= y_star)

    ns = [3, 5, 10, 20]
    seeds = np.arange(20)
    for n in ns:
        for seed in seeds:
            test(n, seed)


def test_nnls_cov_full():
    def run_cvxpy(X, y, l2):
        d = X.shape[1]
        x = cp.Variable(d)
        expr = 0.5 * cp.sum_squares(y - X @ x) + (0.5 * l2) * cp.sum_squares(x)
        constraints = [x >= 0]
        prob = cp.Problem(cp.Minimize(expr), constraints)
        prob.solve()
        return x.value

    def objective(x, X, y, l2):
        return 0.5 * np.sum((y - X @ x) ** 2) + (0.5 * l2) * np.sum(x ** 2)

    def test(d, seed):
        np.random.seed(seed)
        n = 10
        l2 = 0
        X = np.random.normal(0, 1, (n, d))
        y = np.random.normal(0, 1, n)
        X /= np.sqrt(n)
        y /= np.sqrt(n)

        x_cvxpy = run_cvxpy(X, y, l2)

        quad = X.T @ X
        linear = X.T @ y
        x0 = np.zeros(d)
        out = opt.nnls_cov_full(
            quad, linear, l2, 1000000, 1e-24, x0, 
        )
        x = out["x"]

        loss_actual = objective(x, X, y, l2)
        loss_expected = objective(x_cvxpy, X, y, l2)
        assert np.allclose(loss_actual, loss_expected)

    ds = [1, 5, 10, 20]
    seeds = np.arange(20)
    for d in ds:
        for seed in seeds:
            test(d, seed)