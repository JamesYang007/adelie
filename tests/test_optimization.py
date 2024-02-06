import adelie.optimization as opt
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
