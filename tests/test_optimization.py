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