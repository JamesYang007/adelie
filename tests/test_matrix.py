import adelie.matrix as mod
import numpy as np


def test_dense():
    def _test(n, p, dtype, order, seed=0):
        np.random.seed(seed)
        X = np.random.normal(0, 1, (n, p))
        X = np.array(X, order=order)

        for dtype in dtypes:
            wrap = mod.dense(X, dtype=dtype)
            assert np.allclose(X[:n//2, :p//2], wrap._core_mat.block(0, 0, n // 2, p // 2))
            assert np.allclose(X[:, 0], wrap._core_mat.col(0))
            assert wrap._core_mat.rows() == n
            assert wrap._core_mat.cols() == p

    dtypes = [np.float32, np.float64]
    orders = ["C", "F"]
    for dtype in dtypes:
        for order in orders:
            _test(100, 20, dtype, order)
