import adelie.matrix as mod
import numpy as np


def test_dense():
    def _test(n, p, dtype, order, seed=0):
        np.random.seed(seed)
        X = np.random.normal(0, 1, (n, p))

        for dtype in dtypes:
            X = np.array(X, dtype=dtype, order=order)
            wrap = mod.dense(X, n_threads=4)
            cX = wrap._core_mat

            # test cmul
            v = np.random.normal(0, 1, n).astype(dtype)
            out = cX.cmul(p//2, v)
            assert np.allclose(v @ X[:, p//2], out)

            # test ctmul
            v = np.random.normal(0, 1)
            out = np.empty(n, dtype=dtype)
            cX.ctmul(p//2, v, out)
            assert np.allclose(v * X[:, p//2], out)

            # test bmul
            v = np.random.normal(0, 1, n).astype(dtype)
            out = np.empty(p // 2, dtype=dtype)
            cX.bmul(0, 0, n, p // 2, v, out)
            assert np.allclose(v.T @ X[:, :p//2], out)

            # test btmul
            v = np.random.normal(0, 1, p//2).astype(dtype)
            out = np.empty(n, dtype=dtype)
            cX.btmul(0, 0, n, p // 2, v, out)
            assert np.allclose(v.T @ X[:, :p//2].T, out)

            assert cX.rows() == n
            assert cX.cols() == p

    dtypes = [np.float32, np.float64]
    orders = ["C", "F"]
    for dtype in dtypes:
        for order in orders:
            _test(100, 20, dtype, order)
