import adelie.matrix as mod
import numpy as np
import scipy


def test_naive_dense():
    def _test(n, p, dtype, order, seed=0):
        np.random.seed(seed)
        X = np.random.normal(0, 1, (n, p))

        for dtype in dtypes:
            atol = 1e-6 if dtype == np.float32 else 1e-14

            X = np.array(X, dtype=dtype, order=order)
            cX = mod.naive_dense(X, n_threads=1)
            w = np.random.uniform(1, 2, n).astype(dtype)
            w = w / np.sum(w)

            # test cmul
            v = np.random.normal(0, 1, n).astype(dtype)
            out = cX.cmul(p//2, v)
            assert np.allclose(v @ X[:, p//2], out, atol=atol)

            # test ctmul
            v = np.random.normal(0, 1)
            out = np.empty(n, dtype=dtype)
            cX.ctmul(p//2, v, w, out)
            assert np.allclose(v * (w * X[:, p//2]), out, atol=atol)

            # test bmul
            v = np.random.normal(0, 1, n).astype(dtype)
            out = np.empty(p // 2, dtype=dtype)
            cX.bmul(0, p // 2, v, out)
            assert np.allclose(v.T @ X[:, :p//2], out, atol=atol)

            # test btmul
            v = np.random.normal(0, 1, p//2).astype(dtype)
            out = np.empty(n, dtype=dtype)
            cX.btmul(0, p // 2, v, w, out)
            assert np.allclose(v.T @ (w[:, None] * X[:, :p//2]).T, out, atol=atol)

            # test mul
            v = np.random.normal(0, 1, n).astype(dtype)
            out = np.empty(p, dtype=dtype)
            cX.mul(v, out)
            assert np.allclose(v.T @ X, out, atol=atol)

            # test sp_btmul
            v = np.random.normal(0, 1, (2, p // 2)).astype(dtype)
            v[:, :p//4] = 0
            expected = v @ (w[:, None] * X[:, :p//2]).T
            v = scipy.sparse.csr_matrix(v)
            out = np.empty((2, n), dtype=dtype)
            cX.sp_btmul(0, p // 2, v, w, out)
            assert np.allclose(expected, out, atol=atol)

            # test to_dense
            out = np.empty((n, p // 2), dtype=dtype, order="F")
            cX.to_dense(0, p // 2, out)
            assert np.allclose(X[:, :p//2], out)

            # test means
            X_means = np.empty(p, dtype=dtype)
            cX.means(w, X_means)
            assert np.allclose(np.sum(w[:, None] * X, axis=0), X_means)

            assert cX.rows() == n
            assert cX.cols() == p

    dtypes = [np.float32, np.float64]
    orders = ["C", "F"]
    for dtype in dtypes:
        for order in orders:
            _test(2, 2, dtype, order)
            _test(100, 20, dtype, order)
            _test(20, 100, dtype, order)


def test_cov_dense():
    def _test(n, p, dtype, order, seed=0):
        np.random.seed(seed)
        X = np.random.normal(0, 1, (n, p))
        A = X.T @ X / n

        for dtype in dtypes:
            atol = 1e-6 if dtype == np.float32 else 1e-14

            A = np.array(A, dtype=dtype, order=order)
            cA = mod.cov_dense(A, n_threads=4)

            # test bmul
            v = np.random.normal(0, 1, p // 2).astype(dtype)
            out = np.empty(p, dtype=dtype)
            cA.bmul(0, 0, p // 2, p, v, out)
            assert np.allclose(v.T @ A[:p//2, :], out, atol=atol)

            # test to_dense
            out = np.empty((p // 2, p // 2), dtype=dtype, order="F")
            cA.to_dense(0, 0, p//2, p//2, out)
            assert np.allclose(A[:p//2, :p//2], out, atol=atol)

            assert cA.rows() == p
            assert cA.cols() == p

    dtypes = [np.float32, np.float64]
    orders = ["C", "F"]
    for dtype in dtypes:
        for order in orders:
            _test(2, 2, dtype, order)
            _test(100, 20, dtype, order)
            _test(20, 100, dtype, order)


def test_cov_lazy():
    def _test(n, p, dtype, order, seed=0):
        np.random.seed(seed)
        X = np.random.normal(0, 1, (n, p))
        X /= np.sqrt(n)
        A = X.T @ X

        for dtype in dtypes:
            atol = 1e-6 if dtype == np.float32 else 1e-7

            X = np.array(X, dtype=dtype, order=order)
            A = np.array(A, dtype=dtype, order=order)
            cA = mod.cov_lazy(X, n_threads=4)

            # test bmul
            v = np.random.normal(0, 1, p // 2).astype(dtype)
            out = np.empty(p, dtype=dtype)
            cA.bmul(0, 0, p // 2, p, v, out)
            assert np.allclose(v.T @ A[:p//2, :], out, atol=atol)

            # check that cache occured properly
            if p > 2:
                v = np.random.normal(0, 1, p // 2 - 1).astype(dtype)
                out = np.empty(p // 2, dtype=dtype)
                cA.bmul(1, 0, p // 2-1, p // 2, v, out)
                assert np.allclose(v.T @ A[1:p//2, :p//2], out, atol=atol)

            # test to_dense
            out = np.empty((p // 2, p // 2), dtype=dtype, order="F")
            cA.to_dense(0, 0, p//2, p//2, out)
            assert np.allclose(A[:p//2, :p//2], out, atol=atol)

            assert cA.rows() == p
            assert cA.cols() == p

    dtypes = [np.float32, np.float64]
    orders = ["C", "F"]
    for dtype in dtypes:
        for order in orders:
            _test(2, 2, dtype, order)
            _test(100, 20, dtype, order)
            _test(20, 100, dtype, order)
