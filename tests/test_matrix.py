import adelie as ad
import adelie.matrix as mod
import numpy as np
import scipy
import os


def test_naive_dense():
    def _test(n, p, dtype, order, seed=0):
        np.random.seed(seed)
        X = np.random.normal(0, 1, (n, p))

        atol = 1e-6 if dtype == np.float32 else 1e-14

        X = np.array(X, dtype=dtype, order=order)
        cX = mod.dense(X, method="naive", n_threads=1)
        w = np.random.uniform(1, 2, n).astype(dtype)
        w = w / np.sum(w)

        # test cmul
        v = np.random.normal(0, 1, n).astype(dtype)
        for i in range(p):
            out = cX.cmul(i, v)
            assert np.allclose(v @ X[:, i], out, atol=atol)

        # test ctmul
        v = np.random.normal(0, 1)
        out = np.empty(n, dtype=dtype)
        for i in range(p):
            cX.ctmul(i, v, w, out)
            assert np.allclose(v * (w * X[:, i]), out, atol=atol)

        # test bmul
        v = np.random.normal(0, 1, n).astype(dtype)
        for i in range(1, p+1):
            out = np.empty(i, dtype=dtype)
            cX.bmul(0, i, v, out)
            assert np.allclose(v.T @ X[:, :i], out, atol=atol)

        # test btmul
        out = np.empty(n, dtype=dtype)
        for i in range(1, p+1):
            v = np.random.normal(0, 1, i).astype(dtype)
            cX.btmul(0, i, v, w, out)
            assert np.allclose(v.T @ (w[:, None] * X[:, :i]).T, out, atol=atol)

        # test mul
        v = np.random.normal(0, 1, n).astype(dtype)
        out = np.empty(p, dtype=dtype)
        cX.mul(v, out)
        assert np.allclose(v.T @ X, out, atol=atol)

        # test sp_btmul
        out = np.empty((2, n), dtype=dtype)
        for i in range(1, p+1):
            v = np.random.normal(0, 1, (2, i)).astype(dtype)
            v[:, :i//2] = 0
            expected = v @ (w[:, None] * X[:, :i]).T
            v = scipy.sparse.csr_matrix(v)
            cX.sp_btmul(0, i, v, w, out)
            assert np.allclose(expected, out, atol=atol)

        # test to_dense
        for i in range(1, p+1):
            out = np.empty((n, i), dtype=dtype, order="F")
            cX.to_dense(0, i, out)
            assert np.allclose(X[:, :i], out)

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

        atol = 1e-6 if dtype == np.float32 else 1e-14

        A = np.array(A, dtype=dtype, order=order)
        cA = mod.dense(A, method="cov", n_threads=4)

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


def test_snp_unphased():
    def _test(n, p, n_files, dtype, seed=0):
        np.random.seed(seed)
        datas = [
            ad.data.create_snp_unphased(n, p, seed=seed+i)
            for i in range(n_files)
        ]
        filenames = [
            f"/tmp/test_snp_unphased_{i}.snpdat"
            for i in range(n_files)
        ]
        for i in range(n_files):
            handler = ad.io.snp_unphased(filenames[i])
            handler.write(datas[i]["X"])
        cX = mod.snp_unphased(
            filenames=filenames,
            dtype=dtype,
        )
        for f in filenames:
            os.remove(f)

        X = np.concatenate([data["X"] for data in datas], axis=-1, dtype=np.int8)

        atol = 1e-6 if dtype == np.float32 else 1e-14

        # total number of features
        p_total = p * n_files

        # test cmul
        v = np.random.normal(0, 1, n).astype(dtype)
        for idx in range(p_total):
            actual = cX.cmul(idx, v)
            expected = X[:, idx] @ v
            assert np.allclose(actual, expected, atol=atol) 

        # test ctmul
        v = np.random.normal(0, 1)
        w = np.random.uniform(0, 1, n).astype(dtype)
        actual = np.empty(n, dtype=dtype)
        for idx in range(p_total):
            cX.ctmul(idx, v, w, actual)
            expected = v * w * X[:, idx]
            assert np.allclose(actual, expected, atol=atol)

        # test bmul
        v = np.random.normal(0, 1, n).astype(dtype)
        for i in range(1, p+1):
            actual = np.empty(i, dtype=dtype)
            cX.bmul(0, i, v, actual)
            expected = v.T @ X[:, :i]
            assert np.allclose(actual, expected, atol=atol)

        # test btmul
        actual = np.empty(n, dtype=dtype)
        for i in range(1, p+1):
            v = np.random.normal(0, 1, i).astype(dtype)
            cX.btmul(0, i, v, w, actual)
            expected = (v.T @ X[:, :i].T) * w[None]
            assert np.allclose(actual, expected, atol=atol)

        # test sp_btmul
        actual = np.empty((2, n), dtype=dtype)
        for i in range(1, p+1):
            v = np.random.normal(0, 1, (2, i)).astype(dtype)
            v = scipy.sparse.csr_matrix(v)
            cX.sp_btmul(0, i, v, w, actual)
            expected = (v @ X[:, :i].T) * w[None]
            assert np.allclose(actual, expected, atol=atol)

        # test to_dense
        for i in range(1, p+1):
            actual = np.empty((n, i), dtype=dtype, order="F")
            cX.to_dense(0, i, actual)
            expected = X[:, :i]
            assert np.allclose(actual, expected, atol=atol)

        # test means
        actual = np.empty(p_total, dtype=dtype)
        cX.means(w, actual)
        expected = np.concatenate([
            np.sum(data["X"] * w[:, None], axis=0)
            for data in datas
        ], dtype=dtype)
        assert np.allclose(actual, expected, atol=atol)

        assert cX.rows() == n
        assert cX.cols() == p_total

    dtypes = [np.float64, np.float32]
    for dtype in dtypes:
        _test(10, 20, 3, dtype)
        _test(1, 13, 3, dtype)
        _test(144, 1, 3, dtype)