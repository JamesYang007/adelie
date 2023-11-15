import adelie as ad
import adelie.matrix as mod
import numpy as np
import scipy
import os


def run_naive(
    X, 
    cX,
    dtype,
):
    n, p = X.shape

    atol = 1e-5 if dtype == np.float32 else 1e-14

    w = np.random.uniform(1, 2, n).astype(dtype)
    w = w / np.sum(w)

    # test cmul
    v = np.random.normal(0, 1, n).astype(dtype)
    for i in range(p):
        out = cX.cmul(i, v)
        expected = v @ X[:, i]
        assert np.allclose(expected, out, atol=atol)

    # test ctmul
    v = np.random.normal(0, 1)
    out = np.empty(n, dtype=dtype)
    for i in range(p):
        cX.ctmul(i, v, w, out)
        expected = v * (w * X[:, i])
        assert np.allclose(expected, out, atol=atol)

    # test bmul
    v = np.random.normal(0, 1, n).astype(dtype)
    for i in range(1, p+1):
        out = np.empty(i, dtype=dtype)
        cX.bmul(0, i, v, out)
        expected = v.T @ X[:, :i]
        assert np.allclose(expected, out, atol=atol)
    q = min(10, p)
    out = np.empty(q, dtype=dtype)
    for i in range(p-q+1):
        cX.bmul(i, q, v, out)
        expected = v.T @ X[:, i:i+q]
        assert np.allclose(expected, out, atol=atol)

    # test btmul
    out = np.empty(n, dtype=dtype)
    for i in range(1, p+1):
        v = np.random.normal(0, 1, i).astype(dtype)
        cX.btmul(0, i, v, w, out)
        expected = v.T @ (w[:, None] * X[:, :i]).T
        assert np.allclose(expected, out, atol=atol)
    q = min(10, p)
    v = np.random.normal(0, 1, q).astype(dtype)
    for i in range(p-q+1):
        cX.btmul(i, q, v, w, out)
        expected = v.T @ (w[:, None] * X[:, i:i+q]).T
        assert np.allclose(expected, out, atol=atol)

    # test mul
    v = np.random.normal(0, 1, n).astype(dtype)
    out = np.empty(p, dtype=dtype)
    cX.mul(v, out)
    expected = v.T @ X
    assert np.allclose(expected, out, atol=atol)

    # test cov
    q = min(5, p)
    sqrt_weights = np.sqrt(w)
    buffer = np.empty((n, q), dtype=dtype, order="F")
    out = np.empty((q, q), dtype=dtype, order="F")
    for i in range(p-q+1):
        cX.cov(i, q, sqrt_weights, out, buffer)
        expected = X[:, i:i+q].T @ (w[:, None] * X[:, i:i+q])
        assert np.allclose(expected, out, atol=atol)

    assert cX.rows() == n
    assert cX.cols() == p

    # test sp_btmul
    out = np.empty((2, n), dtype=dtype)
    v = np.random.normal(0, 1, (2, p)).astype(dtype)
    v[:, :p//2] = 0
    expected = v @ (w[:, None] * X).T
    v = scipy.sparse.csr_matrix(v)
    cX.sp_btmul(v, w, out)
    assert np.allclose(expected, out, atol=atol)

    # test means
    X_means = np.empty(p, dtype=dtype)
    cX.means(w, X_means)
    expected = np.sum(w[:, None] * X, axis=0)
    assert np.allclose(expected, X_means)


def run_cov(
    A,
    cA,
    dtype,
):
    p = A.shape[0]

    atol = 1e-6 if dtype == np.float32 else 1e-14

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


def test_naive_dense():
    def _test(n, p, dtype, order, seed=0):
        np.random.seed(seed)
        X = np.random.normal(0, 1, (n, p))
        X = np.array(X, dtype=dtype, order=order)
        cX = mod.dense(X, method="naive", n_threads=4)
        run_naive(X, cX, dtype)

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
        A = np.array(A, dtype=dtype, order=order)
        cA = mod.dense(A, method="cov", n_threads=4)
        run_cov(A, cA, dtype)

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
        X = np.array(X, dtype=dtype, order=order)
        A = np.array(A, dtype=dtype, order=order)
        cA = mod.cov_lazy(X, n_threads=4)
        run_cov(A, cA, dtype)

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
            n_threads=15,
        )
        for f in filenames:
            os.remove(f)

        X = np.concatenate([data["X"] for data in datas], axis=-1, dtype=np.int8)
        run_naive(X, cX, dtype)


    dtypes = [np.float64, np.float32]
    for dtype in dtypes:
        _test(10, 20, 3, dtype)
        _test(1, 13, 3, dtype)
        _test(144, 1, 3, dtype)


def test_snp_phased_ancestry():
    # TODO: this was copied from test_io.py
    def create_dense(calldata, ancestries, A):
        n, s = calldata.shape[0], calldata.shape[1] // 2
        dense = np.zeros((n, s * A), dtype=np.int8)
        base_indices = A * np.arange(n * s, dtype=int)[None]
        dense.ravel()[
            base_indices +
            ancestries.reshape(n, s, 2)[:,:,0].ravel()
        ] += calldata.reshape(n, s, 2)[:,:,0].ravel()
        dense.ravel()[
            base_indices +
            ancestries.reshape(n, s, 2)[:,:,1].ravel()
        ] += calldata.reshape(n, s, 2)[:,:,1].ravel()
        return dense

    def _test(n, s, A, n_files, dtype, seed=0):
        np.random.seed(seed)
        datas = [
            ad.data.create_snp_phased_ancestry(n, s, A, seed=seed+i)
            for i in range(n_files)
        ]
        filenames = [
            f"/tmp/test_snp_phased_ancestry_{i}.snpdat"
            for i in range(n_files)
        ]
        for i in range(n_files):
            handler = ad.io.snp_phased_ancestry(filenames[i])
            handler.write(datas[i]["X"], datas[i]["ancestries"], A)
        cX = mod.snp_phased_ancestry(
            filenames=filenames,
            dtype=dtype,
            n_threads=8,
        )
        for f in filenames:
            os.remove(f)

        X = np.concatenate([
            create_dense(data["X"], data["ancestries"], A) 
            for data in datas
        ], axis=-1, dtype=np.int8)
        run_naive(X, cX, dtype)


    dtypes = [np.float64, np.float32]
    for dtype in dtypes:
        _test(10, 20, 4, 3, dtype)
        _test(1, 13, 3, 3, dtype)
        _test(144, 1, 2, 3, dtype)