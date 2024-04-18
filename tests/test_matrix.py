import adelie as ad
import adelie.matrix as mod
import numpy as np
import scipy
import os
import warnings


# ==========================================================================================
# TEST cov
# ==========================================================================================


def run_cov(
    A,
    cA,
    dtype,
):
    p = A.shape[0]

    atol = 1e-6 if dtype == np.float32 else 1e-14

    v = np.random.normal(0, 1, p)
    nnzs = np.random.binomial(1, 0.7, p).astype(bool)
    v[~nnzs] = 0
    indices = np.arange(p)[nnzs]
    values = v[indices]

    # test bmul
    for i in range(1, p+1):
        subset = np.sort(np.random.choice(p, i, replace=False))
        out = np.empty(i, dtype=dtype)
        cA.bmul(subset, indices, values, out)
        expected = v.T @ A[:, subset]
        assert np.allclose(expected, out, atol=atol)

    # test mul
    cA.mul(indices, values, out)
    expected = v.T @ A
    assert np.allclose(expected, out, atol=atol)

    # test to_dense
    q = min(10, p)
    out = np.empty((q, q), dtype=dtype, order="F")
    for i in range(p-q+1):
        cA.to_dense(i, q, out)
        expected = A[i:i+q, i:i+q]
        assert np.allclose(expected, out, atol=atol)

    assert cA.rows() == p
    assert cA.cols() == p


def test_cov_block_diag():
    def _test(n, ps, dtype, seed=0):
        np.random.seed(seed)
        Xs = [np.random.normal(0, 1, (n, p)).astype(dtype) for p in ps]
        As = [X.T @ X / n for X in Xs]
        p = np.sum(ps)
        A = np.zeros((p, p), dtype=dtype)
        pos = 0
        for i, pi in enumerate(ps):
            A[pos:pos+pi, pos:pos+pi] = As[i]
            pos += pi
        cA = mod.block_diag(As, n_threads=4)
        run_cov(A, cA, dtype)

    np.random.seed(69)
    dtypes = [np.float32, np.float64]
    for dtype in dtypes:
        ps = np.random.choice(10, size=4, replace=True)
        _test(2, ps, dtype)
        _test(100, ps, dtype)
        _test(20, ps, dtype)


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


def test_cov_lazy_cov():
    def _test(n, p, dtype, order, seed=0):
        np.random.seed(seed)
        X = np.random.normal(0, 1, (n, p))
        X /= np.sqrt(n)
        X = np.array(X, dtype=dtype, order=order)
        A = X.T @ X
        A = np.array(A, dtype=dtype, order=order)
        cA = mod.lazy_cov(X, n_threads=4)
        run_cov(A, cA, dtype)

    dtypes = [np.float32, np.float64]
    orders = ["C", "F"]
    for dtype in dtypes:
        for order in orders:
            _test(2, 2, dtype, order)
            _test(100, 20, dtype, order)
            _test(20, 100, dtype, order)

def test_cov_sparse():
    def _test(n, p, dtype, order, seed=0):
        np.random.seed(seed)
        X = np.random.normal(0, 1, (n, p))
        X /= np.sqrt(n)
        X = np.array(X, dtype=dtype, order=order)
        A = X.T @ X
        subset = np.sort(np.random.choice(p, p // 2, replace=False))
        A[subset, :] = 0
        A[:, subset] = 0
        A_sp = {
            "C": scipy.sparse.csr_matrix,
            "F": scipy.sparse.csc_matrix,
        }[order](A)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cA = mod.sparse(A_sp, method="cov", n_threads=3)
        run_cov(A, cA, dtype)

    dtypes = [np.float32, np.float64]
    orders = ["C", "F"]
    for dtype in dtypes:
        for order in orders:
            _test(2, 2, dtype, order)
            _test(100, 20, dtype, order)
            _test(20, 100, dtype, order)


# ==========================================================================================
# TEST naive
# ==========================================================================================


def run_naive(
    X, 
    cX,
    dtype,
):
    n, p = X.shape

    atol = 1e-5 if dtype == np.float32 else 1e-14

    w = np.random.uniform(1, 2, n).astype(dtype)

    # test cmul
    v = np.random.normal(0, 1, n).astype(dtype)
    for i in range(p):
        out = cX.cmul(i, v, w)
        expected = (v * w) @ X[:, i]
        assert np.allclose(expected, out, atol=atol)

    # test ctmul
    v = np.random.normal(0, 1)
    out = np.zeros(n, dtype=dtype)
    for i in range(p):
        out_old = out.copy()
        cX.ctmul(i, v, out)
        expected = out_old + v * X[:, i]
        assert np.allclose(expected, out, atol=atol)

    # test bmul
    v = np.random.normal(0, 1, n).astype(dtype)
    for i in range(1, p+1):
        out = np.empty(i, dtype=dtype)
        cX.bmul(0, i, v, w, out)
        expected = (v * w).T @ X[:, :i]
        assert np.allclose(expected, out, atol=atol)
    q = min(10, p)
    out = np.empty(q, dtype=dtype)
    for i in range(p-q+1):
        cX.bmul(i, q, v, w, out)
        expected = (v * w).T @ X[:, i:i+q]
        assert np.allclose(expected, out, atol=atol)

    # test btmul
    out = np.zeros(n, dtype=dtype)
    for i in range(1, p+1):
        out_old = out.copy()
        v = np.random.normal(0, 1, i).astype(dtype)
        cX.btmul(0, i, v, out)
        expected = out_old + v.T @ X[:, :i].T
        assert np.allclose(expected, out, atol=atol)
    q = min(10, p)
    v = np.random.normal(0, 1, q).astype(dtype)
    for i in range(p-q+1):
        out_old = out.copy()
        cX.btmul(i, q, v, out)
        expected = out_old + v.T @ X[:, i:i+q].T
        assert np.allclose(expected, out, atol=atol)

    # test mul
    v = np.random.normal(0, 1, n).astype(dtype)
    out = np.empty(p, dtype=dtype)
    cX.mul(v, w, out)
    expected = (v * w).T @ X
    assert np.allclose(expected, out, atol=atol)

    # test cov
    q = min(5, p)
    sqrt_weights = np.sqrt(w)
    buffer = np.empty((n, q), dtype=dtype, order="F")
    out = np.empty((q, q), dtype=dtype, order="F")
    for i in range(p-q+1):
        try:
            cX.cov(i, q, sqrt_weights, out, buffer)
            expected = X[:, i:i+q].T @ (w[:, None] * X[:, i:i+q])
            assert np.allclose(expected, out, atol=atol)
        except RuntimeError as err:
            err_msg = str(err)
            if "MatrixNaiveCConcatenate::cov() only allows the block to be fully contained in one of the matrices in the list." in err_msg:
                pass
            else:
                raise err

    assert cX.rows() == n
    assert cX.cols() == p

    # test sp_btmul
    out = np.empty((2, n), dtype=dtype)
    v = np.random.normal(0, 1, (2, p)).astype(dtype)
    v[:, :p//2] = 0
    expected = v @ X.T
    v = scipy.sparse.csr_matrix(v)
    cX.sp_btmul(v, out)
    assert np.allclose(expected, out, atol=atol)


def test_naive_cconcatenate():
    def _test(n, ps, dtype, n_threads=7, seed=0):
        np.random.seed(seed)
        Xs = [
            np.random.normal(0, 1, (n, p))
            for p in ps
        ]
        X = np.concatenate(Xs, axis=1, dtype=dtype)
        cX = mod.concatenate(
            [mod.dense(_X.astype(dtype), method="naive", n_threads=n_threads) for _X in Xs], 
            axis=1,
            n_threads=n_threads, 
        )
        run_naive(X, cX, dtype)

    dtypes = [np.float32, np.float64]
    ps = [1, 7, 41, 13, 113]
    for dtype in dtypes:
        _test(20, ps, dtype)


def test_naive_rconcatenate():
    def _test(ns, p, dtype, n_threads=2, seed=0):
        np.random.seed(seed)
        Xs = [
            np.random.normal(0, 1, (n, p))
            for n in ns
        ]
        X = np.concatenate(Xs, axis=0, dtype=dtype)
        cX = mod.concatenate(
            [mod.dense(_X.astype(dtype), method="naive", n_threads=n_threads) for _X in Xs], 
            axis=0,
            n_threads=n_threads, 
        )
        run_naive(X, cX, dtype)

    dtypes = [np.float32, np.float64]
    ns = [1, 7, 41, 13, 113]
    for dtype in dtypes:
        _test(ns, 20, dtype)


def test_naive_dense():
    def _test(n, p, dtype, order, seed=0):
        np.random.seed(seed)
        X = np.random.normal(0, 1, (n, p))
        X = np.array(X, dtype=dtype, order=order)
        cX = mod.dense(X, method="naive", n_threads=15)
        run_naive(X, cX, dtype)

    dtypes = [np.float32, np.float64]
    orders = ["C", "F"]
    for dtype in dtypes:
        for order in orders:
            _test(2, 2, dtype, order)
            _test(100, 20, dtype, order)
            _test(20, 100, dtype, order)


def test_naive_kronecker_eye():
    def _test(n, p, K, dtype, order, seed=0):
        np.random.seed(seed)
        X = np.random.normal(0, 1, (n, p))
        X = np.array(X, dtype=dtype, order=order)
        cX = mod.kronecker_eye(
            mod.dense(X, method="naive"), K, n_threads=2
        )
        X = np.kron(X, np.eye(K))
        run_naive(X, cX, dtype)

    dtypes = [np.float32, np.float64]
    orders = ["C", "F"]
    for dtype in dtypes:
        for order in orders:
            _test(10, 1, 3, dtype, order)
            _test(100, 20, 2, dtype, order)
            _test(20, 100, 4, dtype, order)


def test_naive_kronecker_eye_dense():
    def _test(n, p, K, dtype, order, seed=0):
        np.random.seed(seed)
        X = np.random.normal(0, 1, (n, p))
        X = np.array(X, dtype=dtype, order=order)
        cX = mod.kronecker_eye(
            X, K, n_threads=7
        )
        X = np.kron(X, np.eye(K))
        run_naive(X, cX, dtype)

    dtypes = [np.float32, np.float64]
    orders = ["C", "F"]
    for dtype in dtypes:
        for order in orders:
            _test(10, 1, 3, dtype, order)
            _test(100, 20, 2, dtype, order)


def test_naive_snp_unphased():
    def _test(n, p, read_mode, dtype, seed=0):
        np.random.seed(seed)
        data = ad.data.snp_unphased(n, p, seed=seed)
        filename = "/tmp/test_snp_unphased.snpdat"
        handler = ad.io.snp_unphased(filename)
        handler.write(data["X"])
        cX = mod.snp_unphased(
            filename=filename,
            read_mode=read_mode,
            dtype=dtype,
            n_threads=7,
        )

        X = data["X"].astype(np.int8)
        run_naive(X, cX, dtype)
        os.remove(filename)

    read_modes = ["file", "mmap"]
    dtypes = [np.float64, np.float32]
    for read_mode in read_modes:
        for dtype in dtypes:
            _test(10, 20, read_mode, dtype)
            _test(1, 13, read_mode, dtype)
            _test(144, 1, read_mode, dtype)


def test_naive_snp_phased_ancestry():
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

    def _test(n, s, A, read_mode, dtype, seed=0):
        np.random.seed(seed)
        data = ad.data.snp_phased_ancestry(n, s, A, seed=seed)
        filename = "/tmp/test_snp_phased_ancestry.snpdat"
        handler = ad.io.snp_phased_ancestry(filename)
        handler.write(data["X"], data["ancestries"], A)
        cX = mod.snp_phased_ancestry(
            filename=filename,
            read_mode=read_mode,
            dtype=dtype,
            n_threads=7,
        )
        os.remove(filename)

        X = create_dense(data["X"], data["ancestries"], A) 
        run_naive(X, cX, dtype)


    read_modes = ["file", "mmap"]
    dtypes = [np.float64, np.float32]
    for read_mode in read_modes:
        for dtype in dtypes:
            _test(10, 20, 4, read_mode, dtype)
            _test(1, 13, 3, read_mode, dtype)
            _test(144, 1, 2, read_mode, dtype)


def test_naive_sparse():
    def _test(n, p, dtype, order, seed=0):
        np.random.seed(seed)
        X = np.random.normal(0, 1, (n, p)).astype(dtype)
        X.flat[np.random.binomial(1, 0.3, X.size)] = 0
        X_sp = {
            "C": scipy.sparse.csr_matrix,
            "F": scipy.sparse.csc_matrix,
        }[order](X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cX = mod.sparse(X_sp, method="naive", n_threads=3)
        run_naive(X, cX, dtype)

    dtypes = [np.float32, np.float64]
    orders = ["C", "F"]
    for dtype in dtypes:
        for order in orders:
            _test(2, 2, dtype, order)
            _test(100, 20, dtype, order)
            _test(20, 100, dtype, order)
