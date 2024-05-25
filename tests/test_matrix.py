import adelie as ad
import adelie.matrix as mod
import numpy as np
import scipy
import os
import pytest
import warnings

# Set to a value that will test all cases
# of n_threads being capped to 1 and using the passed-in value.
ad.configs.set_configs("min_bytes", 20)


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


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("ps", [[1, 5, 8, 2]])
@pytest.mark.parametrize("n", [2, 100, 20])
def test_cov_block_diag(n, ps, dtype, seed=0):
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


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("n, p", [
    [2, 2],
    [100, 20],
    [20, 100],
])
def test_cov_dense(n, p, dtype, order, seed=0):
    np.random.seed(seed)
    X = np.random.normal(0, 1, (n, p))
    A = X.T @ X / n
    A = np.array(A, dtype=dtype, order=order)
    cA = mod.dense(A, method="cov", n_threads=4)
    run_cov(A, cA, dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("n, p", [
    [2, 2],
    [100, 20],
    [20, 100],
])
def test_cov_lazy_cov(n, p, dtype, order, seed=0):
    np.random.seed(seed)
    X = np.random.normal(0, 1, (n, p))
    X /= np.sqrt(n)
    X = np.array(X, dtype=dtype, order=order)
    A = X.T @ X
    A = np.array(A, dtype=dtype, order=order)
    cA = mod.lazy_cov(X, n_threads=4)
    run_cov(A, cA, dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("n, p", [
    [2, 2],
    [100, 20],
    [20, 100],
])
def test_cov_sparse(n, p, dtype, order, seed=0):
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


# ==========================================================================================
# TEST naive
# ==========================================================================================


def run_naive(
    X, 
    cX,
    dtype,
):
    n, p = X.shape

    atol = 1e-4 if dtype == np.float32 else 1e-14

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
    v = np.random.normal(0, 1, p).astype(dtype)
    out = cX @ v
    expected = X @ v
    assert np.allclose(expected, out, atol=atol)

    # test mul
    v = np.random.normal(0, 1, n).astype(dtype)
    out = np.empty(p, dtype=dtype)
    cX.mul(v, w, out)
    expected = (v * w).T @ X
    assert np.allclose(expected, out, atol=atol)
    out = cX.T @ (v * w)
    assert np.allclose(expected, out, atol=atol)

    # test cov
    q = min(100, p)
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
            elif "MatrixNaiveCSubset::cov() is not implemented when " in err_msg:
                pass
            elif "MatrixNaiveInteractionDense::cov() not implemented for " in err_msg:
                pass
            elif "MatrixNaiveOneHotDense::cov() not implemented for " in err_msg:
                pass
            else:
                raise err

    # test cov (special cases)
    if isinstance(
        cX,
        (
            ad.adelie_core.matrix.MatrixNaiveInteractionDense64C,
            ad.adelie_core.matrix.MatrixNaiveInteractionDense64F,
            ad.adelie_core.matrix.MatrixNaiveInteractionDense32C,
            ad.adelie_core.matrix.MatrixNaiveInteractionDense32F,
            ad.adelie_core.matrix.MatrixNaiveOneHotDense64C,
            ad.adelie_core.matrix.MatrixNaiveOneHotDense64F,
            ad.adelie_core.matrix.MatrixNaiveOneHotDense32C,
            ad.adelie_core.matrix.MatrixNaiveOneHotDense32F,
        )
    ):
        groups = cX.groups
        group_sizes = cX.group_sizes
        for g, gs in zip(groups, group_sizes):
            out = np.empty((gs, gs), dtype=dtype, order="F")
            buffer = np.empty((n, gs), dtype=dtype, order="F")
            cX.cov(g, gs, sqrt_weights, out, buffer)
            expected = X[:, g:g+gs].T @ (w[:, None] * X[:, g:g+gs])
            assert np.allclose(expected, out, atol=atol)

    assert cX.rows() == n
    assert cX.cols() == p
    assert cX.shape == (n, p)

    # test sp_btmul
    out = np.empty((2, n), dtype=dtype)
    v = np.random.normal(0, 1, (2, p)).astype(dtype)
    v[:, :p//2] = 0
    expected = v @ X.T
    v = scipy.sparse.csr_matrix(v)
    cX.sp_btmul(v, out)
    assert np.allclose(expected, out, atol=atol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("ps", [[1, 7, 41, 13, 113]])
@pytest.mark.parametrize("n", [10, 20, 30])
def test_naive_cconcatenate(n, ps, dtype, n_threads=2, seed=0):
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


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("ns", [[1, 7, 41, 13, 113]])
@pytest.mark.parametrize("p", [10, 20, 30])
def test_naive_rconcatenate(ns, p, dtype, n_threads=2, seed=0):
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


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("n, p", [
    [2, 2],
    [100, 20],
    [20, 100],
])
def test_naive_dense(n, p, dtype, order, seed=0):
    np.random.seed(seed)
    X = np.random.normal(0, 1, (n, p))
    X = np.array(X, dtype=dtype, order=order)
    cX = mod.dense(X, method="naive", n_threads=15)
    run_naive(X, cX, dtype)

    
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("n, d", [
    [1, 2],
    [10, 2],
    [1, 10],
    [20, 30],
    [100, 20],
])
def test_naive_interaction_dense(n, d, dtype, order, seed=0):
    def _expand(x, level):
        n = x.shape[0]
        if level <= 0:
            return np.array([np.ones(n), x]).T
        return np.array([x == k for k in range(level)]).T

    def _create_dense(X, pairs, levels):
        col_lst = []
        for pair in pairs:
            i0, i1 = pair[0], pair[1]
            l0, l1 = levels[i0], levels[i1]
            Y0 = _expand(X[:, i0], l0)
            Y1 = _expand(X[:, i1], l1)
            if (l0 <= 0) and (l1 <= 0):
                col_lst.append(Y0[:, 1])
                col_lst.append(Y1[:, 1])
                col_lst.append(Y0[:, 1] * Y1[:, 1])
            else:
                for j1 in range(Y1.shape[1]):
                    for j0 in range(Y0.shape[1]):
                        col_lst.append(Y0[:, j0] * Y1[:, j1])
        return np.array(col_lst, dtype=dtype, order=order).T

    min_bytes = ad.configs.Configs.min_bytes
    ad.configs.set_configs("min_bytes", None)

    np.random.seed(seed)
    X = np.random.normal(0, 1, (n, d))
    X = np.array(X, dtype=dtype, order=order)
    c_subset = np.random.choice(d, size=d//2, replace=False)
    c_levels = 1 + np.random.choice(10, size=d//2, replace=True)
    for j, level in zip(c_subset, c_levels):
        X[:, j] = np.random.choice(level, size=n, replace=True)
    levels = np.zeros(d, dtype=int)
    levels[c_subset] = c_levels
    intr_map = {}
    for j in range(min(10, d)):
        if np.random.binomial(1, 0.5, 1):
            intr_map[j] = None
        else:
            intr_map[j] = np.random.choice(d, size=d//2, replace=False)
    cX = mod.interaction(X, intr_map, levels=levels, n_threads=2)
    X = _create_dense(X, cX.pairs, levels)
    run_naive(X, cX, dtype)

    ad.configs.set_configs("min_bytes", min_bytes)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("n, p, K", [
    [10, 1, 3],
    [100, 20, 2],
    [20, 100, 4],
])
def test_naive_kronecker_eye(n, p, K, dtype, order, seed=0):
    np.random.seed(seed)
    X = np.random.normal(0, 1, (n, p))
    X = np.array(X, dtype=dtype, order=order)
    cX = mod.kronecker_eye(
        mod.dense(X, method="naive"), K, n_threads=2
    )
    X = np.kron(X, np.eye(K))
    run_naive(X, cX, dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("n, p, K", [
    [10, 1, 3],
    [100, 20, 2],
    [20, 100, 4],
])
def test_naive_kronecker_eye_dense(n, p, K, dtype, order, seed=0):
    np.random.seed(seed)
    X = np.random.normal(0, 1, (n, p))
    X = np.array(X, dtype=dtype, order=order)
    cX = mod.kronecker_eye(
        X, K, n_threads=7
    )
    X = np.kron(X, np.eye(K))
    run_naive(X, cX, dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("n, d", [
    [1, 10],
    [10, 1],
    [100, 20],
])
def test_naive_one_hot_dense(n, d, dtype, order, seed=0):
    def _expand(x, level):
        if level <= 0:
            return x.reshape((-1, 1))
        return np.array([x == k for k in range(level)]).T

    def _create_dense(X, levels):
        return np.concatenate([
            _expand(X[:, i], level)
            for i, level in enumerate(levels)
        ], axis=1, dtype=dtype)

    np.random.seed(seed)
    X = np.random.normal(0, 1, (n, d))
    X = np.array(X, dtype=dtype, order=order)
    c_subset = np.random.choice(d, size=d//2, replace=False)
    c_levels = 1 + np.random.choice(10, size=d//2, replace=True)
    for j, level in zip(c_subset, c_levels):
        X[:, j] = np.random.choice(level, size=n, replace=True)
    levels = np.zeros(d, dtype=int)
    levels[c_subset] = c_levels
    cX = mod.one_hot(X, levels)
    X = _create_dense(X, levels)
    run_naive(X, cX, dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("read_mode", ["file", "mmap"])
@pytest.mark.parametrize("n, p", [
    [10, 20],
    [1, 13],
    [144, 1],
    [10000, 1],
])
def test_naive_snp_unphased(n, p, read_mode, dtype, seed=0):
    min_bytes = ad.configs.Configs.min_bytes
    ad.configs.set_configs("min_bytes", 0)

    np.random.seed(seed)
    data = ad.data.snp_unphased(n, p, seed=seed)
    filename = "/tmp/test_snp_unphased.snpdat"
    handler = ad.io.snp_unphased(filename)
    handler.write(data["X"], impute_method="mean")
    cX = mod.snp_unphased(
        filename=filename,
        read_mode=read_mode,
        dtype=dtype,
        n_threads=7,
    )

    handler.read()
    impute = handler.impute
    X = data["X"].astype(float)
    X = np.where(X == -9, impute[None], X)
    run_naive(X, cX, dtype)
    os.remove(filename)

    ad.configs.set_configs("min_bytes", min_bytes)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("read_mode", ["file", "mmap"])
@pytest.mark.parametrize("n, s, A", [
    [10, 20, 4],
    [1, 13, 3],
    [144, 1, 2],
    [10000, 1, 2],
])
def test_naive_snp_phased_ancestry(n, s, A, read_mode, dtype, seed=0):
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

    min_bytes = ad.configs.Configs.min_bytes
    ad.configs.set_configs("min_bytes", 0)

    np.random.seed(seed)
    data = ad.data.snp_phased_ancestry(n, s, A, seed=seed)
    filename = "/tmp/test_snp_phased_ancestry.snpdat"
    handler = ad.io.snp_phased_ancestry(filename)
    handler.write(data["X"], data["ancestries"], A)
    cX = mod.snp_phased_ancestry(
        filename=filename,
        read_mode=read_mode,
        dtype=dtype,
        n_threads=2,
    )
    os.remove(filename)

    X = create_dense(data["X"], data["ancestries"], A) 
    run_naive(X, cX, dtype)

    ad.configs.set_configs("min_bytes", min_bytes)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("n, p", [
    [2, 2],
    [100, 20],
    [20, 100],
])
def test_naive_sparse(n, p, dtype, order, seed=0):
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


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n, p", [
    [2, 10],
    [10, 1],
    [100, 20],
    [20, 100],
])
def test_naive_standardize(n, p, dtype, seed=0):
    np.random.seed(seed)
    X = np.asfortranarray(np.random.normal(0, 1, (n, p)).astype(dtype))
    dX = mod.dense(X) 

    cX = mod.standardize(dX)
    means = np.mean(X, axis=0)
    scales = np.std(X, axis=0)
    assert np.allclose(cX.centers, means)
    assert np.allclose(cX.scales, scales)

    centers = np.random.normal(0, 1, p)
    cX = mod.standardize(dX, centers)
    scales = np.sqrt(
        np.sum((X - centers[None]) ** 2, axis=0) / n
    )
    assert np.allclose(cX.centers, centers)
    assert np.allclose(cX.scales, scales)

    X = (X - centers[None]) / scales[None]
    run_naive(X, cX, dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("subset_prop", [0, 0.5, 1])
@pytest.mark.parametrize("n, p", [
    [1, 1],
    [1, 2],
    [2, 1],
    [10, 7],
    [100, 20],
    [20, 100],
])
def test_naive_csubset(n, p, subset_prop, dtype, seed=0):
    np.random.seed(seed)
    X = np.asfortranarray(np.random.normal(0, 1, (n, p)).astype(dtype))
    indices = np.random.choice(p, size=max(int(subset_prop * p), 1), replace=False)
    dX = mod.dense(X, method="naive")
    cX = mod.subset(dX, indices, axis=1, n_threads=2)
    X = X[:, indices]
    run_naive(X, cX, dtype)
    # test operator[] works properly
    cX = dX[:, indices]
    run_naive(X, cX, dtype)
    bool_indices = np.zeros(p, dtype=bool)
    bool_indices[indices] = True
    cX = dX[:, bool_indices]
    X = X[:, np.argsort(indices)]
    run_naive(X, cX, dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("subset_prop", [0, 0.5, 1])
@pytest.mark.parametrize("n, p", [
    [1, 1],
    [1, 2],
    [2, 1],
    [10, 7],
    [100, 20],
    [20, 100],
])
def test_naive_rsubset(n, p, subset_prop, dtype, seed=0):
    np.random.seed(seed)
    X = np.asfortranarray(np.random.normal(0, 1, (n, p)).astype(dtype))
    indices = np.random.choice(n, size=max(int(subset_prop * n), 1), replace=False)
    dX = mod.dense(X, method="naive")
    cX = mod.subset(dX, indices, axis=0, n_threads=2)
    sX = X[indices]
    run_naive(sX, cX, dtype)
    # test operator[] works properly
    cX = dX[indices]
    run_naive(sX, cX, dtype)
    col_indices = np.random.choice(p, size=max(p//2, 1), replace=False)
    cX = dX[:, col_indices][indices]
    sX = X[:, col_indices][indices]
    run_naive(sX, cX, dtype)
    bool_indices = np.zeros(n, dtype=bool)
    bool_indices[indices] = True
    cX = dX[bool_indices]
    sX = X[np.sort(indices)]
    run_naive(sX, cX, dtype)


# Reset to default settings
ad.configs.set_configs("min_bytes", None)