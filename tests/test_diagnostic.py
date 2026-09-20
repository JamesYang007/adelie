import adelie as ad
import numpy as np
import pytest
import scipy


def generate_data(n, p, L, K, beta_type, dtype, seed):
    np.random.seed(seed)
    X = np.random.normal(0, 1, (n, p)).astype(dtype)
    y = X @ np.random.normal(0, 1, (p, K)) + np.random.normal(0, 1, (n, K))
    y = y.astype(dtype)
    betas = np.random.uniform(-1, 1, (L, p*K)).astype(dtype)
    if beta_type == "sparse":
        betas = scipy.sparse.csr_matrix(betas, dtype=dtype)
    if K == 1:
        y = y.squeeze(axis=1)
        intercepts = np.random.normal(0, 1, (L,)).astype(dtype)
    else:
        intercepts = np.random.normal(0, 1, (L, K)).astype(dtype)
    lmdas = np.sort(np.random.uniform(0, 1, L))[::-1]

    return {
        "X": X, 
        "y": y, 
        "betas": betas, 
        "intercepts": intercepts,
        "lmdas": lmdas,
    }


def predict(X, betas, intercepts):
    p = X.shape[1]
    L = betas.shape[0]
    K = 1 if len(intercepts.shape) == 1 else intercepts.shape[1]
    if K != 1:
        if isinstance(betas, scipy.sparse.csr_matrix):
            betas = betas.toarray()
        betas = betas.reshape((L, p, K))
        Xbetas = np.einsum("ij,ljk->lik", X, betas)
    else:
        Xbetas = betas @ X.T
    return intercepts[:, None] + Xbetas


@pytest.mark.filterwarnings("ignore: Detected matrix to be C-contiguous.")
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("beta_type", ["dense", "sparse"])
@pytest.mark.parametrize("K", [1, 3])
@pytest.mark.parametrize("L", [1, 2])
@pytest.mark.parametrize("p", [1, 10])
@pytest.mark.parametrize("n", [2, 20])
def test_predict(n, p, L, K, beta_type, dtype, seed=0):
    data = generate_data(n, p, L, K, beta_type, dtype, seed)
    X = data["X"]
    betas = data["betas"]
    intercepts = data["intercepts"]
    actual = ad.diagnostic.predict(X, betas, intercepts)
    expected = predict(X, betas, intercepts)

    assert np.allclose(actual, expected, atol=1e-6)


@pytest.mark.filterwarnings("ignore: Detected matrix to be C-contiguous.")
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("beta_type", ["dense", "sparse"])
@pytest.mark.parametrize("K", [1, 3])
@pytest.mark.parametrize("L", [1, 2])
@pytest.mark.parametrize("p", [1, 10])
@pytest.mark.parametrize("n", [2, 20])
def test_objective(n, p, L, K, beta_type, dtype, seed=0):
    def _objective(X, glm, betas, intercepts, lmdas):
        etas = predict(X, betas, intercepts)
        losses = np.array([glm.loss(eta) - glm.loss_full() for eta in etas])
        if isinstance(betas, scipy.sparse.csr_matrix):
            betas = betas.toarray()
        if K == 1:
            penalty = np.sum(np.abs(betas), axis=-1)
        else:
            betas = betas.reshape((L, p, K))
            penalty = np.sum(np.linalg.norm(betas, axis=-1), axis=-1)
        return losses + lmdas * np.sqrt(K) * penalty
    
    data = generate_data(n, p, L, K, beta_type, dtype, seed)
    X = data["X"]
    y = data["y"]
    betas = data["betas"]
    intercepts = data["intercepts"]
    lmdas = data["lmdas"]
    if K == 1:
        glm = ad.glm.gaussian(y)
    else:
        glm = ad.glm.multigaussian(y)

    actual = ad.diagnostic.objective(X, glm, betas, intercepts, lmdas)
    expected = _objective(X, glm, betas, intercepts, lmdas)

    assert np.allclose(actual, expected, atol=1e-6)


@pytest.mark.filterwarnings("ignore: Detected matrix to be C-contiguous.")
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_diagnostic(dtype, seed=0):
    n = 10
    p = 10
    np.random.seed(seed)
    X = np.random.normal(0, 1, (n, p)).astype(dtype)
    y = X @ np.random.normal(0, 1, p) + np.random.normal(0, 1, n)
    y = y.astype(dtype)
    constraints = [None] * p
    constraints[0] = ad.constraint.lower(np.full(1, -1, dtype=dtype))
    state = ad.grpnet(X, ad.glm.gaussian(y), progress_bar=False)
    ad.diagnostic.diagnostic(state)

# --- coefficients(): vectorised coefficient() used by cv.cv_grpnet --------------------
def _generate_path(p, L, K, beta_type, dtype, seed, density=0.5):
    rng = np.random.default_rng(seed)
    betas = rng.uniform(-1, 1, (L, p * K)).astype(dtype)
    betas[rng.random(betas.shape) > density] = 0
    betas = scipy.sparse.csr_matrix(betas, dtype=dtype)
    if beta_type == "noncanonical" and betas.nnz > 1:
        # break canonical format in the first non-empty row -> fallback path
        for r in range(L):
            s, e = betas.indptr[r], betas.indptr[r + 1]
            if e - s > 1:
                betas.indices[s:e] = betas.indices[s:e][::-1].copy()
                betas.has_canonical_format = False
                break
    if K == 1:
        intercepts = rng.normal(0, 1, (L,)).astype(dtype)
    else:
        intercepts = rng.normal(0, 1, (L, K)).astype(dtype)
    lmdas = np.sort(rng.uniform(0.1, 1, L))[::-1]
    return betas, intercepts, lmdas


def _query_grid(lmdas, rng):
    """On-grid (the cv_grpnet case), interior off-grid, and both boundary sides."""
    lo, hi = lmdas[-1], lmdas[0]
    interior = rng.uniform(lo, hi, 5) if len(lmdas) > 1 else np.array([lo])
    return np.concatenate(
        [lmdas, interior, [hi * 1.5, hi + 1e-12], [lo * 0.5, 0.0, max(lo - 1e-12, 0.0)]]
    )


def _reference(lmdas_new, betas, intercepts, lmdas):
    beta_ints = [
        ad.diagnostic.coefficient(lmda=l, betas=betas, intercepts=intercepts, lmdas=lmdas)
        for l in lmdas_new
    ]
    return (
        scipy.sparse.vstack([x[0] for x in beta_ints]).tocsr(),
        np.array([x[1] for x in beta_ints]),
    )


def _assert_same_csr(a, b):
    a = a.tocsr()
    b = b.tocsr()
    assert a.shape == b.shape
    assert np.array_equal(a.indptr, b.indptr)
    assert np.array_equal(a.indices, b.indices)
    assert a.data.dtype == b.data.dtype
    # byte-level, not allclose: catches -0.0 vs 0.0 and any rounding drift
    assert a.data.tobytes() == b.data.tobytes()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("beta_type", ["sparse", "noncanonical"])
@pytest.mark.parametrize("K", [1, 3])
@pytest.mark.parametrize("L", [1, 2, 7])
@pytest.mark.parametrize("p", [1, 10, 40])
def test_coefficients_matches_coefficient(p, L, K, beta_type, dtype, seed=0):
    rng = np.random.default_rng(seed + 1)
    betas, intercepts, lmdas = _generate_path(p, L, K, beta_type, dtype, seed)
    lmdas_new = _query_grid(lmdas, rng)

    actual_b, actual_i = ad.diagnostic.coefficients(lmdas_new, betas, intercepts, lmdas)
    expected_b, expected_i = _reference(lmdas_new, betas, intercepts, lmdas)

    _assert_same_csr(actual_b, expected_b)
    assert actual_i.shape == expected_i.shape
    assert actual_i.dtype == expected_i.dtype
    assert actual_i.tobytes() == expected_i.tobytes()


def test_coefficients_explicit_and_negative_zeros():
    """Stored zeros / -0.0 in the fitted path must survive identically."""
    betas, intercepts, lmdas = _generate_path(30, 6, 1, "sparse", np.float64, seed=3)
    betas.data[::3] = 0.0
    betas.data[1::5] = -0.0
    lmdas_new = _query_grid(lmdas, np.random.default_rng(4))
    actual = ad.diagnostic.coefficients(lmdas_new, betas, intercepts, lmdas)
    expected = _reference(lmdas_new, betas, intercepts, lmdas)
    _assert_same_csr(actual[0], expected[0])
    assert actual[1].tobytes() == expected[1].tobytes()


def test_coefficients_input_validation():
    betas, intercepts, lmdas = _generate_path(5, 3, 1, "sparse", np.float64, seed=5)
    with pytest.raises(ValueError):
        ad.diagnostic.coefficients(np.ones((2, 2)), betas, intercepts, lmdas)
    with pytest.raises(RuntimeError):
        ad.diagnostic.coefficients(np.ones(2), betas, intercepts, np.array([]))


@pytest.mark.filterwarnings("ignore: Detected matrix to be C-contiguous.")
@pytest.mark.parametrize("early_exit", [False, True])
@pytest.mark.parametrize("glm_name", ["gaussian", "binomial"])
def test_cv_grpnet_unchanged_by_vectorised_extraction(monkeypatch, early_exit, glm_name, seed=0):
    """cv_grpnet's loss surface must be byte-identical whether the per-fold path
    is materialised on the common grid by the vectorised coefficients() or by
    the reference per-lambda coefficient() stacking it replaced. Covers both the
    on-grid regime (early_exit=False, cv default) and the extrapolating regime
    (early_exit=True, folds may stop short of the common path)."""
    import adelie.cv as cv_mod

    np.random.seed(seed)
    n, p = 80, 12
    X = np.random.normal(0, 1, (n, p))
    eta = X[:, :3] @ np.array([1.0, -2.0, 0.5])
    if glm_name == "gaussian":
        y = eta + np.random.normal(0, 0.5, n)
        glm = ad.glm.gaussian(y)
    else:
        y = (np.random.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)
        glm = ad.glm.binomial(y)

    kwargs = dict(X=X, glm=glm, n_folds=3, seed=seed, early_exit=early_exit, min_ratio=1e-2)
    vectorised = ad.cv_grpnet(**kwargs)

    def stacked(lmdas_new, betas, intercepts, lmdas):
        return _reference(lmdas_new, betas, intercepts, lmdas)

    monkeypatch.setattr(cv_mod, "coefficients", stacked)
    reference = ad.cv_grpnet(**kwargs)

    assert vectorised.lmdas.tobytes() == reference.lmdas.tobytes()
    assert vectorised.losses.tobytes() == reference.losses.tobytes()
    assert vectorised.avg_losses.tobytes() == reference.avg_losses.tobytes()
    assert vectorised.best_idx == reference.best_idx

