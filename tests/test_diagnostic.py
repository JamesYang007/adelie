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