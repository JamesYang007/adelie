import adelie as ad
import numpy as np
import os
import pytest


@pytest.mark.parametrize("impute_method", ["mean"])
@pytest.mark.parametrize("read_mode", ["file", "mmap"])
@pytest.mark.parametrize("n, p", [
    [1, 1],
    [200, 32],
    [2000, 3000],
    [1421, 927],
])
def test_io_snp_unphased(
    n, p, impute_method, read_mode, 
    seed=0,
):
    def create_calldata(
        n, p, seed
    ):
        np.random.seed(seed)
        calldata = np.zeros((n, p), dtype=np.int8)
        calldata.ravel()[
            np.random.choice(np.arange(n * p), int(0.25 * n * p), replace=False)
        ] = -9
        calldata.ravel()[
            np.random.choice(np.arange(n * p), int(0.25 * n * p), replace=False)
        ] = 1
        calldata.ravel()[
            np.random.choice(np.arange(n * p), int(0.05 * n * p), replace=False)
        ] = 2
        calldata = np.asfortranarray(calldata)
        return calldata

    calldata = create_calldata(n, p, seed)

    filename = "/tmp/dummy_snp_unphased.snpdat"
    handler = ad.io.snp_unphased(filename, read_mode=read_mode)
    w_bytes, _ = handler.write(calldata, impute_method, n_threads=2)
    r_bytes = handler.read()
    r_bytes = handler.read() # try double-reading

    expected_nnms = np.sum(calldata >= 0, axis=0)
    assert np.allclose(handler.nnm, expected_nnms)
    if impute_method == "mean":
        means = np.mean(calldata, axis=0, where=calldata >= 0)
        assert np.allclose(handler.impute, means)
    else:
        raise NotImplementedError()

    assert w_bytes == r_bytes
    assert handler.rows == n
    assert handler.cols == p
    assert handler.snps == p

    expected_nnzs = np.sum(calldata != 0, axis=0)
    assert np.allclose(handler.nnz, expected_nnzs)

    dense = handler.to_dense()
    assert np.allclose(dense, calldata)
    os.remove(filename)


@pytest.mark.parametrize("read_mode", ["file", "mmap"])
@pytest.mark.parametrize("n, s, A", [
    [1, 1, 1],
    [200, 32, 4],
    [2000, 3000, 7],
    [1421, 927, 8],
])
def test_io_snp_phased_ancestry(
    n, s, A, read_mode, 
    seed=0,
):
    def create_dense(calldata, ancestries, A, hap=None):
        n, s = calldata.shape[0], calldata.shape[1] // 2
        dense = np.zeros((n, s * A), dtype=np.int8)
        base_indices = A * np.arange(n * s, dtype=int)[None]
        if (hap is None) or (hap == 0):
            dense.ravel()[
                base_indices +
                ancestries.reshape(n, s, 2)[:,:,0].ravel()
            ] += calldata.reshape(n, s, 2)[:,:,0].ravel()
        if (hap is None) or (hap == 1):
            dense.ravel()[
                base_indices +
                ancestries.reshape(n, s, 2)[:,:,1].ravel()
            ] += calldata.reshape(n, s, 2)[:,:,1].ravel()
        return dense

    data = ad.data.snp_phased_ancestry(n, s, A, seed=seed)
    calldata = data["X"]
    ancestries = data["ancestries"]
    dense = create_dense(calldata, ancestries, A)

    filename = "/tmp/dummy_snp_phased_ancestry.snpdat"
    handler = ad.io.snp_phased_ancestry(filename, read_mode=read_mode)
    w_bytes, _ = handler.write(calldata, ancestries, A, n_threads=2)
    r_bytes = handler.read()
    r_bytes = handler.read() # try double-reading

    assert w_bytes == r_bytes
    assert handler.rows == n
    assert handler.snps == s
    assert handler.ancestries == A
    assert handler.cols == s * A

    dense0 = create_dense(calldata, ancestries, A, hap=0)
    expected_nnz0s = np.sum(dense0 != 0, axis=0)
    assert np.allclose(handler.nnz0, expected_nnz0s)

    dense1 = create_dense(calldata, ancestries, A, hap=1)
    expected_nnz1s = np.sum(dense1 != 0, axis=0)
    assert np.allclose(handler.nnz1, expected_nnz1s)

    my_dense = handler.to_dense()
    assert np.allclose(my_dense, dense)
    os.remove(filename)