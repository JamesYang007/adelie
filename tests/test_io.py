import adelie as ad
import numpy as np
import os


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


def test_io_snp_unphased():
    def _test(n, p, impute_method, read_mode, seed=0):
        calldata = create_calldata(n, p, seed)

        filename = "/tmp/dummy_snp_unphased.snpdat"
        handler = ad.io.snp_unphased(filename, read_mode=read_mode)
        w_bytes, _ = handler.write(calldata, impute_method)
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

    impute_methods = ["mean"]
    read_modes = ["file", "mmap"]
    for impute_method in impute_methods:
        for read_mode in read_modes:
            _test(1, 1, impute_method, read_mode)
            _test(200, 32, impute_method, read_mode)
            _test(2000, 3000, impute_method, read_mode)
            _test(1421, 927, impute_method, read_mode)


def test_io_snp_phased_ancestry():
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

    def _test(n, s, A, read_mode, seed=0):
        data = ad.data.snp_phased_ancestry(n, s, A, seed=seed)
        calldata = data["X"]
        ancestries = data["ancestries"]
        dense = create_dense(calldata, ancestries, A)

        filename = "/tmp/dummy_snp_phased_ancestry.snpdat"
        handler = ad.io.snp_phased_ancestry(filename, read_mode=read_mode)
        w_bytes = handler.write(calldata, ancestries, A, n_threads=8)
        r_bytes = handler.read()
        r_bytes = handler.read() # try double-reading

        assert w_bytes == r_bytes
        assert handler.rows() == n
        assert handler.cols() == s * A

        outer = handler.outer()
        nnzs = np.array([handler.nnz(j, h) for j in range(s) for h in range(2)])
        inners = [handler.inner(j, h) for j in range(s) for h in range(2)]
        my_ancestries = [handler.ancestry(j, h) for j in range(s) for h in range(2)]

        assert np.allclose((outer[1:] - outer[:-1]) / 5, nnzs)
        for j in range(calldata.shape[-1]):
            assert np.allclose(
                np.arange(n)[calldata[:, j] != 0],
                inners[j],
            )
            assert np.allclose(
                ancestries[:, j][calldata[:, j] != 0],
                my_ancestries[j],
            )

        my_dense = handler.to_dense()
        assert np.allclose(my_dense, dense)
        os.remove(filename)

    read_modes = ["file", "mmap"]
    for read_mode in read_modes:
        _test(1, 1, 1, read_mode)
        _test(200, 32, 4, read_mode)
        _test(2000, 3000, 7, read_mode)
        _test(1421, 927, 8, read_mode)