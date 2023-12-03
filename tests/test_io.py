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
    ] = 1
    calldata.ravel()[
        np.random.choice(np.arange(n * p), int(0.05 * n * p), replace=False)
    ] = 2
    calldata = np.asfortranarray(calldata)
    return calldata


def test_io_snp_unphased():
    def _test(n, p, seed=0):
        calldata = create_calldata(n, p, seed)

        filename = "/tmp/dummy_snp_unphased.snpdat"
        handler = ad.io.snp_unphased(filename)
        w_bytes = handler.write(calldata)
        r_bytes = handler.read()
        os.remove(filename)

        total_bytes_exp = (
            1 + 2 * 4 + 8 * (p + 1) + 5 * np.sum(calldata != 0)
        )
        assert w_bytes == total_bytes_exp
        assert w_bytes == r_bytes
        assert handler.rows() == n
        assert handler.cols() == p

        outer = handler.outer()
        nnzs = np.array([handler.nnz(j) for j in range(p)])
        inners = [handler.inner(j) for j in range(p)]
        values = [handler.value(j) for j in range(p)]

        assert np.allclose((outer[1:] - outer[:-1]) / 5, nnzs)
        for j in range(p):
            assert np.allclose(
                np.arange(n)[calldata[:, j] != 0],
                inners[j],
            )
            assert np.allclose(
                calldata[:, j][calldata[:, j] != 0],
                values[j],
            )

        dense = handler.to_dense()
        assert np.allclose(dense, calldata)

    _test(1, 1)
    _test(200, 32)
    _test(2000, 3000)
    _test(1421, 927)


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

    def _test(n, s, A, seed=0):
        data = ad.data.create_snp_phased_ancestry(n, s, A, seed=seed)
        calldata = data["X"]
        ancestries = data["ancestries"]
        dense = create_dense(calldata, ancestries, A)

        filename = "/tmp/dummy_snp_phased_ancestry.snpdat"
        handler = ad.io.snp_phased_ancestry(filename)
        w_bytes = handler.write(calldata, ancestries, A, n_threads=8)
        r_bytes = handler.read()
        os.remove(filename)

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

    _test(1, 1, 1)
    _test(200, 32, 4)
    _test(2000, 3000, 7)
    _test(1421, 927, 8)