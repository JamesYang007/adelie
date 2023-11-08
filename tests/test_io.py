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
