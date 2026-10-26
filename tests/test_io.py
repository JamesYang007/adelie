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


@pytest.mark.parametrize("read_mode", ["file", "mmap"])
@pytest.mark.parametrize("n, s, A", [
    [1, 1, 1],
    [200, 32, 4],
    [2000, 3000, 7],
    [1421, 927, 8],
])
def test_io_snp_combine_r(n, s, A, read_mode, seed=0):
    data = ad.data.snp_combine_r(n, s, A, seed=seed)
    calldata = data["X"]
    ancestries = data["ancestries"]

    # build expected dense: for each SNP j,
    #  - first column is the genotype calldata[:, j]
    #  - next A columns are the unphased dosage per ancestry
    X_expected = np.zeros((n, s * (1 + A)), dtype=np.int8)
    for j in range(s):
        # genotype column
        X_expected[:, j * (1 + A)] = calldata[:, j]
        # ancestry dosage columns
        for k in range(A):
            dosage = (
                (ancestries[:, 2 * j] == k).astype(np.int8)
                + (ancestries[:, 2 * j + 1] == k).astype(np.int8)
            )
            X_expected[:, j * (1 + A) + 1 + k] = dosage

    filename = "/tmp/dummy_snp_combine_r.snpdat"
    handler = ad.io.snp_combine_r(filename, read_mode=read_mode)

    # print matrix and dtype
    print(f"calldata:\n{calldata}")
    print(f"ancestries:\n{ancestries}")
    print(f"calldata.dtype:\n{calldata.dtype}")
    print(f"ancestries.dtype:\n{ancestries.dtype}")
    
    # write & read twice
    w_bytes, _ = handler.write(calldata, ancestries, A, n_threads=2)
    r_bytes1 = handler.read()
    r_bytes2 = handler.read()
    assert w_bytes == r_bytes1 == r_bytes2

    # basic metadata
    assert handler.rows == n
    assert handler.snps == s
    assert handler.ancestries == A
    assert handler.cols == s * (1 + A)

    # full dense round‐trip
    dense = handler.to_dense()
    print(f"dense:\n{dense}")
    print(f"X_expected:\n{X_expected}")
    assert np.allclose(dense, X_expected)

    # nnz per column (non‐zero counts)
    expected_nnz = np.sum(X_expected != 0, axis=0)
    print(f"expected_nnz:\n{expected_nnz}")
    print(f"handler.nnz:\n{handler.nnz}")
    assert np.allclose(handler.nnz, expected_nnz)

    os.remove(filename)


@pytest.mark.parametrize("read_mode", ["file", "mmap"])
@pytest.mark.parametrize("n, s, A", [
    [1, 1, 1],
    [37, 13, 3],
    [200, 32, 4],
])
def test_io_snp_combine_s(n, s, A, read_mode, seed=0):
    # Generate phased haplotype calldata (0/1 per hap) and phased ancestries
    data = ad.data.snp_phased_ancestry(n, s, A, seed=seed)
    calldata = data["X"]        # shape (n, 2*s), entries in {0,1}
    ancestries = data["ancestries"]  # shape (n, 2*s), entries in [0,A)

    # Build expected dense (n, s*(2*A))
    X_expected = np.zeros((n, s * (2 * A)), dtype=np.int8)
    for j in range(s):
        c0 = calldata[:, 2*j]
        c1 = calldata[:, 2*j + 1]
        a0 = ancestries[:, 2*j]
        a1 = ancestries[:, 2*j + 1]
        for a in range(A):
            # mutated hap counts per ancestry (first A columns)
            mut = (c0 == 1).astype(np.int8) * (a0 == a).astype(np.int8) \
                  + (c1 == 1).astype(np.int8) * (a1 == a).astype(np.int8)
            X_expected[:, j * (2*A) + a] = mut
            # ancestry dosage per ancestry (last A columns)
            dos = (a0 == a).astype(np.int8) + (a1 == a).astype(np.int8)
            X_expected[:, j * (2*A) + A + a] = dos

    filename = "/tmp/dummy_snp_combine_s.snpdat"
    handler = ad.io.snp_combine_s(filename, read_mode=read_mode)

    # write & read twice
    w_bytes, _ = handler.write(calldata, ancestries, A, n_threads=2)
    r_bytes1 = handler.read()
    r_bytes2 = handler.read()
    assert w_bytes == r_bytes1 == r_bytes2

    # basic metadata
    assert handler.rows == n
    assert handler.snps == s
    assert handler.ancestries == A
    assert handler.cols == s * (2 * A)

    # dense round-trip
    dense = handler.to_dense(n_threads=2)
    assert np.allclose(dense, X_expected)

    # nnz per column (non-zero sample counts)
    expected_nnz = np.sum(X_expected != 0, axis=0)
    assert np.allclose(handler.nnz, expected_nnz)

    os.remove(filename)


@pytest.mark.parametrize("read_mode", ["file", "mmap"])
@pytest.mark.parametrize("n, s, A", [
    [37, 13, 3],
    [200, 32, 4],
])
def test_io_snp_combine_s_selected(n, s, A, read_mode, seed=1):
    data = ad.data.snp_phased_ancestry(n, s, A, seed=seed)
    calldata = data["X"]
    ancestries = data["ancestries"]

    # Select a subset of ancestries (preserve order, unique)
    if A >= 3:
        selected = np.array([0, A-1], dtype=np.uint32)
    else:
        selected = np.array([0], dtype=np.uint32)
    B = len(selected)

    # Build expected dense (n, s*(2*B)) restricted to selected ancestries
    X_expected = np.zeros((n, s * (2 * B)), dtype=np.int8)
    for j in range(s):
        c0 = calldata[:, 2*j]
        c1 = calldata[:, 2*j + 1]
        a0 = ancestries[:, 2*j]
        a1 = ancestries[:, 2*j + 1]
        for bi, a in enumerate(selected):
            # mutated block
            mut = (c0 == 1).astype(np.int8) * (a0 == a).astype(np.int8) \
                  + (c1 == 1).astype(np.int8) * (a1 == a).astype(np.int8)
            X_expected[:, j * (2*B) + bi] = mut
            # dosage block
            dos = (a0 == a).astype(np.int8) + (a1 == a).astype(np.int8)
            X_expected[:, j * (2*B) + B + bi] = dos

    filename = "/tmp/dummy_snp_combine_s_sel.snpdat"
    handler = ad.io.snp_combine_s(filename, read_mode=read_mode)
    w_bytes, _ = handler.write(calldata, ancestries, A, n_threads=2, selected_ancestries=selected)
    r_bytes = handler.read()
    assert w_bytes == r_bytes

    # Metadata reflects selection
    assert handler.rows == n
    assert handler.snps == s
    assert handler.ancestries == B
    assert handler.cols == s * (2 * B)

    dense = handler.to_dense(n_threads=2)
    assert np.allclose(dense, X_expected)

    expected_nnz = np.sum(X_expected != 0, axis=0)
    assert np.allclose(handler.nnz, expected_nnz)

    os.remove(filename)
