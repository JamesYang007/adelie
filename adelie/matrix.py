from . import adelie_core as core
from .adelie_core.matrix import (
    MatrixNaiveBase64,
    MatrixNaiveBase32,
    MatrixCovBase64,
    MatrixCovBase32,
)
from scipy.sparse import (
    csc_matrix,
    csr_matrix,
)
from typing import Union
import numpy as np
import warnings


def _to_dtype(mat):
    if (
        isinstance(mat, MatrixNaiveBase32) or
        isinstance(mat, MatrixCovBase32)
    ): 
        return np.float32
    elif (
        isinstance(mat, MatrixNaiveBase64) or
        isinstance(mat, MatrixCovBase64)
    ): 
        return np.float64
    return None


def block_diag(
    mats: list,
    *,
    n_threads: int =1,
):
    """Creates a block-diagonal matrix given by the list of matrices.

    If ``mats`` represents a list of matrices :math:`A_1,\\ldots, A_L`,
    then the resulting matrix represents the block-diagonal matrix given by

    .. math::
        \\begin{align*}
            \\begin{bmatrix}
                A_1 & 0 & \\cdots & 0 \\\\
                0 & A_2 & \\cdots & 0 \\\\
                \\vdots & \\vdots & \\ddots & \\vdots \\\\
                0 & 0 & \\cdots & A_L
            \\end{bmatrix}
        \\end{align*}

    .. note::
        This matrix only works for covariance method!

    Parameters
    ----------
    mats : list
        List of matrices to represent the block diagonal matrix.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.

    Returns
    -------
    wrap
        Wrapper matrix object.

    See Also
    --------
    adelie.matrix.MatrixCovBase64
    """
    mats = [
        dense(mat, method="cov", n_threads=1)
        if isinstance(mat, np.ndarray) else
        mat
        for mat in mats
    ]
    dtype = _to_dtype(mats[0])

    for mat in mats:
        if dtype == _to_dtype(mat): continue
        raise ValueError("All matrices must have the same underlying data type.")

    dispatcher = {
        np.float64: core.matrix.MatrixCovBlockDiag64,
        np.float32: core.matrix.MatrixCovBlockDiag32,
    }

    core_base = dispatcher[dtype]

    class _block_diag(core_base):
        def __init__(self):
            self.mats = mats
            core_base.__init__(self, self.mats, n_threads)

    return _block_diag()


def concatenate(
    mats: list,
    *,
    n_threads: int =1,
):
    """Creates a column-wise concatenation of the matrices.

    If ``mats`` represents a list of matrices :math:`X_1,\\ldots, X_L`,
    then the resulting matrix represents the column-wise concatenated matrix
    given by

    .. math::
        \\begin{align*}
            \\begin{bmatrix}
                \\vert & \\vert & \\cdots & \\vert \\\\
                X_1 & X_2 & \\cdots & X_L \\\\
                \\vert & \\vert & \\cdots & \\vert
            \\end{bmatrix}
        \\end{align*}

    .. note::
        This matrix only works for naive method!

    Parameters
    ----------
    mats : list
        List of matrices to concatenate along the columns.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.

    Returns
    -------
    wrap
        Wrapper matrix object.

    See Also
    --------
    adelie.matrix.MatrixNaiveBase64
    """
    mats = [
        dense(mat, method="naive", n_threads=n_threads)
        if isinstance(mat, np.ndarray) else
        mat
        for mat in mats
    ]
    dtype = _to_dtype(mats[0])

    for mat in mats:
        if dtype == _to_dtype(mat): continue
        raise ValueError("All matrices must have the same underlying data type.")

    dispatcher = {
        np.float64: core.matrix.MatrixNaiveConcatenate64,
        np.float32: core.matrix.MatrixNaiveConcatenate32,
    }

    core_base = dispatcher[dtype]

    class _concatenate(core_base):
        def __init__(self):
            self.mats = mats
            core_base.__init__(self, self.mats, n_threads)

    return _concatenate()


def dense(
    mat: np.ndarray,
    *,
    method: str ="naive",
    n_threads: int =1,
):
    """Creates a viewer of a dense matrix.
    
    Parameters
    ----------
    mat : np.ndarray
        The dense matrix to view.
    method : str, optional
        Method type. It must be one of the following:

            - ``"naive"``: naive method.
            - ``"cov"``: covariance method.

        Default is ``"naive"``.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.

    Returns
    -------
    wrap
        Wrapper matrix object.

    See Also
    --------
    adelie.matrix.MatrixCovBase64
    adelie.matrix.MatrixNaiveBase64
    """
    naive_dispatcher = {
        np.dtype("float64"): {
            "C": core.matrix.MatrixNaiveDense64C,
            "F": core.matrix.MatrixNaiveDense64F,
        },
        np.dtype("float32"): {
            "C": core.matrix.MatrixNaiveDense32C,
            "F": core.matrix.MatrixNaiveDense32F,
        },
    }

    cov_dispatcher = {
        np.dtype("float64"): {
            "C": core.matrix.MatrixCovDense64C,
            "F": core.matrix.MatrixCovDense64F,
        },
        np.dtype("float32"): {
            "C": core.matrix.MatrixCovDense32C,
            "F": core.matrix.MatrixCovDense32F,
        },
    }

    dispatcher = {
        "naive" : naive_dispatcher,
        "cov" : cov_dispatcher,
    }

    dtype = mat.dtype
    order = (
        "F"
        # prioritize choosing Fortran contiguity
        if mat.flags.f_contiguous else
        "C"
    )
    if order == "C":
        warnings.warn(
            "Detected matrix to be C-contiguous. "
            "Performance may improve with F-contiguous matrix."
        )
    core_base = dispatcher[method][dtype][order]

    class _dense(core_base):
        def __init__(self):
            self.mat = mat
            core_base.__init__(self, self.mat, n_threads)

    return _dense()


def kronecker_eye(
    mat: Union[np.ndarray, MatrixNaiveBase32, MatrixNaiveBase64],
    K: int,
    *,
    n_threads: int =1,
):
    """Creates a viewer of a matrix Kronecker product identity matrix.

    The matrix is represented as :math:`X \\otimes I_K`
    where :math:`X` is the underlying dense matrix and 
    :math:`I_K` is the identity matrix of dimension :math:`K`.

    .. note::
        This matrix only works for naive method!
    
    Parameters
    ----------
    mat : Union[np.ndarray, MatrixNaiveBase32, MatrixNaiveBase64]
        The matrix to view as a Kronecker product with identity matrix.
        If ``np.ndarray``, a specialized class is created with more optimized routines.
    K : int
        Dimension of the identity matrix.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.

    Returns
    -------
    wrap
        Wrapper matrix object.

    See Also
    --------
    adelie.matrix.MatrixNaiveBase64
    """
    if isinstance(mat, np.ndarray):
        dispatcher = {
            np.dtype("float64"): {
                "C": core.matrix.MatrixNaiveKroneckerEyeDense64C,
                "F": core.matrix.MatrixNaiveKroneckerEyeDense64F,
            },
            np.dtype("float32"): {
                "C": core.matrix.MatrixNaiveKroneckerEyeDense32C,
                "F": core.matrix.MatrixNaiveKroneckerEyeDense32F,
            },
        }
        dtype = mat.dtype
        order = (
            "C"
            if mat.flags.c_contiguous else
            "F"
        )
        core_base = dispatcher[dtype][order]
    else:
        dispatcher = {
            np.float64: core.matrix.MatrixNaiveKroneckerEye64,
            np.float32: core.matrix.MatrixNaiveKroneckerEye32,
        }
        dtype = _to_dtype(mat)
        core_base = dispatcher[dtype]

    class _kronecker_eye(core_base):
        def __init__(self):
            self.mat = mat
            core_base.__init__(self, self.mat, K, n_threads)

    return _kronecker_eye()


def lazy_cov(
    mat: np.ndarray,
    *,
    n_threads: int =1,
):
    """Creates a viewer of a lazy covariance matrix.

    The lazy covariance matrix :math:`A` uses 
    the underlying matrix :math:`X` given by ``mat``
    to compute the values of :math:`A` dynamically.
    It only computes rows of :math:`A`
    on-the-fly that are needed when calling its member functions.
    This is useful in ``adelie.solver.gaussian_cov`` where
    the covariance method must be used but the dimensions of :math:`A`
    is too large to construct the entire matrix as a dense matrix.
    
    .. note::
        This matrix only works for covariance method!

    Parameters
    ----------
    mat : (n, p) np.ndarray
        The data matrix from which to lazily compute the covariance.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.

    Returns
    -------
    wrap
        Wrapper matrix object.

    See Also
    --------
    adelie.matrix.MatrixCovBase64
    """
    dispatcher = {
        np.dtype("float64"): {
            "C": core.matrix.MatrixCovLazyCov64C,
            "F": core.matrix.MatrixCovLazyCov64F,
        },
        np.dtype("float32"): {
            "C": core.matrix.MatrixCovLazyCov32C,
            "F": core.matrix.MatrixCovLazyCov32F,
        },
    }

    dtype = mat.dtype
    order = (
        "C"
        if mat.flags.c_contiguous else
        "F"
    )
    core_base = dispatcher[dtype][order]

    class _lazy_cov(core_base):
        def __init__(self):
            self.mat = mat
            core_base.__init__(self, self.mat, n_threads)

    return _lazy_cov()


def snp_phased_ancestry(
    filename: str,
    *,
    read_mode: str ="auto",
    n_threads: int =1,
    dtype: Union[np.float32, np.float64] =np.float64,
):
    """Creates a SNP phased ancestry matrix.

    The SNP phased ancestry matrix is represented by a file with name ``filename``.
    It must be in the same format as described in ``adelie.io.snp_phased_ancestry``.
    Typically, the user first writes into the file ``filename`` 
    using ``adelie.io.snp_phased_ancestry`` and then loads the matrix using this function.

    .. note::
        This matrix only works for naive method!
    
    Parameters
    ----------
    filename : str
        File name that contains phased calldata with ancestry information in ``.snpdat`` format.
    read_mode : str, optional
        See the corresponding parameter in ``adelie.io.snp_phased_ancestry``.
        Default is ``"auto"``.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.
    dtype : Union[np.float32, np.float64], optional
        Underlying value type.
        Default is ``np.float64``.

    Returns
    -------
    wrap
        Wrapper matrix object.

    See Also
    --------
    adelie.io.snp_phased_ancestry
    adelie.matrix.MatrixNaiveBase64
    """
    dispatcher = {
        np.float64: core.matrix.MatrixNaiveSNPPhasedAncestry64,
        np.float32: core.matrix.MatrixNaiveSNPPhasedAncestry32,
    }
    core_base = dispatcher[dtype]

    class _snp_phased_ancestry(core_base):
        def __init__(self):
            core_base.__init__(self, filename, read_mode, n_threads)

    return _snp_phased_ancestry()


def snp_unphased(
    filename: str,
    *,
    read_mode: str ="auto",
    n_threads: int =1,
    dtype: Union[np.float32, np.float64] =np.float64,
):
    """Creates a SNP unphased matrix.

    The SNP unphased matrix is represented by a file with name ``filename``.
    It must be in the same format as described in ``adelie.io.snp_unphased``.
    Typically, the user first writes into the file ``filename`` 
    using ``adelie.io.snp_unphased`` and then loads the matrix using this function.

    .. note::
        This matrix only works for naive method!
    
    Parameters
    ----------
    filename : str
        File name that contains unphased calldata in ``.snpdat`` format.
    read_mode : str, optional
        See the corresponding parameter in ``adelie.io.snp_unphased``.
        Default is ``"auto"``.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.
    dtype : Union[np.float32, np.float64], optional
        Underlying value type.
        Default is ``np.float64``.

    Returns
    -------
    wrap
        Wrapper matrix object.

    See Also
    --------
    adelie.io.snp_unphased
    adelie.matrix.MatrixNaiveBase64
    """
    dispatcher = {
        np.float64: core.matrix.MatrixNaiveSNPUnphased64,
        np.float32: core.matrix.MatrixNaiveSNPUnphased32,
    }
    core_base = dispatcher[dtype]

    class _snp_unphased(core_base):
        def __init__(self):
            core_base.__init__(self, filename, read_mode, n_threads)

    return _snp_unphased()


def sparse(
    mat: Union[csc_matrix, csr_matrix],
    *,
    method: str ="naive",
    n_threads: int =1,
):
    """Creates a viewer of a sparse matrix.

    .. note::
        Regardless of the storage order of the input matrix,
        we internally convert the matrix to CSC order for performance reasons.
    
    Parameters
    ----------
    mat : Union[scipy.sparse.csc_matrix, scipy.sparse.csr_matrix]
        The sparse matrix to view.
    method : str, optional
        Method type. It must be one of the following:

            - ``"naive"``: naive method.
            - ``"cov"``: covariance method.

        Default is ``"naive"``.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.

    Returns
    -------
    wrap
        Wrapper matrix object.

    See Also
    --------
    adelie.matrix.MatrixCovBase64
    adelie.matrix.MatrixNaiveBase64
    """
    if not (isinstance(mat, csr_matrix) or isinstance(mat, csc_matrix)):
        raise TypeError("mat must be scipy.sparse.csr_matrix or scipy.sparse.csc_matrix.")

    mat.prune()
    mat.sort_indices()

    if isinstance(mat, csr_matrix):
        warnings.warn("Converting to CSC format.")
        mat = mat.tocsc(copy=True)

    naive_dispatcher = {
        np.dtype("float64"): core.matrix.MatrixNaiveSparse64F,
        np.dtype("float32"): core.matrix.MatrixNaiveSparse32F,
    }
    
    cov_dispatcher = {
        np.dtype("float64"): core.matrix.MatrixCovSparse64F,
        np.dtype("float32"): core.matrix.MatrixCovSparse32F,
    }

    dispatcher = {
        "naive" : naive_dispatcher,
        "cov" : cov_dispatcher,
    }

    dtype = mat.dtype
    core_base = dispatcher[method][dtype]

    class _sparse(core_base):
        def __init__(self):
            self.mat = mat
            core_base.__init__(
                self, 
                self.mat.shape[0], 
                self.mat.shape[1], 
                self.mat.nnz,
                self.mat.indptr,
                self.mat.indices,
                self.mat.data,
                n_threads,
            )

    return _sparse()
