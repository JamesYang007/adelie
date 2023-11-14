from . import adelie_core as core
from .adelie_core.matrix import (
    MatrixNaiveBase64,
    MatrixNaiveBase32,
    MatrixCovBase64,
    MatrixCovBase32,
)
import numpy as np


def dense(
    mat: np.ndarray,
    *,
    method: str,
    n_threads: int =1,
):
    """Creates a viewer of a dense matrix.
    
    Parameters
    ----------
    mat : np.ndarray
        The matrix to view.
    method : str
        Method type. It must be one of the following:

            - ``"naive"``: naive method.
            - ``"cov"``: covariance method.

    n_threads : int, optional
        Number of threads.
        Default is ``1``.

    Returns
    -------
    wrap
        Wrapper matrix object.
    """
    if n_threads < 1:
        raise ValueError("Number of threads must be >= 1.")

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
        "C"
        if mat.flags.c_contiguous else
        "F"
    )
    core_base = dispatcher[method][dtype][order]

    class _dense(core_base):
        def __init__(
            self,
            mat: np.ndarray,
            n_threads: int =1,
        ):
            self.mat = mat
            core_base.__init__(self, self.mat, n_threads)

    return _dense(mat, n_threads)


def snp_unphased(
    filenames: list,
    *,
    n_threads: int =1,
    dtype: np.float32 | np.float64 =np.float64,
):
    """Creates a SNP unphased matrix.

    .. note::
        This matrix only works for naive method!
    
    Parameters
    ----------
    filenames : list
        List of file names that contain column-block slices of the matrix.
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
    """
    if n_threads < 1:
        raise ValueError("Number of threads must be >= 1.")

    dispatcher = {
        np.float64: core.matrix.MatrixNaiveSNPUnphased64,
        np.float32: core.matrix.MatrixNaiveSNPUnphased32,
    }
    core_base = dispatcher[dtype]

    class _snp_unphased(core_base):
        def __init__(
            self,
            filenames: list,
            n_threads: int,
        ):
            core_base.__init__(self, filenames, n_threads)

    return _snp_unphased(filenames, n_threads)


def snp_phased_ancestry(
    filenames: list,
    *,
    n_threads: int =1,
    dtype: np.float32 | np.float64 =np.float64,
):
    """Creates a SNP phased ancestry matrix.

    .. note::
        This matrix only works for naive method!
    
    Parameters
    ----------
    filenames : list
        List of file names that contain column-block slices of the matrix.
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
    """
    if n_threads < 1:
        raise ValueError("Number of threads must be >= 1.")

    dispatcher = {
        np.float64: core.matrix.MatrixNaiveSNPPhasedAncestry64,
        np.float32: core.matrix.MatrixNaiveSNPPhasedAncestry32,
    }
    core_base = dispatcher[dtype]

    class _snp_phased_ancestry(core_base):
        def __init__(
            self,
            filenames: list,
            n_threads: int,
        ):
            core_base.__init__(self, filenames, n_threads)

    return _snp_phased_ancestry(filenames, n_threads)


def cov_lazy(
    mat: np.ndarray,
    *,
    n_threads: int =1,
):
    """Creates a viewer of a lazy covariance matrix.
    
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
    """
    if n_threads < 1:
        raise ValueError("Number of threads must be >= 1.")

    dispatcher = {
        np.dtype("float64"): {
            "C": core.matrix.MatrixCovLazy64C,
            "F": core.matrix.MatrixCovLazy64F,
        },
        np.dtype("float32"): {
            "C": core.matrix.MatrixCovLazy32C,
            "F": core.matrix.MatrixCovLazy32F,
        },
    }

    dtype = mat.dtype
    order = (
        "C"
        if mat.flags.c_contiguous else
        "F"
    )
    core_base = dispatcher[dtype][order]

    class _cov_lazy(core_base):
        def __init__(
            self,
            mat: np.ndarray,
            n_threads: int =1,
        ):
            self.mat = mat
            core_base.__init__(self, self.mat, n_threads)

    return _cov_lazy(mat, n_threads)