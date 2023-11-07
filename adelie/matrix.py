from . import adelie_core as core
from .adelie_core.matrix import (
    MatrixNaiveBase64,
    MatrixNaiveBase32,
    MatrixCovBase64,
    MatrixCovBase32,
)
import numpy as np


def naive_dense(
    mat: np.ndarray,
    *,
    n_threads: int =1,
):
    """Creates a viewer of a dense matrix for naive method.
    
    Parameters
    ----------
    mat : np.ndarray
        The matrix to view.
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
            "C": core.matrix.MatrixNaiveDense64C,
            "F": core.matrix.MatrixNaiveDense64F,
        },
        np.dtype("float32"): {
            "C": core.matrix.MatrixNaiveDense32C,
            "F": core.matrix.MatrixNaiveDense32F,
        },
    }
    dtype = mat.dtype
    order = (
        "C"
        if mat.flags.c_contiguous else
        "F"
    )
    core_base = dispatcher[dtype][order]

    class _naive_dense(core_base):
        def __init__(
            self,
            mat: np.ndarray,
            n_threads: int =1,
        ):
            self.mat = mat
            core_base.__init__(self, self.mat, n_threads)

    return _naive_dense(mat, n_threads)


def cov_dense(
    mat: np.ndarray,
    *,
    n_threads: int =1,
):
    """Creates a viewer of a dense matrix for covariance method.
    
    Parameters
    ----------
    mat : np.ndarray
        The matrix to view.
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
            "C": core.matrix.MatrixCovDense64C,
            "F": core.matrix.MatrixCovDense64F,
        },
        np.dtype("float32"): {
            "C": core.matrix.MatrixCovDense32C,
            "F": core.matrix.MatrixCovDense32F,
        },
    }

    dtype = mat.dtype
    order = (
        "C"
        if mat.flags.c_contiguous else
        "F"
    )
    core_base = dispatcher[dtype][order]

    class _cov_dense(core_base):
        def __init__(
            self,
            mat: np.ndarray,
            n_threads: int =1,
        ):
            self.mat = mat
            core_base.__init__(self, self.mat, n_threads)

    return _cov_dense(mat, n_threads)


def cov_lazy(
    mat: np.ndarray,
    *,
    n_threads: int =1,
):
    """Creates a viewer of a lazy matrix for covariance method.
    
    Parameters
    ----------
    mat : np.ndarray
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