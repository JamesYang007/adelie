from . import adelie_core as core
from .adelie_core.matrix import (
    MatrixNaiveBase64,
    MatrixNaiveBase32,
    MatrixCovBase64,
    MatrixCovBase32,
)
import numpy as np


class base:
    """Base matrix wrapper class.

    All Python matrix classes must inherit from this class.

    Parameters
    ----------
    core_mat
        Usually a C++ matrix object.
    """
    def __init__(self, core_mat):
        self._core_mat = core_mat

    def internal(self):
        """Returns the core matrix object."""
        return self._core_mat


class naive_dense(base):
    """Creates a viewer of a dense matrix for naive method.
    
    Parameters
    ----------
    mat : np.ndarray
        The matrix to view.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.
    """
    def __init__(
        self,
        mat: np.ndarray,
        n_threads: int =1,
    ):
        if n_threads < 1:
            raise ValueError("Number of threads must be >= 1.")
        self.mat = mat
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

        dtype = self.mat.dtype
        order = (
            "C"
            if self.mat.flags.c_contiguous else
            "F"
        )
        super().__init__(dispatcher[dtype][order](self.mat, n_threads))


class cov_dense(base):
    """Creates a viewer of a dense matrix for covariance method.
    
    Parameters
    ----------
    mat : np.ndarray
        The matrix to view.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.
    """
    def __init__(
        self,
        mat: np.ndarray,
        n_threads: int =1,
    ):
        if n_threads < 1:
            raise ValueError("Number of threads must be >= 1.")
        self.mat = mat
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

        dtype = self.mat.dtype
        order = (
            "C"
            if self.mat.flags.c_contiguous else
            "F"
        )
        super().__init__(dispatcher[dtype][order](self.mat, n_threads))


class cov_lazy(base):
    """Creates a viewer of a lazy matrix for covariance method.
    
    Parameters
    ----------
    mat : np.ndarray
        The data matrix from which to lazily compute the covariance.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.
    """
    def __init__(
        self,
        mat: np.ndarray,
        n_threads: int =1,
    ):
        if n_threads < 1:
            raise ValueError("Number of threads must be >= 1.")
        self.mat = mat
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

        dtype = self.mat.dtype
        order = (
            "C"
            if self.mat.flags.c_contiguous else
            "F"
        )
        super().__init__(dispatcher[dtype][order](self.mat, n_threads))


class basil_naive_dense(base):
    """Creates a viewer of a dense matrix for basil, naive method.
    
    Parameters
    ----------
    mat : np.ndarray
        The matrix to view.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.
    """
    def __init__(
        self,
        mat: np.ndarray,
        n_threads: int =1,
    ):
        if n_threads < 1:
            raise ValueError("Number of threads must be >= 1.")
        self.mat = mat
        dispatcher = {
            np.dtype("float64"): {
                "C": core.matrix.MatrixBasilNaiveDense64C,
                "F": core.matrix.MatrixBasilNaiveDense64F,
            },
            np.dtype("float32"): {
                "C": core.matrix.MatrixBasilNaiveDense32C,
                "F": core.matrix.MatrixBasilNaiveDense32F,
            },
        }

        dtype = self.mat.dtype
        order = (
            "C"
            if self.mat.flags.c_contiguous else
            "F"
        )
        super().__init__(dispatcher[dtype][order](self.mat, n_threads))
