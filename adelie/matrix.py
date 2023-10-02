from . import adelie_core as core
from .adelie_core.matrix import (
    Base64,
    Base32,
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


class dense(base):
    """Creates a viewer of a dense matrix.
    
    Parameters
    ----------
    mat : np.ndarray
        The matrix to view.
    """
    def __init__(
        self,
        mat: np.ndarray,
    ):
        self.mat = mat
        dispatcher = {
            np.dtype("float64"): {
                "C": core.matrix.Dense64C,
                "F": core.matrix.Dense64F,
            },
            np.dtype("float32"): {
                "C": core.matrix.Dense32C,
                "F": core.matrix.Dense32F,
            },
        }

        dtype = self.mat.dtype
        order = (
            "C"
            if self.mat.flags.c_contiguous else
            "F"
        )
        super().__init__(dispatcher[dtype][order](self.mat))
