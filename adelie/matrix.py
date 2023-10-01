from . import adelie_core as core
from .adelie_core.matrix import (
    Base64,
    Base32,
)
import numpy as np


class base:
    pass


class dense(base):
    """Creates a viewer of a dense matrix.
    
    Parameters
    ----------
    mat : np.ndarray
        The matrix to view. A copy is made if ``dtype``
        is inconsistent with that of ``mat`` or the storage order is not column-major. 
        See documentation for ``numpy.array`` for details
        on when exactly a copy is made.
    dtype : Union[np.float64, np.float32], optional
        The underlying data type.
        Default is ``np.float64``.
    """
    def __init__(
        self,
        mat: np.ndarray,
        *,
        dtype: np.float64 | np.float32 =np.float64,
    ):
        self.mat = np.array(mat, copy=False, dtype=dtype, order="F")
        dispatcher = {
            np.float64: core.matrix.Dense64,
            np.float32: core.matrix.Dense32,
        }
        self._core_mat = dispatcher[dtype](self.mat)
