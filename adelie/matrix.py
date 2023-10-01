from . import adelie_core as core
from .adelie_core.matrix import (
    Base64F,
    Base64C,
    Base32F,
    Base32C,
)
import numpy as np


class dense:
    """Creates a viewer of a dense matrix.
    
    Parameters
    ----------
    mat : np.ndarray
    dtype : Union[np.float64, np.float32], optional
        The underlying data type.
        Default is ``np.float64``.
    order : str, optional
        The storage order.
        It must be one of the following (using ``numpy`` convention): ``["C", "F"]``.
        Default is ``"F"``.
    """
    def __init__(
        self,
        mat: np.ndarray,
        dtype: np.float64 | np.float32 =np.float64,
        order: str ="F",
    ):
        self.mat = np.array(mat, copy=False, dtype=dtype, order=order)
        dispatcher = {
            np.float64: {
                "F": core.matrix.Dense64F,
                "C": core.matrix.Dense64C,
            },
            np.float32: {
                "F": core.matrix.Dense32F,
                "C": core.matrix.Dense32C,
            },
        }
        self._core_mat = dispatcher[dtype][order](self.mat)
