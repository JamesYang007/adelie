from .adelie_core.constraint import (
    ConstraintBase32,
    ConstraintBase64,
)
from .configs import Configs
from .glm import _coerce_dtype
from . import adelie_core as core
from typing import Union
import numpy as np


def lower(
    b: np.ndarray,
    *,
    max_iters: int =100,
    tol: float =1e-7,
    nnls_max_iters: int =10000,
    nnls_tol: float =1e-7,
    slack: float =1e-7,
    dtype: Union[np.float32, np.float64] =None,
):
    """Creates a lower bound constraint.

    The lower bound constraint is given by :math:`x \\geq -b` where :math:`b \\geq 0`.

    Parameters
    ----------
    b : (d,) np.ndarray
        Bound :math:`b`.
    max_iters : int, optional
        Maximum number of proximal Newton iterations.
        Default is ``100``.
    tol : float, optional
        Convergence tolerance for proximal Newton.
        Default is ``1e-7``.
    nnls_max_iters : int, optional
        Maximum number of non-negative least squares iterations.
        Default is ``10000``.
    nnls_tol : float, optional
        Maximum number of non-negative least squares iterations.
        Default is ``1e-7``.
    slack : float, optional
        Slackness for backtracking when proximal Newton overshoots
        the boundary where primal is zero.
        The smaller the value, the less slack so that the
        backtrack takes the iterates closer to (but outside) the boundary.

        .. warning::
            If this value is too small, ``solve()`` may not converge!

        Default is ``1e-7``.
    dtype : Union[np.float32, np.float64], optional
        The underlying data type.
        If ``None``, it is inferred from ``b``,
        in which case ``b`` must have an underlying data type of
        ``np.float32`` or ``np.float64``.
        Default is ``None``.

    Returns
    -------
    wrap
        Wrapper constraint object.

    See Also
    --------
    adelie.constraint.one_sided
    """
    D = np.full(b.shape[0], -1.0)
    return one_sided(
        D=D, 
        b=b, 
        max_iters=max_iters,
        tol=tol,
        nnls_max_iters=nnls_max_iters,
        nnls_tol=nnls_tol,
        slack=slack,
        dtype=dtype,
    )


def one_sided(
    D: np.ndarray,
    b: np.ndarray,
    *,
    max_iters: int =100,
    tol: float =1e-7,
    nnls_max_iters: int =10000,
    nnls_tol: float =1e-7,
    slack: float =1e-7,
    dtype: Union[np.float32, np.float64] =None,
):
    """Creates a one-sided bound constraint.

    The one-sided bound constraint is given by 
    :math:`D x \\leq b` where 
    :math:`D` is a diagonal matrix with :math:`\\pm 1` along the diagonal,
    and :math:`b \\geq 0`.

    Parameters
    ----------
    D : (d,) np.ndarray
        Diagonal matrix :math:`D`.
    b : (d,) np.ndarray
        Bound :math:`b`.
    max_iters : int, optional
        Maximum number of proximal Newton iterations.
        Default is ``100``.
    tol : float, optional
        Convergence tolerance for proximal Newton.
        Default is ``1e-7``.
    nnls_max_iters : int, optional
        Maximum number of non-negative least squares iterations.
        Default is ``10000``.
    nnls_tol : float, optional
        Maximum number of non-negative least squares iterations.
        Default is ``1e-7``.
    slack : float, optional
        Slackness for backtracking when proximal Newton overshoots
        the boundary where primal is zero.
        The smaller the value, the less slack so that the
        backtrack takes the iterates closer to (but outside) the boundary.

        .. warning::
            If this value is too small, ``solve()`` may not converge!

        Default is ``1e-7``.
    dtype : Union[np.float32, np.float64], optional
        The underlying data type.
        If ``None``, it is inferred from ``b``,
        in which case ``b`` must have an underlying data type of
        ``np.float32`` or ``np.float64``.
        Default is ``None``.

    Returns
    -------
    wrap
        Wrapper constraint object.

    See Also
    --------
    adelie.adelie_core.constraint.ConstraintOneSided64
    """
    b, dtype = _coerce_dtype(b, dtype)
    b = np.minimum(b, Configs.max_solver_value)

    core_base = {
        np.float32: core.constraint.ConstraintOneSided32,
        np.float64: core.constraint.ConstraintOneSided64,
    }[dtype]

    class _one_sided(core_base):
        def __init__(self):
            self._D = np.array(D, copy=True, dtype=dtype)
            self._b = np.array(b, copy=True, dtype=dtype)
            core_base.__init__(
                self, 
                sgn=self._D,
                b=self._b, 
                max_iters=max_iters, 
                tol=tol, 
                nnls_max_iters=nnls_max_iters, 
                nnls_tol=nnls_tol, 
                slack=slack,
            )
        
    return _one_sided()
