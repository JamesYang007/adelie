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
    **kwargs,
):
    """Creates a lower bound constraint.

    The lower bound constraint is given by :math:`x \\geq -b` where :math:`b \\geq 0`.

    Parameters
    ----------
    b : (d,) ndarray
        Bound :math:`b`.
    **kwargs : optional
        Keyword arguments to :class:`adelie.constraint.one_sided`.

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
        **kwargs,
    )


def one_sided(
    D: np.ndarray,
    b: np.ndarray,
    *,
    method: str ="proximal-newton",
    configs: dict =None,
    dtype: Union[np.float32, np.float64] =None,
):
    """Creates a one-sided bound constraint.

    The one-sided bound constraint is given by 
    :math:`D x \\leq b` where 
    :math:`D` is a diagonal matrix with :math:`\\pm 1` along the diagonal,
    and :math:`b \\geq 0`.

    Parameters
    ----------
    D : (d,) ndarray
        Diagonal matrix :math:`D`.
    b : (d,) ndarray
        Bound :math:`b`.
    method : str, optional
        Method for :func:`~adelie.adelie_core.constraint.ConstraintBase64.solve`.
        It must be one of the following:

            - ``"proximal-newton"``: proximal Newton algorithm.
            - ``"admm"``: ADMM algorithm.

        Default is ``"proximal-newton"``.
    configs : dict, optional
        Configurations specific to ``method``.
        For each method type, the following arguments are used:

            - ``"proximal-newton"``:
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
                cs_tol : float, optional
                    Complementary slackness tolerance.
                    Default is ``1e-16``.
                slack : float, optional
                    Slackness for backtracking when proximal Newton overshoots
                    the boundary where primal is zero.
                    The smaller the value, the less slack so that the
                    backtrack takes the iterates closer to (but outside) the boundary.

                    .. warning::
                        If this value is too small, 
                        :func:`~adelie.adelie_core.constraint.ConstraintBase64.solve`
                        may not converge!

                    Default is ``1e-4``.

            - ``"admm"``:
                max_iters : int, optional
                    Maximum number of ADMM iterations.
                    Default is ``10000``.
                tol_abs : float, optional
                    Absolute convergence tolerance.
                    Default is ``1e-7``.
                tol_rel : float, optional
                    Relative convergence tolerance.
                    Default is ``1e-7``.
                rho : float, optional
                    ADMM penalty parameter.
                    Default is ``1``.

        If ``None``, the default values are used.
        Default is ``None``.
    dtype : Union[float32, float64], optional
        The underlying data type.
        If ``None``, it is inferred from ``b``,
        in which case ``b`` must have an underlying data type of
        :class:`numpy.float32` or :class:`numpy.float64`.
        Default is ``None``.

    Returns
    -------
    wrap
        Wrapper constraint object.

    See Also
    --------
    adelie.adelie_core.constraint.ConstraintOneSidedADMM32
    adelie.adelie_core.constraint.ConstraintOneSidedADMM64
    adelie.adelie_core.constraint.ConstraintOneSidedProximalNewton32
    adelie.adelie_core.constraint.ConstraintOneSidedProximalNewton64
    """
    b, dtype = _coerce_dtype(b, dtype)
    b = np.minimum(b, Configs.max_solver_value)

    core_base = {
        "proximal-newton": {
            np.float32: core.constraint.ConstraintOneSidedProximalNewton32,
            np.float64: core.constraint.ConstraintOneSidedProximalNewton64,
        },
        "admm": {
            np.float32: core.constraint.ConstraintOneSidedADMM32,
            np.float64: core.constraint.ConstraintOneSidedADMM64,
        },
    }[method][dtype]

    user_configs = configs
    configs = {
        "proximal-newton": {
            "max_iters": 100,
            "tol": 1e-7,
            "nnls_max_iters": 10000,
            "nnls_tol": 1e-7,
            "cs_tol": 1e-16,
            "slack": 1e-4,
        },
        "admm": {
            "max_iters": 10000,
            "tol_abs": 1e-7,
            "tol_rel": 1e-7,
            "rho": 1,
        },
    }[method]
    if not (user_configs is None):
        for key, val in user_configs.items():
            configs[key] = val

    class _one_sided(core_base):
        def __init__(self):
            self._D = np.array(D, copy=True, dtype=dtype)
            self._b = np.array(b, copy=True, dtype=dtype)
            core_base.__init__(
                self, 
                sgn=self._D,
                b=self._b, 
                **configs,
            )
        
    return _one_sided()


def upper(
    b: np.ndarray,
    **kwargs,
):
    """Creates an upper bound constraint.

    The upper bound constraint is given by :math:`x \\leq b` where :math:`b \\geq 0`.

    Parameters
    ----------
    b : (d,) ndarray
        Bound :math:`b`.
    **kwargs : optional
        Keyword arguments to :class:`adelie.constraint.one_sided`.

    Returns
    -------
    wrap
        Wrapper constraint object.

    See Also
    --------
    adelie.constraint.one_sided
    """
    D = np.full(b.shape[0], 1.0)
    return one_sided(
        D=D, 
        b=b, 
        **kwargs,
    )