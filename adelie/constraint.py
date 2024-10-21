from . import adelie_core as core
from . import matrix 
from .adelie_core.constraint import (
    ConstraintBase32,
    ConstraintBase64,
)
from .configs import Configs
from .glm import _coerce_dtype
from .matrix import (
    MatrixConstraintBase32,
    MatrixConstraintBase64,
)
from scipy.sparse import csr_matrix
from typing import Union
import numpy as np


def box(
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    method: str ="proximal-newton",
    configs: dict =None,
    dtype: Union[np.float32, np.float64] =None,
):
    """Creates a box constraint.

    The box constraint is given by 
    :math:`\\ell \\leq x \\leq u` where 
    :math:`\\ell \\leq 0 \\leq u`.

    Parameters
    ----------
    lower : (d,) ndarray
        Lower bound :math:`\\ell`.
    upper : (d,) ndarray
        Upper bound :math:`u`.
    method : str, optional
        Method for :func:`~adelie.adelie_core.constraint.ConstraintBase64.solve`.
        It must be one of the following:

            - ``"proximal-newton"``: proximal Newton algorithm.

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
                    Default is ``1e-9``.
                pinball_max_iters : int, optional
                    Maximum number of coordinate descent iterations for the pinball least squares solver.
                    Default is ``int(1e5)``.
                pinball_tol : float, optional
                    Convergence tolerance for the pinball least squares solver.
                    Default is ``1e-7``.
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

        If ``None``, the default values are used.
        Default is ``None``.
    dtype : Union[float32, float64], optional
        The underlying data type.
        If ``None``, it is inferred from ``lower`` or ``upper``,
        in which case one of them must have an underlying data type of
        :class:`numpy.float32` or :class:`numpy.float64`.
        Default is ``None``.

    Returns
    -------
    wrap
        Wrapper constraint object.

    See Also
    --------
    adelie.adelie_core.constraint.ConstraintBox32
    adelie.adelie_core.constraint.ConstraintBox64
    """
    lower, l_dtype = _coerce_dtype(lower, dtype)
    upper, u_dtype = _coerce_dtype(upper, dtype)
    assert l_dtype == u_dtype
    dtype = l_dtype

    lower = np.minimum(-lower, Configs.max_solver_value)
    upper = np.minimum(upper, Configs.max_solver_value)

    core_base = {
        "proximal-newton": {
            np.float32: core.constraint.ConstraintBox32,
            np.float64: core.constraint.ConstraintBox64,
        },
    }[method][dtype]

    user_configs = configs
    configs = {
        "proximal-newton": {
            "max_iters": 100,
            "tol": 1e-9,
            "pinball_max_iters": int(1e5),
            "pinball_tol": 1e-7,
            "slack": 1e-4,
        },
    }[method]
    if not (user_configs is None):
        for key, val in user_configs.items():
            configs[key] = val

    class _box(core_base):
        def __init__(self):
            self._lower = np.array(lower, dtype=dtype)
            self._upper = np.array(upper, dtype=dtype)
            core_base.__init__(
                self, 
                lower=self._lower,
                upper=self._upper,
                **configs,
            )
        
    return _box()


def linear(
    A: Union[np.ndarray, csr_matrix, MatrixConstraintBase32, MatrixConstraintBase64],
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    vars: np.ndarray =None,
    copy: bool =False,
    method: str ="proximal-newton",
    configs: dict =None,
    dtype: Union[np.float32, np.float64] =None,
):
    """Creates a linear constraint.

    The linear constraint is given by :math:`\\ell \\leq A x \\leq u`
    where :math:`\\ell \\leq 0 \\leq u`.
    This constraint object is intended to support general linear constraints.
    If :math:`A` is structured (e.g. box constraint),
    it is more efficient to use specialized constraint types.

    Parameters
    ----------
    A : (m, d) Union[ndarray, csr_matrix, MatrixConstraintBase32, MatrixConstraintBase64]
        Constraint matrix :math:`A`.
    lower : (m,) ndarray
        Lower bound :math:`\\ell`.
    upper : (m,) ndarray
        Upper bound :math:`u`.
    vars : ndarray, optional
        Equivalent to :math:`\\mathrm{diag}(AA^\\top)`.
        If ``None`` and ``A`` is ``ndarray`` or ``csr_matrix``, it is computed internally.
        Otherwise, it must be explicitly provided by the user.
        Default is ``None``.
    copy : bool, optional
        If ``True``, a copy of the inputs are stored internally.
        Otherwise, a reference is stored instead.
        Default is ``False``.
    method : str, optional
        Method for :func:`~adelie.adelie_core.constraint.ConstraintBase64.solve`.
        It must be one of the following:

            - ``"proximal-newton"``: proximal Newton algorithm.

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
                    Default is ``1e-9``.
                nnls_max_iters : int, optional
                    Maximum number of coordinate descent iterations for the non-negative least squares solver.
                    Default is ``int(1e5)``.
                nnls_tol : float, optional
                    Convergence tolerance for the non-negative least squares solver.
                    Default is ``1e-7``.
                pinball_max_iters : int, optional
                    Maximum number of coordinate descent iterations for the pinball least squares solver.
                    Default is ``int(1e5)``.
                pinball_tol : float, optional
                    Convergence tolerance for the pinball least squares solver.
                    Default is ``1e-7``.
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
                n_threads : int, optional
                    Number of threads.
                    Default is ``1``.

        If ``None``, the default values are used.
        Default is ``None``.
    dtype : Union[float32, float64], optional
        The underlying data type.
        If ``None``, it is inferred from ``A``, ``lower``, or ``upper``,
        in which case one of them must have an underlying data type of
        :class:`numpy.float32` or :class:`numpy.float64`.
        Default is ``None``.

    Returns
    -------
    wrap
        Wrapper constraint object.

    See Also
    --------
    adelie.adelie_core.constraint.ConstraintLinear32
    adelie.adelie_core.constraint.ConstraintLinear64
    """
    if isinstance(A, np.ndarray):
        if vars is None:
            vars = np.sum(A ** 2, axis=1)

        A, _ = _coerce_dtype(A, dtype)
        A = matrix.dense(A, method="constraint", copy=copy)
    elif isinstance(A, csr_matrix):
        if vars is None:
            vars = (A ** 2).sum(axis=1)

        A = matrix.sparse(A, method="constraint", copy=copy)
    else:
        assert not (vars is None)

    A_dtype = (
        np.float32
        if isinstance(A, MatrixConstraintBase32) else
        np.float64
    )

    lower, l_dtype = _coerce_dtype(lower, dtype)
    upper, u_dtype = _coerce_dtype(upper, dtype)
    assert A_dtype == l_dtype
    assert A_dtype == u_dtype
    dtype = A_dtype

    lower = np.minimum(-lower, Configs.max_solver_value)
    upper = np.minimum(upper, Configs.max_solver_value)

    core_base = {
        "proximal-newton": {
            np.float32: core.constraint.ConstraintLinear32,
            np.float64: core.constraint.ConstraintLinear64,
        },
    }[method][dtype]

    user_configs = configs
    configs = {
        "proximal-newton": {
            "max_iters": 100,
            "tol": 1e-9,
            "nnls_max_iters": int(1e5),
            "nnls_tol": 1e-7,
            "pinball_max_iters": int(1e5),
            "pinball_tol": 1e-7,
            "slack": 1e-4,
            "n_threads": 1,
        },
    }[method]
    if not (user_configs is None):
        for key, val in user_configs.items():
            configs[key] = val

    class _linear(core_base):
        def __init__(self):
            self._A = A
            self._lower = np.array(lower, dtype=dtype)
            self._upper = np.array(upper, dtype=dtype)
            self._vars = np.array(vars, copy=copy, dtype=dtype)
            core_base.__init__(
                self,
                A=self._A,
                lower=self._lower,
                upper=self._upper,
                A_vars=self._vars,
                **configs,
            )
    
    return _linear()


def lower(
    b: np.ndarray,
    **kwargs,
):
    """Creates a lower bound constraint.

    The lower bound constraint is given by :math:`x \\geq b` where :math:`b \\leq 0`.

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
        b=-b, 
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
    :math:`D` is a diagonal matrix with :math:`\\pm 1` along the diagonal
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
                    Default is ``1e-9``.
                pinball_max_iters : int, optional
                    Maximum number of coordinate descent iterations for the pinball least squares solver.
                    Default is ``int(1e5)``.
                pinball_tol : float, optional
                    Convergence tolerance for the pinball least squares solver.
                    Default is ``1e-7``.
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
                    Default is ``int(1e5)``.
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
    adelie.adelie_core.constraint.ConstraintOneSided32
    adelie.adelie_core.constraint.ConstraintOneSided64
    """
    b, dtype = _coerce_dtype(b, dtype)
    b = np.minimum(b, Configs.max_solver_value)

    core_base = {
        "proximal-newton": {
            np.float32: core.constraint.ConstraintOneSided32,
            np.float64: core.constraint.ConstraintOneSided64,
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
            "tol": 1e-9,
            "pinball_max_iters": int(1e5),
            "pinball_tol": 1e-7,
            "slack": 1e-4,
        },
        "admm": {
            "max_iters": int(1e5),
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
            self._D = np.array(D, dtype=dtype)
            self._b = np.array(b, dtype=dtype)
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