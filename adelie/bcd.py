from . import adelie_core as core
import numpy as np


def root_lower_bound(
    *,
    quad: np.ndarray,
    linear: np.ndarray,
    l1: float,
):
    """Computes a lower bound on the root of BCD root function.

    The lower bound :math:`h` is guaranteed to be non-negative
    and satisfies :math:`\\varphi(h) \\geq 0` where :math:`\\varphi`
    is given by ``root_function()`` whenever :math:`\\|v\\|_2 > \\lambda`.

    Parameters
    ----------
    quad : (p,) np.ndarray
        See ``root_function()``.
    linear : (p,) np.ndarray
        See ``root_function()``.
    l1 : float
        See ``root_function()``.
    
    See Also
    --------
    adelie.bcd.root_function

    Returns
    -------
    out : float
        Lower bound on the root.
    """
    return core.bcd.root_lower_bound(quad, linear, l1)


def root_upper_bound(
    *,
    quad: np.ndarray,
    linear: np.ndarray,
    zero_tol: float=1e-10,
):
    """Computes an upper bound on the root of BCD root function.

    The upper bound :math:`h` is guaranteed to be non-negative.
    However, it *may not satisfy* :math:`\\varphi(h) \\leq 0` where :math:`\\varphi`
    is given by ``root_function()`` if ``zero_tol`` is too large.
    Even when ``zero_tol`` is small enough, 
    we assume that :math:`\\|v\\|_2 > \\lambda` and :math:`v_i=0` whenever :math:`\\Sigma_{ii} = 0`.

    Parameters
    ----------
    quad : (p,) np.ndarray
        See ``root_function()``.
    linear : (p,) np.ndarray
        See ``root_function()``.
    zero_tol : float, optional
        A value is considered zero if its absolute value is less than or equal to ``zero_tol``.
        Default is ``1e-10``.
    
    See Also
    --------
    adelie.bcd.root_function

    Returns
    -------
    out : float
        Upper bound on the root.
    """
    return core.bcd.root_upper_bound(quad, linear, zero_tol)


def root_function(
    h: float, 
    *,
    quad: np.ndarray,
    linear: np.ndarray,
    l1: float,
):
    """Computes the BCD root function.

    The BCD root function is given by

    .. math::
        \\begin{align*}
            \\varphi(h) := 
            \\sum\\limits_{i=1}^p \\frac{v_i^2}{(\\Sigma_{ii} h + \\lambda)^2} - 1
        \\end{align*}

    where
    :math:`h \\geq 0`,
    :math:`\\Sigma \\in \\mathbb{R}_+^{p\\times p}` diagonal,
    :math:`v \\in \\mathbb{R}^p`,
    and :math:`\\lambda \\geq 0`.

    Parameters
    ----------
    h : float
        The value at which to evaluate the BCD root function.
    quad : (p,) np.ndarray
        The quadratic component :math:`\\Sigma`.
    linear : (p,) np.ndarray
        The linear component :math:`v`.
    l1 : float
        The :math:`\\ell_1` component :math:`\\lambda`.

    Returns
    -------
    out : float
        The BCD root function value.
    """
    return core.bcd.root_function(h, quad, linear, l1)


def objective(
    beta: np.ndarray,
    *,
    quad: np.ndarray,
    linear: np.ndarray,
    l1: float,
    l2: float,
):
    """Computes the BCD objective.

    The BCD objective is given by

    .. math::
        \\begin{align*}
            \\frac{1}{2} \\beta^\\top \\Sigma \\beta - v^\\top \\beta 
            + \\lambda_1 \\|\\beta\\|_2
            + \\frac{\\lambda_2}{2} \\|\\beta\\|_2^2
        \\end{align*}

    where 
    :math:`\\beta \\in \\mathbb{R}^p`,
    :math:`\\Sigma \\in \\mathbb{R}_+^{p\\times p}` diagonal,
    :math:`v \\in \\mathbb{R}^p`,
    and :math:`\\lambda_i \\geq 0`.

    Parameters
    ----------
    beta : (p,) np.ndarray
        The input value in which the objective is evaluated.
    quad : (p,) np.ndarray
        The quadratic component :math:`\\Sigma`.
    linear : (p,) np.ndarray
        The linear component :math:`v`.
    l1 : float
        The :math:`\\ell_1` component :math:`\\lambda_1`.
    l2 : float
        The :math:`\\ell_2` component :math:`\\lambda_2`.

    Returns
    -------
    out : float
        The BCD objective.
    """
    beta_norm = np.linalg.norm(beta)
    return 0.5 * quad @ beta ** 2 - linear @ beta + l1 * beta_norm + 0.5 * l2 * beta_norm ** 2


_solver_dict = {
    "brent":            core.bcd.brent_solver,
    "newton":           core.bcd.newton_solver,
    "newton_brent":     core.bcd.newton_brent_solver,
    "newton_abs":       core.bcd.newton_abs_solver,
    "newton_abs_debug": core.bcd.newton_abs_debug_solver,
    "ista":             core.bcd.ista_solver,
    "fista":            core.bcd.fista_solver,
    "fista_adares":     core.bcd.fista_adares_solver,
}


def solve(
    *,
    quad: np.ndarray,
    linear: np.ndarray,
    l1: float,
    l2: float,
    tol: float =1e-12,
    max_iters: int =1000,
    solver: str ="newton_abs",
    **kwargs,
):
    """Solves the BCD update.

    The BCD update for the group elastic net is obtained by minimizing
    the BCD objective given in ``objective()``.
    The solution only exists when :math:`\\|v\\|_2 \\geq \\lambda_1`
    and :math:`\\|v_S\\|_2 < \\lambda_1`. where :math:`S` is the subset of indices
    where :math:`\\Sigma_{ii} + \\lambda_2 = 0`.

    Parameters
    ----------
    quad : (p,) np.ndarray
        See ``objective()``.
    linear : (p,) np.ndarray
        See ``objective()``.
    l1 : float
        See ``objective()``.
    l2 : float
        See ``objective()``.
    tol : float, optional
        Convergence tolerance. Default is ``1e-12``.
    max_iters : int, optional
        Max number of iterations. Default is ``1000``.
    solver : str, optional
        Solver type must be one of 
        ``["brent", "newton", "newton_brent", "newton_abs", "newton_abs_debug", "ista", "fista", "fista_adares"]``.
        Default is ``"newton_abs"``.

        .. warning::
            The following methods are known to have poor convergence:
            ``["brent", "ista", "fista", "fista_adares"]``.
    smart_init : bool
        If ``True``, the ABS method is invoked to find a smart initial point before starting Newton's method.
        It is only used when ``solver`` is ``"newton_abs_debug"``.

    Returns
    -------
    out : Dict[str, Any]
        - ``out["beta"]``: solution vector.
        - ``out["iters"]``: number of iterations taken.

    See Also
    --------
    adelie.bcd.objective
    adelie.bcd.root
    """
    return _solver_dict[solver](quad, linear, l1, l2, tol, max_iters, **kwargs)


def root(
    *,
    quad: np.ndarray,
    linear: np.ndarray,
    l1: float,
    l2: float,
    tol: float =1e-12,
    max_iters: int =1000,
    solver: str ="newton_abs",
):
    """Solves the non-negative root of the BCD root function.

    The BCD root function is given in ``root_function()``.
    The non-negative root only exists ``solve()`` is well-defined.

    Parameters
    ----------
    quad : (p,) np.ndarray
        See ``objective()``.
    linear : (p,) np.ndarray
        See ``objective()``.
    l1 : float
        See ``objective()``.
    l2 : float
        See ``objective()``.
    tol : float, optional
        Convergence tolerance. Default is ``1e-12``.
    max_iters : int, optional
        Max number of iterations. Default is ``1000``.
    solver : str, optional
        Solver type must be one of 
        ``["brent", "newton", "newton_brent", "newton_abs", "ista", "fista", "fista_adares"]``.
        Default is ``"newton_abs"``.

        .. warning::
            The following methods are known to have poor convergence:
            ``["brent", "ista", "fista", "fista_adares"]``.

    Returns
    -------
    out : Dict[str, Any]
        - ``out["root"]``: root.
        - ``out["iters"]``: number of iterations taken.

    See Also
    --------
    adelie.bcd.objective
    adelie.bcd.solve
    """
    if (np.linalg.norm(linear) < l1) or \
        (np.linalg.norm(linear[quad + l2 <= 0]) >= l1):
        return {
            "root": None,
            "iters": 0,
        }
    out = solve(
        quad=quad,
        linear=linear,
        l1=l1,
        l2=l2,
        tol=tol,
        max_iters=max_iters,
        solver=solver,
    )
    return {
        "root": np.linalg.norm(out["beta"]),
        "iters": out["iters"],
    }