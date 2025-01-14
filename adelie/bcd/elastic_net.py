from ..adelie_core.bcd import elastic_net as core
import numpy as np


def root_lower_bound(
    *,
    quad: np.ndarray,
    linear: np.ndarray,
    l1: float,
):
    """Computes a lower bound on the root of BCD root function.

    The lower bound :math:`h_\\star` is guaranteed to be non-negative
    and satisfies :math:`\\varphi(h_\\star) \\geq 0` where :math:`\\varphi`
    is given by :func:`adelie.bcd.elastic_net.root_function` whenever :math:`\\|v\\|_2 > \\lambda`.
    It is undefined behavior if the condition is not satisfied.

    Parameters
    ----------
    quad : (p,) ndarray
        See :func:`adelie.bcd.elastic_net.root_function`.
    linear : (p,) ndarray
        See :func:`adelie.bcd.elastic_net.root_function`.
    l1 : float
        See :func:`adelie.bcd.elastic_net.root_function`.
    
    See Also
    --------
    adelie.bcd.elastic_net.root_upper_bound
    adelie.bcd.elastic_net.root_function

    Returns
    -------
    lower : float
        Lower bound on the root.
    """
    return core.root_lower_bound(quad, linear, l1)


def root_upper_bound(
    *,
    quad: np.ndarray,
    linear: np.ndarray,
    l1: float,
    zero_tol: float=1e-14,
):
    """Computes an upper bound on the root of BCD root function.

    The upper bound :math:`h^\\star` is guaranteed to be non-negative.
    However, it *may not satisfy* :math:`\\varphi(h^\\star) \\leq 0` where :math:`\\varphi`
    is given by :func:`adelie.bcd.elastic_net.root_function` if ``zero_tol`` is too large.
    We assume that :math:`\\|v_S\\|_2 < \\lambda` 
    where :math:`S = \\{i : \\Sigma_{ii} = 0\\}`.
    It is undefined behavior if the condition is not satisfied.

    Parameters
    ----------
    quad : (p,) ndarray
        See :func:`adelie.bcd.elastic_net.root_function`.
    linear : (p,) ndarray
        See :func:`adelie.bcd.elastic_net.root_function`.
    l1 : float
        See :func:`adelie.bcd.elastic_net.root_function`.
    zero_tol : float, optional
        A value is considered zero if its absolute value is less than or equal to ``zero_tol``.
        Default is ``1e-14``.
    
    See Also
    --------
    adelie.bcd.elastic_net.root_lower_bound
    adelie.bcd.elastic_net.root_function

    Returns
    -------
    upper : float
        Upper bound on the root.
    """
    return core.root_upper_bound(quad, linear, l1, zero_tol)


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
    quad : (p,) ndarray
        The quadratic component :math:`\\Sigma`.
    linear : (p,) ndarray
        The linear component :math:`v`.
    l1 : float
        The :math:`\\ell_1` component :math:`\\lambda`.

    Returns
    -------
    func : float
        The BCD root function value.
    """
    return core.root_function(h, quad, linear, l1)


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
    beta : (p,) ndarray
        The value :math:`\\beta` at which the objective is computed.
    quad : (p,) ndarray
        The quadratic component :math:`\\Sigma`.
    linear : (p,) ndarray
        The linear component :math:`v`.
    l1 : float
        The :math:`\\ell_1` component :math:`\\lambda_1`.
    l2 : float
        The :math:`\\ell_2` component :math:`\\lambda_2`.

    Returns
    -------
    obj : float
        The BCD objective.
    """
    beta_norm = np.linalg.norm(beta)
    return 0.5 * quad @ beta ** 2 - linear @ beta + l1 * beta_norm + 0.5 * l2 * beta_norm ** 2


_solver_dict = {
    "brent":            core.unconstrained_brent_solver,
    "newton":           core.unconstrained_newton_solver,
    "newton_brent":     core.unconstrained_newton_brent_solver,
    "newton_abs":       core.unconstrained_newton_abs_solver,
    "newton_abs_debug": core.unconstrained_newton_abs_debug_solver,
    "ista":             core.unconstrained_ista_solver,
    "fista":            core.unconstrained_fista_solver,
    "fista_adares":     core.unconstrained_fista_adares_solver,
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
    smart_init: bool =True,
):
    """Solves the BCD update.

    The BCD update for the group elastic net is obtained by minimizing
    the BCD objective given in :func:`adelie.bcd.elastic_net.objective`.
    The solution exists finitely if and only if 
    :math:`\\|v\\|_2 \\leq \\lambda_1`
    or :math:`\\|v_S\\|_2 < \\lambda_1`,
    where :math:`S` is the subset of indices
    such that :math:`\\Sigma_{ii} + \\lambda_2 = 0`.
    If the solution exists, it is unique.

    Parameters
    ----------
    quad : (p,) ndarray
        See :func:`adelie.bcd.elastic_net.objective`.
    linear : (p,) ndarray
        See :func:`adelie.bcd.elastic_net.objective`.
    l1 : float
        See :func:`adelie.bcd.elastic_net.objective`.
    l2 : float
        See :func:`adelie.bcd.elastic_net.objective`.
    tol : float, optional
        Convergence tolerance. Default is ``1e-12``.
    max_iters : int, optional
        Max number of iterations. Default is ``1000``.
    solver : str, optional
        Solver type must be one of the following: 

            - ``"brent"``: Brent method.
            - ``"ista"``: ISTA method.
            - ``"fista"``: FISTA method.
            - ``"fista_adares"``: FISTA with Adaptive Restarts method.
            - ``"newton"``: Newton method.
            - ``"newton_abs"``: Newton method combined with Adaptive Bisection Starts for initialization.
            - ``"newton_abs_debug"``: same as ``"newton_abs"`` but with more debug information.
            - ``"newton_brent"``: Newton method combined with Brent method for initialization.

        Default is ``"newton_abs"``.

        .. warning::
            The following methods are known to have poor convergence:

                - ``"brent"``
                - ``"ista"``
                - ``"fista"``
                - ``"fista_adares"``

    smart_init : bool, optional
        If ``True``, the ABS method is invoked to find a smart initial point before starting Newton's method.
        It is only used when ``solver`` is ``"newton_abs_debug"``.
        Default is ``True``.

    Returns
    -------
    result : Dict[str, Any]
        A dictionary containing the output:

            - ``"beta"``: solution vector.
            - ``"iters"``: number of iterations taken.
            - ``"time_elapsed"``: time elapsed to run the solver.

    See Also
    --------
    adelie.bcd.elastic_net.objective
    adelie.bcd.elastic_net.root
    """
    if solver == "newton_abs_debug":
        return _solver_dict[solver](quad, linear, l1, l2, tol, max_iters, smart_init)
    return _solver_dict[solver](quad, linear, l1, l2, tol, max_iters)


def root(
    *,
    quad: np.ndarray,
    linear: np.ndarray,
    l1: float,
    tol: float =1e-12,
    max_iters: int =1000,
    solver: str ="newton_abs",
):
    """Solves the non-negative root of the BCD root function.

    The BCD root function is given in :func:`adelie.bcd.elastic_net.root_function`.
    The non-negative root only exists when
    :math:`\\|v_S\\|_2 < \\lambda_1 < \\|v\\|_2`
    where :math:`S` is the subset of indices
    such that :math:`\\Sigma_{ii} = 0`.

    Parameters
    ----------
    quad : (p,) ndarray
        See :func:`adelie.bcd.elastic_net.root_function`.
    linear : (p,) ndarray
        See :func:`adelie.bcd.elastic_net.root_function`.
    l1 : float
        See :func:`adelie.bcd.elastic_net.root_function`.
    tol : float, optional
        Convergence tolerance. Default is ``1e-12``.
    max_iters : int, optional
        Max number of iterations. Default is ``1000``.
    solver : str, optional
        Solver type must be one of the following: 

            - ``"brent"``: Brent method.
            - ``"newton"``: Newton method.
            - ``"newton_brent"``: Newton method combined with Brent method for initialization.
            - ``"newton_abs"``: Newton method combined with Adaptive Bisection Starts for initialization.
            - ``"ista"``: ISTA method.
            - ``"fista"``: FISTA method.
            - ``"fista_adares"``: FISTA with Adaptive Restarts method.

        Default is ``"newton_abs"``.

        .. warning::
            The following methods are known to have poor convergence:

                - ``"brent"``
                - ``"ista"``
                - ``"fista"``
                - ``"fista_adares"``

    Returns
    -------
    result : Dict[str, Any]
        A dictionary containing the output:

            - ``"root"``: the non-negative root.
            - ``"iters"``: number of iterations taken.

    See Also
    --------
    adelie.bcd.elastic_net.root_function
    """
    if (np.linalg.norm(linear) <= l1) or \
        (np.linalg.norm(linear[quad <= 0]) >= l1):
        return {
            "root": None,
            "iters": 0,
        }
    out = solve(
        quad=quad,
        linear=linear,
        l1=l1,
        l2=0,
        tol=tol,
        max_iters=max_iters,
        solver=solver,
    )
    return {
        "root": np.linalg.norm(out["beta"]),
        "iters": out["iters"],
    }