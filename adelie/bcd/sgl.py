from ..adelie_core.bcd import sgl as core
import numpy as np


def quartic_roots(
    A: float, 
    B: float, 
    C: float, 
    D: float, 
    E: float,
):
    """Computes roots of a quartic polynomial.

    A quartic polynomial is given by

    .. math::
        \\begin{align*}
            p(x) = A x^4 + B x^3 + C x^2 + D x + E
        \\end{align*}

    Parameters
    ----------
    A : float
        First coefficient :math:`A`.
    B : float
        Second coefficient :math:`B`.
    C : float
        Third coefficient :math:`C`.
    D : float
        Fourth coefficient :math:`D`.
    E : float
        Fifth coefficient :math:`E`.

    Returns
    -------
    roots : (4,) ndarray
        Roots of :math:`p(x)`.
    """
    return core.quartic_roots(A, B, C, D, E)


def root_secular(
    y: float,
    *,
    m: float,
    a: float,
    b: float,
    tol: float =1e-12,
    max_iters: int =1000,
):
    """Computes the positive root of the SGL secular equation.

    The SGL secular equation is given by

    .. math::
        \\begin{align*}
            \\left(
                m + \\frac{b}{\\sqrt{x^2 + a}}
            \\right) x - y
        \\end{align*}

    It is only well-defined when :math:`m,a,b,y \\geq 0`.

    Parameters
    ----------
    y : float
        Function level :math:`y` at which to compute the root.
    m : float
        Linear slope :math:`m`.
    a : float
        Shift :math:`a`.
    b : float
        Non-linear scale :math:`b`.
    tol : float, optional
        Convergence tolerance.
        Default is ``1e-12``.
    max_iters : int, optional
        Max number of iterations.
        Default is ``1000``.

    Returns
    -------
    root : float
        The positive root of the secular equation.
    """
    return core.root_secular(y, m, a, b, tol, max_iters)


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
            + \\lambda_1 \\|\\beta\\|_1
            + \\lambda_2 \\|\\beta\\|_2
        \\end{align*}

    where 
    :math:`\\beta \\in \\mathbb{R}^p`,
    :math:`\\Sigma \\in \\mathbb{R}^{p\\times p}` positive semi-definite,
    :math:`v \\in \\mathbb{R}^p`,
    and :math:`\\lambda_i \\geq 0`.

    Parameters
    ----------
    beta : (p,) ndarray
        The value :math:`\\beta` at which the objective is computed.
    quad : (p, p) ndarray
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
    return 0.5 * beta.T @ quad @ beta - linear @ beta + l1 * np.linalg.norm(beta, 1) + l2 * np.linalg.norm(beta)


def solve(
    *,
    quad: np.ndarray,
    linear: np.ndarray,
    l1: float,
    l2: float,
    tol: float =1e-12,
    max_iters: int =1000,
):
    """Solves the BCD update.

    The BCD update for SGL is obtained by minimizing
    the BCD objective given in :func:`adelie.bcd.sgl.objective`.

    Parameters
    ----------
    quad : (p, p) ndarray
        See :func:`adelie.bcd.sgl.objective`.
    linear : (p,) ndarray
        See :func:`adelie.bcd.sgl.objective`.
    l1 : float
        See :func:`adelie.bcd.sgl.objective`.
    l2 : float
        See :func:`adelie.bcd.sgl.objective`.
    tol : float, optional
        Convergence tolerance. Default is ``1e-12``.
    max_iters : int, optional
        Max number of iterations. Default is ``1000``.

    Returns
    -------
    result : Dict[str, Any]
        A dictionary containing the output:

            - ``"beta"``: solution vector.
            - ``"iters"``: number of iterations taken.
            - ``"time_elapsed"``: time elapsed to run the solver.

    See Also
    --------
    adelie.bcd.sgl.objective
    """
    return core.unconstrained_coordinate_descent_solver(quad, linear, l1, l2, tol, max_iters)