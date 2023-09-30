from . import grpglmnet_core as core
import numpy as np
from dataclasses import dataclass, asdict


def bcd_update(
    quad: np.ndarray,
    linear: np.ndarray,
    l1: float,
    l2: float,
    tol: float,
    max_iters: int,
    *,
    solver: str ="newton_abs",
    **kwargs,
):
    """Solves the block-coordinate descent update.

    The block-coordinate descent update for the group elastic net is obtained by solving:

    .. math::
        \\begin{align*}
            \\mathrm{minimize}_\\beta \\quad&
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
    quad : (p,) np.ndarray
        The quadratic component :math:`\\Sigma`.
    linear : (p,) np.ndarray
        The linear component :math:`v`.
    l1 : float
        The :math:`\\ell_1` component :math:`\\lambda_1`.
    l2 : float
        The :math:`\\ell_2` component :math:`\\lambda_2`.
    tol : float
        Convergence tolerance.
    max_iters : int
        Max number of iterations.
    solver : str, optional
        Solver type must be one of 
        ``["brent", "newton", "newton_brent", "newton_abs", "newton_abs_debug", "ista", "fista", "fista_adares"]``.
        Default is ``"newton_abs"``.
    smart_init : bool
        If ``True``, the ABS method is invoked to find a smart initial point before starting Newton's method.
        It is only used when ``solver`` is ``"newton_abs_debug"``.

    Returns
    -------
    out : Dict[str, Any]
        - ``out["x"]``: solution vector.
        - ``out["it"]``: number of iterations taken.
    """
    solver_dict = {
        "brent": core.brent_solver,
        "newton": core.newton_solver,
        "newton_brent": core.newton_brent_solver,
        "newton_abs": core.newton_abs_solver,
        "newton_abs_debug": core.newton_abs_debug_solver,
        "ista": core.ista_solver,
        "fista": core.fista_solver,
        "fista_adares": core.fista_adares_solver,
    }
    return solver_dict[solver](L, v, l1, l2, tol, max_iters, **kwargs)


def bcd_root_lower_bound(
    quad: np.ndarray,
    linear: np.ndarray,
    l1: float,
):
    """Computes a lower bound on the root of BCD root function.

    The lower bound :math:`h` is guaranteed to be non-negative
    and satisfies :math:`\\varphi(h) \\geq 0` where :math:`\\varphi`
    is given by ``bcd_root_function()``.

    Parameters
    ----------
    quad : (p,) np.ndarray
        See ``bcd_root_function()``.
    linear : (p,) np.ndarray
        See ``bcd_root_function()``.
    l1 : float
        See ``bcd_root_function()``.
    
    See Also
    --------
    grpglmnet.group_elnet.bcd_root_function

    Returns
    -------
    out : float
        Lower bound on the root.
    """
    return core.bcd_root_lower_bound(quad, linear, l1)


def bcd_root_upper_bound(
    quad: np.ndarray,
    linear: np.ndarray,
    zero_tol: float=1e-10,
):
    """Computes an upper bound on the root of BCD root function.

    The upper bound :math:`h` is guaranteed to be non-negative.
    However, it *may not satisfy* :math:`\\varphi(h) \\leq 0` where :math:`\\varphi`
    is given by ``bcd_root_function()`` if ``zero_tol`` is too large.

    Parameters
    ----------
    quad : (p,) np.ndarray
        See ``bcd_root_function()``.
    linear : (p,) np.ndarray
        See ``bcd_root_function()``.
    zero_tol : float
        A value is considered zero if its absolute value is less than or equal to ``zero_tol``.
    
    See Also
    --------
    grpglmnet.group_elnet.bcd_root_function

    Returns
    -------
    out : float
        Upper bound on the root.
    """
    return core.bcd_root_upper_bound(quad, linear, zero_tol)


def bcd_root_function(
    h: float, 
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
    return core.bcd_root_function(h, quad, linear, l1)


def objective(
    beta, X, y, groups, group_sizes, lmda, alpha, penalty
):
    """Group elastic net objective.

    Computes the group elastic net objective.

    Parameters
    ----------
    beta : (p,) np.ndarray
        Coefficient vector.
    X : (n, p) np.ndarray
        Feature matrix.
    y : (n,) np.ndarray
        Response vector.
    groups : (G,) array-like
        List of starting indices to each group.
    group_sizes : (G,) array-like
        List of group sizes corresponding to the same position as in ``groups``.
    lmda : float
        Regularization parameter.
    alpha : float
        Elastic net parameter.
    penalty : (G,) np.ndarray
        List of penalties for each group corresponding to the same position as in ``groups``.
    
    Returns
    -------
    out : float
        Group elastic net objective
    """
    return core.group_elnet_objective(
        beta, X, y, groups, group_sizes, lmda, alpha, penalty
    )


@dataclass
class GroupElnetNaiveBaseState:
    """Group elastic net naive base state class.

    This is the base state class for all naive method states.

    Parameters
    ----------
    groups : (G,) np.ndarray
        List of starting indices to each group.
    group_sizes : (G,) np.ndarray
        List of group sizes corresponding to each element in ``groups``.
    alpha : float
        Elastic net parameter.
    penalty : (G,) np.ndarray
        Penalty factor for each group.
    strong_set : (s,) np.ndarray
        List of strong groups taking on values in ``[0, G)``.
        ``strong_set[i]`` is ``i`` th strong group.
    strong_g1 : (s1,) np.ndarray
        Subset of ``strong_set`` that correspond to groups of size ``1``.
        ``strong_g1[i]`` is the ``i`` th strong group of size ``1``.
    strong_g2 : (s2,) np.ndarray
        Subset of ``strong_set`` that correspond to groups more than size ``1``.
        ``strong_g2[i]`` is the ``i`` th strong group of size more than ``1``.
    strong_begins : (s,) np.ndarray
        List of indices that index a corresponding list of values for each strong group.
        ``strong_begins[i]`` is the index to start reading ``strong_beta``, for example,
        for the ``i`` th group.
    strong_A_diag : (ws,) np.ndarray
        List of the diagonal of :math:`X_k^\\top X_k` along the strong groups.
        ``strong_A_diag[b:b+p]`` is the diagonal of :math:`X_k^\\top X_k` for the ``i`` th strong group where
        ``k = strong_set[i]``,
        ``b = strong_begins[i]``,
        and ``p = group_sizes[k]``.
    lmdas : (l,) np.ndarray
        List of the regularization sequence.
    max_cds : int
        Maximum number of coordinate descents.
    thr : float
        Convergence tolerance.
    cond_0_thresh : float
        Early stopping rule check on slope.
    cond_1_thresh : float
        Early stopping rule check on curvature.
    newton_tol : float
        Tolerance for the Newton step.
    newton_max_iters : int
        Maximum number of iterations for the Newton step.
    rsq : float
        Initial :math:`R^2` value at ``strong_beta``.
    resid : (n,) np.ndarray
        Initial residual :math:`y-X\\beta` at ``strong_beta``.
    strong_beta : (ws,) np.ndarray
        Initial coefficient vector on the strong set.
        ``strong_beta[b:b+p]`` is the coefficient for the ``i`` th strong group 
        where
        ``k = strong_set[i]``,
        ``b = strong_begins[i]``,
        and ``p = group_sizes[k]``.
    strong_grad : (ws,) np.ndarray
        Initial gradient :math:`X_k^\\top (y-X\\beta)` at ``strong_beta`` on the strong set.
        ``strong_grad[b:b+p]`` is the gradient for the ``i`` th strong group
        where 
        ``k = strong_set[i]``,
        ``b = strong_begins[i]``,
        and ``p = group_sizes[k]``.
    active_set : (a,) np.ndarray
        Initial active set among the strong groups.
        ``active_set[i]`` is the *index* to ``strong_set`` that indicates the ``i`` th active group.
    active_g1 : (a1,) np.ndarray
        Initial active set with group sizes of ``1``.
        Similar description as ``active_set``.
    active_g2 : (a2,) np.ndarray
        Initial active set with group sizes more than ``1``.
        Similar description as ``active_set``.
    active_begins : (a,) np.ndarray
        Initial indices to values for the active set.
        ``active_begins[i]`` is the starting index to read values in ``strong_beta``
        for the ``i`` th active group.
    active_order : (a,) np.ndarray
        Initial indices in such that ``strong_set`` is sorted in ascending order for the active groups.
        ``active_order[i]`` is the ``i`` th active group in sorted order
        such that ``strong_set[active_order[i]]`` is the corresponding group number.
    is_active : (s,) np.ndarray
        Initial boolean vector that indicates whether each strong group is active or not.
        ``is_active[i]`` is True if and only if ``strong_set[i]`` is active.
    betas : (l, p) np.ndarray
        ``betas[i]`` corresponds to the solution as a dense vector corresponding to ``lmdas[i]``.
    rsqs : (l,) np.ndarray
        ``rsqs[i]`` corresponds to the solution :math:`R^2` corresponding to ``lmdas[i]``.
    resids : (l, n) np.ndarray
        ``resids[i]`` corresponds to the solution residual corresponding to ``lmdas[i]``.
    """
    groups: np.ndarray
    group_sizes: np.ndarray
    alpha: float
    penalty: np.ndarray
    strong_set: np.ndarray
    strong_g1: np.ndarray
    strong_g2: np.ndarray
    strong_begins: np.ndarray
    strong_A_diag: np.ndarray
    lmdas: np.ndarray
    max_cds: int
    thr: float
    cond_0_thresh: float
    cond_1_thresh: float
    newton_tol: float
    newton_max_iters: int
    rsq: float
    resid: np.ndarray
    strong_beta: np.ndarray
    strong_grad: np.ndarray
    active_set: np.ndarray
    active_g1: np.ndarray
    active_g2: np.ndarray
    active_begins: np.ndarray
    active_order: np.ndarray
    is_active: np.ndarray
    betas: list[np.ndarray]
    rsqs: list[float]
    resids: list[np.ndarray]


class GroupElnetNaiveDenseState(GroupElnetNaiveBaseState):
    """Group elastic net state class for naive method dense feature matrix.

    This state class is for the case of using the naive method to solve group elastic net
    and :math:`X` feature matrix is given as a full dense matrix.

    Parameters
    ----------
    X : (n, p) np.ndarray
        Feature matrix.
    args
        See ``grpglmnet.group_elnet.GroupElnetNaiveBaseState``.
    kwargs
        See ``grpglmnet.group_elnet.GroupElnetNaiveBaseState``.

    See Also
    --------
    grpglmnet.group_elnet.GroupElnetNaiveBaseState
    grpglmnet.group_elnet.objective
    """
    def __init__(
        self,
        X: np.ndarray,
        *args,
        **kwargs,
    ):
        super(GroupElnetNaiveBaseState, self).__init__(*args, **kwargs)
        self._state = core.GroupElnetNaiveDenseState(
            X=X,
            **asdict(super(GroupElnetNaiveBaseState, self)),
        )


@dataclass
class GroupElnetResult:
    """Group elastic net result pack.

    Parameters
    ----------
    rsq : float
        :math:`R^2` value using ``strong_beta``.
    resid : (n,) np.ndarray
        The residual :math:`y-X\\beta` using ``strong_beta``.
    strong_beta : (w,) np.ndarray
        The last-updated coefficient for strong groups.
    strong_grad : (w,) np.ndarray
        The last-updated gradient :math:`X_k^\\top (y - X\\beta)` for all strong groups :math:`k`.
    active_set : (a,) np.ndarray
        The last-updated active set among the strong groups.
    active_g1 : (a1,) np.ndarray
        The last-updated active set with group sizes of 1.
    active_g2 : (a2,) np.ndarray
        The last-updated active set with group sizes > 1.
    active_begins : (a,) np.ndarray
        The last-updated indices to values for the active set.
    active_order : (a,) np.ndarray
        The last-updated indices in such that ``strong_set`` is sorted in ascending order for the active groups.
    is_active : (s,) np.ndarray
        The last-updated boolean vector that indicates whether each strong group is active or not.
    betas : (l, p) np.ndarray
        ``betas[i]`` corresponds to the solution as a dense vector corresponding to ``lmdas[i]``.
    rsqs : (l,) np.ndarray
        ``rsqs[i]`` corresponds to the solution :math:`R^2` corresponding to ``lmdas[i]``.
    resids : (l, n) np.ndarray
        ``resids[i]`` corresponds to the solution residual corresponding to ``lmdas[i]``.
    n_cds : int
        Number of coordinate descents taken.
    """
    rsq: float
    resid: np.ndarray
    strong_beta: np.ndarray
    strong_grad: np.ndarray
    active_set: np.ndarray
    active_g1: np.ndarray
    active_g2: np.ndarray
    active_begins: np.ndarray
    active_order: np.ndarray
    is_active: np.ndarray
    betas: np.ndarray
    rsqs: np.ndarray
    resids: np.ndarray
    n_cds: int


def group_elnet(state):
    """Group elastic net solver.

    The group elastic net solves the following problem:
    
    .. math::
        \\begin{align*}
            \\mathrm{minimize}_\\beta \\quad&
            \\frac{1}{2} \\|y - X\\beta\\|_2^2
            + \\lambda \\sum\\limits_{j=1}^G w_j \\left(
                \\alpha \\|\\beta_j\\|_2 + \\frac{1-\\alpha}{2} \\|\\beta_j\\|_2^2
            \\right)
        \\end{align*}

    where 
    :math:`X` is the feature matrix,
    :math:`y` is the response vector,
    :math:`w` is the penalty factor,
    :math:`\\lambda` is the regularization parameter,
    :math:`\\alpha` is the elastic net parameter,
    :math:`G` is the number of groups,
    and :math:`\\beta_j` are the coefficients for the :math:`j` th group.

    Parameters
    ----------
    state : GroupElnetNaiveDenseState
        See the documentation for one of the listed types.
    """
    # mapping of each state type to the corresponding solver
    f_dict = {
        GroupElnetNaiveDenseState: core.group_elnet_naive_dense,
        #'full_cov': group_elnet__,
        #'data': group_elnet_data__,
        #'data_newton': group_elnet_data_newton__,
    }

    # solve group elastic net
    f = f_dict[type(state)]
    out = f(state)

    # raise any errors
    if out["error"] != "":
        raise RuntimeError(out["error"])

    # return a subsetted Python result object
    state = out["state"]
    return GroupElnetResult(
        rsq=state.rsq,
        resid=np.array(state.resid),
        strong_beta=np.array(state.strong_beta),
        strong_grad=np.array(state.strong_grad),
        active_set=np.array(state.active_set),
        active_g1=np.array(state.active_g1),
        active_g2=np.array(state.active_g2),
        active_begins=np.array(state.active_begins),
        active_order=np.array(state.active_order),
        is_active=np.array(state.is_active),
        betas=np.array(state.betas),
        rsqs=np.array(state.rsqs),
        resids=np.array(state.resids),
        n_cds=state.n_cds,
    )
