from . import adelie_core as core
from . import logger
from . import matrix
import os
import adelie as ad
import numpy as np


def gaussian_naive_objective(
    beta0: float,
    beta: np.ndarray, 
    *,
    X: np.ndarray, 
    y: np.ndarray, 
    groups: np.ndarray, 
    group_sizes: np.ndarray, 
    lmda: float, 
    alpha: float, 
    penalty: np.ndarray,
    weights: np.ndarray,
):
    """Computes the group elastic net objective.

    The group elastic net objective is given by:

    .. math::
        \\begin{align*}
            \\frac{1}{2} \\|y - X\\beta - \\beta_0 \\textbf{1}\\|_{W}^2
            + \\lambda \\sum\\limits_{g=1}^G w_g \\left(
                \\alpha \\|\\beta_g\\|_2 + \\frac{1-\\alpha}{2} \\|\\beta_g\\|_2^2
            \\right)
        \\end{align*}

    where 
    :math:`\\beta_0` is the intercept,
    :math:`\\beta` is the coefficient vector,
    :math:`X` is the feature matrix,
    :math:`y` is the response vector,
    :math:`W` is the observation weight diagonal matrix,
    :math:`\\lambda \\geq 0` is the regularization parameter,
    :math:`G` is the number of groups,
    :math:`w \\geq 0` is the penalty factor,
    :math:`\\alpha \\in [0,1]` is the elastic net parameter,
    and :math:`\\beta_g` are the coefficients for the :math:`g` th group.

    Parameters
    ----------
    beta0 : float
        Intercept.
    beta : (p,) np.ndarray
        Coefficient vector :math:`\\beta`.
    X : (n, p) np.ndarray
        Feature matrix :math:`X`.
    y : (n,) np.ndarray
        Response vector :math:`y`.
    groups : (G,) np.ndarray
        List of starting column indices of ``X`` for each group.
    group_sizes : (G,) np.ndarray
        List of group sizes corresponding to each element of ``groups``.
    lmda : float
        Regularization parameter :math:`\\lambda`.
    alpha : float
        Elastic net parameter :math:`\\alpha`.
    penalty : (G,) np.ndarray
        List of penalty factors :math:`w_g` corresponding to each element of ``groups``.
    weights : (G,) np.ndarray
        Observation weights :math:`W`.
    
    Returns
    -------
    obj : float
        Group elastic net objective.
    """
    return core.solver.gaussian_naive_objective(
        beta0, beta, X, y, groups, group_sizes, lmda, alpha, penalty, weights,
    )


def solve_gaussian_pin(state):
    """Solves the pinned group elastic net problem.

    The pinned group elastic net problem is given by
    minimizing the objective defined in ``objective()``
    with the constraint that :math:`\\beta_{-S} = 0`
    where :math:`S` denotes the screen set,
    that is, the coefficient vector is forced to be zero
    for groups outside the screen set.

    Parameters
    ----------
    state
        See the documentation for one of the listed types.

    Returns
    -------
    result
        The resulting state after running the solver.
        The type is the same as that of ``state``.

    See Also
    --------
    adelie.state.gaussian_pin_naive
    adelie.solver.objective
    """
    # mapping of each state type to the corresponding solver
    f_dict = {
        core.state.StateGaussianPinNaive64: core.solver.solve_gaussian_pin_naive_64,
        core.state.StateGaussianPinNaive32: core.solver.solve_gaussian_pin_naive_32,
        core.state.StateGaussianPinCov64: core.solver.solve_gaussian_pin_cov_64,
        core.state.StateGaussianPinCov32: core.solver.solve_gaussian_pin_cov_32,
    }

    # solve group elastic net
    f = f_dict[state._core_type]
    out = f(state)

    # raise any errors
    if out["error"] != "":
        logger.logger.warning(RuntimeError(out["error"]))

    # return a subsetted Python result object
    core_state = out["state"]
    state = type(state).create_from_core(state, core_state)

    return state


def solve_gaussian(state, progress_bar: bool =False):
    """Solves the group elastic net problem with Gaussian loss.

    Parameters
    ----------
    state
        A state object.
    progress_bar : bool, optional
        ``True`` to enable progress bar.
        Default is ``False``.

    Returns
    -------
    result
        The resulting state after running the solver.
        The type is the same as that of ``state``.

    See Also
    --------
    adelie.state.gaussian_naive
    adelie.solver.objective
    """
    # mapping of each state type to the corresponding solver
    f_dict = {
        core.state.StateGaussianNaive64: core.solver.solve_gaussian_naive_64,
        core.state.StateGaussianNaive32: core.solver.solve_gaussian_naive_32,
    }

    # solve group elastic net
    f = f_dict[state._core_type]
    out = f(state, progress_bar)

    # raise any errors
    if out["error"] != "":
        logger.logger.warning(RuntimeError(out["error"]))

    # return a subsetted Python result object
    core_state = out["state"]
    state = type(state).create_from_core(state, core_state)

    return state


def grpnet(
    *,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray =None,
    alpha: float =1,
    penalty: np.ndarray =None,
    weights: np.ndarray =None,
    lmda_path: np.ndarray =None,
    max_iters: int =int(1e5),
    tol: float =1e-7,
    rsq_tol: float =0.9,
    rsq_slope_tol: float =1e-3,
    rsq_curv_tol: float =1e-3,
    newton_tol: float =1e-12,
    newton_max_iters: int =1000,
    n_threads: int =1,
    early_exit: bool =True,
    intercept: bool =True,
    screen_rule: str ="pivot",
    min_ratio: float =1e-2,
    lmda_path_size: int =100,
    max_screen_size: int =None,
    pivot_subset_ratio: float =0.1,
    pivot_subset_min: int =1,
    pivot_slack_ratio: float =1.25,
    check_state: bool =False,
    progress_bar: bool =False,
):
    """Group elastic net solver.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
        It is typically one of the matrices defined in ``adelie.matrix`` sub-module
        or a ``numpy`` array.
    y : (n,) np.ndarray
        Response vector.
    groups : (G,) np.ndarray, optional
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
        Default is ``np.arange(p)``.
    alpha : float, optional
        Elastic net parameter.
        It must be in the range :math:`[0,1]`.
        Default is ``1``.
    penalty : (G,) np.ndarray, optional
        Penalty factor for each group in the same order as ``groups``.
        It must be a non-negative vector.
        Default is ``None``, in which case, it is set to ``np.sqrt(group_sizes)``.
    weights : (n,) np.ndarray, optional
        Observation weights.
        Default is ``None``, in which case, it is set to ``np.full(n, 1/n)``.
    lmda_path : (l,) np.ndarray, optional
        The regularization path to solve for.
        The full path is not considered if ``early_exit`` is ``True``.
        It is recommended that the path is sorted in decreasing order.
        If ``None``, the path will be generated.
        Default is ``None``.
    max_iters : int, optional
        Maximum number of coordinate descents.
        Default is ``int(1e5)``.
    tol : float, optional
        Convergence tolerance.
        Default is ``1e-7``.
    rsq_tol : float, optional
        Early stopping rule check on :math:`R^2`.
        Default is ``0.9``.
    rsq_slope_tol : float, optional
        Early stopping rule check on slope of :math:`R^2`.
        Default is ``1e-3``.
    rsq_curv_tol : float, optional
        Early stopping rule check on curvature of :math:`R^2`.
        Default is ``1e-3``.
    newton_tol : float, optional
        Convergence tolerance for the BCD update.
        Default is ``1e-12``.
    newton_max_iters : int, optional
        Maximum number of iterations for the BCD update.
        Default is ``1000``.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.
    early_exit : bool, optional
        ``True`` if the function should early exit based on training :math:`R^2`.
        Default is ``True``.
    min_ratio : float, optional
        The ratio between the largest and smallest :math:`\\lambda` in the regularization sequence
        if it is to be generated.
        Default is ``1e-2``.
    lmda_path_size : int, optional
        Number of regularizations in the path if it is to be generated.
        Default is ``100``.
    intercept : bool, optional 
        ``True`` if the function should fit with intercept.
        Default is ``True``.
    screen_rule : str, optional
        The type of screening rule to use. It must be one of the following options:

            - ``"strong"``: adds groups whose active scores are above the strong threshold.
            - ``"pivot"``: adds groups whose active scores are above the pivot cutoff with slack.

        Default is ``"pivot"``.
    max_screen_size: int, optional
        Maximum number of strong groups allowed.
        The function will return a valid state and guaranteed to have strong set size
        less than or equal to ``max_screen_size``.
        If ``None``, it will be set to the total number of groups.
        Default is ``None``.
    pivot_subset_ratio : float, optional
        If screening takes place, then the ``(1 + pivot_subset_ratio) * s``
        largest gradient norms are used to determine the pivot point
        where ``s`` is the current strong set size.
        It is only used if ``screen_rule == "pivot"``.
        Default is ``0.1``.
    pivot_subset_min : int, optional
        If screening takes place, then at least ``pivot_subset_min``
        number of active scores are used to determine the pivot point.
        It is only used if ``screen_rule == "pivot"``.
        Default is ``1``.
    pivot_slack_ratio : float, optional
        If screening takes place, then ``pivot_slack_ratio``
        number of groups with next smallest (new) active scores 
        below the pivot point are also added to the strong set as slack.
        It is only used if ``screen_rule == "pivot"``.
        Default is ``1.25``.
    check_state : bool, optional 
        ``True`` is state should be checked for inconsistencies before calling solver.
        Default is ``False``.
    progress_bar : bool, optional
        ``True`` to enable progress bar.
        Default is ``False``.

    Returns
    -------
    result
        The resulting state after running the solver.
        The type is the same as that of ``state``.
    """
    if isinstance(X, np.ndarray):
        X = ad.matrix.dense(X, method="naive", n_threads=n_threads)

    assert (
        isinstance(X, matrix.MatrixNaiveBase64) or
        isinstance(X, matrix.MatrixNaiveBase32)
    )

    dtype = (
        np.float64
        if isinstance(X, matrix.MatrixNaiveBase64) else
        np.float32
    )

    n, p = X.rows(), X.cols()

    if groups is None:
        groups = np.arange(p, dtype=int)
    group_sizes = np.concatenate([groups, [p]], dtype=int)
    group_sizes = group_sizes[1:] - group_sizes[:-1]

    G = len(groups)

    if weights is None:
        weights = np.full(n, 1/n)
    else:
        weights = weights / np.sum(weights)

    X_means = np.empty(p, dtype=dtype)
    X.means(weights, X_means)

    y_mean = np.sum(y * weights)
    yc = y
    if intercept:
        yc = yc - y_mean
    y_var = np.sum(weights * yc ** 2)
    resid = weights * yc

    if penalty is None:
        penalty = np.sqrt(group_sizes)

    screen_set = np.arange(G)[(penalty <= 0) | (alpha <= 0)]
    screen_beta = np.zeros(np.sum(group_sizes[screen_set]), dtype=dtype)
    screen_is_active = np.ones(screen_set.shape[0], dtype=bool)

    rsq = 0
    lmda = np.inf
    grad = np.empty(p, dtype=dtype)
    X.mul(resid, grad)

    if not (lmda_path is None):
        lmda_path = np.flip(np.sort(lmda_path))

    state = ad.state.gaussian_naive(
        X=X,
        X_means=X_means,
        y_mean=y_mean,
        y_var=y_var,
        resid=resid,
        groups=groups,
        group_sizes=group_sizes,
        alpha=alpha,
        penalty=penalty,
        weights=weights,
        screen_set=screen_set,
        screen_beta=screen_beta,
        screen_is_active=screen_is_active,
        rsq=rsq,
        lmda=lmda,
        grad=grad,
        lmda_path=lmda_path,
        lmda_max=None,
        max_iters=max_iters,
        tol=tol,
        rsq_tol=rsq_tol,
        rsq_slope_tol=rsq_slope_tol,
        rsq_curv_tol=rsq_curv_tol,
        newton_tol=newton_tol,
        newton_max_iters=newton_max_iters,
        n_threads=n_threads,
        early_exit=early_exit,
        intercept=intercept,
        screen_rule=screen_rule,
        min_ratio=min_ratio,
        lmda_path_size=lmda_path_size,
        max_screen_size=max_screen_size,
        pivot_subset_ratio=pivot_subset_ratio,
        pivot_subset_min=pivot_subset_min,
        pivot_slack_ratio=pivot_slack_ratio,
    )

    if check_state:
        state.check(method="assert")

    return solve_gaussian(
        state=state, 
        progress_bar=progress_bar,
    )
