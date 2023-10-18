from . import adelie_core as core
from . import logger
from . import matrix
import os
import adelie as ad
import numpy as np


def objective(
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
):
    """Computes the group elastic net objective.

    The group elastic net objective is given by:

    .. math::
        \\begin{align*}
            \\frac{1}{2} \\|y - X\\beta - \\beta_0 \\textbf{1}\\|_2^2
            + \\lambda \\sum\\limits_{j=1}^G w_j \\left(
                \\alpha \\|\\beta_j\\|_2 + \\frac{1-\\alpha}{2} \\|\\beta_j\\|_2^2
            \\right)
        \\end{align*}

    where 
    :math:`\\beta_0` is the intercept,
    :math:`\\beta` is the coefficient vector,
    :math:`X` is the feature matrix,
    :math:`y` is the response vector,
    :math:`\\lambda \\geq 0` is the regularization parameter,
    :math:`G` is the number of groups,
    :math:`w \\geq 0` is the penalty factor,
    :math:`\\alpha \\in [0,1]` is the elastic net parameter,
    and :math:`\\beta_j` are the coefficients for the :math:`j` th group.

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
        List of penalty factors corresponding to each element of ``groups``.
    
    Returns
    -------
    obj : float
        Group elastic net objective.
    """
    return core.solver.objective(
        beta0, beta, X, y, groups, group_sizes, lmda, alpha, penalty,
    )


def solve_pin(
    state,
    logger=logger.logger,
):
    """Solves the pinned group elastic net problem.

    The pinned group elastic net problem is given by
    minimizing the objective defined in ``objective()``
    with the constraint that :math:`\\beta_{-S} = 0`
    where :math:`S` denotes the strong set,
    that is, the coefficient vector is forced to be zero
    for groups outside the strong set.

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
    adelie.state.pin_naive
    adelie.solver.objective
    """
    # mapping of each state type to the corresponding solver
    f_dict = {
        ad.state.pin_naive_64: core.solver.solve_pin_naive_64,
        ad.state.pin_naive_32: core.solver.solve_pin_naive_32,
        ad.state.pin_cov_64: core.solver.solve_pin_cov_64,
        ad.state.pin_cov_32: core.solver.solve_pin_cov_32,
    }

    # solve group elastic net
    f = f_dict[type(state)]
    out = f(state)

    # raise any errors
    if out["error"] != "":
        logger.warning(RuntimeError(out["error"]))

    # return a subsetted Python result object
    core_state = out["state"]
    state = type(state).create_from_core(state, core_state)

    return state


def solve_basil(
    state,
    logger=logger.logger,
):
    """Solves the group elastic net problem using BASIL.

    Parameters
    ----------
    state
        A basil state object.

    Returns
    -------
    result
        The resulting state after running the solver.
        The type is the same as that of ``state``.

    See Also
    --------
    adelie.state.basil_naive
    adelie.solver.objective
    """
    # mapping of each state type to the corresponding solver
    f_dict = {
        ad.state.basil_naive_64: core.solver.solve_basil_naive_64,
        ad.state.basil_naive_32: core.solver.solve_basil_naive_32,
    }

    # solve group elastic net
    f = f_dict[type(state)]
    out = f(state)

    # raise any errors
    if out["error"] != "":
        logger.error(RuntimeError(out["error"]))

    # return a subsetted Python result object
    core_state = out["state"]
    state = type(state).create_from_core(state, core_state)

    return state


def grpnet(
    *,
    X: np.ndarray | matrix.base,
    y: np.ndarray,
    groups: np.ndarray,
    group_sizes: np.ndarray,
    alpha: float =1,
    penalty: np.ndarray =None,
    lmda_path: np.ndarray =None,
    max_iters: int =int(1e5),
    tol: float =1e-12,
    rsq_tol: float =0.9,
    rsq_slope_tol: float =1e-3,
    rsq_curv_tol: float =1e-3,
    newton_tol: float =1e-12,
    newton_max_iters: int =1000,
    n_threads: int =max(os.cpu_count() // 2, 1),
    early_exit: bool =True,
    intercept: bool =True,
    strong_rule: str ="default",
    min_ratio: float =1e-2,
    lmda_path_size: int =100,
    delta_lmda_path_size: int =5,
    delta_strong_size: int =5,
    max_strong_size: int =None,
    use_edpp: bool =True,
    check_state: bool =False,
):
    """TODO
    """
    if isinstance(X, np.ndarray):
        X_raw = X
        X = ad.matrix.naive_dense(X_raw, n_threads=n_threads)
        _X = X.internal()
    else:
        assert (
            isinstance(X, matrix.MatrixNaiveBase64) or
            isinstance(X, matrix.MatrixNaiveBase32)
        )
        _X = X

    dtype = (
        np.float64
        if isinstance(_X, matrix.MatrixNaiveBase64) else
        np.float32
    )

    n, p = _X.rows(), _X.cols()
    G = len(groups)

    X_means = np.empty(p, dtype=dtype)
    _X.means(X_means)

    X_group_norms = np.empty(G, dtype=dtype)
    _X.group_norms(
        groups,
        group_sizes,
        X_means,
        intercept,
        X_group_norms,
    )

    y_mean = np.mean(y)
    yc = y
    if intercept:
        yc = yc - y_mean
    y_var = np.sum(yc ** 2)
    resid = yc

    if penalty is None:
        penalty = np.sqrt(group_sizes)

    strong_set = np.arange(G)[(penalty <= 0) | (alpha <= 0)]
    strong_beta = np.zeros(np.sum(group_sizes[strong_set]), dtype=dtype)
    strong_is_active = np.ones(strong_set.shape[0], dtype=bool)

    rsq = 0
    lmda = np.inf
    grad = np.empty(p, dtype=dtype)
    _X.bmul(0, p, resid, grad)

    if not (lmda_path is None):
        lmda_path = np.flip(np.sort(lmda_path))

    edpp_safe_set = None
    edpp_v1_0 = None
    edpp_resid_0 = None
    if not use_edpp:
        edpp_safe_set = np.arange(G)
        edpp_v1_0 = np.empty(n, dtype=dtype)
        edpp_resid_0 = np.empty(n, dtype=dtype)

    state = ad.state.basil_naive(
        X=X,
        X_means=X_means,
        X_group_norms=X_group_norms,
        y_mean=y_mean,
        y_var=y_var,
        resid=resid,
        groups=groups,
        group_sizes=group_sizes,
        alpha=alpha,
        penalty=penalty,
        strong_set=strong_set,
        strong_beta=strong_beta,
        strong_is_active=strong_is_active,
        rsq=rsq,
        lmda=lmda,
        grad=grad,
        lmda_path=lmda_path,
        lmda_max=None,
        edpp_safe_set=edpp_safe_set,
        edpp_v1_0=edpp_v1_0,
        edpp_resid_0=edpp_resid_0,
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
        strong_rule=strong_rule,
        min_ratio=min_ratio,
        lmda_path_size=lmda_path_size,
        delta_lmda_path_size=delta_lmda_path_size,
        delta_strong_size=delta_strong_size,
        max_strong_size=max_strong_size,
    )

    if check_state:
        assert isinstance(X_raw, np.ndarray)
        state.check(X_raw, y, method="assert")

    return solve_basil(state)
