from typing import Union
from . import adelie_core as core
from . import logger
from . import matrix
from . import glm
import adelie as ad
import numpy as np
from scipy.sparse import csr_matrix


def objective(
    *,
    X: Union[matrix.MatrixNaiveBase32, matrix.MatrixNaiveBase64], 
    y: np.ndarray, 
    groups: np.ndarray, 
    group_sizes: np.ndarray, 
    lmda: float, 
    alpha: float, 
    penalty: np.ndarray,
    weights: np.ndarray,
    beta0: float,
    beta: np.ndarray | csr_matrix, 
    glm: Union[glm.GlmBase32, glm.GlmBase64] =glm.gaussian(),
    relative: bool =True,
    add_penalty: bool =True,
):
    """Computes the group elastic net objective.

    See ``adelie.solver.grpnet`` for details.

    Parameters
    ----------
    X : (n, p) Union[adelie.matrix.MatrixNaiveBase32, adelie.matrix.MatrixNaiveBase64]
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
        List of penalty factors :math:`p_g` corresponding to each element of ``groups``.
    weights : (G,) np.ndarray
        Observation weights :math:`w`.
    beta0 : float
        Intercept.
    beta : (p,) np.ndarray, (1, p) scipy.sparse.csr_matrix
        Coefficient vector :math:`\\beta`.
    glm : Union[glm.GlmBase32, glm.GlmBase64], optional
        GLM object.
        Default is ``adelie.glm.gaussian()``.
    relative : bool, optional
        If ``True``, then the deviance, :math:`D^\\star`, is computed at the saturated model
        and the difference :math:`\\mathcal{L}-D^\\star` is provided,
        which will always be non-negative.
        This effectively computes deviance *relative* to the saturated model.
        Default is ``True``.
    add_penalty : bool, optional
        If ``False``, the regularization term is removed. 
        Default is ``True``.
    
    Returns
    -------
    obj : float
        Group elastic net objective.

    See Also
    --------
    adelie.solver.grpnet
    """
    n, p = X.rows(), X.cols()

    # if numpy array, add an extra dimension to beta 
    # for code consistency with sparse case.
    eta = np.empty(n)
    if isinstance(beta, np.ndarray):
        X.btmul(0, p, beta, np.ones(n), eta)
    elif isinstance(beta, csr_matrix):
        X.sp_btmul(beta, np.ones(n), eta[None]) 
        beta = beta.toarray()
    else:
        raise RuntimeError("beta is not one of np.ndarray or scipy.sparse.csr_matrix.")
    eta += beta0

    # compute deviance part
    deviance = np.empty(n)
    glm.deviance(y, eta, deviance)
    obj = np.sum(weights * deviance)

    # relative to saturated model
    if relative:
        eta_sat = np.empty(n)
        glm.gradient_inverse(y, eta_sat)
        deviance_sat = np.empty(n)
        glm.deviance(y, eta_sat, deviance_sat)
        obj -= np.sum(weights * deviance_sat)

    # compute regularization part
    if add_penalty:
        reg = lmda * np.sum([
            pi * (
                alpha * np.linalg.norm(beta[g:g+gs])
                +
                0.5 * (1-alpha) * np.sum(beta[g:g+gs] ** 2)
            )
            for g, gs, pi in zip(groups, group_sizes, penalty)
        ])
        obj += reg

    return obj


def solve_gaussian_pin(state):
    """Solves the pinned gaussian group elastic net problem.

    The gaussian pin group elastic net problem is given by
    minimizing the objective defined in ``adelie.solver.objective``
    for the Gaussian GLM object
    with the additional constraint that :math:`\\beta_{-S} = 0`
    where :math:`S` denotes the screen set,
    that is, the coefficient vector is forced to be zero
    for groups outside the screen set.

    Parameters
    ----------
    state
        See the documentation for one of the states below.

    Returns
    -------
    result
        The resulting state after running the solver.
        The type is the same as that of ``state``.

    See Also
    --------
    adelie.state.gaussian_pin_cov
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
    """Solves the gaussian group elastic net problem.

    Parameters
    ----------
    state
        See the documentation for one of the states below.
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

    # add extra total time information
    state.total_time = out["total_time"]

    return state


def solve_glm(state, progress_bar: bool =False):
    """Solves the GLM group elastic net problem.

    Parameters
    ----------
    state
        See the documentation for one of the states below.
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
    adelie.state.glm_naive
    adelie.solver.objective
    """
    # mapping of each state type to the corresponding solver
    f_dict = {
        core.state.StateGlmNaive64: core.solver.solve_glm_naive_64,
        core.state.StateGlmNaive32: core.solver.solve_glm_naive_32,
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

    # add extra total time information
    state.total_time = out["total_time"]

    return state


def grpnet(
    *,
    X: np.ndarray,
    y: np.ndarray,
    glm: Union[glm.GlmBase32, glm.GlmBase64] =None,
    groups: np.ndarray =None,
    alpha: float =1,
    penalty: np.ndarray =None,
    weights: np.ndarray =None,
    lmda_path: np.ndarray =None,
    irls_max_iters: int =int(1e2),
    irls_tol: float =1e-7,
    max_iters: int =int(1e5),
    tol: float =1e-7,
    adev_tol: float =0.9,
    ddev_tol: float =1e-4,
    newton_tol: float =1e-12,
    newton_max_iters: int =1000,
    n_threads: int =1,
    early_exit: bool =True,
    intercept: bool =True,
    screen_rule: str ="pivot",
    min_ratio: float =1e-2,
    lmda_path_size: int =100,
    max_screen_size: int =None,
    max_active_size: int =None,
    pivot_subset_ratio: float =0.1,
    pivot_subset_min: int =1,
    pivot_slack_ratio: float =1.25,
    check_state: bool =False,
    progress_bar: bool =False,
    warm_start =None,
):
    """Group elastic net solver.

    The group elastic net problem minimizes 
    the objective given by :math:`\\mathcal{L}(\\beta_0, \\beta)`:

    .. math::
        \\begin{align*}
            \\sum_{i=1}^n w_i \\left(
                -y_i (x_i^\\top \\beta + \\beta_0) + A(x_i^\\top \\beta + \\beta_0)
            \\right)
            + \\lambda \\sum\\limits_{g=1}^G p_g \\left(
                \\alpha \\|\\beta_g\\|_2 + \\frac{1-\\alpha}{2} \\|\\beta_g\\|_2^2
            \\right)
        \\end{align*}

    where 
    :math:`\\beta_0` is the intercept,
    :math:`\\beta` is the coefficient vector,
    :math:`X` is the feature matrix,
    :math:`y` is the response vector,
    :math:`w \\geq 0` is the observation weight vector (that sum to 1),
    :math:`\\lambda \\geq 0` is the regularization parameter,
    :math:`G` is the number of groups,
    :math:`p \\geq 0` is the penalty factor,
    :math:`\\alpha \\in [0,1]` is the elastic net parameter,
    :math:`\\beta_g` are the coefficients for the :math:`g` th group,
    and :math:`A` is the log-partition function defining the GLM family.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
        It is typically one of the matrices defined in ``adelie.matrix`` sub-module
        or a ``numpy`` array.
    y : (n,) np.ndarray
        Response vector.
    glm : Union[adelie.glm.GlmBase32, adelie.glm.GlmBase64], optional
        GLM object.
        If ``None``, then it is assumed that the Gaussian GLM is used
        and an optimized routine is used to fit the group lasso problem.
        Otherwise, a more general-purpose solver is used.
        We recommend users to use ``None`` instead of supplying the Gaussian GLM object
        for best performance.
        Default is ``None``.
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
    irls_max_iters : int, optional
        Maximum number of IRLS iterations.
        This parameter is only used if ``glm`` is not ``None``.
        Default is ``100``.
    irls_tol : float, optional
        IRLS convergence tolerance.
        This parameter is only used if ``glm`` is not ``None``.
        Default is ``1e-7``.
    max_iters : int, optional
        Maximum number of coordinate descents.
        Default is ``int(1e5)``.
    tol : float, optional
        Convergence tolerance.
        Default is ``1e-7``.
    adev_tol : float, optional
        Percent deviance explained tolerance.
        Default is ``0.9``.
    ddev_tol : float, optional
        Difference in percent deviance explained tolerance.
        Default is ``1e-4``.
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
    max_screen_size : int, optional
        Maximum number of screen groups allowed.
        The function will return a valid state and guarantees to have screen set size
        less than or equal to ``max_screen_size``.
        If ``None``, it will be set to the total number of groups.
        Default is ``None``.
    max_active_size : int, optional
        Maximum number of active groups allowed.
        The function will return a valid state and guarantees to have active set size
        less than or equal to ``max_active_size``.
        If ``None``, it will be set to the total number of groups.
        Default is ``None``.
    pivot_subset_ratio : float, optional
        If screening takes place, then the ``(1 + pivot_subset_ratio) * s``
        largest gradient norms are used to determine the pivot point
        where ``s`` is the current screen set size.
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
        below the pivot point are also added to the screen set as slack.
        It is only used if ``screen_rule == "pivot"``.
        Default is ``1.25``.
    check_state : bool, optional 
        ``True`` is state should be checked for inconsistencies before calling solver.
        Default is ``False``.
    progress_bar : bool, optional
        ``True`` to enable progress bar.
        Default is ``False``.
    warm_start : optional
        If no warm-start is provided, the initial solution is set to 0
        and other invariance quantities are set accordingly.
        Otherwise, the warm-start is used to extract all necessary state variables.
        If warm-start is used, the user *must* still provide consistent inputs,
        that is, warm-start will not overwrite the arguments passed into this function.
        For example, if ``X`` were a different matrix object from ``warm_start.X``,
        it is undefined behavior.
        However, changing configuration settings such as tolerance levels is well-defined.
        Default is ``None``.

        .. note::
            The primary use-case is when a user already called ``grpnet`` with ``warm_start=False``
            but would like to continue fitting down a longer path of regularizations.
            This way, the user does not have to restart the fit at the beginning,
            but can simply continue from the last state ``grpnet`` returned.

        .. warning::
            We have only tested warm-starts in the setting described in the note above,
            that is, when ``lmda_path`` is changed and no other arguments are changed.
            Use with caution in other settings!

    Returns
    -------
    state
        The resulting state after running the solver.

    See Also
    --------
    adelie.solver.solve_gaussian
    adelie.solver.solve_glm
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

    # compute common quantities

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

    if penalty is None:
        penalty = np.sqrt(group_sizes)

    if not (lmda_path is None):
        # MUST evaluate the flip to be able to pass into C++ backend.
        lmda_path = np.array(np.flip(np.sort(lmda_path)))

    if warm_start is None:
        lmda = np.inf
        lmda_max = None
        screen_set = np.arange(G)[(penalty <= 0) | (alpha <= 0)]
        screen_beta = np.zeros(np.sum(group_sizes[screen_set]), dtype=dtype)
        screen_is_active = np.ones(screen_set.shape[0], dtype=bool)
    else:
        lmda = warm_start.lmda
        lmda_max = warm_start.lmda_max
        screen_set = warm_start.screen_set
        screen_beta = warm_start.screen_beta
        screen_is_active = warm_start.screen_is_active

    solver_args = {
        "X": X,
        "y": y,
        "groups": groups,
        "group_sizes": group_sizes,
        "alpha": alpha,
        "penalty": penalty,
        "weights": weights,
        "screen_set": screen_set,
        "screen_beta": screen_beta,
        "screen_is_active": screen_is_active,
        "lmda": lmda,
        "lmda_path": lmda_path,
        "lmda_max": lmda_max, 
        "max_iters": max_iters,
        "tol": tol,
        "adev_tol": adev_tol,
        "ddev_tol": ddev_tol,
        "newton_tol": newton_tol,
        "newton_max_iters": newton_max_iters,
        "n_threads": n_threads,
        "early_exit": early_exit,
        "intercept": intercept,
        "screen_rule": screen_rule,
        "min_ratio": min_ratio,
        "lmda_path_size": lmda_path_size,
        "max_screen_size": max_screen_size,
        "max_active_size": max_active_size,
        "pivot_subset_ratio": pivot_subset_ratio,
        "pivot_subset_min": pivot_subset_min,
        "pivot_slack_ratio": pivot_slack_ratio,
    }

    # compute quantities specific to each method 

    if glm is None:
        if warm_start is None:
            X_means = np.empty(p, dtype=dtype)
            X.means(weights, X_means)
            y_mean = np.sum(y * weights)
            yc = y
            if intercept:
                yc = yc - y_mean
            y_var = np.sum(weights * yc ** 2)
            rsq = 0
            resid = weights * yc
            grad = np.empty(p, dtype=dtype)
            X.mul(resid, grad)
        else:
            X_means = warm_start.X_means
            y_mean = warm_start.y_mean
            y_var = warm_start.y_var
            rsq = warm_start.rsq
            resid = warm_start.resid
            grad = warm_start.grad

        solver_args["X_means"] = X_means
        solver_args["y_mean"] = y_mean
        solver_args["y_var"] = y_var
        solver_args["rsq"] = rsq
        solver_args["resid"] = resid
        solver_args["grad"] = grad

        state = ad.state.gaussian_naive(**solver_args)

    else:
        if warm_start is None:
            beta0 = 0
            eta = np.zeros(n)
            mu = np.empty(n); glm.gradient(eta, mu)
            resid = weights * (y - mu)
            grad = np.empty(p); X.mul(resid, grad)
            dev = np.empty(n); glm.deviance(y, eta, dev)
            dev_null = np.sum(weights * dev)
            eta_full = np.empty(n); glm.gradient_inverse(y, eta_full)
            glm.deviance(y, eta_full, dev)
            dev_full = np.sum(weights * dev)
        else:
            beta0 = warm_start.beta0
            eta = warm_start.eta
            mu = warm_start.mu
            grad = warm_start.grad
            dev_null = warm_start.dev_null
            dev_full = warm_start.dev_full

        solver_args["glm"] = glm
        solver_args["beta0"] = beta0
        solver_args["grad"] = grad
        solver_args["eta"] = eta
        solver_args["mu"] = mu
        solver_args["dev_null"] = dev_null
        solver_args["dev_full"] = dev_full
        solver_args["irls_max_iters"] = irls_max_iters
        solver_args["irls_tol"] = irls_tol
        state = ad.state.glm_naive(**solver_args)

    if check_state:
        state.check(method="assert")

    solve_fun = (
        solve_gaussian
        if glm is None else
        solve_glm
    )
    return solve_fun(
        state=state,
        progress_bar=progress_bar,
    )
