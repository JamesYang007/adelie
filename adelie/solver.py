from typing import Union
from . import adelie_core as core
from . import logger
from . import matrix
from . import glm
import adelie as ad
import numpy as np
import warnings


def _solve(state, progress_bar: bool =False):
    """Solves the group elastic net problem.

    The gaussian pin group elastic net problem is given by
    minimizing the objective defined in ``adelie.solver.grpnet``
    for the Gaussian GLM with the additional constraint that :math:`\\beta_{-S} = 0`
    where :math:`S` denotes the screen set,
    that is, the coefficient vector is forced to be zero
    for groups outside the screen set.

    For details on the other group elastic net problems, see ``adelie.solver.grpnet``.

    Parameters
    ----------
    state
        See the documentation for one of the states below.
    progress_bar : bool, optional
        ``True`` to enable progress bar.
        It is ignored for gaussian pin methods.
        Default is ``False``.

    Returns
    -------
    result
        The resulting state after running the solver.
        The type is the same as that of ``state``.

    See Also
    --------
    adelie.state.gaussian_naive
    adelie.state.gaussian_pin_cov
    adelie.state.gaussian_pin_naive
    adelie.state.glm_naive
    adelie.state.multigaussian_naive
    adelie.state.multiglm_naive
    """
    # mapping of each state type to the corresponding solver
    f_dict = {
        # cov methods
        core.state.StateGaussianPinCov64: core.solver.solve_gaussian_pin_cov_64,
        core.state.StateGaussianPinCov32: core.solver.solve_gaussian_pin_cov_32,
        core.state.StateGaussianCov64: core.solver.solve_gaussian_cov_64,
        core.state.StateGaussianCov32: core.solver.solve_gaussian_cov_32,
        # naive methods
        core.state.StateGaussianPinNaive64: core.solver.solve_gaussian_pin_naive_64,
        core.state.StateGaussianPinNaive32: core.solver.solve_gaussian_pin_naive_32,
        core.state.StateGaussianNaive64: core.solver.solve_gaussian_naive_64,
        core.state.StateGaussianNaive32: core.solver.solve_gaussian_naive_32,
        core.state.StateMultiGaussianNaive64: core.solver.solve_multigaussian_naive_64,
        core.state.StateMultiGaussianNaive32: core.solver.solve_multigaussian_naive_32,
        core.state.StateGlmNaive64: core.solver.solve_glm_naive_64,
        core.state.StateGlmNaive32: core.solver.solve_glm_naive_32,
        core.state.StateMultiGlmNaive64: core.solver.solve_multiglm_naive_64,
        core.state.StateMultiGlmNaive32: core.solver.solve_multiglm_naive_32,
    }

    is_gaussian_pin = (
        isinstance(state, core.state.StateGaussianPinCov64) or
        isinstance(state, core.state.StateGaussianPinCov32) or
        isinstance(state, core.state.StateGaussianPinNaive64) or
        isinstance(state, core.state.StateGaussianPinNaive32)
    )
    is_gaussian = (
        isinstance(state, core.state.StateGaussianCov64) or
        isinstance(state, core.state.StateGaussianCov32) or
        isinstance(state, core.state.StateGaussianNaive64) or
        isinstance(state, core.state.StateGaussianNaive32) or
        isinstance(state, core.state.StateMultiGaussianNaive64) or
        isinstance(state, core.state.StateMultiGaussianNaive32)
    )
    is_glm = (
        isinstance(state, core.state.StateGlmNaive64) or
        isinstance(state, core.state.StateGlmNaive32) or
        isinstance(state, core.state.StateMultiGlmNaive64) or
        isinstance(state, core.state.StateMultiGlmNaive32)
    )

    # solve group elastic net
    f = f_dict[state._core_type]
    if is_gaussian_pin:
        out = f(state)
    elif is_gaussian:
        out = f(state, progress_bar)
    elif is_glm:
        out = f(state, state._glm, progress_bar)
    else:
        raise RuntimeError("Unexpected state type.")

    # raise any errors
    if out["error"] != "":
        logger.logger.warning(RuntimeError(out["error"]))

    # return a subsetted Python result object
    core_state = out["state"]
    state = type(state).create_from_core(state, core_state)

    # add extra total time information
    state.total_time = out["total_time"]

    return state


def gaussian_cov(
    *,
    A: np.ndarray,
    v: np.ndarray,
    groups: np.ndarray =None,
    alpha: float =1,
    penalty: np.ndarray =None,
    lmda_path: np.ndarray =None,
    max_iters: int =int(1e5),
    tol: float =1e-7,
    rdev_tol: float =1e-4,
    newton_tol: float =1e-12,
    newton_max_iters: int =1000,
    n_threads: int =1,
    early_exit: bool =True,
    screen_rule: str ="pivot",
    min_ratio: float =1e-2,
    lmda_path_size: int =100,
    max_screen_size: int =None,
    max_active_size: int =None,
    pivot_subset_ratio: float =0.1,
    pivot_subset_min: int =1,
    pivot_slack_ratio: float =1.25,
    check_state: bool =False,
    progress_bar: bool =True,
    warm_start =None,
):
    """Gaussian elastic net solver via covariance method.

    The Gaussian elastic net problem via covariance method minimizes the following:

    .. math::
        \\begin{align*}
            \\mathrm{minimize}_{\\beta} \\quad&
            \\frac{1}{2} \\beta^\\top A \\beta - v^\\top \\beta
            + \\lambda \\sum\\limits_{g=1}^G \\omega_g \\left(
                \\alpha \\|\\beta_g\\|_2 + \\frac{1-\\alpha}{2} \\|\\beta_g\\|_2^2
            \\right)
        \\end{align*}

    where 
    :math:`\\beta` is the coefficient vector,
    :math:`A` is any positive semi-definite matrix,
    :math:`v` is any vector,
    :math:`\\lambda \\geq 0` is the regularization parameter,
    :math:`G` is the number of groups,
    :math:`\\omega \\geq 0` is the penalty factor,
    :math:`\\alpha \\in [0,1]` is the elastic net parameter,
    and :math:`\\beta_g` are the coefficients for the :math:`g` th group.

    Parameters
    ----------
    A : (p, p) Union[adelie.matrix.MatrixCovBase64, adelie.matrix.MatrixCovBase32]
        Positive semi-definite matrix.
        It is typically one of the matrices defined in ``adelie.matrix`` submodule.
    v : (p,) np.ndarray
        Linear term.
    groups : (G,) np.ndarray, optional
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
        Default is ``None``, in which case it is set to ``np.arange(p)``.
    alpha : float, optional
        Elastic net parameter.
        It must be in the range :math:`[0,1]`.
        Default is ``1``.
    penalty : (G,) np.ndarray, optional
        Penalty factor for each group in the same order as ``groups``.
        It must be a non-negative vector.
        Default is ``None``, in which case, it is set to ``np.sqrt(group_sizes)``.
    lmda_path : (L,) np.ndarray, optional
        The regularization path to solve for.
        The full path is not considered if ``early_exit`` is ``True``.
        It is recommended that the path is sorted in decreasing order.
        If ``None``, the path will be generated.
        Default is ``None``.
    max_iters : int, optional
        Maximum number of coordinate descents.
        Default is ``int(1e5)``.
    tol : float, optional
        Coordinate descent convergence tolerance.
        Default is ``1e-7``.
    rdev_tol : float, optional
        Relative percent deviance explained tolerance.
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
        ``True`` if the function should early exit based on training deviance explained.
        Default is ``True``.
    min_ratio : float, optional
        The ratio between the largest and smallest :math:`\\lambda` in the regularization sequence
        if it is to be generated.
        Default is ``1e-2``.
    lmda_path_size : int, optional
        Number of regularizations in the path if it is to be generated.
        Default is ``100``.
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
        largest active scores are used to determine the pivot point
        where ``s`` is the current screen set size.
        It is only used if ``screen_rule="pivot"``.
        Default is ``0.1``.
    pivot_subset_min : int, optional
        If screening takes place, then at least ``pivot_subset_min``
        number of active scores are used to determine the pivot point.
        It is only used if ``screen_rule="pivot"``.
        Default is ``1``.
    pivot_slack_ratio : float, optional
        If screening takes place, then ``pivot_slack_ratio``
        number of groups with next smallest (new) active scores 
        below the pivot point are also added to the screen set as slack.
        It is only used if ``screen_rule="pivot"``.
        Default is ``1.25``.
    check_state : bool, optional 
        ``True`` is state should be checked for inconsistencies before calling solver.
        Default is ``False``.

        .. warning::
            The check may take a long time if the inputs are big!

    progress_bar : bool, optional
        ``True`` to enable progress bar.
        Default is ``True``.
    warm_start : optional
        If no warm-start is provided, the initial solution is set to 0
        and other invariance quantities are set accordingly.
        Otherwise, the warm-start is used to extract all necessary state variables.
        If warm-start is used, the user *must* still provide consistent inputs,
        that is, warm-start will not overwrite most arguments passed into this function.
        However, changing configuration settings such as tolerance levels is well-defined.
        Default is ``None``.

        .. note::
            The primary use-case is when a user already called the function with ``warm_start=False``
            but would like to continue fitting down a longer path of regularizations.
            This way, the user does not have to restart the fit at the beginning,
            but can simply continue from the last returned state.

        .. warning::
            We have only tested warm-starts in the setting described in the note above,
            that is, when ``lmda_path`` and possibly static configurations have changed.
            Use with caution in other settings!

    Returns
    -------
    state
        The resulting state after running the solver.

    See Also
    --------
    adelie.adelie_core.state.StateGaussianCov64
    """
    if isinstance(A, np.ndarray):
        A = ad.matrix.dense(A, method="cov", n_threads=n_threads)

    assert (
        isinstance(A, matrix.MatrixCovBase64) or
        isinstance(A, matrix.MatrixCovBase32)
    )

    dtype = (
        np.float64
        if isinstance(A, matrix.MatrixCovBase64) else
        np.float32
    )
    
    p = A.cols()

    if not (lmda_path is None):
        # MUST evaluate the flip to be able to pass into C++ backend.
        lmda_path = np.array(np.flip(np.sort(lmda_path)))

    if groups is None:
        groups = np.arange(p, dtype=int)
    group_sizes = np.concatenate([groups, [p]], dtype=int)
    group_sizes = group_sizes[1:] - group_sizes[:-1]

    G = len(groups)

    if penalty is None:
        penalty = np.sqrt(group_sizes)

    if warm_start is None:
        lmda = np.inf
        lmda_max = None
        screen_set = np.arange(G)[(penalty <= 0) | (alpha <= 0)]
        screen_beta = np.zeros(np.sum(group_sizes[screen_set]), dtype=dtype)
        screen_is_active = np.ones(screen_set.shape[0], dtype=bool)
        rsq = 0

        grad = v
        _tmp = np.empty(p, dtype=dtype)
        _pos = 0
        for i in range(screen_set.shape[0]):
            ss = screen_set[i]
            g, gs = groups[ss], group_sizes[ss]
            A.mul(g, gs, screen_beta[_pos:_pos+gs], _tmp)
            grad -= _tmp
            _pos += gs
    else:
        lmda = warm_start.lmda
        lmda_max = warm_start.lmda_max
        screen_set = warm_start.screen_set
        screen_beta = warm_start.screen_beta
        screen_is_active = warm_start.screen_is_active
        rsq = warm_start.rsq
        grad = warm_start.grad

    state = ad.state.gaussian_cov(
        A=A,
        v=v,
        groups=groups,
        group_sizes=group_sizes,
        alpha=alpha,
        penalty=penalty,
        screen_set=screen_set,
        screen_beta=screen_beta,
        screen_is_active=screen_is_active,
        rsq=rsq,
        lmda=lmda,
        grad=grad,
        lmda_path=lmda_path,
        lmda_max=lmda_max,
        max_iters=max_iters,
        tol=tol,
        rdev_tol=rdev_tol,
        newton_tol=newton_tol,
        newton_max_iters=newton_max_iters,
        n_threads=n_threads,
        early_exit=early_exit,
        screen_rule=screen_rule,
        min_ratio=min_ratio,
        lmda_path_size=lmda_path_size,
        max_screen_size=max_screen_size,
        max_active_size=max_active_size,
        pivot_subset_ratio=pivot_subset_ratio,
        pivot_subset_min=pivot_subset_min,
        pivot_slack_ratio=pivot_slack_ratio,
    )

    if check_state:
        state.check(method="assert")

    return _solve(
        state=state, 
        progress_bar=progress_bar,
    )


def grpnet(
    *,
    X: np.ndarray,
    glm: Union[glm.GlmBase32, glm.GlmBase64],
    groups: np.ndarray =None,
    alpha: float =1,
    penalty: np.ndarray =None,
    offsets: np.ndarray =None,
    lmda_path: np.ndarray =None,
    irls_max_iters: int =int(1e4),
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
    progress_bar: bool =True,
    warm_start =None,
):
    """Group elastic net solver.

    The group elastic net problem minimizes the following:

    .. math::
        \\begin{align*}
            \\mathrm{minimize}_{\\beta, \\beta_0} \\quad&
            \\ell(\\eta)
            + \\lambda \\sum\\limits_{g=1}^G \\omega_g \\left(
                \\alpha \\|\\beta_g\\|_2 + \\frac{1-\\alpha}{2} \\|\\beta_g\\|_2^2
            \\right)
            \\\\
            \\text{subject to} \\quad&
            \\eta = X\\beta + \\beta_0 \\mathbf{1} + \\eta^0
        \\end{align*}

    where 
    :math:`\\beta_0` is the intercept,
    :math:`\\beta` is the coefficient vector,
    :math:`X` is the feature matrix,
    :math:`\\eta^0` is a fixed offset vector,
    :math:`\\lambda \\geq 0` is the regularization parameter,
    :math:`G` is the number of groups,
    :math:`\\omega \\geq 0` is the penalty factor,
    :math:`\\alpha \\in [0,1]` is the elastic net parameter,
    :math:`\\beta_g` are the coefficients for the :math:`g` th group,
    and :math:`\\ell(\\cdot)` is the loss function defined by a GLM.

    For multi-response problems (i.e. when :math:`y` is 2-dimensional)
    such as in multigaussian or multinomial,
    the group elastic net problem minimizes the following:

    .. math::
        \\begin{align*}
            \\mathrm{minimize}_{\\beta, \\beta_0} \\quad&
            \\ell(\\eta)
            + \\lambda \\sum\\limits_{g=1}^G \\omega_g \\left(
                \\alpha \\|\\beta_g\\|_2 + \\frac{1-\\alpha}{2} \\|\\beta_g\\|_2^2
            \\right)
            \\\\
            \\text{subject to} \\quad&
            \\mathrm{vec}(\\eta^\\top) 
            = 
            (X\\otimes I_K) \\beta + (\\mathbf{1}\\otimes I_K) \\beta_0 
            + 
            \\mathrm{vec}(\\eta^{0\\top})
        \\end{align*}

    where :math:`\\mathrm{vec}(x)` is the operator that flattens the input as column-major.
    Note that if ``intercept`` is ``True``, then an intercept for each class is provided
    as additional unpenalized features in the data matrix and the global intercept is turned off.

    .. note::
        Multi-response GLMs that are symmetric across the class coefficients (e.g. multinomial)
        may observe slow convergence due to the Hessian being singular.
        This becomes especially pronounced when ``X`` has strongly correlated features.
        Counter-intuitively, increasing ``tol`` such as ``1e-8`` may result in faster convergence
        since the quadratic approximation finds a more accurate solution,
        thereby warm-starting with a better initialization.

    Parameters
    ----------
    X : (n, p) matrix-like
        Feature matrix.
        It is typically one of the matrices defined in ``adelie.matrix`` submodule or ``np.ndarray``.
    glm : Union[adelie.glm.GlmBase32, adelie.glm.GlmBase64, adelie.glm.GlmMultiBase32, adelie.glm.GlmMultiBase64]
        GLM object.
        It is typically one of the GLM classes defined in ``adelie.glm`` submodule.
    groups : (G,) np.ndarray, optional
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
        If ``glm`` is multi-response type, then we only allow two types of groupings:

            - ``"grouped"``: coefficients for each predictor is grouped across the classes.
            - ``"ungrouped"``: every coefficient is its own group.

        Default is ``None``, in which case it is set to
        ``np.arange(p)`` if ``y`` is single-response
        and ``"grouped"`` if multi-response.
    alpha : float, optional
        Elastic net parameter.
        It must be in the range :math:`[0,1]`.
        Default is ``1``.
    penalty : (G,) np.ndarray, optional
        Penalty factor for each group in the same order as ``groups``.
        It must be a non-negative vector.
        Default is ``None``, in which case, it is set to ``np.sqrt(group_sizes)``.
    offsets : (n,) or (n, K) np.ndarray, optional
        Observation offsets :math:`\\eta^0`.
        Default is ``None``, in which case, it is set to 
        ``np.zeros(n)`` if ``y`` is single-response
        and ``np.zeros((n, K))`` if multi-response.
    lmda_path : (L,) np.ndarray, optional
        The regularization path to solve for.
        The full path is not considered if ``early_exit`` is ``True``.
        It is recommended that the path is sorted in decreasing order.
        If ``None``, the path will be generated.
        Default is ``None``.
    irls_max_iters : int, optional
        Maximum number of IRLS iterations.
        This parameter is only used if ``glm`` is not of gaussian type.
        Default is ``int(1e4)``.
    irls_tol : float, optional
        IRLS convergence tolerance.
        This parameter is only used if ``glm`` is not of gaussian type.
        Default is ``1e-7``.
    max_iters : int, optional
        Maximum number of coordinate descents.
        Default is ``int(1e5)``.
    tol : float, optional
        Coordinate descent convergence tolerance.
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
        ``True`` if the function should early exit based on training deviance explained.
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
        If ``y`` is multi-response, then an intercept for each class is added
        and the global intercept is turned off.
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
        largest active scores are used to determine the pivot point
        where ``s`` is the current screen set size.
        It is only used if ``screen_rule="pivot"``.
        Default is ``0.1``.
    pivot_subset_min : int, optional
        If screening takes place, then at least ``pivot_subset_min``
        number of active scores are used to determine the pivot point.
        It is only used if ``screen_rule="pivot"``.
        Default is ``1``.
    pivot_slack_ratio : float, optional
        If screening takes place, then ``pivot_slack_ratio``
        number of groups with next smallest (new) active scores 
        below the pivot point are also added to the screen set as slack.
        It is only used if ``screen_rule="pivot"``.
        Default is ``1.25``.
    check_state : bool, optional 
        ``True`` is state should be checked for inconsistencies before calling solver.
        Default is ``False``.

        .. warning::
            The check may take a long time if the inputs are big!

    progress_bar : bool, optional
        ``True`` to enable progress bar.
        Default is ``True``.
    warm_start : optional
        If no warm-start is provided, the initial solution is set to 0
        and other invariance quantities are set accordingly.
        Otherwise, the warm-start is used to extract all necessary state variables.
        If warm-start is used, the user *must* still provide consistent inputs,
        that is, warm-start will not overwrite most arguments passed into this function.
        However, changing configuration settings such as tolerance levels is well-defined.
        Default is ``None``.

        .. note::
            The primary use-case is when a user already called the function with ``warm_start=False``
            but would like to continue fitting down a longer path of regularizations.
            This way, the user does not have to restart the fit at the beginning,
            but can simply continue from the last returned state.

        .. warning::
            We have only tested warm-starts in the setting described in the note above,
            that is, when ``lmda_path`` and possibly static configurations have changed.
            Use with caution in other settings!

    Returns
    -------
    state
        The resulting state after running the solver.

    See Also
    --------
    adelie.adelie_core.state.StateGaussianNaive64
    adelie.adelie_core.state.StateGlmNaive64
    adelie.adelie_core.state.StateMultiGaussianNaive64
    adelie.adelie_core.state.StateMultiGlmNaive64
    """
    X_raw = X

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

    # compute common quantities
    if not (offsets is None): 
        if offsets.shape != glm.y.shape:
            raise RuntimeError("offsets must be same shape as y if not None.")
        offsets = np.array(offsets, order="C", copy=False, dtype=dtype)
    else:
        offsets = np.zeros(glm.y.shape, dtype=dtype)

    if not (lmda_path is None):
        # MUST evaluate the flip to be able to pass into C++ backend.
        lmda_path = np.array(np.flip(np.sort(lmda_path)))

    solver_args = {
        "X": X_raw,
        "alpha": alpha,
        "offsets": offsets,
        "lmda_path": lmda_path,
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

    # do special routine for optimized gaussian
    is_gaussian_opt = (
        (glm.name in ["gaussian", "multigaussian"]) and
        glm.opt
    )

    # add a few more configs in GLM case
    if not is_gaussian_opt:
        solver_args["glm"] = glm
        solver_args["irls_max_iters"] = irls_max_iters
        solver_args["irls_tol"] = irls_tol
    else:
        solver_args["y"] = glm.y
        solver_args["weights"] = glm.weights

    # multi-response GLMs
    if glm.is_multi:
        K = glm.y.shape[-1]

        if groups is None:
            groups = "grouped"
        if groups == "grouped":
            groups = K * np.arange(p, dtype=int)
        elif groups == "ungrouped":
            groups = np.arange(K * p, dtype=int)
        else:
            raise RuntimeError(
                "groups must be one of \"grouped\" or \"ungrouped\" for multi-response."
            )
        if intercept:
            groups = np.concatenate([np.arange(K), K + groups], dtype=int)
        group_sizes = np.concatenate([groups, [(p+intercept)*K]], dtype=int)
        group_sizes = group_sizes[1:] - group_sizes[:-1]

        if penalty is None:
            penalty = np.sqrt(group_sizes)
            if intercept:
                penalty[:K] = 0
        else:
            if intercept:
                penalty = np.concatenate([np.zeros(K), penalty])

        if warm_start is None:
            lmda = np.inf
            lmda_max = None
            screen_set = np.arange(groups.shape[0])[(penalty <= 0) | (alpha <= 0)]
            screen_beta = np.zeros(np.sum(group_sizes[screen_set]), dtype=dtype)
            screen_is_active = np.ones(screen_set.shape[0], dtype=bool)
        else:
            lmda = warm_start.lmda
            lmda_max = warm_start.lmda_max
            screen_set = warm_start.screen_set
            screen_beta = warm_start.screen_beta
            screen_is_active = warm_start.screen_is_active

        solver_args["groups"] = groups
        solver_args["group_sizes"] = group_sizes
        solver_args["penalty"] = penalty
        solver_args["lmda"] = lmda
        solver_args["lmda_max"] = lmda_max
        solver_args["screen_set"] = screen_set
        solver_args["screen_beta"] = screen_beta
        solver_args["screen_is_active"] = screen_is_active

        # represent the augmented X matrix as used in single-response reformatted problem.
        X_aug = matrix.kronecker_eye(X_raw, K, n_threads=n_threads)
        if intercept:
            X_aug = matrix.concatenate([
                matrix.kronecker_eye(
                    np.ones((n, 1)), K, n_threads=n_threads
                ),
                X_aug,
            ], n_threads=n_threads)

        # special gaussian case
        if is_gaussian_opt:
            y = glm.y
            weights = glm.weights
            weights_mscaled = weights / K
            if warm_start is None:
                ones = np.ones(n, dtype=dtype)
                X_means = np.empty(p, dtype=dtype)
                X.mul(ones, weights_mscaled, X_means)
                X_means = np.repeat(X_means, K)
                if intercept:
                    X_means = np.concatenate([
                        np.full(K, 1/K),
                        X_means,
                    ])
                y_off = y - offsets
                # variance of y that gaussian solver expects
                y_var = np.sum(weights_mscaled[:, None] * y_off ** 2)
                # variance for the null model with multi-intercept
                # R^2 can be initialized to MSE under intercept-model minus y_var.
                # This is a negative quantity in general, but will be corrected to 0
                # when the model fits the unpenalized (including intercept) term.
                # Then, supplying y_var as the normalization will result in R^2 
                # relative to the intercept-model.
                if intercept:
                    y_off_c = y_off - (y_off.T @ weights)[None] # NOT a typo: weights
                    yc_var = np.sum(weights_mscaled[:, None] * y_off_c ** 2)
                    rsq = yc_var - y_var
                    y_var = yc_var
                else:
                    rsq = 0
                resid = y_off.ravel()
                resid_sum = np.sum(weights_mscaled[:, None] * y_off)
                grad = np.empty(X_aug.cols(), dtype=dtype)
                weights_mscaled = np.repeat(weights_mscaled, K)
                X_aug.mul(resid, weights_mscaled, grad)
            else:
                X_means = warm_start.X_means
                y_var = warm_start.y_var
                rsq = warm_start.rsq
                resid = warm_start.resid
                resid_sum = warm_start.resid_sum
                grad = warm_start.grad

            solver_args["X_means"] = X_means
            solver_args["y_var"] = y_var
            solver_args["rsq"] = rsq
            solver_args["resid"] = resid
            solver_args["resid_sum"] = resid_sum
            solver_args["grad"] = grad

            state = ad.state.multigaussian_naive(**solver_args)
        
        # GLM case
        else:
            if warm_start is None:
                ones = np.ones(offsets.size, dtype=dtype)
                eta = offsets
                resid = np.empty(eta.shape, dtype=dtype)
                glm.gradient(eta, resid)
                resid = resid.ravel()
                grad = np.empty(X_aug.cols(), dtype=dtype)
                X_aug.mul(resid, ones, grad)
                loss_null = None
                loss_full = glm.loss_full()
                eta = eta.ravel()
            else:
                eta = warm_start.eta
                resid = warm_start.resid
                grad = warm_start.grad
                loss_null = warm_start.loss_null
                loss_full = warm_start.loss_full

            solver_args["grad"] = grad
            solver_args["eta"] = eta
            solver_args["resid"] = resid
            solver_args["loss_null"] = loss_null
            solver_args["loss_full"] = loss_full

            state = ad.state.multiglm_naive(**solver_args)

    # single-response GLMs
    else:
        if groups is None:
            groups = np.arange(p, dtype=int)
        group_sizes = np.concatenate([groups, [p]], dtype=int)
        group_sizes = group_sizes[1:] - group_sizes[:-1]

        G = len(groups)

        if penalty is None:
            penalty = np.sqrt(group_sizes)

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

        solver_args["groups"] = groups
        solver_args["group_sizes"] = group_sizes
        solver_args["penalty"] = penalty
        solver_args["lmda"] = lmda
        solver_args["lmda_max"] = lmda_max
        solver_args["screen_set"] = screen_set
        solver_args["screen_beta"] = screen_beta
        solver_args["screen_is_active"] = screen_is_active

        # special gaussian case
        if is_gaussian_opt:
            y = glm.y
            weights = glm.weights
            if warm_start is None:
                ones = np.ones(n, dtype=dtype)
                X_means = np.empty(p, dtype=dtype)
                X.mul(ones, weights, X_means)
                y_off = y - offsets
                y_mean = np.sum(y_off * weights)
                yc = y_off
                if intercept:
                    yc = yc - y_mean
                y_var = np.sum(weights * yc ** 2)
                rsq = 0
                resid = yc
                resid_sum = np.sum(weights * resid)
                grad = np.empty(p, dtype=dtype)
                X.mul(resid, weights, grad)
            else:
                X_means = warm_start.X_means
                y_mean = warm_start.y_mean
                y_var = warm_start.y_var
                rsq = warm_start.rsq
                resid = warm_start.resid
                resid_sum = warm_start.resid_sum
                grad = warm_start.grad

            solver_args["X_means"] = X_means
            solver_args["y_mean"] = y_mean
            solver_args["y_var"] = y_var
            solver_args["rsq"] = rsq
            solver_args["resid"] = resid
            solver_args["resid_sum"] = resid_sum
            solver_args["grad"] = grad

            state = ad.state.gaussian_naive(**solver_args)

        # GLM case
        else:
            if warm_start is None:
                ones = np.ones(n, dtype=dtype)
                beta0 = 0
                eta = offsets
                resid = np.empty(n, dtype=dtype)
                glm.gradient(eta, resid)
                grad = np.empty(p, dtype=dtype)
                X.mul(resid, ones, grad)
                loss_null = None
                loss_full = glm.loss_full()
            else:
                beta0 = warm_start.beta0
                eta = warm_start.eta
                resid = warm_start.resid
                grad = warm_start.grad
                loss_null = warm_start.loss_null
                loss_full = warm_start.loss_full

            solver_args["beta0"] = beta0
            solver_args["grad"] = grad
            solver_args["eta"] = eta
            solver_args["resid"] = resid
            solver_args["loss_null"] = loss_null
            solver_args["loss_full"] = loss_full
            state = ad.state.glm_naive(**solver_args)

    if check_state:
        state.check(method="assert")

    return _solve(
        state=state,
        progress_bar=progress_bar,
    )
