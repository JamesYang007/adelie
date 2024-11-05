from . import matrix
from .configs import Configs
from .constraint import (
    ConstraintBase32,
    ConstraintBase64,
)
from .glm import (
    GlmBase32,
    GlmBase64,
    GlmMultiBase32,
    GlmMultiBase64,
)
from .matrix import (
    MatrixConstraintBase32,
    MatrixConstraintBase64,
    MatrixCovBase32,
    MatrixCovBase64,
    MatrixNaiveBase32,
    MatrixNaiveBase64,
)
from .state import (
    bvls as state_bvls,
    css_cov as state_css_cov,
    gaussian_cov as state_gaussian_cov,
    gaussian_naive as state_gaussian_naive,
    glm_naive as state_glm_naive,
    multigaussian_naive as state_multigaussian_naive,
    multiglm_naive as state_multiglm_naive,
    pinball as state_pinball,
) 
from scipy.sparse import csr_matrix
from typing import (
    Callable,
    Union,
)
import numpy as np


def gaussian_cov(
    A: Union[np.ndarray, MatrixCovBase32, MatrixCovBase64],
    v: np.ndarray,
    *,
    constraints: list[Union[ConstraintBase32, ConstraintBase64]] =None,
    groups: np.ndarray =None,
    alpha: float =1,
    penalty: np.ndarray =None,
    lmda_path: np.ndarray =None,
    max_iters: int =int(1e5),
    tol: float =1e-7,
    rdev_tol: float =1e-3,
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
    exit_cond: Callable =None,
):
    """Solves Gaussian group elastic net via covariance method.

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
    A : (p, p) Union[ndarray, MatrixCovBase64, MatrixCovBase32]
        Positive semi-definite matrix.
        It is typically one of the matrices defined in :mod:`adelie.matrix` submodule or :class:`numpy.ndarray`.
    v : (p,) ndarray
        Linear term.
    constraints : (G,) list[Union[ConstraintBase32, ConstraintBase64]], optional
        List of constraints for each group.
        ``constraints[i]`` is the constraint object corresponding to group ``i``.
        If ``constraints[i]`` is ``None``, then the ``i`` th group is unconstrained.
        If ``None``, every group is unconstrained.
        Default is ``None``.
    groups : (G,) ndarray, optional
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
        Default is ``None``, in which case it is set to ``np.arange(p)``.
    alpha : float, optional
        Elastic net parameter.
        It must be in the range :math:`[0,1]`.
        Default is ``1``.
    penalty : (G,) ndarray, optional
        Penalty factor for each group in the same order as ``groups``.
        It must be a non-negative vector.
        Default is ``None``, in which case, it is set to ``np.sqrt(group_sizes)``.
    lmda_path : (L,) ndarray, optional
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

    exit_cond : Callable, optional
        If not ``None``, it must be a callable object that takes in a single argument.
        The argument is the current state object of the same type as the return value.
        During the optimization, 
        after obtaining the solution at each regularization value,
        ``exit_cond(state)`` is evaluated as an opportunity
        for the user to early exit the program based on their own rule.
        Default is ``None``.

        .. note::
            The algorithm early exits if ``exit_cond(state)``
            evaluates to ``True`` *or* the built-in early exit
            function evaluates to ``True`` (if ``early_exit`` is ``True``).

    Returns
    -------
    state
        The resulting state after running the solver.

    See Also
    --------
    adelie.adelie_core.state.StateGaussianCov32
    adelie.adelie_core.state.StateGaussianCov64
    """
    if isinstance(A, np.ndarray):
        A = matrix.dense(A, method="cov", n_threads=n_threads)

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

    # clear cached information for every constraint object
    if isinstance(constraints, list):
        for c in constraints:
            if c is None: continue
            c.clear()

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
        active_set_size = screen_set.shape[0]
        active_set = np.empty(G, dtype=int)
        active_set[:active_set_size] = np.arange(active_set_size)
        rsq = 0

        subset = np.array([
            np.arange(groups[ss], groups[ss] + group_sizes[ss]) 
            for ss in screen_set
        ])
        order = np.argsort(subset)
        indices = subset[order]
        values = screen_beta[order]

        grad = np.empty(p, dtype=dtype)
        A.mul(indices, values, grad)
        grad = v - grad

    else:
        lmda = warm_start.lmda
        lmda_max = warm_start.lmda_max
        screen_set = warm_start.screen_set
        screen_beta = warm_start.screen_beta
        screen_is_active = warm_start.screen_is_active
        active_set_size = warm_start.active_set_size
        active_set = warm_start.active_set
        rsq = warm_start.rsq
        grad = warm_start.grad

    state = state_gaussian_cov(
        A=A,
        v=v,
        constraints=constraints,
        groups=groups,
        group_sizes=group_sizes,
        alpha=alpha,
        penalty=penalty,
        screen_set=screen_set,
        screen_beta=screen_beta,
        screen_is_active=screen_is_active,
        active_set_size=active_set_size,
        active_set=active_set,
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

    return state.solve(
        progress_bar=progress_bar,
        exit_cond=exit_cond,
    )


def grpnet(
    X: Union[np.ndarray, MatrixNaiveBase32, MatrixNaiveBase64],
    glm: Union[GlmBase32, GlmBase64, GlmMultiBase32, GlmMultiBase64],
    *,
    constraints: list[Union[ConstraintBase32, ConstraintBase64]] =None,
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
    ddev_tol: float =0,
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
    exit_cond: Callable =None,
):
    """Solves group elastic net via naive method.

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

    where :math:`\\mathrm{vec}(\\cdot)` is the operator that flattens the input as column-major.
    Note that if ``intercept`` is ``True``, then an intercept for each class is provided
    as additional unpenalized features in the data matrix and the global intercept is turned off.

    Parameters
    ----------
    X : (n, p) Union[ndarray, MatrixNaiveBase32, MatrixNaiveBase64]
        Feature matrix.
        It is typically one of the matrices defined in :mod:`adelie.matrix` submodule or :class:`numpy.ndarray`.
    glm : Union[GlmBase32, GlmBase64, GlmMultiBase32, GlmMultiBase64]
        GLM object.
        It is typically one of the GLM classes defined in :mod:`adelie.glm` submodule.
    constraints : (G,) list[Union[ConstraintBase32, ConstraintBase64]], optional
        List of constraints for each group.
        ``constraints[i]`` is the constraint object corresponding to group ``i``.
        If ``constraints[i]`` is ``None``, then the ``i`` th group is unconstrained.
        If ``None``, every group is unconstrained.
        Default is ``None``.
    groups : (G,) ndarray, optional
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
        If ``glm`` is of multi-response type, then
        ``groups[i]`` is the starting *feature* index of the ``i`` th group.
        In either case, ``groups[i]`` must then be a value in the range :math:`\\{1,\\ldots, p\\}`.
        Default is ``None``, in which case it is set to ``np.arange(p)``.
    alpha : float, optional
        Elastic net parameter.
        It must be in the range :math:`[0,1]`.
        Default is ``1``.
    penalty : (G,) ndarray, optional
        Penalty factor for each group in the same order as ``groups``.
        It must be a non-negative vector.
        Default is ``None``, in which case, it is set to ``np.sqrt(group_sizes)``.
    offsets : (n,) or (n, K) ndarray, optional
        Observation offsets :math:`\\eta^0`.
        Default is ``None``, in which case, it is set to 
        ``np.zeros(n)`` if ``y`` is single-response
        and ``np.zeros((n, K))`` if multi-response.
    lmda_path : (L,) ndarray, optional
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
        Default is ``0``.
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

    exit_cond : Callable, optional
        If not ``None``, it must be a callable object that takes in a single argument.
        The argument is the current state object of the same type as the return value.
        During the optimization, 
        after obtaining the solution at each regularization value,
        ``exit_cond(state)`` is evaluated as an opportunity
        for the user to early exit the program based on their own rule.
        Default is ``None``.

        .. note::
            The algorithm early exits if ``exit_cond(state)``
            evaluates to ``True`` *or* the built-in early exit
            function evaluates to ``True`` (if ``early_exit`` is ``True``).
            The latter can be disabled with ``early_exit=False``.

    Returns
    -------
    state
        The resulting state after running the solver.

    See Also
    --------
    adelie.adelie_core.state.StateGaussianNaive32
    adelie.adelie_core.state.StateGaussianNaive64
    adelie.adelie_core.state.StateGlmNaive32
    adelie.adelie_core.state.StateGlmNaive64
    adelie.adelie_core.state.StateMultiGaussianNaive32
    adelie.adelie_core.state.StateMultiGaussianNaive64
    adelie.adelie_core.state.StateMultiGlmNaive32
    adelie.adelie_core.state.StateMultiGlmNaive64
    """
    X_raw = X

    if isinstance(X, np.ndarray):
        X = matrix.dense(X, method="naive", n_threads=n_threads)

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

    # clear cached information for every constraint object
    if isinstance(constraints, list):
        for c in constraints:
            if c is None: continue
            c.clear()

    # compute common quantities
    if not (offsets is None): 
        if offsets.shape != glm.y.shape:
            raise RuntimeError("offsets must be same shape as y if not None.")
        offsets = np.array(offsets, order="C", copy=False, dtype=dtype)
    else:
        offsets = np.zeros(glm.y.shape, dtype=dtype)

    if not (lmda_path is None):
        # MUST evaluate the flip to be able to pass into C++ backend.
        lmda_path = np.array(np.flip(np.sort(lmda_path)), dtype=dtype)

    solver_args = {
        "X": X_raw,
        "constraints": constraints,
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

    if groups is None:
        groups = np.arange(p, dtype=int)

    # multi-response GLMs
    if glm.is_multi:
        K = glm.y.shape[-1]

        # flatten the grouping index across the classes
        groups = groups * K

        if intercept:
            groups = np.concatenate([np.arange(K), K + groups], dtype=int)
        group_sizes = np.concatenate([groups, [(p+intercept)*K]], dtype=int)
        group_sizes = group_sizes[1:] - group_sizes[:-1]

        if penalty is None:
            penalty = np.sqrt(group_sizes).astype(dtype)
            if intercept:
                penalty[:K] = 0
        else:
            if intercept:
                penalty = np.concatenate([np.zeros(K), penalty], dtype=dtype)

        if warm_start is None:
            lmda = np.inf
            lmda_max = None
            screen_set = np.arange(groups.shape[0])[(penalty <= 0) | (alpha <= 0)]
            screen_beta = np.zeros(np.sum(group_sizes[screen_set]), dtype=dtype)
            screen_is_active = np.ones(screen_set.shape[0], dtype=bool)
            active_set_size = screen_set.shape[0]
            active_set = np.empty(groups.shape[0], dtype=int)
            active_set[:active_set_size] = np.arange(active_set_size)
        else:
            lmda = warm_start.lmda
            lmda_max = warm_start.lmda_max
            screen_set = warm_start.screen_set
            screen_beta = warm_start.screen_beta
            screen_is_active = warm_start.screen_is_active
            active_set_size = warm_start.active_set_size
            active_set = warm_start.active_set

        solver_args["groups"] = groups
        solver_args["group_sizes"] = group_sizes
        solver_args["penalty"] = penalty
        solver_args["lmda"] = lmda
        solver_args["lmda_max"] = lmda_max
        solver_args["screen_set"] = screen_set
        solver_args["screen_beta"] = screen_beta
        solver_args["screen_is_active"] = screen_is_active
        solver_args["active_set_size"] = active_set_size
        solver_args["active_set"] = active_set

        # represent the augmented X matrix as used in single-response reformatted problem.
        X_aug = matrix.kronecker_eye(X_raw, K, n_threads=n_threads)
        if intercept:
            X_aug = matrix.concatenate(
                [
                    matrix.kronecker_eye(
                        np.ones((n, 1), dtype=dtype), 
                        K, 
                        n_threads=n_threads,
                    ),
                    X_aug,
                ], 
                axis=1, 
                n_threads=n_threads,
            )

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
                    ], dtype=dtype)
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

            state = state_multigaussian_naive(**solver_args)
        
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

            state = state_multiglm_naive(**solver_args)

    # single-response GLMs
    else:
        group_sizes = np.concatenate([groups, [p]], dtype=int)
        group_sizes = group_sizes[1:] - group_sizes[:-1]

        G = len(groups)

        if penalty is None:
            penalty = np.sqrt(group_sizes).astype(dtype)

        if warm_start is None:
            lmda = np.inf
            lmda_max = None
            screen_set = np.arange(G)[(penalty <= 0) | (alpha <= 0)]
            screen_beta = np.zeros(np.sum(group_sizes[screen_set]), dtype=dtype)
            screen_is_active = np.ones(screen_set.shape[0], dtype=bool)
            active_set_size = screen_set.shape[0]
            active_set = np.empty(groups.shape[0], dtype=int)
            active_set[:active_set_size] = np.arange(active_set_size)

        else:
            lmda = warm_start.lmda
            lmda_max = warm_start.lmda_max
            screen_set = warm_start.screen_set
            screen_beta = warm_start.screen_beta
            screen_is_active = warm_start.screen_is_active
            active_set_size = warm_start.active_set_size
            active_set = warm_start.active_set

        solver_args["groups"] = groups
        solver_args["group_sizes"] = group_sizes
        solver_args["penalty"] = penalty
        solver_args["lmda"] = lmda
        solver_args["lmda_max"] = lmda_max
        solver_args["screen_set"] = screen_set
        solver_args["screen_beta"] = screen_beta
        solver_args["screen_is_active"] = screen_is_active
        solver_args["active_set_size"] = active_set_size
        solver_args["active_set"] = active_set

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

            state = state_gaussian_naive(**solver_args)

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
            state = state_glm_naive(**solver_args)

    if check_state:
        state.check(method="assert")

    return state.solve(
        progress_bar=progress_bar,
        exit_cond=exit_cond,
    )


def bvls(
    X: Union[np.ndarray, MatrixNaiveBase32, MatrixNaiveBase64],
    y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    weights: np.ndarray =None,
    kappa: int =None,
    max_iters: int =int(1e5),
    tol: float =1e-7,
    n_threads: int =1,
    warm_start =None,
):
    """Solves bounded variable least squares.

    The bounded variable least squares is given by

    .. math::
        \\begin{align*}
            \\mathrm{minimize}_{\\beta} &\\quad
            \\frac{1}{2} \\|y - X \\beta\\|_{W}^2 \\\\
            \\text{subject to} &\\quad
            \\ell \\leq \\beta \\leq u
        \\end{align*}

    where 
    :math:`X \\in \\mathbb{R}^{n \\times p}` is the feature matrix,
    :math:`y \\in \\mathbb{R}^n` is the response vector,
    :math:`W \\in \\mathbb{R}_+^{n \\times n}` is the (diagonal) observation weights,
    and :math:`\\ell \\leq u \\in \\mathbb{R}^p` are the lower and upper bounds, respectively.

    Parameters
    ----------
    X : (n, p) Union[ndarray, MatrixNaiveBase32, MatrixNaiveBase64]
        Feature matrix.
        It is typically one of the matrices defined in :mod:`adelie.matrix` submodule or :class:`numpy.ndarray`.
    y : (n,) ndarray
        Response vector.
    lower : (p,) ndarray
        Lower bound for each variable.
    upper : (p,) ndarray
        Upper bound for each variable.
    weights : (n,) ndarray, optional
        Observation weights.
        If ``None``, it is set to ``np.full(n, 1/n)``.
        Default is ``None``.
    kappa : int, optional
        Violation batching size.
        If ``None``, it is set to ``min(n, p)``.
        Default is ``None``.
    max_iters : int, optional
        Maximum number of coordinate descents.
        Default is ``int(1e5)``.
    tol : float, optional
        Convergence tolerance.
        Default is ``1e-7``.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.
    warm_start : optional
        If no warm-start is provided, the initial solution is set to the vertex of the box closest to the origin.
        Otherwise, the warm-start is used to extract all necessary state variables.
        Default is ``None``.

    Returns
    -------
    state
        The resulting state after running the solver.

    See Also
    --------
    adelie.adelie_core.state.StateBVLS32
    adelie.adelie_core.state.StateBVLS64
    """
    X_raw = X

    if isinstance(X, np.ndarray):
        X = matrix.dense(X, method="naive", n_threads=n_threads)

    assert (
        isinstance(X, matrix.MatrixNaiveBase64) or
        isinstance(X, matrix.MatrixNaiveBase32)
    )

    dtype = (
        np.float64
        if isinstance(X, matrix.MatrixNaiveBase64) else
        np.float32
    )

    n, p = X.shape

    if weights is None:
        weights = np.full(n, 1 / n)
    if kappa is None:
        kappa = min(n, p)
    y_var = np.sum(y ** 2 * weights)

    if isinstance(X_raw, np.ndarray):
        X_vars = np.sum(weights[:, None] * X_raw ** 2, axis=0)

    else:
        X_vars = np.zeros(X.shape[1], dtype)
        X.sq_mul(weights, X_vars)

    lower = np.maximum(lower, -Configs.max_solver_value)
    upper = np.minimum(upper,  Configs.max_solver_value)

    if warm_start is None:
        beta = np.where(np.abs(lower) < np.abs(upper), lower, upper)
        active_set = np.empty(p, dtype=int)
        active_set_size = 0
        is_active = np.zeros(p, dtype=bool)
        screen_set = np.empty(p, dtype=int)
        screen_set_size = 0
        is_screen = np.zeros(p, dtype=bool)

    else:
        beta = warm_start.beta
        active_set = warm_start.active_set
        active_set_size = warm_start.active_set_size
        is_active = warm_start.is_active
        screen_set = warm_start.active_set
        screen_set_size = warm_start.active_set_size
        is_screen = warm_start.is_active

    if isinstance(X_raw, np.ndarray):
        resid = y - X_raw @ beta
    else:
        resid = y - (X @ csr_matrix(beta[None]).T)[:, 0]
    grad = np.empty(p, dtype=dtype)
    loss = 0.5 * np.sum(resid ** 2 * weights)

    state = state_bvls(
        X=X,
        y_var=y_var,
        X_vars=X_vars,
        lower=lower,
        upper=upper,
        weights=weights,
        kappa=kappa,
        max_iters=max_iters,
        tol=tol,
        screen_set_size=screen_set_size,
        screen_set=screen_set,
        is_screen=is_screen,
        active_set_size=active_set_size,
        active_set=active_set,
        is_active=is_active,
        beta=beta,
        resid=resid,
        grad=grad,
        loss=loss,
    )

    return state.solve()


def pinball(
    A: Union[np.ndarray, MatrixConstraintBase32, MatrixConstraintBase64],
    S: np.ndarray,
    v: np.ndarray,
    penalty_neg: np.ndarray,
    penalty_pos: np.ndarray,
    *,
    kappa: int =None,
    max_iters: int =int(1e5),
    tol: float =1e-7,
    n_threads: int =1,
    warm_start =None,
):
    """Solves pinball least squares.

    The pinball least squares is given by

    .. math::
        \\begin{align*}
            \\mathrm{minimize}_{\\beta} &\\quad
            \\frac{1}{2} \\|S^{-\\frac{1}{2}} v - S^{\\frac{1}{2}} A^\\top \\beta\\|_{2}^2 
            + \\ell^\\top \\beta_- + u^\\top \\beta_+
        \\end{align*}

    where 
    :math:`A \\in \\mathbb{R}^{m \\times d}` is a constraint matrix,
    :math:`S \\in \\mathbb{R}^{d \\times d}` is a positive semi-definite matrix,
    :math:`v \\in \\mathbb{R}^d` is the linear term,
    and :math:`\\ell, u \\in \\mathbb{R}^m` are the penalty factors 
    for the negative and positive parts of :math:`\\beta`, respectively.

    Parameters
    ----------
    A : (m, d) Union[ndarray, MatrixConstraintBase32, MatrixConstraintBase64]
        Constraint matrix :math:`A`.
        It is typically one of the matrices defined in :mod:`adelie.matrix` submodule.
    S : (d, d) ndarray
        Positive semi-definite matrix :math:`S`.
    v : (n,) ndarray
        Linear term :math:`v`.
    penalty_neg : (m,) ndarray
        Penalty :math:`\\ell` on the negative part of :math:`\\beta`.
    penalty_pos : (m,) ndarray
        Penalty :math:`u` on the positive part of :math:`\\beta`.
    kappa : int, optional
        Violation batching size.
        If ``None``, it is set to ``min(m, d)``.
        Default is ``None``.
    max_iters : int, optional
        Maximum number of coordinate descents.
        Default is ``int(1e5)``.
    tol : float, optional
        Coordinate descent convergence tolerance.
        Default is ``1e-7``.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.
    warm_start : optional
        If no warm-start is provided, the initial solution is set to all zeros.
        Otherwise, the warm-start is used to extract all necessary state variables.
        Default is ``None``.

    Returns
    -------
    state
        The resulting state after running the solver.

    See Also
    --------
    adelie.adelie_core.state.StatePinball32
    adelie.adelie_core.state.StatePinball64
    """
    if isinstance(A, np.ndarray):
        A = matrix.dense(A, method="constraint", n_threads=n_threads)

    assert (
        isinstance(A, matrix.MatrixConstraintBase64) or
        isinstance(A, matrix.MatrixConstraintBase32)
    )

    dtype = (
        np.float64
        if isinstance(A, matrix.MatrixConstraintBase64) else
        np.float32
    )

    m, d = A.shape

    if kappa is None:
        kappa = min(m, d)
    y_var = v @ np.linalg.solve(S, v)

    penalty_neg = np.minimum(penalty_neg, Configs.max_solver_value)
    penalty_pos = np.minimum(penalty_pos, Configs.max_solver_value)

    if warm_start is None:
        screen_set_size = 0
        screen_set = np.empty(m, dtype=int)
        is_screen = np.zeros(m, dtype=bool)
        screen_ASAT_diag = np.empty(m, dtype=dtype)
        screen_AS = np.empty((m, d), dtype=dtype, order="C")
        active_set_size = 0
        active_set = np.empty(m, dtype=int)
        is_active = np.zeros(m, dtype=bool)
        beta = np.zeros(m, dtype=dtype)
        resid = np.array(v, dtype=dtype)
        loss = 0.5 * y_var

    else:
        screen_set_size = warm_start.active_set_size
        screen_set = warm_start.active_set
        is_screen = warm_start.is_active
        screen_ASAT_diag = np.empty(m, dtype=dtype)
        screen_AS = np.empty((m, d), dtype=dtype, order="C")
        for i in range(screen_set_size):
            k = screen_set[i]
            A.rmmul(k, S, screen_AS[k])
            screen_ASAT_diag[k] = max(A.rvmul(k, screen_AS[k]), 0)
        active_set_size = warm_start.active_set_size
        active_set = warm_start.active_set
        is_active = warm_start.is_active
        beta = warm_start.beta
        resid = np.empty(d, dtype=dtype)
        A.mul(beta, resid)
        resid = v - S @ resid
        loss = 0.5 * resid @ np.linalg.solve(S, resid)

    grad = np.empty(m, dtype=dtype)

    state = state_pinball(
        A=A,
        y_var=y_var,
        S=S,
        penalty_neg=penalty_neg,
        penalty_pos=penalty_pos,
        kappa=kappa,
        max_iters=max_iters,
        tol=tol,
        screen_set_size=screen_set_size,
        screen_set=screen_set,
        is_screen=is_screen,
        screen_ASAT_diag=screen_ASAT_diag,
        screen_AS=screen_AS,
        active_set_size=active_set_size,
        active_set=active_set,
        is_active=is_active,
        beta=beta,
        resid=resid,
        grad=grad,
        loss=loss,
    )

    return state.solve()


def css_cov(
    S: np.ndarray,
    subset_size: int =None,
    *,
    subset: np.ndarray =None,
    method: str ="swapping",
    loss: str ="least_squares",
    n_threads: int =1,
):
    """Solves column subset selection via covariance method.

    Column subset selection via covariance method solves

    .. math::
        \\begin{align*}
            \\mathrm{minimize}_{T \\subseteq [p] : |T|=k} &\\quad
            \\ell\\left(
                \\Sigma, T
            \\right)
        \\end{align*}

    for a variety of loss functions :math:`\\ell`
    where 
    :math:`\\Sigma \\in \\mathbb{R}^{p \\times p}` is a positive semi-definite matrix
    and :math:`T` is an index set of size :math:`k`.

    The least squares loss is given by

    .. math::
        \\begin{align*}
            \\ell(\\Sigma, T)
            =
            \\mathrm{Tr}\\left(
                \\Sigma - \\Sigma_{\\cdot T} \\Sigma_{T,T}^\\dagger \\Sigma_{T \\cdot}
            \\right)
        \\end{align*}

    The subset factor loss is given by

    .. math::
        \\begin{align*}
            \\ell(\\Sigma, T)
            =
            \\log|\\Sigma_T| 
            +
            \\log(|\\mathrm{diag}(
                \\Sigma_{-T,-T} - \\Sigma_{-T,T} \\Sigma_{T,T}^\\dagger \\Sigma_{T,-T}
            )|)
        \\end{align*}

    The minimum determinant loss is given by

    .. math::
        \\begin{align*}
            \\ell(\\Sigma, T)
            =
            \\left|\\Sigma_T\\right|
        \\end{align*}

    .. note::
        The greedy method is generally significantly faster than the swapping method.
        However, the swapping method yields a much more accurate solution to the CSS problem.
        We recommend using the greedy method only if the swapping method is too time-consuming
        or an accurate solution is not necessary.

    Parameters
    ----------
    S : (p, p) ndarray
        Positive semi-definite matrix :math:`\\Sigma`.
    subset_size : int, optional
        Subset size :math:`k`.
        It must satisfy the following conditions for each method type:

            - ``"greedy"``: must be an integer.
            - ``"swapping"``: must satisfy the conditions for ``"greedy"`` 
              if ``subset`` is ``None``.
              Otherwise, it is ignored.

        Default is ``None``.  
    subset : ndarray, optional
        Initial subset :math:`T`.
        This argument is only used by the swapping method.
        If ``None``, the greedy method is used 
        to first initialize a subset of size ``subset_size``.
        Default is ``None``.
    method : str, optional
        Search method to identify the optimal :math:`T`. 
        It must be one of the following:
        
            - ``"greedy"``: greedy method.
            - ``"swapping"``: swapping method.

        Default is ``"swapping"``.
    loss : str, optional
        Loss type. It must be one of the following:

            - ``"least_squares"``: least squares loss.
            - ``"subset_factor"``: subset factor loss.
            - ``"min_det"``: minimum determinant loss.

        Default is ``"least_squares"``.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.

    Returns
    -------
    state
        The resulting state after running the solver.

    See Also
    --------
    adelie.adelie_core.state.StateCSSCov32
    adelie.adelie_core.state.StateCSSCov64
    """
    if method == "greedy":
        if not isinstance(subset_size, int):
            raise ValueError("subset_size must be an integer for the greedy method.")
        subset = np.empty(0, dtype=int)

    if method == "swapping":
        if subset is None:
            subset = css_cov(
                S=S, 
                subset_size=subset_size,
                method="greedy",
                loss=loss,
                n_threads=n_threads,
            ).subset
        subset = np.array(subset, dtype=int)
        subset_size = subset.size

    state = state_css_cov(
        S=S,
        subset_size=subset_size,
        subset=subset,
        method=method,
        loss=loss,
        n_threads=n_threads
    )

    return state.solve()