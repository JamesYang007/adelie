from . import adelie_core as core
from . import logger
from .state import (
    pin_naive_64,
    pin_naive_32,
    pin_cov_64,
    pin_cov_32,
)
import numpy as np


def objective(
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
            \\frac{1}{2} \\|y - X\\beta\\|_2^2
            + \\lambda \\sum\\limits_{j=1}^G w_j \\left(
                \\alpha \\|\\beta_j\\|_2 + \\frac{1-\\alpha}{2} \\|\\beta_j\\|_2^2
            \\right)
        \\end{align*}

    where 
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
    return core.grpnet.objective(
        beta, X, y, groups, group_sizes, lmda, alpha, penalty,
    )


def lambda_max(
    *,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    group_sizes: np.ndarray,
    alpha: float,
    penalty: np.ndarray,
):
    """Computes the largest :math:`\\lambda` in the regularization sequence.

    The largest :math:`\\lambda` in the sequence is the smallest
    :math:`\\lambda` such that the optimal coefficient vector :math:`\\beta \\equiv 0`.

    TODO
    """
    grad = X.T @ y
    abs_grad = np.array([
        np.linalg.norm(grad[g:g+gs])
        for g, gs in zip(groups, group_sizes)
    ])
    return core.grpnet.lambda_max(
        abs_grad, alpha, penalty,
    )


def create_lambdas(
    *,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    group_sizes: np.ndarray,
    alpha: float,
    penalty: np.ndarray,
    n_lambdas: int =100,
    min_ratio: float =1e-4,
):
    """Creates a log-spaced regularization sequence.

    The generated sequence has length ``n_lambdas``.
    It is decreasing and equally-spaced
    in log-scale such that the ratio of the smallest to largest is given by ``min_ratio``.

    TODO
    """
    lmda_max = lambda_max(
        X=X, 
        y=y, 
        groups=groups, 
        group_sizes=group_sizes, 
        alpha=alpha, 
        penalty=penalty,
    )
    return core.grpnet.create_lambdas(
        n_lambdas,
        min_ratio,
        lmda_max,
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
    We also assume that :math:`X` is such that the column blocks :math:`X_k`
    defined by the groups have diagonal :math:`X_k^\\top X_k`.

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
    adelie.grpnet.objective
    """
    # mapping of each state type to the corresponding solver
    f_dict = {
        pin_naive_64: core.grpnet.solve_pin_naive_64,
        pin_naive_32: core.grpnet.solve_pin_naive_32,
        pin_cov_64: core.grpnet.solve_pin_cov_64,
        pin_cov_32: core.grpnet.solve_pin_cov_32,
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