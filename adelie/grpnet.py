from . import adelie_core as core
from .state import (
    pin_naive,
)
from copy import copy
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


def solve_pin(
    state: pin_naive,
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
    state : pin_naive
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
        pin_naive: core.grpnet.solve_pin_naive_64,
    }

    # solve group elastic net
    f = f_dict[type(state)]
    out = f(state.internal())

    # raise any errors
    if out["error"] != "":
        raise RuntimeError(out["error"])

    # return a subsetted Python result object
    core_state = out["state"]
    state = copy(state)
    state.initialize(core_state)

    return state