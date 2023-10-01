from . import adelie_core as core
import numpy as np
from dataclasses import dataclass


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
    groups : (G,) np.ndarray
        List of starting indices to each group.
    group_sizes : (G,) np.ndarray
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
    return core.solver.objective(
        beta, X, y, groups, group_sizes, lmda, alpha, penalty
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


def grpelnet_pin(
    state: pin_naive_dense,
):
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
    state : pin_naive_dense
        See the documentation for one of the listed types.

    Returns
    -------
    result : GroupElnetResult
        See ``adelie.solver.GroupElnetResult`` for details.

    See Also
    --------
    adelie.state.pin_naive_dense
    adelie.solver.GroupElnetResult
    """
    # mapping of each state type to the corresponding solver
    f_dict = {
        pin_naive_dense: core.solver.grpelnet_pin_naive_dense,
    }

    # solve group elastic net
    f = f_dict[type(state)]
    out = f(state._state)

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
