from . import adelie_core as core
import numpy as np
import scipy
from dataclasses import dataclass, asdict


@dataclass
class pin_base:
    """Base class for all state classes.

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
    """
    # Static states
    X: np.ndarray # TODO: add more types
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

    # Configuration
    max_cds: int
    thr: float
    cond_0_thresh: float
    cond_1_thresh: float
    newton_tol: float
    newton_max_iters: int

    # Dynamic states
    rsq: float
    strong_beta: np.ndarray
    strong_grad: np.ndarray
    active_set: np.ndarray
    active_g1: np.ndarray
    active_g2: np.ndarray
    active_begins: np.ndarray
    active_order: np.ndarray
    is_active: np.ndarray
    betas: scipy.sparse.csr_matrix
    rsqs: np.ndarray
    n_cds: int
    time_strong_cd: np.ndarray
    time_active_cd: np.ndarray


@dataclass
class pin_naive(pin_base):
    """Base class for pin naive state classes.

    Parameters
    ----------
    resid : (n,) np.ndarray
        Initial residual :math:`y-X\\beta` at ``strong_beta``.
    resids : (l, n) np.ndarray
        ``resids[i]`` corresponds to the solution residual corresponding to ``lmdas[i]``.
    """
    resid: np.ndarray
    resids: list[np.ndarray]

    def __init__(
        self, 
        X: np.ndarray,
        groups: np.ndarray,
        group_sizes: np.ndarray,
        alpha: float,
        penalty: np.ndarray,
        strong_set: np.ndarray,
        strong_g1: np.ndarray,
        strong_g2: np.ndarray,
        strong_begins: np.ndarray,
        strong_A_diag: np.ndarray,
        lmdas: np.ndarray,
        max_cds: int,
        thr: float,
        cond_0_thresh: float,
        cond_1_thresh: float,
        newton_tol: float,
        newton_max_iters: int,
        rsq: float,
        resid: np.ndarray,
        strong_beta: np.ndarray,
        strong_grad: np.ndarray,
        active_set: np.ndarray,
        active_g1: np.ndarray,
        active_g2: np.ndarray,
        active_begins: np.ndarray,
        active_order: np.ndarray,
        is_active: np.ndarray,
        betas: scipy.sparse.csr_matrix,
        rsqs: np.ndarray,
        resids: np.ndarray
    ):
        self._core_state = core.state.PinNaive(
            X=X,
            groups=groups,
            group_sizes=group_sizes,
            alpha=alpha,
            penalty=penalty,
            strong_set=strong_set,
            strong_g1=strong_g1,
            strong_g2=strong_g2,
            strong_begins=strong_begins,
            strong_A_diag=strong_A_diag,
            lmdas=lmdas,
            max_cds=max_cds,
            thr=thr,
            cond_0_thresh=cond_0_thresh,
            cond_1_thresh=cond_1_thresh,
            newton_tol=newton_tol,
            newton_max_iters=newton_max_iters,
            rsq=rsq,
            resid=resid,
            strong_beta=strong_beta,
            strong_grad=strong_grad,
            active_set=active_set,
            active_g1=active_g1,
            active_g2=active_g2,
            active_begins=active_begins,
            active_order=active_order,
            is_active=is_active,
            betas=betas,
            rsqs=rsqs,
            resids=resids,
        )
        self.cache_core_state(core_state)
