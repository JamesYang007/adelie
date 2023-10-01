from . import adelie_core as core
import numpy as np
import scipy
from dataclasses import dataclass 


@dataclass
class pin_base:
    """Base state class for pin methods.

    Parameters
    ----------
    X : (n, p) array-like
        Feature matrix where each column blocks :math:`X_k` defined by the groups
        is such that :math:`X_k^\\top X_k` is diagonal.
    groups : (G,) np.ndarray
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
    group_sizes : (G,) np.ndarray
        List of group sizes corresponding to each element in ``groups``.
        ``group_sizes[i]`` is the group size of the ``i`` th group. 
    alpha : float
        Elastic net parameter.
    penalty : (G,) np.ndarray
        Penalty factor for each group in the same order as ``groups``.
    strong_set : (s,) np.ndarray
        List of strong groups taking on values in ``[0, G)``.
        ``strong_set[i]`` is ``i`` th strong group.
    strong_g1 : (s1,) np.ndarray
        Subset of ``strong_set`` that correspond to groups of size ``1``.
        ``strong_g1[i]`` is the ``i`` th strong group of size ``1``
        such that ``group_sizes[strong_g1[i]]`` is ``1``.
    strong_g2 : (s2,) np.ndarray
        Subset of ``strong_set`` that correspond to groups more than size ``1``.
        ``strong_g2[i]`` is the ``i`` th strong group of size more than ``1``
        such that ``group_sizes[strong_g2[i]]`` is more than ``1``.
    strong_begins : (s,) np.ndarray
        List of indices that index a corresponding list of values for each strong group.
        ``strong_begins[i]`` is the starting index corresponding to the ``i`` th strong group.
    strong_A_diag : (ws,) np.ndarray
        List of the diagonal of :math:`X_k^\\top X_k` along the strong groups ``k``.
        ``strong_A_diag[b:b+p]`` is the diagonal of :math:`X_k^\\top X_k` for the ``i`` th strong group where
        ``k = strong_set[i]``,
        ``b = strong_begins[i]``,
        and ``p = group_sizes[k]``.
    lmdas : (l,) np.ndarray
        Regularization sequence to fit on.
    max_cds : int
        Maximum number of coordinate descents.
    thr : float
        Convergence tolerance.
    cond_0_thresh : float
        Early stopping rule check on slope of :math:`R^2`.
    cond_1_thresh : float
        Early stopping rule check on curvature of :math:`R^2`.
    newton_tol : float
        Convergence tolerance for the BCD update.
    newton_max_iters : int
        Maximum number of iterations for the BCD update.
    rsq : float
        :math:`R^2` value at ``strong_beta``.
    strong_beta : (ws,) np.ndarray
        Coefficient vector on the strong set.
        ``strong_beta[b:b+p]`` is the coefficient for the ``i`` th strong group 
        where
        ``k = strong_set[i]``,
        ``b = strong_begins[i]``,
        and ``p = group_sizes[k]``.
    strong_grad : (ws,) np.ndarray
        Gradient :math:`X_k^\\top (y-X\\beta)` on the strong groups ``k`` where :math:`beta` is given by ``strong_beta``.
        ``strong_grad[b:b+p]`` is the gradient for the ``i`` th strong group
        where 
        ``k = strong_set[i]``,
        ``b = strong_begins[i]``,
        and ``p = group_sizes[k]``.
    active_set : (a,) np.ndarray
        List of active groups taking on values in ``[0, s)``.
        ``strong_set[active_set[i]]`` is the ``i`` th active group.
        An active group is one with non-zero coefficient block.
    active_g1 : (a1,) np.ndarray
        Subset of ``active_set`` that correspond to groups of size ``1``.
    active_g2 : (a2,) np.ndarray
        Subset of ``active_set`` that correspond to groups of size more than ``1``.
    active_begins : (a,) np.ndarray
        List of indices that index a corresponding list of values for each active group.
        ``active_begins[i]`` is the starting index corresponding to the ``i`` th active group.
    active_order : (a,) np.ndarray
        Ordering such that ``strong_set`` is sorted in ascending order for the active groups.
        ``strong_set[active_order[i]]`` is the ``i`` th active group in ascending order.
    is_active : (s,) np.ndarray
        Boolean vector that indicates whether each strong group in ``groups`` is active or not.
        ``is_active[i]`` is ``True`` if and only if ``strong_set[i]`` is active.
    betas : (l, p) scipy.sparse.csr_matrix
        ``betas[i]`` corresponds to the solution corresponding to ``lmdas[i]``.
    rsqs : (l,) np.ndarray
        ``rsqs[i]`` corresponds to the :math:`R^2` at ``betas[i]``.
    n_cds : int
        Number of coordinate descents taken.
    time_strong_cd : np.ndarray # TODO
        Benchmark time for performing coordinate-descent on the strong set at every iteration.
    time_active_cd : np.ndarray # TODO
        Benchmark time for performing coordinate-descent on the active set at every iteration.
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
    
    def propagate_core_state(self, core_state):
        self.X = core_state.X
        self.groups = core_state.groups
        self.group_sizes = core_state.group_sizes
        self.alpha = core_state.alpha
        self.penalty = core_state.penalty
        self.strong_set = core_state.strong_set
        self.strong_g1 = core_state.strong_g1
        self.strong_g2 = core_state.strong_g2
        self.strong_begins = core_state.strong_begins
        self.strong_A_diag = core_state.strong_A_diag
        self.lmdas = core_state.lmdas
        self.max_cds = core_state.max_cds
        self.thr = core_state.thr
        self.cond_0_thresh = core_state.cond_0_thresh
        self.cond_1_thresh = core_state.cond_1_thresh
        self.newton_tol = core_state.newton_tol
        self.newton_max_iters = core_state.newton_max_iters
        self.rsq = core_state.rsq
        self.strong_beta = core_state.strong_beta
        self.strong_grad = core_state.strong_grad
        self.active_set = core_state.active_set
        self.active_g1 = core_state.active_g1
        self.active_g2 = core_state.active_g2
        self.active_begins = core_state.active_begins
        self.active_order = core_state.active_order
        self.is_active = core_state.is_active
        self.betas = core_state.betas
        self.rsqs = core_state.rsqs
        self.n_cds = core_state.n_cds
        self.time_strong_cd = core_state.time_strong_cd
        self.time_active_cd = core_state.time_active_cd


@dataclass
class pin_naive(pin_base):
    """State class for pin, naive method.
    
    For descriptions on the parameters,
    see ``pin_base``.

    Parameters
    ----------
    X : np.ndarray
    groups : np.ndarray
    group_sizes : np.ndarray
    alpha : float
    penalty : np.ndarray
    strong_set : np.ndarray
    strong_g1 : np.ndarray
    strong_g2 : np.ndarray
    strong_begins : np.ndarray
    strong_A_diag : np.ndarray
    lmdas : np.ndarray
    max_cds : int
    thr : float
    cond_0_thresh : float
    cond_1_thresh : float
    newton_tol : float
    newton_max_iters : int
    rsq : float
    resid : np.ndarray
    strong_beta : np.ndarray
    strong_grad : np.ndarray
    active_set : np.ndarray
    active_g1 : np.ndarray
    active_g2 : np.ndarray
    active_begins : np.ndarray
    active_order : np.ndarray
    is_active : np.ndarray
        
    See Also
    --------
    adelie.state.pin_base
    """
    #: Residual :math:`y-X\\beta` at ``strong_beta``.
    resid: np.ndarray
    #: ``resids[i]`` corresponds to the residual at ``betas[i]``.
    resids: np.ndarray

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
            betas=[],
            rsqs=[],
            resids=[],
        )

        super().__init__(
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
            strong_beta=strong_beta,
            strong_grad=strong_grad,
            active_set=active_set,
            active_g1=active_g1,
            active_g2=active_g2,
            active_begins=active_begins,
            active_order=active_order,
            is_active=is_active,
            betas=self._core_state.betas,
            rsqs=self._core_state.rsqs,
            n_cds=self._core_state.n_cds,
            time_strong_cd=self._core_state.time_strong_cd,
            time_active_cd=self._core_state.time_active_cd,
        )

    def propagate_core_state(self, core_state):
        super().propagate_core_state(core_state)
        self.resid = core_state.resid 
        self.resids = core_state.resids
