from . import adelie_core as core
from . import matrix
from . import logger
import numpy as np
import scipy
import os
from dataclasses import dataclass 


def deduce_states(
    *,
    groups: np.ndarray,
    group_sizes: np.ndarray,
    strong_set: np.ndarray,
    active_set: np.ndarray,
):
    """Deduce state variables.

    Parameters
    ----------
    groups : (G,) np.ndarray
        See ``pin_base``.
    group_sizes : (G,) np.ndarray
        See ``pin_base``.
    strong_set : (s,) np.ndarray
        See ``pin_base``.
    active_set : (a,) np.ndarray
        See ``pin_base``.

    Returns
    -------
    strong_g1 : (s1,) np.ndarray
        See ``pin_base``.
    strong_g2 : (s2,) np.ndarray
        See ``pin_base``.
    strong_begins : (s,) np.ndarray
        See ``pin_base``.
    strong_var : (ws,) np.ndarray
        See ``pin_base``.
    active_g1 : (a1,) np.ndarray
        See ``pin_base``.
    active_g2 : (a2,) np.ndarray
        See ``pin_base``.
    active_begins : (a,) np.ndarray
        See ``pin_base``.
    active_order : (a,) np.ndarray
        See ``pin_base``.
    is_active : (s,) np.ndarray
        See ``pin_base``.

    See Also
    --------
    adelie.state.pin_base
    """
    S = strong_set.shape[0]
    strong_g1 = np.arange(S)[group_sizes[strong_set] == 1]
    strong_g2 = np.arange(S)[group_sizes[strong_set] > 1]
    strong_begins = np.cumsum(
        np.concatenate([[0], group_sizes[strong_set]]),
        dtype=int,
    )[:-1]
    active_g1 = active_set[group_sizes[strong_set[active_set]] == 1]
    active_g2 = active_set[group_sizes[strong_set[active_set]] > 1]
    active_begins = np.cumsum(
        np.concatenate([[0], group_sizes[strong_set[active_set]]]),
        dtype=int,
    )[:-1]
    active_order = np.argsort(groups[strong_set[active_set]])
    is_active = np.zeros(strong_set.shape[0], dtype=bool)
    is_active[active_set] = True
    return (
        strong_g1,
        strong_g2,
        strong_begins,
        active_g1,
        active_g2,
        active_begins,
        active_order,
        is_active,
    )

class base:
    """Base wrapper state class.

    All Python wrapper classes for core state classes must inherit from this class.

    Parameters
    ----------
    core_state
        Usually a C++ exported state class.
    """
    def __init__(self, core_state):
        self._core_state = core_state

    def internal(self):
        """Returns the core state object."""
        return self._core_state

    def initialize(self, core_state):
        """Initializes the current object with core state object.
        
        A reference to the core state object will be saved so that
        the lifetime of the object remains until the lifetime of the current object.
        The members of the core state object will then be exposed
        as members of the current object, often viewing rather than copying them.
        
        .. note::
            The core state object will usually reference existing objects.
            It is the user's responsibility that the lifetime of the referenced objects
            surpasses that of the core state object.
            
        
        Parameters
        ----------
        core_state
            Usually a C++ exported state class.
        """
        self._core_state = core_state

    def _check(self, passed, msg, method, logger, *args, **kwargs):
        if passed:
            logger.info(msg, *args, **kwargs)
        else:
            logger.error(msg, *args, **kwargs)
            if method == "assert":
                assert False

    def check(
        self, 
        method: str =None, 
        logger=logger.logger,
    ):
        """Checks consistency of the members.

        All information is logged using the ``logging`` module.

        Parameters
        ----------
        method : str, optional
            Must be one of the following options:

                - ``None``: every check is logged only. 
                  The function does not raise exceptions or assert on failed checks.
                - ``"assert"``: every check is logged and asserted.
                  The function does not raise exceptions on failed checks.

            Default is ``None``.
        logger : optional
            Logger object that behaves like a logger object in ``logging``.
            Default is ``logging.getLogger()``.
        """
        return


@dataclass(kw_only=True)
class pin_base(base):
    """Base state class for pin methods.

    Parameters
    ----------
    groups : (G,) np.ndarray
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
    group_sizes : (G,) np.ndarray
        List of group sizes corresponding to each element in ``groups``.
        ``group_sizes[i]`` is the group size of the ``i`` th group. 
    alpha : float
        Elastic net parameter.
        It must be in the range :math:`[0,1]`.
    penalty : (G,) np.ndarray
        Penalty factor for each group in the same order as ``groups``.
        It must be a non-negative vector.
    strong_set : (s,) np.ndarray
        List of indices into ``groups`` that correspond to the strong groups.
        ``strong_set[i]`` is ``i`` th strong group.
    strong_g1 : (s1,) np.ndarray
        List of indices into ``strong_set`` that correspond to groups of size ``1``.
        ``strong_set[strong_g1[i]]`` is the ``i`` th strong group of size ``1``
        such that ``group_sizes[strong_set[strong_g1[i]]]`` is ``1``.
    strong_g2 : (s2,) np.ndarray
        List of indices into ``strong_set`` that correspond to groups more than size ``1``.
        ``strong_set[strong_g2[i]]`` is the ``i`` th strong group of size more than ``1``
        such that ``group_sizes[strong_set[strong_g2[i]]]`` is more than ``1``.
    strong_begins : (s,) np.ndarray
        List of indices that index a corresponding list of values for each strong group.
        ``strong_begins[i]`` is the starting index corresponding to the ``i`` th strong group.
        From this index, reading ``group_sizes[strong_set[i]]`` number of elements
        will grab values corresponding to the full ``i`` th strong group block.
    strong_var : (ws,) np.ndarray
        List of the diagonal of :math:`X_k^\\top X_k` along the strong groups :math:`k`.
        ``strong_var[b:b+p]`` is the diagonal of :math:`X_k^\\top X_k` for the ``i`` th strong group where
        ``k = strong_set[i]``,
        ``b = strong_begins[i]``,
        and ``p = group_sizes[k]``.
    lmdas : (l,) np.ndarray
        Regularization sequence to fit on.
    max_iters : int
        Maximum number of coordinate descents.
    tol : float
        Convergence tolerance.
    rsq_slope_tol : float
        Early stopping rule check on slope of :math:`R^2`.
    rsq_curv_tol : float
        Early stopping rule check on curvature of :math:`R^2`.
    newton_tol : float
        Convergence tolerance for the BCD update.
    newton_max_iters : int
        Maximum number of iterations for the BCD update.
    n_threads : int
        Number of threads.
    rsq : float
        Unnormalized :math:`R^2` value at ``strong_beta``.
        The unnormalized :math:`R^2` is given by :math:`\\|y\\|_2^2 - \\|y-X\\beta\\|_2^2`.
    strong_beta : (ws,) np.ndarray
        Coefficient vector on the strong set.
        ``strong_beta[b:b+p]`` is the coefficient for the ``i`` th strong group 
        where
        ``k = strong_set[i]``,
        ``b = strong_begins[i]``,
        and ``p = group_sizes[k]``.
    strong_grad : (ws,) np.ndarray
        Gradient :math:`X_k^\\top (y-X\\beta)` on the strong groups :math:`k` where :math:`beta` is given by ``strong_beta``.
        ``strong_grad[b:b+p]`` is the gradient for the ``i`` th strong group
        where 
        ``k = strong_set[i]``,
        ``b = strong_begins[i]``,
        and ``p = group_sizes[k]``.
    active_set : (a,) np.ndarray
        List of indices into ``strong_set`` that correspond to active groups.
        ``strong_set[active_set[i]]`` is the ``i`` th active group.
        An active group is one with non-zero coefficient block,
        that is, for every ``i`` th active group, 
        ``strong_beta[b:b+p] == 0`` where 
        ``j = active_set[i]``,
        ``k = strong_set[j]``,
        ``b = strong_begins[j]``,
        and ``p = group_sizes[k]``.
    active_g1 : (a1,) np.ndarray
        Subset of ``active_set`` that correspond to groups of size ``1``.
    active_g2 : (a2,) np.ndarray
        Subset of ``active_set`` that correspond to groups of size more than ``1``.
    active_begins : (a,) np.ndarray
        List of indices that index a corresponding list of values for each active group.
        ``active_begins[i]`` is the starting index corresponding to the ``i`` th active group.
    active_order : (a,) np.ndarray
        Ordering such that ``groups`` is sorted in ascending order for the active groups.
        ``groups[strong_set[active_order[i]]]`` is the ``i`` th active group in ascending order.
    is_active : (s,) np.ndarray
        Boolean vector that indicates whether each strong group in ``groups`` is active or not.
        ``is_active[i]`` is ``True`` if and only if ``strong_set[i]`` is active.
    betas : (l, p) scipy.sparse.csr_matrix
        ``betas[i]`` corresponds to the solution corresponding to ``lmdas[i]``.
    rsqs : (l,) np.ndarray
        ``rsqs[i]`` corresponds to the unnormalized :math:`R^2` at ``betas[i]``.
    iters : int
        Number of coordinate descents taken.
    time_strong_cd : np.ndarray
        Benchmark time for performing coordinate-descent on the strong set at every iteration.
    time_active_cd : np.ndarray
        Benchmark time for performing coordinate-descent on the active set at every iteration.
    """
    # Static states
    groups: np.ndarray
    """
    List of starting indices to each group where `G` is the number of groups.
    ``groups[i]`` is the starting index of the ``i`` th group. 
    """
    group_sizes: np.ndarray
    """
    List of group sizes corresponding to each element in ``groups``.
    ``group_sizes[i]`` is the group size of the ``i`` th group. 
    """
    alpha: float
    """
    Elastic net parameter.
    It must be in the range :math:`[0,1]`.
    """
    penalty: np.ndarray
    """
    Penalty factor for each group in the same order as ``groups``.
    It must be a non-negative vector.
    """
    strong_set: np.ndarray
    """
    List of indices into ``groups`` that correspond to the strong groups.
    ``strong_set[i]`` is ``i`` th strong group.
    """
    strong_g1: np.ndarray
    """
    List of indices into ``strong_set`` that correspond to groups of size ``1``.
    ``strong_set[strong_g1[i]]`` is the ``i`` th strong group of size ``1``
    such that ``group_sizes[strong_set[strong_g1[i]]]`` is ``1``.
    """
    strong_g2: np.ndarray
    """
    List of indices into ``strong_set`` that correspond to groups more than size ``1``.
    ``strong_set[strong_g2[i]]`` is the ``i`` th strong group of size more than ``1``
    such that ``group_sizes[strong_set[strong_g2[i]]]`` is more than ``1``.
    """
    strong_begins: np.ndarray
    """
    List of indices that index a corresponding list of values for each strong group.
    ``strong_begins[i]`` is the starting index corresponding to the ``i`` th strong group.
    From this index, reading ``group_sizes[strong_set[i]]`` number of elements
    will grab values corresponding to the full ``i`` th strong group block.
    """
    strong_var: np.ndarray
    """
    List of the diagonal of :math:`X_k^\\top X_k` along the strong groups :math:`k`.
    ``strong_var[b:b+p]`` is the diagonal of :math:`X_k^\\top X_k` for the ``i`` th strong group where
    ``k = strong_set[i]``,
    ``b = strong_begins[i]``,
    and ``p = group_sizes[k]``.
    """
    lmdas: np.ndarray
    """
    Regularization sequence to fit on.
    """

    # Configuration
    max_iters: int
    """
    Maximum number of coordinate descents.
    """
    tol: float
    """
    Convergence tolerance.
    """
    rsq_slope_tol: float
    """
    Early stopping rule check on slope of :math:`R^2`.
    """
    rsq_curv_tol: float
    """
    Early stopping rule check on curvature of :math:`R^2`.
    """
    newton_tol: float
    """
    Convergence tolerance for the BCD update.
    """
    newton_max_iters: int
    """
    Maximum number of iterations for the BCD update.
    """
    n_threads: int
    """
    Number of threads.
    """

    # Dynamic states
    rsq: float
    """
    Unnormalized :math:`R^2` value at ``strong_beta``.
    The unnormalized :math:`R^2` is given by :math:`\\|y\\|_2^2 - \\|y-X\\beta\\|_2^2`.
    """
    strong_beta: np.ndarray
    """
    Coefficient vector on the strong set.
    ``strong_beta[b:b+p]`` is the coefficient for the ``i`` th strong group 
    where
    ``k = strong_set[i]``,
    ``b = strong_begins[i]``,
    and ``p = group_sizes[k]``.
    """
    strong_grad: np.ndarray
    """
    Gradient :math:`X_k^\\top (y-X\\beta)` on the strong groups :math:`k` where :math:`beta` is given by ``strong_beta``.
    ``strong_grad[b:b+p]`` is the gradient for the ``i`` th strong group
    where 
    ``k = strong_set[i]``,
    ``b = strong_begins[i]``,
    and ``p = group_sizes[k]``.
    """
    active_set: np.ndarray
    """
    List of indices into ``strong_set`` that correspond to active groups.
    ``strong_set[active_set[i]]`` is the ``i`` th active group.
    An active group is one with non-zero coefficient block,
    that is, for every ``i`` th active group, 
    ``strong_beta[b:b+p] == 0`` where 
    ``j = active_set[i]``,
    ``k = strong_set[j]``,
    ``b = strong_begins[j]``,
    and ``p = group_sizes[k]``.
    """
    active_g1: np.ndarray
    """
    Subset of ``active_set`` that correspond to groups of size ``1``.
    """
    active_g2: np.ndarray
    """
    Subset of ``active_set`` that correspond to groups of size more than ``1``.
    """
    active_begins: np.ndarray
    """
    List of indices that index a corresponding list of values for each active group.
    ``active_begins[i]`` is the starting index corresponding to the ``i`` th active group.
    """
    active_order: np.ndarray
    """
    Ordering such that ``groups`` is sorted in ascending order for the active groups.
    ``groups[strong_set[active_order[i]]]`` is the ``i`` th active group in ascending order.
    """
    is_active: np.ndarray
    """
    Boolean vector that indicates whether each strong group in ``groups`` is active or not.
    ``is_active[i]`` is ``True`` if and only if ``strong_set[i]`` is active.
    """
    betas: scipy.sparse.csr_matrix
    """
    ``betas[i]`` corresponds to the solution corresponding to ``lmdas[i]``.
    """
    rsqs: np.ndarray
    """
    ``rsqs[i]`` corresponds to the :math:`R^2` at ``betas[i]``.
    """
    iters: int
    """
    Number of coordinate descents taken.
    """
    time_strong_cd: np.ndarray
    """
    Benchmark time for performing coordinate-descent on the strong set at every iteration.
    """
    time_active_cd: np.ndarray
    """
    Benchmark time for performing coordinate-descent on the active set at every iteration.
    """
    
    def initialize(self, core_state):
        super().initialize(core_state)
        self.groups = core_state.groups
        self.group_sizes = core_state.group_sizes
        self.alpha = core_state.alpha
        self.penalty = core_state.penalty
        self.strong_set = core_state.strong_set
        self.strong_g1 = core_state.strong_g1
        self.strong_g2 = core_state.strong_g2
        self.strong_begins = core_state.strong_begins
        self.strong_var = core_state.strong_var
        self.lmdas = core_state.lmdas
        self.max_iters = core_state.max_iters
        self.tol = core_state.tol
        self.rsq_slope_tol = core_state.rsq_slope_tol
        self.rsq_curv_tol = core_state.rsq_curv_tol
        self.newton_tol = core_state.newton_tol
        self.newton_max_iters = core_state.newton_max_iters
        self.n_threads = core_state.n_threads
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
        self.iters = core_state.iters
        self.time_strong_cd = core_state.time_strong_cd
        self.time_active_cd = core_state.time_active_cd

    def check(
        self, 
        method: str =None, 
        logger=logger.logger,
    ):
        p = np.sum(self.group_sizes)

        # ================ groups check ====================
        self._check(
            np.all((0 <= self.groups) & (self.groups <= p)),
            "check groups is in [0, p)",
            method, logger,
        )
        self._check(
            len(self.groups) == len(np.unique(self.groups)),
            "check groups has unique values",
            method, logger,
        )
        self._check(
            self.groups.dtype == np.dtype("int"),
            "check groups dtype is int",
            method, logger,
        )
        G = len(self.groups)

        # ================ group_sizes check ====================
        self._check(
            len(self.group_sizes) == G,
            "check groups and group_sizes have same length",
            method, logger,
        )
        self._check(
            np.sum(self.group_sizes) == p,
            "check sum of group_sizes is p",
            method, logger,
        )
        self._check(
            np.all((0 < self.group_sizes) & (self.group_sizes <= p)),
            "check group_sizes is in (0, p]",
            method, logger,
        )
        self._check(
            self.group_sizes.dtype == np.dtype("int"),
            "check group_sizes dtype is int",
            method, logger,
        )

        # ================ alpha check ====================
        self._check(
            (self.alpha >= 0) and (self.alpha <= 1),
            "check alpha is in [0, 1]",
            method, logger,
        )

        # ================ penalty check ====================
        self._check(
            np.all(self.penalty >= 0),
            "check penalty is non-negative",
            method, logger,
        )
        self._check(
            len(self.penalty) == G,
            "check penalty and groups have same length",
            method, logger,
        )

        # ================ strong_set check ====================
        self._check(
            np.all((0 <= self.strong_set) & (self.strong_set < G)),
            "check strong_set is a subset of [0, G)",
            method, logger,
        )
        self._check(
            len(self.strong_set) == len(np.unique(self.strong_set)),
            "check strong_set has unique values",
            method, logger,
        )
        self._check(
            self.strong_set.dtype == np.dtype("int"),
            "check strong_set dtype is int",
            method, logger,
        )
        S = len(self.strong_set)

        # ================ strong_g1 check ====================
        self._check(
            np.all((0 <= self.strong_g1) & (self.strong_g1 < S)),
            "check strong_g1 is in [0, S)",
            method, logger,
        )
        self._check(
            len(self.strong_g1) == len(np.unique(self.strong_g1)),
            "check strong_g1 has unique values",
            method, logger,
        )
        self._check(
            np.all(self.group_sizes[self.strong_set[self.strong_g1]] == 1),
            "check strong_g1 has group sizes of 1",
            method, logger,
        )
        self._check(
            self.strong_g1.dtype == np.dtype("int"),
            "check strong_g1 dtype is int",
            method, logger,
        )

        # ================ strong_g2 check ====================
        self._check(
            np.all((0 <= self.strong_g2) & (self.strong_g2 < S)),
            "check strong_g2 is in [0, S)",
            method, logger,
        )
        self._check(
            len(self.strong_g2) == len(np.unique(self.strong_g2)),
            "check strong_g2 has unique values",
            method, logger,
        )
        self._check(
            np.all(self.group_sizes[self.strong_set[self.strong_g2]] > 1),
            "check strong_g2 has group sizes more than 1",
            method, logger,
        )
        self._check(
            self.strong_g2.dtype == np.dtype("int"),
            "check strong_g2 dtype is int",
            method, logger,
        )
        self._check(
            len(self.strong_g1) + len(self.strong_g2) == S,
            "check strong_g1 and strong_g2 combined have length S",
            method, logger,
        )

        # ================ strong_begins check ====================
        expected = np.cumsum(
            np.concatenate([[0], self.group_sizes[self.strong_set]], dtype=int)
        )
        WS = expected[-1]
        expected = expected[:-1]
        self._check(
            np.all(self.strong_begins == expected),
            "check strong_begins is [0, g1, g2, ...] where gi is the group size of (i-1)th strong group.",
            method, logger,
        )
        self._check(
            self.strong_begins.dtype == np.dtype("int"),
            "check strong_begins dtype is int",
            method, logger,
        )

        # ================ strong_var check ====================
        self._check(
            len(self.strong_var) == WS,
            "check strong_var size",
            method, logger,
        )
        self._check(
            np.all(0 <= self.strong_var),
            "check strong_var is non-negative",
            method, logger,
        )
        
        # ================ lmdas check ====================
        self._check(
            np.all(0 <= self.lmdas),
            "check lmdas is non-negative",
            method, logger,
        )
        # if not sorted in decreasing order
        if np.any(self.lmdas != np.sort(self.lmdas)[::-1]):
            logger.warning("lmdas are not sorted in decreasing order")

        # ================ max_iters check ====================
        self._check(
            self.max_iters >= 0,
            "check max_iters >= 0",
            method, logger,
        )

        # ================ tol check ====================
        self._check(
            self.tol >= 0,
            "check tol >= 0",
            method, logger,
        )

        # ================ rsq_slope_tol check ====================
        self._check(
            self.rsq_slope_tol >= 0,
            "check rsq_slope_tol >= 0",
            method, logger,
        )

        # ================ rsq_curve_tol check ====================
        self._check(
            self.rsq_curv_tol >= 0,
            "check rsq_curv_tol >= 0",
            method, logger,
        )

        # ================ newton_tol check ====================
        self._check(
            self.newton_tol >= 0,
            "check newton_tol >= 0",
            method, logger,
        )

        # ================ newton_max_iters check ====================
        self._check(
            self.newton_max_iters >= 0,
            "check newton_max_iters >= 0",
            method, logger,
        )

        # ================ newton_max_iters check ====================
        self._check(
            self.n_threads > 0,
            "check n_threads > 0",
            method, logger,
        )

        # ================ rsq check ====================
        self._check(
            self.rsq >= 0,
            "check rsq >= 0",
            method, logger,
        )

        # ================ strong_beta check ====================
        self._check(
            len(self.strong_beta) == WS,
            "check strong_beta size",
            method, logger,
        )

        # ================ strong_grad check ====================
        self._check(
            len(self.strong_grad) == WS,
            "check strong_grad size",
            method, logger,
        )

        # ================ active_set check ====================
        self._check(
            np.all((0 <= self.active_set) & (self.active_set < S)),
            "check active_set is in [0, S)",
            method, logger,
        )
        self._check(
            len(self.active_set) == len(np.unique(self.active_set)),
            "check active_set is unique",
            method, logger,
        )
        self._check(
            self.active_set.dtype == np.dtype("int"),
            "check active_set dtype is int",
            method, logger,
        )
        A = len(self.active_set)

        # ================ active_g1 check ====================
        self._check(
            set(self.active_g1) <= set(self.active_set),
            "check active_g1 subset of active_set",
            method, logger,
        )
        self._check(
            len(self.active_g1) == len(np.unique(self.active_g1)),
            "check active_g1 is unique",
            method, logger,
        )
        self._check(
            np.all(self.group_sizes[self.strong_set[self.active_g1]] == 1),
            "check active_g1 has group sizes of 1",
            method, logger,
        )
        self._check(
            self.active_g1.dtype == np.dtype("int"),
            "check active_g1 dtype is int",
            method, logger,
        )

        # ================ active_g2 check ====================
        self._check(
            set(self.active_g2) <= set(self.active_set),
            "check active_g2 subset of active_set",
            method, logger,
        )
        self._check(
            len(self.active_g2) == len(np.unique(self.active_g2)),
            "check active_g2 is unique",
            method, logger,
        )
        self._check(
            np.all(self.group_sizes[self.strong_set[self.active_g2]] > 1),
            "check active_g2 has group sizes more than 1",
            method, logger,
        )
        self._check(
            self.active_g2.dtype == np.dtype("int"),
            "check active_g2 dtype is int",
            method, logger,
        )
        self._check(
            len(self.active_g1) + len(self.active_g2) == A,
            "check active_g1 and active_g2 combined have length A",
            method, logger,
        )

        # ================ active_begins check ====================
        expected = np.cumsum(
            np.concatenate([[0], self.group_sizes[self.strong_set[self.active_set]]], dtype=int)
        )
        WA = expected[-1]
        expected = expected[:-1]
        self._check(
            np.all(self.active_begins == expected),
            "check active_begins is [0, g1, g2, ...] where gi is the group size of (i-1)th active group.",
            method, logger,
        )
        self._check(
            self.active_begins.dtype == np.dtype("int"),
            "check active_begins dtype is int",
            method, logger,
        )

        # ================ active_order check ====================
        actual = self.groups[self.strong_set[self.active_set[self.active_order]]]
        self._check(
            np.all(actual == np.sort(actual)),
            "check active_order orders active_set such that groups is ordered",
            method, logger,
        )
        self._check(
            self.active_order.dtype == np.dtype("int"),
            "check active_order dtype is int",
            method, logger,
        )

        # ================ is_active check ====================
        self._check(
            np.all(np.arange(S)[self.is_active] == np.sort(self.active_set)),
            "check is_active is consistent with active_set",
            method, logger,
        )
        self._check(
            self.is_active.dtype == np.dtype("bool"),
            "check is_active dtype is bool",
            method, logger,
        )

        # ================ betas check ====================
        self._check(
            self.betas.shape[0] <= self.lmdas.shape[0],
            "check betas rows is no more than the number of lmdas",
            method, logger,
        )
        self._check(
            isinstance(self.betas, scipy.sparse.csr_matrix),
            "check betas type",
            method, logger,
        )
        self._check(
            self.betas.shape[1] == p,
            "check betas shape",
            method, logger,
        )

        # ================ rsqs check ====================
        self._check(
            self.rsqs.shape == (self.betas.shape[0],),
            "check rsqs shape",
            method, logger,
        )
        self._check(
            np.all(0 <= self.rsqs),
            "check rsqs is non-negative",
            method, logger,
        )
        

@dataclass
class pin_naive(pin_base):
    """State class for pin, naive method.

    Parameters
    ----------
    X : Union[adelie.matrix.Base64, adelie.matrix.Base32]
        Feature matrix where each column block :math:`X_k` defined by the groups
        is such that :math:`X_k^\\top X_k` is diagonal.
        It is typically one of the matrices defined in ``adelie.matrix`` sub-module.
    groups : (G,) np.ndarray
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
    group_sizes : (G,) np.ndarray
        List of group sizes corresponding to each element in ``groups``.
        ``group_sizes[i]`` is the group size of the ``i`` th group. 
    alpha : float
        Elastic net parameter.
        It must be in the range :math:`[0,1]`.
    penalty : (G,) np.ndarray
        Penalty factor for each group in the same order as ``groups``.
        It must be a non-negative vector.
    strong_set : (s,) np.ndarray
        List of indices into ``groups`` that correspond to the strong groups.
        ``strong_set[i]`` is ``i`` th strong group.
    lmdas : (l,) np.ndarray
        Regularization sequence to fit on.
    rsq : float
        Unnormalized :math:`R^2` value at ``strong_beta``.
        The unnormalized :math:`R^2` is given by :math:`\\|y\\|_2^2 - \\|y-X\\beta\\|_2^2`.
    resid : np.ndarray
        Residual :math:`y-X\\beta` at ``strong_beta``.
    strong_beta : (ws,) np.ndarray
        Coefficient vector on the strong set.
        ``strong_beta[b:b+p]`` is the coefficient for the ``i`` th strong group 
        where
        ``k = strong_set[i]``,
        ``b = strong_begins[i]``,
        and ``p = group_sizes[k]``.
    active_set : (a,) np.ndarray
        List of indices into ``strong_set`` that correspond to active groups.
        ``strong_set[active_set[i]]`` is the ``i`` th active group.
        An active group is one with non-zero coefficient block,
        that is, for every ``i`` th active group, 
        ``strong_beta[b:b+p] == 0`` where 
        ``j = active_set[i]``,
        ``k = strong_set[j]``,
        ``b = strong_begins[j]``,
        and ``p = group_sizes[k]``.
    max_iters : int, optional
        Maximum number of coordinate descents.
        Default is ``int(1e5)``.
    tol : float, optional
        Convergence tolerance.
        Default is ``1e-12``.
    rsq_slope_tol : float, optional
        Early stopping rule check on slope of :math:`R^2`.
        Default is ``1e-2``.
    rsq_curv_tol : float, optional
        Early stopping rule check on curvature of :math:`R^2`.
        Default is ``1e-2``.
    newton_tol : float, optional
        Convergence tolerance for the BCD update.
        Default is ``1e-12``.
    newton_max_iters : int, optional
        Maximum number of iterations for the BCD update.
        Default is ``1000``.
    n_threads : int, optional
        Number of threads.
        Default is ``os.cpu_count()``.
        
    See Also
    --------
    adelie.state.pin_base
    """
    X: matrix.Base64 | matrix.Base32
    """
    Feature matrix where each column block :math:`X_k` defined by the groups
    is such that :math:`X_k^\\top X_k` is diagonal.
    It is typically one of the matrices defined in ``adelie.matrix`` sub-module.
    """
    resid: np.ndarray
    """Residual :math:`y-X\\beta` at ``strong_beta``."""
    resids: np.ndarray
    """``resids[i]`` corresponds to the residual at ``betas[i]``."""

    def __init__(
        self, 
        *,
        X: matrix.base | matrix.Base64 | matrix.Base32,
        groups: np.ndarray,
        group_sizes: np.ndarray,
        alpha: float,
        penalty: np.ndarray,
        strong_set: np.ndarray,
        lmdas: np.ndarray,
        rsq: float,
        resid: np.ndarray,
        strong_beta: np.ndarray,
        active_set: np.ndarray,
        max_iters: int =int(1e5),
        tol: float =1e-12,
        rsq_slope_tol: float =1e-2,
        rsq_curv_tol: float =1e-2,
        newton_tol: float =1e-12,
        newton_max_iters: int =1000,
        n_threads: int =os.cpu_count(),
    ):
        ## save inputs due to lifetime issues
        # static inputs require a reference to input
        # or copy if it must be made
        self._X = X

        if isinstance(X, matrix.base):
            X = X.internal()

        dtype = (
            np.float64
            if isinstance(X, matrix.Base64) else
            np.float32
        )

        self._groups = np.array(groups, copy=False, dtype=int)
        self._group_sizes = np.array(group_sizes, copy=False, dtype=int)
        self._penalty = np.array(penalty, copy=False, dtype=dtype)
        self._strong_set = np.array(strong_set, copy=False, dtype=int)
        self._lmdas = np.array(lmdas, copy=False, dtype=dtype)

        # dynamic inputs require a copy to not modify user's inputs
        self._resid = np.copy(resid).astype(dtype)
        self._strong_beta = np.copy(strong_beta).astype(dtype)
        self._active_set = np.copy(active_set).astype(int)

        (
            self._strong_g1,
            self._strong_g2,
            self._strong_begins,
            self._active_g1,
            self._active_g2,
            self._active_begins,
            self._active_order,
            self._is_active,
        ) = deduce_states(
            groups=groups,
            group_sizes=group_sizes,
            strong_set=strong_set,
            active_set=active_set,
        )

        self._strong_var = np.concatenate([
            [X.cnormsq(jj) for jj in range(g, g + gs)]
            for g, gs in zip(groups[strong_set], group_sizes[strong_set])
        ])

        self._strong_grad = []
        for k in strong_set:
            out = np.empty(group_sizes[k], dtype=dtype)
            X.bmul(0, groups[k], X.rows(), group_sizes[k], resid, out)
            self._strong_grad.append(out)
        self._strong_grad = np.concatenate(self._strong_grad, dtype=dtype)

        State = (
            core.state.PinNaive64 
            if isinstance(X, matrix.Base64) else 
            core.state.PinNaive32
        )

        self._core_state = State(
            X=X,
            groups=self._groups,
            group_sizes=self._group_sizes,
            alpha=alpha,
            penalty=self._penalty,
            strong_set=self._strong_set,
            strong_g1=self._strong_g1,
            strong_g2=self._strong_g2,
            strong_begins=self._strong_begins,
            strong_var=self._strong_var,
            lmdas=self._lmdas,
            max_iters=max_iters,
            tol=tol,
            rsq_slope_tol=rsq_slope_tol,
            rsq_curv_tol=rsq_curv_tol,
            newton_tol=newton_tol,
            newton_max_iters=newton_max_iters,
            n_threads=n_threads,
            rsq=rsq,
            resid=self._resid,
            strong_beta=self._strong_beta,
            strong_grad=self._strong_grad,
            active_set=self._active_set,
            active_g1=self._active_g1,
            active_g2=self._active_g2,
            active_begins=self._active_begins,
            active_order=self._active_order,
            is_active=self._is_active,
            betas=[],
            rsqs=[],
            resids=[],
        )

        self.initialize(self._core_state)

    def initialize(self, core_state):
        super().initialize(core_state)
        self.X = core_state.X
        self.resid = core_state.resid 
        self.resids = core_state.resids

    def check(
        self, 
        method: str =None, 
        logger=logger.logger,
    ):
        super().check(method=method, logger=logger)
        self._check(
            isinstance(self.X, matrix.Base32) or isinstance(self.X, matrix.Base64),
            "check X type",
            method, logger,
        )
        self._check(
            self.resid.shape[0] == self.X.rows(),
            "check resid shape",
            method, logger,
        )
        self._check(
            self.resids.shape == (self.betas.shape[0], self.X.rows()),
            "check resids shape",
            method, logger,
        )


@dataclass
class pin_cov(pin_base):
    """State class for pin, cov method.

    Parameters
    ----------
    A : Union[adelie.matrix.Base64, adelie.matrix.Base32]
        Covariance matrix where each diagonal block :math:`A_{kk}` defined by the groups
        is a diagonal matrix.
        It is typically one of the matrices defined in ``adelie.matrix`` sub-module.
    groups : (G,) np.ndarray
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
    group_sizes : (G,) np.ndarray
        List of group sizes corresponding to each element in ``groups``.
        ``group_sizes[i]`` is the group size of the ``i`` th group. 
    alpha : float
        Elastic net parameter.
        It must be in the range :math:`[0,1]`.
    penalty : (G,) np.ndarray
        Penalty factor for each group in the same order as ``groups``.
        It must be a non-negative vector.
    strong_set : (s,) np.ndarray
        List of indices into ``groups`` that correspond to the strong groups.
        ``strong_set[i]`` is ``i`` th strong group.
    lmdas : (l,) np.ndarray
        Regularization sequence to fit on.
    rsq : float
        Unnormalized :math:`R^2` value at ``strong_beta``.
        The unnormalized :math:`R^2` is given by :math:`\\|y\\|_2^2 - \\|y-X\\beta\\|_2^2`.
    strong_beta : (ws,) np.ndarray
        Coefficient vector on the strong set.
        ``strong_beta[b:b+p]`` is the coefficient for the ``i`` th strong group 
        where
        ``k = strong_set[i]``,
        ``b = strong_begins[i]``,
        and ``p = group_sizes[k]``.
    strong_grad : (ws,) np.ndarray
        Gradient :math:`X_k^\\top (y-X\\beta)` on the strong groups :math:`k` where :math:`beta` is given by ``strong_beta``.
        ``strong_grad[b:b+p]`` is the gradient for the ``i`` th strong group
        where 
        ``k = strong_set[i]``,
        ``b = strong_begins[i]``,
        and ``p = group_sizes[k]``.
    active_set : (a,) np.ndarray
        List of indices into ``strong_set`` that correspond to active groups.
        ``strong_set[active_set[i]]`` is the ``i`` th active group.
        An active group is one with non-zero coefficient block,
        that is, for every ``i`` th active group, 
        ``strong_beta[b:b+p] == 0`` where 
        ``j = active_set[i]``,
        ``k = strong_set[j]``,
        ``b = strong_begins[j]``,
        and ``p = group_sizes[k]``.
    max_iters : int, optional
        Maximum number of coordinate descents.
        Default is ``int(1e5)``.
    tol : float, optional
        Convergence tolerance.
        Default is ``1e-12``.
    rsq_slope_tol : float, optional
        Early stopping rule check on slope of :math:`R^2`.
        Default is ``1e-2``.
    rsq_curv_tol : float, optional
        Early stopping rule check on curvature of :math:`R^2`.
        Default is ``1e-2``.
    newton_tol : float, optional
        Convergence tolerance for the BCD update.
        Default is ``1e-12``.
    newton_max_iters : int, optional
        Maximum number of iterations for the BCD update.
        Default is ``1000``.
    n_threads : int, optional
        Number of threads.
        Default is ``os.cpu_count()``.
        
    See Also
    --------
    adelie.state.pin_base
    """
    A: matrix.base | matrix.Base64 | matrix.Base32

    def __init__(
        self, 
        *,
        A: matrix.base | matrix.Base64 | matrix.Base32,
        groups: np.ndarray,
        group_sizes: np.ndarray,
        alpha: float,
        penalty: np.ndarray,
        strong_set: np.ndarray,
        lmdas: np.ndarray,
        rsq: float,
        strong_beta: np.ndarray,
        strong_grad: np.ndarray,
        active_set: np.ndarray,
        max_iters: int =int(1e5),
        tol: float =1e-12,
        rsq_slope_tol: float =1e-2,
        rsq_curv_tol: float =1e-2,
        newton_tol: float =1e-12,
        newton_max_iters: int =1000,
        n_threads: int =os.cpu_count(),
    ):
        ## save inputs due to lifetime issues
        # static inputs require a reference to input
        # or copy if it must be made
        self._A = A

        if isinstance(A, matrix.base):
            A = A.internal()

        dtype = (
            np.float64
            if isinstance(A, matrix.Base64) else
            np.float32
        )

        self._groups = np.array(groups, copy=False, dtype=int)
        self._group_sizes = np.array(group_sizes, copy=False, dtype=int)
        self._penalty = np.array(penalty, copy=False, dtype=dtype)
        self._strong_set = np.array(strong_set, copy=False, dtype=int)
        self._lmdas = np.array(lmdas, copy=False, dtype=dtype)

        # dynamic inputs require a copy to not modify user's inputs
        self._strong_beta = np.copy(strong_beta).astype(dtype)
        self._strong_grad = np.copy(strong_grad).astype(dtype)
        self._active_set = np.copy(active_set).astype(int)

        (
            self._strong_g1,
            self._strong_g2,
            self._strong_begins,
            self._active_g1,
            self._active_g2,
            self._active_begins,
            self._active_order,
            self._is_active,
        ) = deduce_states(
            groups=groups,
            group_sizes=group_sizes,
            strong_set=strong_set,
            active_set=active_set,
        )

        self._strong_var = np.array([
            A.coeff(j, j) for j in range(A.cols())
        ])

        State = (
            core.state.PinCov64 
            if isinstance(A, matrix.Base64) else 
            core.state.PinCov32
        )

        self._core_state = State(
            A=A,
            groups=self._groups,
            group_sizes=self._group_sizes,
            alpha=alpha,
            penalty=self._penalty,
            strong_set=self._strong_set,
            strong_g1=self._strong_g1,
            strong_g2=self._strong_g2,
            strong_begins=self._strong_begins,
            strong_var=self._strong_var,
            lmdas=self._lmdas,
            max_iters=max_iters,
            tol=tol,
            rsq_slope_tol=rsq_slope_tol,
            rsq_curv_tol=rsq_curv_tol,
            newton_tol=newton_tol,
            newton_max_iters=newton_max_iters,
            n_threads=n_threads,
            rsq=rsq,
            strong_beta=self._strong_beta,
            strong_grad=self._strong_grad,
            active_set=self._active_set,
            active_g1=self._active_g1,
            active_g2=self._active_g2,
            active_begins=self._active_begins,
            active_order=self._active_order,
            is_active=self._is_active,
            betas=[],
            rsqs=[],
        )

        self.initialize(self._core_state)

    def initialize(self, core_state):
        super().initialize(core_state)
        self.A = core_state.A

    def check(
        self, 
        method: str =None, 
        logger=logger.logger,
    ):
        super().check(method=method, logger=logger)
        self._check(
            isinstance(self.A, matrix.Base32) or isinstance(self.X, matrix.Base64),
            "check A type",
            method, logger,
        )
        self._check(
            self.resid.shape[0] == self.X.rows(),
            "check resid shape",
            method, logger,
        )
        self._check(
            self.resids.shape == (self.betas.shape[0], self.X.rows()),
            "check resids shape",
            method, logger,
        )
