from . import adelie_core as core
from . import matrix
from . import logger
import numpy as np
import scipy
from dataclasses import dataclass 


def deduce_states(
    *,
    X: matrix.Base64 | matrix.Base32,
    groups: np.ndarray,
    group_sizes: np.ndarray,
    strong_set: np.ndarray,
    active_set: np.ndarray,
):
    """Deduce state variables.

    Parameters
    ----------
    X : matrix.Base64 | matrix.Base32
        See ``pin_base``.
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
    strong_g1 = strong_set[group_sizes[strong_set] == 1]
    strong_g2 = strong_set[group_sizes[strong_set] > 1]
    strong_begins = np.cumsum(
        np.concatenate([[0], group_sizes[strong_set]]),
        dtype=int,
    )[:-1]
    strong_var = np.array([X.cnormsq(j) for j in groups[strong_set]])
    active_g1 = active_set[group_sizes[strong_set[active_set]] == 1]
    active_g2 = active_set[group_sizes[strong_set[active_set]] > 1]
    active_begins = np.cumsum(
        np.concatenate([[0], group_sizes[strong_set[active_set]]]),
        dtype=int,
    )[:-1]
    active_order = np.argsort(strong_set[active_set])
    is_active = np.zeros(strong_set.shape[0], dtype=bool)
    is_active[active_set] = True
    return (
        strong_g1,
        strong_g2,
        strong_begins,
        strong_var,
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
        """Saves the core state object."""
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

        All information is logged using ``logging``.

        Parameters
        ----------
        method : str, optional
            Must be one of the following options:

                - ``None``: every check is logged only. 
                    The function does not raise or assert any failed checks.
                - ``"assert"``: every check is logged and asserted.

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
    max_cds : int
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
        Gradient :math:`X_k^\\top (y-X\\beta)` on the strong groups :math:`k` where :math:`beta` is given by ``strong_beta``.
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
        Ordering such that ``groups`` is sorted in ascending order for the active groups.
        ``groups[strong_set[active_order[i]]]`` is the ``i`` th active group in ascending order.
    is_active : (s,) np.ndarray
        Boolean vector that indicates whether each strong group in ``groups`` is active or not.
        ``is_active[i]`` is ``True`` if and only if ``strong_set[i]`` is active.
    betas : (l, p) scipy.sparse.csr_matrix
        ``betas[i]`` corresponds to the solution corresponding to ``lmdas[i]``.
    rsqs : (l,) np.ndarray
        ``rsqs[i]`` corresponds to the :math:`R^2` at ``betas[i]``.
    n_cds : int
        Number of coordinate descents taken.
    time_strong_cd : np.ndarray
        Benchmark time for performing coordinate-descent on the strong set at every iteration.
    time_active_cd : np.ndarray
        Benchmark time for performing coordinate-descent on the active set at every iteration.
    """
    # Static states
    X: matrix.Base64 | matrix.Base32
    """
    Feature matrix where each column block :math:`X_k` defined by the groups
    is such that :math:`X_k^\\top X_k` is diagonal.
    It is typically one of the matrices defined in ``adelie.matrix`` sub-module.
    """
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
    Subset of ``strong_set`` that correspond to groups of size ``1``.
    ``strong_g1[i]`` is the ``i`` th strong group of size ``1``
    such that ``group_sizes[strong_g1[i]]`` is ``1``.
    """
    strong_g2: np.ndarray
    """
    Subset of ``strong_set`` that correspond to groups more than size ``1``.
    ``strong_g2[i]`` is the ``i`` th strong group of size more than ``1``
    such that ``group_sizes[strong_g2[i]]`` is more than ``1``.
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
    max_cds: int
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

    # Dynamic states
    rsq: float
    """
    :math:`R^2` value at ``strong_beta``.
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
    List of active groups taking on values in ``[0, s)``.
    ``strong_set[active_set[i]]`` is the ``i`` th active group.
    An active group is one with non-zero coefficient block.
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
    n_cds: int
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
        """Propagate core state members to current object's members.

        Most values view the contents of the members of ``core_state``.

        Parameters
        ----------
        core_state
            The core state object to reflect.
        """
        super().initialize(core_state)
        self.X = core_state.X
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
        self.max_cds = core_state.max_cds
        self.tol = core_state.tol
        self.rsq_slope_tol = core_state.rsq_slope_tol
        self.rsq_curv_tol = core_state.rsq_curv_tol
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

    def check(
        self, 
        method: str =None, 
        logger=logger.logger,
    ):
        # ================ X check ====================
        self._check(
            isinstance(self.X, matrix.Base32) or isinstance(self.X, matrix.Base64),
            "check X type",
            method, logger,
        )
        n, p = self.X.rows(), self.X.cols()

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
            np.all(self.groups == np.sort(self.groups)),
            "check groups is in increasing order",
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
            set(self.strong_g1) <= set(self.strong_set),
            "check strong_g1 subset of strong_set",
            method, logger,
        )
        self._check(
            np.all(self.group_sizes[self.strong_g1] == 1),
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
            set(self.strong_g2) <= set(self.strong_set),
            "check strong_g2 subset of strong_set",
            method, logger,
        )
        self._check(
            np.all(self.group_sizes[self.strong_g2] > 1),
            "check strong_g2 has group sizes more than 1",
            method, logger,
        )
        self._check(
            self.strong_g2.dtype == np.dtype("int"),
            "check strong_g2 dtype is int",
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
            len(self.strong_var) == S,
            "check strong_var and strong_set have same length",
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

        # ================ max_cds check ====================
        self._check(
            self.max_cds >= 0,
            "check max_cds >= 0",
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
            np.all(self.group_sizes[self.strong_set[self.active_g2]] > 1),
            "check active_g2 has group sizes more than 1",
            method, logger,
        )
        self._check(
            self.active_g2.dtype == np.dtype("int"),
            "check active_g2 dtype is int",
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
        actual = self.groups[self.strong_set[self.active_order]]
        self._check(
            np.all(actual == np.sort(actual)),
            "check active_order is an increasing ordering of active_set",
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
        :math:`R^2` value at ``strong_beta``.
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
        List of active groups taking on values in ``[0, s)``.
        ``strong_set[active_set[i]]`` is the ``i`` th active group.
        An active group is one with non-zero coefficient block.
    max_cds : int, optional
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
        
    See Also
    --------
    adelie.state.pin_base
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
        max_cds: int =int(1e5),
        tol: float =1e-12,
        rsq_slope_tol: float =1e-2,
        rsq_curv_tol: float =1e-2,
        newton_tol: float =1e-12,
        newton_max_iters: int =1000,
    ):
        # save inputs due to lifetime issues
        self._X = X
        self._groups = groups
        self._group_sizes = group_sizes
        self._penalty = penalty
        self._strong_set = strong_set
        self._lmdas = lmdas
        self._resid = resid
        self._strong_beta = strong_beta
        self._active_set = active_set

        if isinstance(X, matrix.base):
            X = X.internal()

        (
            self._strong_g1,
            self._strong_g2,
            self._strong_begins,
            self._strong_var,
            self._active_g1,
            self._active_g2,
            self._active_begins,
            self._active_order,
            self._is_active,
        ) = deduce_states(
            X=X,
            groups=groups,
            group_sizes=group_sizes,
            strong_set=strong_set,
            active_set=active_set,
        )

        self._strong_grad = []
        for k in strong_set:
            out = np.empty(group_sizes[k])
            X.bmul(0, groups[k], X.rows(), group_sizes[k], resid, out)
            self._strong_grad.append(out)
        self._strong_grad = np.concatenate(self._strong_grad)

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
            max_cds=max_cds,
            tol=tol,
            rsq_slope_tol=rsq_slope_tol,
            rsq_curv_tol=rsq_curv_tol,
            newton_tol=newton_tol,
            newton_max_iters=newton_max_iters,
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
        self.resid = core_state.resid 
        self.resids = core_state.resids

    def check(
        self, 
        method: str =None, 
        logger=logger.logger,
    ):
        super().check(method=method, logger=logger)
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
