from . import adelie_core as core
from . import matrix
from . import logger
from . import glm
from .constraint import (
    ConstraintBase32,
    ConstraintBase64,
)
from .matrix import (
    MatrixCovBase32,
    MatrixCovBase64,
    MatrixNaiveBase32,
    MatrixNaiveBase64,
)
from .glm import (
    GlmBase32,
    GlmBase64,
    GlmMultiBase32,
    GlmMultiBase64,
)
from typing import Union
import numpy as np
import scipy


def render_constraints(
    n_groups: int,
    constraints: list,
    dtype,
):
    if constraints is None:
        constraints = [None] * n_groups
    if len(constraints) > n_groups:
        raise RuntimeError("constraints is unexpectedly larger than the number of groups!")
    elif len(constraints) < n_groups:
        # Assume this is because we are in multi-response state.
        # Must prepend with None since the missing constraints are for the intercept columns.
        constraints = [None] * (n_groups - len(constraints)) + constraints

    return {
        np.float32: core.constraint.VectorConstraintBase32,
        np.float64: core.constraint.VectorConstraintBase64,
    }[dtype](constraints)


def render_dual_groups(
    constraints: list,
):
    return np.cumsum(np.concatenate(
        [[0] + [0 if c is None else c.dual_size for c in constraints]], 
        dtype=int,
    ))[:-1]


def deduce_states(
    *,
    constraints: list,
    group_sizes: np.ndarray,
    screen_set: np.ndarray,
    dtype,
):
    constraints = render_constraints(
        group_sizes.shape[0],
        constraints,
        dtype,
    )
    dual_groups = render_dual_groups(constraints)
    screen_begins = np.cumsum(
        np.concatenate([[0], group_sizes[screen_set]]),
        dtype=int,
    )[:-1]
    screen_dual_begins = np.cumsum(
        np.concatenate([
            [0], 
            [
                0
                if constraints[k] is None else
                constraints[k].dual_size 
                for k in screen_set
            ],
        ]),
        dtype=int,
    )[:-1]
    return (
        constraints,
        dual_groups,
        screen_begins,
        screen_dual_begins,
    )


class base:
    """Base wrapper state class.

    All Python wrapper classes for core state classes must inherit from this class.
    """
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

        All information is logged using the :mod:`logging` module.

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
            Logger object that behaves like a logger object in :mod:`logging`.
            Default is ``logging.getLogger()``.
        """
        return

    @staticmethod
    def create_from_core(
        cls, 
        state, 
        core_state, 
        pycls, 
        corecls,
    ):
        """Create a new instance of a state using an existing state and new (C++) core state.

        Parameters
        ---------- 
        cls
            Class to instantiate an object for.
        state
            State object to grab static members from.
        core_state
            New core state to initialize with.
        pycls
            Python derived class type.
        corecls
            Core state class type.

        Returns
        -------
        new_state : cls
            New state object.
        """
        # allocate new object cls casted to pycls type
        obj = super(pycls, cls).__new__(cls)
        # keep reference to static members to extend lifetime
        # all static members are of the form: _name.
        for a in dir(state):
            if a.startswith("_") and not a.startswith("__") and not callable(getattr(state, a)):
                setattr(obj, a, getattr(state, a))
        # initialize core state
        corecls.__init__(obj, core_state)
        return obj


class gaussian_pin_base(base):
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
        self._check(
            np.allclose(self.groups, np.cumsum(np.concatenate([[0], self.group_sizes]))[:-1]),
            "check groups and group_sizes consistency",
            method, logger,
        )

        # ================ penalty check ====================
        self._check(
            np.all(self.penalty >= 0),
            "check penalty is non-negative",
            method, logger,
        )

        # ================ screen_set check ====================
        self._check(
            np.all((0 <= self.screen_set) & (self.screen_set < G)),
            "check screen_set is a subset of [0, G)",
            method, logger,
        )
        self._check(
            len(self.screen_set) == len(np.unique(self.screen_set)),
            "check screen_set has unique values",
            method, logger,
        )
        self._check(
            self.screen_set.dtype == np.dtype("int"),
            "check screen_set dtype is int",
            method, logger,
        )
        S = len(self.screen_set)

        # ================ screen_begins check ====================
        expected = np.cumsum(
            np.concatenate([[0], self.group_sizes[self.screen_set]], dtype=int)
        )
        WS = expected[-1]
        expected = expected[:-1]
        self._check(
            np.all(self.screen_begins == expected),
            "check screen_begins is [0, g1, g2, ...] where gi is the group size of (i-1)th screen group.",
            method, logger,
        )
        self._check(
            self.screen_begins.dtype == np.dtype("int"),
            "check screen_begins dtype is int",
            method, logger,
        )

        # ================ screen_vars check ====================
        self._check(
            len(self.screen_vars) == WS,
            "check screen_vars size",
            method, logger,
        )
        self._check(
            np.all(0 <= self.screen_vars),
            "check screen_vars is non-negative",
            method, logger,
        )

        # ================ screen_transforms check ====================
        self._check(
            len(self.screen_transforms) == S,
            "check screen_transforms size",
            method, logger,
        )
        
        # ================ lmda_path check ====================
        self._check(
            np.all(0 <= self.lmda_path),
            "check lmda_path is non-negative",
            method, logger,
        )
        # if not sorted in decreasing order
        if np.any(self.lmda_path != np.sort(self.lmda_path)[::-1]):
            logger.warning("lmda_path are not sorted in decreasing order")

        # ================ screen_is_active check ====================
        active_set = self.active_set[:self.active_set_size]
        self._check(
            np.all(np.arange(S)[self.screen_is_active] == np.sort(active_set)),
            "check screen_is_active is consistent with active_set",
            method, logger,
        )
        self._check(
            self.screen_is_active.dtype == np.dtype("bool"),
            "check screen_is_active dtype is bool",
            method, logger,
        )

        # ================ active_set check ====================
        self._check(
            np.all((0 <= active_set) & (active_set < S)),
            "check active_set is in [0, S)",
            method, logger,
        )
        self._check(
            len(active_set) == len(np.unique(active_set)),
            "check active_set is unique",
            method, logger,
        )
        self._check(
            active_set.dtype == np.dtype("int"),
            "check active_set dtype is int",
            method, logger,
        )
        A = len(active_set)

        # ================ active_begins check ====================
        expected = np.cumsum(
            np.concatenate([[0], self.group_sizes[self.screen_set[active_set]]], dtype=int)
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
        actual = self.groups[self.screen_set[active_set[self.active_order]]]
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

        # ================ betas check ====================
        self._check(
            self.betas.shape[0] <= self.lmda_path.shape[0],
            "check betas rows is no more than the number of lmda_path",
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

        # ================ lmdas check ====================
        self._check(
            self.lmdas.shape == (self.betas.shape[0],),
            "check lmdas shape",
            method, logger,
        )


class gaussian_pin_naive_base(gaussian_pin_base):
    """State wrapper base class for all gaussian pin naive method."""
    def check(
        self, 
        method: str =None, 
        logger=logger.logger,
    ):
        super().check(method=method, logger=logger)
        self._check(
            isinstance(self.X, matrix.MatrixNaiveBase64) or isinstance(self.X, matrix.MatrixNaiveBase32),
            "check X type",
            method, logger,
        )
        self._check(
            self.resid.shape[0] == self.X.rows(),
            "check resid shape",
            method, logger,
        )


def gaussian_pin_naive(
    *,
    X: Union[MatrixNaiveBase32, MatrixNaiveBase64],
    y_mean: float,
    y_var: float,
    constraints: list[Union[ConstraintBase32, ConstraintBase64]],
    groups: np.ndarray,
    alpha: float,
    penalty: np.ndarray,
    weights: np.ndarray,
    screen_set: np.ndarray,
    lmda_path: np.ndarray,
    rsq: float,
    resid: np.ndarray,
    screen_beta: np.ndarray,
    screen_is_active: np.ndarray,
    screen_dual: np.ndarray,
    active_set_size: int,
    active_set: np.ndarray,
    intercept: bool =True,
    max_active_size: int =None,
    max_iters: int =int(1e5),
    tol: float =1e-7,
    adev_tol: float =0.9,
    ddev_tol: float =0,
    newton_tol: float =1e-12,
    newton_max_iters: int =1000,
    n_threads: int =1,
):
    """Creates a Gaussian, pin, naive method state object.

    Define the following quantities:

        - :math:`X_c` as :math:`X` if ``intercept`` is ``False`` and otherwise 
          the column-centered version.
        - :math:`y_c` as :math:`y` if ``intercept`` is ``False`` and otherwise
          the centered version.

    Parameters
    ----------
    X : Union[MatrixNaiveBase32, MatrixNaiveBase64]
        Feature matrix.
        It is typically one of the matrices defined in :mod:`adelie.matrix` submodule.
    y_mean : float
        Mean of the response vector :math:`y` (weighted by :math:`W`),
        i.e. :math:`\\mathbf{1}^\\top W y`.
    y_var : float
        :math:`\\ell_2` norm squared (weighted by :math:`W`) of :math:`y_c`, i.e. :math:`\\|y_c\\|_{W}^2`.
        Variance of the response vector :math:`y` (weighted by :math:`W`), 
        i.e. :math:`\\|y_c\\|_{W}^2`.
        This is only used to check convergence as a relative measure,
        i.e. this quantity is the "null" model MSE.
    constraints : (G,) list[Union[ConstraintBase32, ConstraintBase64]], optional
        List of constraints for each group.
        ``constraints[i]`` is the constraint object corresponding to group ``i``.
        If ``constraints[i]`` is ``None``, then the ``i`` th group is unconstrained.
        If ``None``, every group is unconstrained.
    groups : (G,) ndarray
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
    alpha : float
        Elastic net parameter.
        It must be in the range :math:`[0,1]`.
    penalty : (G,) ndarray
        Penalty factor for each group in the same order as ``groups``.
        It must be a non-negative vector.
    weights : (n,) ndarray
        Observation weights :math:`W`.
        The weights must sum to 1.
    screen_set : (s,) ndarray
        List of indices into ``groups`` that correspond to the screen groups.
        ``screen_set[i]`` is ``i`` th screen group.
        ``screen_set`` must contain at least the true (optimal) active groups
        when the regularization is given by ``lmda``.
    lmda_path : (L,) ndarray
        The regularization path to solve for.
        It is recommended that the path is sorted in decreasing order.
    rsq : float
        The change in unnormalized :math:`R^2` given by 
        :math:`\\|y_c-X_c\\beta_{\\mathrm{old}}\\|_{W}^2 - \\|y_c-X_c\\beta_{\\mathrm{curr}}\\|_{W}^2`.
        Usually, :math:`\\beta_{\\mathrm{old}} = 0` 
        and :math:`\\beta_{\\mathrm{curr}}` is given by ``screen_beta``.
    resid : (n,) ndarray
        Residual :math:`y_c - X \\beta` where :math:`\\beta` is given by ``screen_beta``.
    screen_beta : (ws,) ndarray
        Coefficient vector on the screen set.
        ``screen_beta[b:b+p]`` is the coefficient for the ``i`` th screen group 
        where
        ``k = screen_set[i]``,
        ``b = screen_begins[i]``,
        and ``p = group_sizes[k]``.
        The values can be arbitrary but it is recommended to be close to the solution at ``lmda``.
    screen_is_active : (s,) ndarray
        Boolean vector that indicates whether each screen group in ``groups`` is active or not.
        ``screen_is_active[i]`` is ``True`` if and only if ``screen_set[i]`` is active.
    screen_dual : (ds,) ndarray
        Dual vector on the screen set.
        ``screen_dual[b:b+p]`` is the dual for the ``i`` th screen group 
        where
        ``k = screen_set[i]``,
        ``b = screen_dual_begins[i]``,
        and ``p = constraints[k].dual_size``.
    active_set_size : int
        Number of active groups.
        ``active_set[i]`` is only well-defined
        for ``i`` in the range ``[0, active_set_size)``.
    active_set : (G,) ndarray
        List of indices into ``screen_set`` that correspond to active groups.
        ``screen_set[active_set[i]]`` is the ``i`` th active group.
        An active group is one with non-zero coefficient block,
        that is, for every ``i`` th active group, 
        ``screen_beta[b:b+p] == 0`` where 
        ``j = active_set[i]``,
        ``k = screen_set[j]``,
        ``b = screen_begins[j]``,
        and ``p = group_sizes[k]``.
    intercept : bool, optional
        ``True`` to fit with intercept.
        Default is ``True``.
    max_active_size : int, optional
        Maximum number of active groups allowed.
        The function will return a valid state and guarantees to have active set size
        less than or equal to ``max_active_size``.
        If ``None``, it will be set to the total number of groups.
        Default is ``None``.
    max_iters : int, optional
        Maximum number of coordinate descents.
        Default is ``int(1e5)``.
    tol : float, optional
        Coordinate descent convergence tolerance.
        Default is ``1e-7``.
    adev_tol : float, optional
        Percent deviance explained tolerance.
        If the training percent deviance explained exceeds this quantity, 
        then the solver terminates.
        Default is ``0.9``.
    ddev_tol : float, optional
        Difference in percent deviance explained tolerance.
        If the difference of the last two training percent deviance explained exceeds this quantity, 
        then the solver terminates.
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

    Returns
    -------
    wrap
        Wrapper state object.

    See Also
    --------
    adelie.adelie_core.state.StateGaussianPinNaive32
    adelie.adelie_core.state.StateGaussianPinNaive64
    """
    if not (
        isinstance(X, matrix.MatrixNaiveBase64) or
        isinstance(X, matrix.MatrixNaiveBase32)
    ):
        raise ValueError(
            "X must be an instance of MatrixNaiveBase32 or MatrixNaiveBase64."
        )

    p = X.cols()
    group_sizes = np.concatenate([groups, [p]], dtype=int)
    group_sizes = group_sizes[1:] - group_sizes[:-1]

    dtype = (
        np.float64
        if isinstance(X, matrix.MatrixNaiveBase64) else
        np.float32
    )
        
    dispatcher = {
        np.float64: core.state.StateGaussianPinNaive64,
        np.float32: core.state.StateGaussianPinNaive32,
    }

    core_base = dispatcher[dtype]

    class _gaussian_pin_naive(gaussian_pin_naive_base, core_base):
        def __init__(self):
            self._core_type = core_base
            ## save inputs due to lifetime issues
            # static inputs require a reference to input
            # or copy if it must be made
            self._X = X
            self._groups = np.array(groups, copy=True, dtype=int)
            self._group_sizes = np.array(group_sizes, copy=True, dtype=int)
            self._penalty = np.array(penalty, copy=True, dtype=dtype)
            self._weights = np.array(weights, copy=True, dtype=dtype)
            self._screen_set = np.array(screen_set, copy=True, dtype=int)
            self._lmda_path = np.array(lmda_path, copy=True, dtype=dtype)
            # dynamic inputs require a copy to not modify user's inputs
            self._resid = np.array(resid, copy=True, dtype=dtype)
            self._screen_beta = np.array(screen_beta, copy=True, dtype=dtype)
            self._screen_is_active = np.array(screen_is_active, copy=True, dtype=bool)
            self._screen_dual = np.array(screen_dual, copy=True, dtype=dtype)
            self._active_set = np.array(active_set, copy=True, dtype=int)

            (
                self._constraints,
                self._dual_groups,
                self._screen_begins,
                self._screen_dual_begins,
            ) = deduce_states(
                constraints=constraints,
                group_sizes=group_sizes,
                screen_set=screen_set,
                dtype=dtype,
            )

            self._max_active_size = (
                len(self._groups)
                if max_active_size is None else
                np.minimum(max_active_size, len(self._groups))
            )

            n, p = X.rows(), X.cols()
            sqrt_weights = np.sqrt(self._weights)
            ones = np.ones(n, dtype=dtype)
            X_means = np.empty(p, dtype=dtype)
            X.mul(ones, self._weights, X_means)

            self._screen_vars = []
            self._screen_X_means = []
            self._screen_transforms = []
            for i in self._screen_set:
                g, gs = groups[i], group_sizes[i]
                XiTXi = np.empty((gs, gs), dtype=dtype, order="F")
                buffer = np.empty((n, gs), dtype=dtype, order="F")
                X.cov(g, gs, sqrt_weights, XiTXi, buffer)
                Xi_means = X_means[g:g+gs]
                if intercept:
                    XiTXi -= Xi_means[:, None] @ Xi_means[None]
                evars, v = np.linalg.eigh(XiTXi)
                self._screen_vars.append(np.maximum(evars, 0))
                self._screen_X_means.append(Xi_means)
                self._screen_transforms.append(np.array(v, copy=False, dtype=dtype, order="F"))
            self._screen_vars = np.concatenate(self._screen_vars, dtype=dtype)
            self._screen_X_means = np.concatenate(self._screen_X_means, dtype=dtype)
            vecmat_type = (
                core.VectorMatrix64
                if dtype == np.float64 else
                core.VectorMatrix32
            )
            self._screen_transforms = vecmat_type(self._screen_transforms)

            resid_sum = np.sum(self._weights * self._resid)

            # MUST call constructor directly and not use super()!
            # https://pybind11.readthedocs.io/en/stable/advanced/classes.html#forced-trampoline-class-initialisation
            core_base.__init__(
                self,
                X=X,
                y_mean=y_mean,
                y_var=y_var,
                constraints=self._constraints,
                groups=self._groups,
                group_sizes=self._group_sizes,
                dual_groups=self._dual_groups,
                alpha=alpha,
                penalty=self._penalty,
                weights=self._weights,
                screen_set=self._screen_set,
                screen_begins=self._screen_begins,
                screen_vars=self._screen_vars,
                screen_X_means=self._screen_X_means,
                screen_transforms=self._screen_transforms,
                screen_dual_begins=self._screen_dual_begins,
                lmda_path=self._lmda_path,
                intercept=intercept,
                max_active_size=self._max_active_size,
                max_iters=max_iters,
                tol=tol,
                adev_tol=adev_tol,
                ddev_tol=ddev_tol,
                newton_tol=newton_tol,
                newton_max_iters=newton_max_iters,
                n_threads=n_threads,
                rsq=rsq,
                resid=self._resid,
                resid_sum=resid_sum,
                screen_beta=self._screen_beta,
                screen_is_active=self._screen_is_active,
                screen_dual=self._screen_dual,
                active_set_size=active_set_size,
                active_set=self._active_set,
            )

        @classmethod
        def create_from_core(cls, state, core_state):
            obj = base.create_from_core(
                cls, state, core_state, _gaussian_pin_naive, core_base,
            )
            gaussian_pin_naive_base.__init__(obj)
            return obj

    return _gaussian_pin_naive()


class gaussian_pin_cov_base(gaussian_pin_base):
    """State wrapper base class for all pin, covariance method."""

    def check(
        self, 
        method: str =None, 
        logger=logger.logger,
    ):
        super().check(method=method, logger=logger)
        self._check(
            isinstance(self.A, matrix.MatrixCovBase32) or isinstance(self.A, matrix.MatrixCovBase64),
            "check A type",
            method, logger,
        )


def gaussian_pin_cov(
    *,
    A: Union[MatrixCovBase32, MatrixCovBase64],
    constraints: list[Union[ConstraintBase32, ConstraintBase64]],
    groups: np.ndarray,
    alpha: float,
    penalty: np.ndarray,
    screen_set: np.ndarray,
    lmda_path: np.ndarray,
    rsq: float,
    screen_beta: np.ndarray,
    screen_grad: np.ndarray,
    screen_is_active: np.ndarray,
    screen_dual: np.ndarray,
    active_set_size: int,
    active_set: np.ndarray,
    max_active_size: int =None,
    max_iters: int =int(1e5),
    tol: float =1e-7,
    rdev_tol: float =1e-4,
    newton_tol: float =1e-12,
    newton_max_iters: int =1000,
    n_threads: int =1,
):
    """Creates a Gaussian, pin, covariance method state object.

    Define the following quantities:

        - :math:`X_c` as :math:`X` if ``intercept`` is ``False`` and otherwise 
          the column-centered version.
        - :math:`y_c` as :math:`y - \\eta^0` if ``intercept`` is ``False`` and otherwise
          the centered version.

    Parameters
    ----------
    A : Union[MatrixCovBase32, MatrixCovBase64]
        Covariance matrix :math:`X_c^\\top W X_c`.
        It is typically one of the matrices defined in :mod:`adelie.matrix` submodule.
    constraints : (G,) list[Union[ConstraintBase32, ConstraintBase64]], optional
        List of constraints for each group.
        ``constraints[i]`` is the constraint object corresponding to group ``i``.
        If ``constraints[i]`` is ``None``, then the ``i`` th group is unconstrained.
        If ``None``, every group is unconstrained.
    groups : (G,) ndarray
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
    alpha : float
        Elastic net parameter.
        It must be in the range :math:`[0,1]`.
    penalty : (G,) ndarray
        Penalty factor for each group in the same order as ``groups``.
        It must be a non-negative vector.
    screen_set : (s,) ndarray
        List of indices into ``groups`` that correspond to the screen groups.
        ``screen_set[i]`` is ``i`` th screen group.
        ``screen_set`` must contain at least the true (optimal) active groups
        when the regularization is given by ``lmda``.
    lmda_path : (L,) ndarray
        The regularization path to solve for.
        It is recommended that the path is sorted in decreasing order.
    rsq : float
        The change in unnormalized :math:`R^2` given by 
        :math:`\\|y_c-X_c\\beta_{\\mathrm{old}}\\|_{W}^2 - \\|y_c-X_c\\beta_{\\mathrm{curr}}\\|_{W}^2`.
        Usually, :math:`\\beta_{\\mathrm{old}} = 0` 
        and :math:`\\beta_{\\mathrm{curr}}` is given by ``screen_beta``.
    screen_beta : (ws,) ndarray
        Coefficient vector on the screen set.
        ``screen_beta[b:b+p]`` is the coefficient for the ``i`` th screen group 
        where
        ``k = screen_set[i]``,
        ``b = screen_begins[i]``,
        and ``p = group_sizes[k]``.
        The values can be arbitrary but it is recommended to be close to the solution at ``lmda``.
    screen_grad : (ws,) ndarray
        Gradient :math:`X_{c,k}^\\top W (y_c-X_c\\beta)` on the screen groups :math:`k` where :math:`\\beta` is given by ``screen_beta``.
        ``screen_grad[b:b+p]`` is the gradient for the ``i`` th screen group
        where 
        ``k = screen_set[i]``,
        ``b = screen_begins[i]``,
        and ``p = group_sizes[k]``.
    screen_is_active : (s,) ndarray
        Boolean vector that indicates whether each screen group in ``groups`` is active or not.
        ``screen_is_active[i]`` is ``True`` if and only if ``screen_set[i]`` is active.
    screen_dual : (ds,) ndarray
        Dual vector on the screen set.
        ``screen_dual[b:b+p]`` is the dual for the ``i`` th screen group 
        where
        ``k = screen_set[i]``,
        ``b = screen_dual_begins[i]``,
        and ``p = constraints[k].dual_size``.
    active_set_size : int
        Number of active groups.
        ``active_set[i]`` is only well-defined
        for ``i`` in the range ``[0, active_set_size)``.
    active_set : (G,) ndarray
        List of indices into ``screen_set`` that correspond to active groups.
        ``screen_set[active_set[i]]`` is the ``i`` th active group.
        An active group is one with non-zero coefficient block,
        that is, for every ``i`` th active group, 
        ``screen_beta[b:b+p] == 0`` where 
        ``j = active_set[i]``,
        ``k = screen_set[j]``,
        ``b = screen_begins[j]``,
        and ``p = group_sizes[k]``.
    max_active_size : int, optional
        Maximum number of active groups allowed.
        The function will return a valid state and guarantees to have active set size
        less than or equal to ``max_active_size``.
        If ``None``, it will be set to the total number of groups.
        Default is ``None``.
    max_iters : int, optional
        Maximum number of coordinate descents.
        Default is ``int(1e5)``.
    tol : float, optional
        Coordinate descent convergence tolerance.
        Default is ``1e-7``.
    rdev_tol : float, optional
        Relative percent deviance explained tolerance.
        If the difference of the last two training percent deviance explained exceeds 
        the last training percent deviance explained scaled by this quantity,
        then the solver terminates.
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

    Returns
    -------
    wrap
        Wrapper state object.

    See Also
    --------
    adelie.adelie_core.state.StateGaussianPinCov32
    adelie.adelie_core.state.StateGaussianPinCov64
    """
    if not (
        isinstance(A, matrix.MatrixCovBase64) or 
        isinstance(A, matrix.MatrixCovBase32)
    ):
        raise ValueError(
            "A must be an instance of MatrixCovBase32 or MatrixCovBase64."
        )

    p = A.cols()
    group_sizes = np.concatenate([groups, [p]], dtype=int)
    group_sizes = group_sizes[1:] - group_sizes[:-1]

    dtype = (
        np.float64
        if isinstance(A, matrix.MatrixCovBase64) else
        np.float32
    )
        
    dispatcher = {
        np.float64: core.state.StateGaussianPinCov64,
        np.float32: core.state.StateGaussianPinCov32,
    }

    core_base = dispatcher[dtype]

    class _gaussian_pin_cov(gaussian_pin_cov_base, core_base):
        def __init__(self):
            self._core_type = core_base
            ## save inputs due to lifetime issues
            # static inputs require a reference to input
            # or copy if it must be made
            self._A = A
            self._groups = np.array(groups, copy=True, dtype=int)
            self._group_sizes = np.array(group_sizes, copy=True, dtype=int)
            self._penalty = np.array(penalty, copy=True, dtype=dtype)
            self._screen_set = np.array(screen_set, copy=True, dtype=int)
            self._lmda_path = np.array(lmda_path, copy=True, dtype=dtype)
            # dynamic inputs require a copy to not modify user's inputs
            self._screen_beta = np.array(screen_beta, copy=True, dtype=dtype)
            self._screen_grad = np.array(screen_grad, copy=True, dtype=dtype)
            self._screen_is_active = np.array(screen_is_active, copy=True, dtype=bool)
            self._screen_dual = np.array(screen_dual, copy=True, dtype=dtype)
            self._active_set = np.array(active_set, copy=True, dtype=int)

            (
                self._constraints,
                self._dual_groups,
                self._screen_begins,
                self._screen_dual_begins,
            ) = deduce_states(
                constraints=constraints,
                group_sizes=group_sizes,
                screen_set=screen_set,
                dtype=dtype,
            )

            self._max_active_size = (
                len(self._groups)
                if max_active_size is None else
                np.minimum(max_active_size, len(self._groups))
            )

            self._screen_vars = []
            self._screen_transforms = []
            for i in self._screen_set:
                g, gs = groups[i], group_sizes[i]
                Aii = np.empty((gs, gs), dtype=dtype, order="F")
                A.to_dense(g, gs, Aii)  
                dsq, v = np.linalg.eigh(Aii)
                self._screen_vars.append(np.maximum(dsq, 0))
                self._screen_transforms.append(v)
            self._screen_vars = np.concatenate(self._screen_vars, dtype=dtype)
            vecmat_type = (
                core.VectorMatrix64
                if dtype == np.float64 else
                core.VectorMatrix32
            )
            self._screen_transforms = vecmat_type(self._screen_transforms)

            self._screen_subset = np.concatenate([
                np.arange(groups[i], groups[i] + group_sizes[i])
                for i in self._screen_set
            ], dtype=int)
            self._screen_subset_order = np.argsort(self._screen_subset)
            self._screen_subset_ordered = self._screen_subset[self._screen_subset_order]

            core_base.__init__(
                self,
                A=A,
                constraints=self._constraints,
                groups=self._groups,
                group_sizes=self._group_sizes,
                dual_groups=self._dual_groups,
                alpha=alpha,
                penalty=self._penalty,
                screen_set=self._screen_set,
                screen_begins=self._screen_begins,
                screen_vars=self._screen_vars,
                screen_transforms=self._screen_transforms,
                screen_dual_begins=self._screen_dual_begins,
                screen_subset_order=self._screen_subset_order,
                screen_subset_ordered=self._screen_subset_ordered,
                lmda_path=self._lmda_path,
                max_active_size=self._max_active_size,
                max_iters=max_iters,
                tol=tol,
                rdev_tol=rdev_tol,
                newton_tol=newton_tol,
                newton_max_iters=newton_max_iters,
                n_threads=n_threads,
                rsq=rsq,
                screen_beta=self._screen_beta,
                screen_grad=self._screen_grad,
                screen_is_active=self._screen_is_active,
                screen_dual=self._screen_dual,
                active_set_size=active_set_size,
                active_set=self._active_set,
            )

        @classmethod
        def create_from_core(cls, state, core_state):
            obj = base.create_from_core(
                cls, state, core_state, _gaussian_pin_cov, core_base,
            )
            gaussian_pin_cov_base.__init__(obj)
            return obj

    return _gaussian_pin_cov()


def _render_gaussian_inputs(
    *,
    groups,
    lmda_max,
    lmda_path,
    lmda_path_size,
    max_screen_size,
    max_active_size,
    dtype,
):
    if max_screen_size is None:
        max_screen_size = len(groups)
    if max_active_size is None:
        max_active_size = len(groups)
    max_screen_size = np.minimum(max_screen_size, len(groups))
    max_active_size = np.minimum(max_active_size, len(groups))

    lmda_path_size = (
        lmda_path_size
        if lmda_path is None else
        len(lmda_path)
    )

    setup_lmda_max = lmda_max is None
    setup_lmda_path = lmda_path is None

    if setup_lmda_max: lmda_max = -1
    if setup_lmda_path: lmda_path = np.empty(0, dtype=dtype)

    return (
        max_screen_size,
        max_active_size,
        lmda_path_size,
        setup_lmda_max,
        setup_lmda_path,
        lmda_max,
        lmda_path,
        dtype,
    )


def _render_gaussian_cov_inputs(
    *,
    A,
    **kwargs,
):
    if not (
        isinstance(A, matrix.MatrixCovBase64) or 
        isinstance(A, matrix.MatrixCovBase32) or
        isinstance(A, np.ndarray)
    ):
        raise ValueError(
            "A must be an instance of MatrixCovBase32, MatrixCovBase64, or np.ndarray."
        )

    dtype = (
        np.float64
        if (
            isinstance(A, matrix.MatrixCovBase64) or
            (isinstance(A, np.ndarray) and A.dtype == np.dtype("float64"))
        ) else
        np.float32
    )

    return _render_gaussian_inputs(dtype=dtype, **kwargs)


def _render_gaussian_naive_inputs(
    *,
    X,
    **kwargs,
):
    if not (
        isinstance(X, matrix.MatrixNaiveBase64) or 
        isinstance(X, matrix.MatrixNaiveBase32) or
        isinstance(X, np.ndarray)
    ):
        raise ValueError(
            "X must be an instance of MatrixNaiveBase32, MatrixNaiveBase64, or np.ndarray."
        )

    dtype = (
        np.float64
        if (
            isinstance(X, matrix.MatrixNaiveBase64) or
            (isinstance(X, np.ndarray) and X.dtype == np.dtype("float64"))
        ) else
        np.float32
    )

    return _render_gaussian_inputs(dtype=dtype, **kwargs)


def _render_multi_inputs(
    *,
    X,
    groups,
    offsets,
    intercept,
    n_threads,
    dtype,
):
    offsets = np.array(offsets, order="C", copy=False, dtype=dtype)
    n, n_classes = offsets.shape
    X = matrix.kronecker_eye(X, n_classes, n_threads=n_threads)
    if intercept:
        ones_kron = matrix.kronecker_eye(
            np.ones((n, 1), dtype=dtype), 
            n_classes, 
            n_threads=n_threads,
        )
        X = matrix.concatenate(
            [ones_kron, X], 
            axis=1,
            n_threads=n_threads,
        )

    # G == (p+intercept) * K if and only if ungrouped
    if groups.shape[0] == X.cols():
        group_type = "ungrouped"
    else:
        if groups.shape[0] != X.cols() // n_classes + (n_classes - 1) * intercept:
            raise RuntimeError("groups must be of the \"grouped\" or \"ungrouped\" type.")
        group_type = "grouped"

    return (
        X, offsets, group_type
    )


def gaussian_cov(
    *,
    A: Union[MatrixCovBase32, matrix.MatrixCovBase64],
    v: np.ndarray,
    constraints: list[Union[ConstraintBase32, ConstraintBase64]],
    groups: np.ndarray,
    group_sizes: np.ndarray,
    alpha: float,
    penalty: np.ndarray,
    screen_set: np.ndarray,
    screen_beta: np.ndarray,
    screen_is_active: np.ndarray,
    screen_dual: np.ndarray,
    active_set_size: int,
    active_set: np.ndarray,
    rsq: float,
    lmda: float,
    grad: np.ndarray,
    lmda_path: np.ndarray =None,
    lmda_max: float =None,
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
):
    """Creates a Gaussian, covariance method state object.

    Parameters
    ----------
    A : (p, p) Union[MatrixCovBase32, MatrixCovBase64]
        Positive semi-definite matrix.
        It is typically one of the matrices defined in :mod:`adelie.matrix` submodule.
    v : (p,) ndarray
        Linear term.
    constraints : (G,) list[Union[ConstraintBase32, ConstraintBase64]], optional
        List of constraints for each group.
        ``constraints[i]`` is the constraint object corresponding to group ``i``.
        If ``constraints[i]`` is ``None``, then the ``i`` th group is unconstrained.
        If ``None``, every group is unconstrained.
    groups : (G,) ndarray
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
    group_sizes : (G,) ndarray
        List of group sizes corresponding to each element of ``groups``.
        ``group_sizes[i]`` is the size of the ``i`` th group.
    alpha : float
        Elastic net parameter.
        It must be in the range :math:`[0,1]`.
    penalty : (G,) ndarray
        Penalty factor for each group in the same order as ``groups``.
        It must be a non-negative vector.
    screen_set : (s,) ndarray
        List of indices into ``groups`` that correspond to the screen groups.
        ``screen_set[i]`` is ``i`` th screen group.
        ``screen_set`` must contain at least the true (optimal) active groups
        when the regularization is given by ``lmda``.
    screen_beta : (ws,) ndarray
        Coefficient vector on the screen set.
        ``screen_beta[b:b+p]`` is the coefficient for the ``i`` th screen group 
        where
        ``k = screen_set[i]``,
        ``b = screen_begins[i]``,
        and ``p = group_sizes[k]``.
        The values can be arbitrary but it is recommended to be close to the solution at ``lmda``.
    screen_is_active : (s,) ndarray
        Boolean vector that indicates whether each screen group in ``groups`` is active or not.
        ``screen_is_active[i]`` is ``True`` if and only if ``screen_set[i]`` is active.
    screen_dual : (ds,) ndarray
        Dual vector on the screen set.
        ``screen_dual[b:b+p]`` is the dual for the ``i`` th screen group 
        where
        ``k = screen_set[i]``,
        ``b = screen_dual_begins[i]``,
        and ``p = constraints[k].dual_size``.
    active_set_size : int
        Number of active groups.
        ``active_set[i]`` is only well-defined
        for ``i`` in the range ``[0, active_set_size)``.
    active_set : (G,) ndarray
        List of indices into ``screen_set`` that correspond to active groups.
        ``screen_set[active_set[i]]`` is the ``i`` th active group.
        An active group is one with non-zero coefficient block,
        that is, for every ``i`` th active group, 
        ``screen_beta[b:b+p] == 0`` where 
        ``j = active_set[i]``,
        ``k = screen_set[j]``,
        ``b = screen_begins[j]``,
        and ``p = group_sizes[k]``.
    rsq : float
        The change in unnormalized :math:`R^2` given by 
        :math:`2(\\ell(\\beta_{\\mathrm{old}}) - \\ell(\\beta_{\\mathrm{curr}}))`.
        Usually, :math:`\\beta_{\\mathrm{old}} = 0` 
        and :math:`\\beta_{\\mathrm{curr}}` is given by ``screen_beta``.
    lmda : float
        The last regularization parameter that was attempted to be solved.
    grad : (p,) ndarray
        The full gradient :math:`v - A \\beta` where
        :math:`\\beta` is given by ``screen_beta``.
    lmda_path : (L,) ndarray, optional
        The regularization path to solve for.
        The full path is not considered if ``early_exit`` is ``True``.
        It is recommended that the path is sorted in decreasing order.
        If ``None``, the path will be generated.
        Default is ``None``.
    lmda_max : float, optional
        The smallest :math:`\\lambda` such that the true solution is zero
        for all coefficients that have a non-vanishing group lasso penalty (:math:`\\ell_2`-norm).
        If ``None``, it will be computed.
        Default is ``None``.
    max_iters : int, optional
        Maximum number of coordinate descents.
        Default is ``int(1e5)``.
    tol : float, optional
        Coordinate descent convergence tolerance.
        Default is ``1e-7``.
    rdev_tol : float, optional
        Relative percent deviance explained tolerance.
        If the difference of the last two training percent deviance explained exceeds 
        the last training percent deviance explained scaled by this quantity,
        then the solver terminates.
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
        ``True`` if the function should early exit based on training percent deviance explained.
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

    Returns
    -------
    wrap
        Wrapper state object.

    See Also
    --------
    adelie.adelie_core.state.StateGaussianCov32
    adelie.adelie_core.state.StateGaussianCov64
    """
    (
        max_screen_size,
        max_active_size,
        lmda_path_size,
        setup_lmda_max,
        setup_lmda_path,
        lmda_max,
        lmda_path,
        dtype,
    ) = _render_gaussian_cov_inputs(
        A=A,
        groups=groups,
        lmda_max=lmda_max,
        lmda_path=lmda_path,
        lmda_path_size=lmda_path_size,
        max_screen_size=max_screen_size,
        max_active_size=max_active_size,
    )

    dispatcher = {
        np.float64: core.state.StateGaussianCov64,
        np.float32: core.state.StateGaussianCov32,
    }
    core_base = dispatcher[dtype]

    if isinstance(A, np.ndarray):
        A = matrix.dense(A, method="cov", n_threads=n_threads)

    class _gaussian_cov(base, core_base):
        def __init__(self):
            self._core_type = core_base
            ## save inputs due to lifetime issues
            # static inputs require a reference to input
            # or copy if it must be made
            self._A = A
            self._v = np.array(v, copy=True, dtype=dtype)
            self._constraints = render_constraints(groups.shape[0], constraints, dtype)
            self._groups = np.array(groups, copy=True, dtype=int)
            self._group_sizes = np.array(group_sizes, copy=True, dtype=int)
            self._dual_groups = render_dual_groups(self._constraints)
            self._penalty = np.array(penalty, copy=True, dtype=dtype)
            self._lmda_path = np.array(lmda_path, copy=False, dtype=dtype)
            self._screen_set = np.array(screen_set, copy=False, dtype=int)
            self._screen_beta = np.array(screen_beta, copy=False, dtype=dtype)
            self._screen_is_active = np.array(screen_is_active, copy=False, dtype=bool)
            self._screen_dual = np.array(screen_dual, copy=False, dtype=dtype)
            self._grad = np.array(grad, copy=False, dtype=dtype)
            self._active_set = np.array(active_set, copy=False, dtype=int)

            # MUST call constructor directly and not use super()!
            # https://pybind11.readthedocs.io/en/stable/advanced/classes.html#forced-trampoline-class-initialisation
            core_base.__init__(
                self,
                A=self._A,
                v=self._v,
                constraints=self._constraints,
                groups=self._groups,
                group_sizes=self._group_sizes,
                dual_groups=self._dual_groups,
                alpha=alpha,
                penalty=self._penalty,
                lmda_path=self._lmda_path,
                lmda_max=lmda_max,
                min_ratio=min_ratio,
                lmda_path_size=lmda_path_size,
                max_screen_size=max_screen_size,
                max_active_size=max_active_size,
                pivot_subset_ratio=pivot_subset_ratio,
                pivot_subset_min=pivot_subset_min,
                pivot_slack_ratio=pivot_slack_ratio,
                screen_rule=screen_rule,
                max_iters=max_iters,
                tol=tol,
                rdev_tol=rdev_tol,
                newton_tol=newton_tol,
                newton_max_iters=newton_max_iters,
                early_exit=early_exit,
                setup_lmda_max=setup_lmda_max,
                setup_lmda_path=setup_lmda_path,
                n_threads=n_threads,
                screen_set=self._screen_set,
                screen_beta=self._screen_beta,
                screen_is_active=self._screen_is_active,
                screen_dual=self._screen_dual,
                active_set_size=active_set_size,
                active_set=self._active_set,
                rsq=rsq,
                lmda=lmda,
                grad=self._grad,
            )

        @classmethod
        def create_from_core(cls, state, core_state):
            obj = base.create_from_core(
                cls, state, core_state, _gaussian_cov, core_base,
            )
            return obj

    return _gaussian_cov()
    

class gaussian_naive_base(base):
    def check(
        self,
        method: str =None, 
        logger=logger.logger,
    ):
        n, p = self.X.rows(), self.X.cols()

        yc = self._glm.y
        if self.intercept:
            yc = yc - np.sum(yc * self.weights)

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
        self._check(
            np.allclose(self.groups, np.cumsum(np.concatenate([[0], self.group_sizes]))[:-1]),
            "check groups and group_sizes consistency",
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

        # ================ weights check ====================
        self._check(
            np.all(self.weights >= 0),
            "check weights is non-negative",
            method, logger,
        )
        self._check(
            np.allclose(np.sum(self.weights), 1),
            "check weights sum to 1",
            method, logger,
        )

        # ================ screen_set check ====================
        self._check(
            np.all((0 <= self.screen_set) & (self.screen_set < G)),
            "check screen_set is a subset of [0, G)",
            method, logger,
        )
        self._check(
            len(self.screen_set) == len(np.unique(self.screen_set)),
            "check screen_set has unique values",
            method, logger,
        )
        self._check(
            self.screen_set.dtype == np.dtype("int"),
            "check screen_set dtype is int",
            method, logger,
        )
        S = len(self.screen_set)

        # ================ screen_begins check ====================
        expected = np.cumsum(
            np.concatenate([[0], self.group_sizes[self.screen_set]], dtype=int)
        )
        WS = expected[-1]
        expected = expected[:-1]
        self._check(
            np.all(self.screen_begins == expected),
            "check screen_begins is [0, g1, g2, ...] where gi is the group size of (i-1)th screen group.",
            method, logger,
        )
        self._check(
            self.screen_begins.dtype == np.dtype("int"),
            "check screen_begins dtype is int",
            method, logger,
        )

        # ================ screen_beta check ====================
        self._check(
            len(self.screen_beta) == WS,
            "check screen_beta size",
            method, logger,
        )

        # ================ screen_is_active check ====================
        # This one is tricky! Since we keep track of ever-active set,
        # some coefficients may have once been active but now zero'ed out.
        # We can only check that if the non-zero coefficient blocks are active.
        nnz_idxs = np.array([
            i 
            for i, sb, gs in zip(
                np.arange(len(self.screen_set)),
                self.screen_begins, 
                self.group_sizes[self.screen_set],
            )
            if np.any(self.screen_beta[sb:sb+gs] != 0)
        ], dtype=int)
        self._check(
            np.all(self.screen_is_active[nnz_idxs]),
            "check screen_is_active is only active on non-zeros of screen_beta",
            method, logger,
        )

        # ================ rsq check ====================
        screen_indices = []
        Xbeta = np.zeros(n)
        for g, gs, b in zip(
            self.groups[self.screen_set], 
            self.group_sizes[self.screen_set],
            self.screen_begins,
        ):
            screen_indices.append(np.arange(g, g + gs))
            self.X.btmul(g, gs, self.screen_beta[b:b+gs], Xbeta)

        if len(screen_indices) == 0:
            screen_indices = np.array(screen_indices, dtype=int)
        else:
            screen_indices = np.concatenate(screen_indices, dtype=int)

        resid = yc - Xbeta
        grad = np.empty(p)
        self.X.mul(resid, self.weights, grad)
        if self.intercept:
            grad -= self.X_means * np.sum(self.weights * resid)
        WXcbeta = self.weights * (Xbeta - self.screen_X_means @ self.screen_beta)
        expected = 2 * np.sum(yc * WXcbeta) - np.linalg.norm(WXcbeta / np.sqrt(self.weights)) ** 2
        self._check(
            np.allclose(self.rsq, expected),
            "check rsq",
            method, logger,
        )

        # ================ grad check ====================
        self._check(
            np.allclose(self.grad, grad),
            "check grad",
            method, logger,
        )

        # ================ abs_grad check ====================
        grad_corr = np.copy(grad)
        for i, b, db, g, gs in zip(
            self.screen_set,
            self.screen_begins,
            self.screen_dual_begins,
            self.groups[self.screen_set],
            self.group_sizes[self.screen_set],
        ):
            lmda = 1e35 if np.isinf(self.lmda) else self.lmda
            constraint_grad = 0
            if not (self.constraints[i] is None):
                ds = self.constraints[i].dual_size
                constraint_grad = np.empty(ds)
                self.constraints[i].gradient(
                    self.screen_beta[b:b+gs],
                    self.screen_dual[db:db+ds],
                    constraint_grad,
                )
            grad_corr[g:g+gs] -= (
                lmda * (1-self.alpha) * self.penalty[i] * self.screen_beta[b:b+gs] + 
                constraint_grad
            )
        abs_grad = np.array([
            np.linalg.norm(grad_corr[g:g+gs])
            for g, gs in zip(self.groups, self.group_sizes)
        ])
        self._check(
            (self.lmda_max == -1) or np.allclose(self.abs_grad, abs_grad),
            "check abs_grad",
            method, logger,
        )

        # ================ resid check ====================
        self._check(
            np.allclose(self.resid, resid),
            "check resid",
            method, logger,
        )

        # ================ resid_sum check ====================
        self._check(
            np.allclose(self.resid_sum, np.sum(self.weights * resid)),
            "check resid_sum",
            method, logger,
        )

        # ================ screen_X_means check ====================
        self._check(
            np.allclose(self.screen_X_means, self.X_means[screen_indices]),
            "check screen_X_means",
            method, logger,
        )

        # ================ screen_transforms / screen_vars check ====================
        sqrt_weights = np.sqrt(self.weights)
        for ss_idx in range(len(self.screen_set)):
            i = self.screen_set[ss_idx]
            g, gs = self.groups[i], self.group_sizes[i]
            Xi = np.empty((n, gs), order="F")
            XiTXi = np.empty((gs, gs), order="F")
            self.X.cov(g, gs, sqrt_weights, XiTXi, Xi)
            if self.intercept:
                Xi_means = self.X_means[g:g+gs]
                XiTXi -= Xi_means[:, None] @ Xi_means[None]
            V = self.screen_transforms[ss_idx]
            Dsq = V.T @ XiTXi @ V
            sb = self.screen_begins[ss_idx]
            self._check(
                np.allclose(self.screen_vars[sb:sb+gs], np.diag(Dsq)),
                f"check screen_vars[{sb}:{sb}+{gs}]",
                method, logger,
            )
            np.fill_diagonal(Dsq, 0)
            self._check(
                np.allclose(Dsq, 0),
                "check VT Xi V is nearly 0 after zeroing the diagonal",
                method, logger,
            )


def gaussian_naive(
    *,
    X: Union[MatrixNaiveBase32, MatrixNaiveBase64],
    y: np.ndarray,
    X_means: np.ndarray,
    y_mean: float,
    y_var: float,
    resid: np.ndarray,
    resid_sum: float,
    constraints: list[Union[ConstraintBase32, ConstraintBase64]],
    groups: np.ndarray,
    group_sizes: np.ndarray,
    alpha: float,
    penalty: np.ndarray,
    weights: np.ndarray,
    offsets: np.ndarray,
    screen_set: np.ndarray,
    screen_beta: np.ndarray,
    screen_is_active: np.ndarray,
    screen_dual: np.ndarray,
    active_set_size: int,
    active_set: np.ndarray,
    rsq: float,
    lmda: float,
    grad: np.ndarray,
    lmda_path: np.ndarray =None,
    lmda_max: float =None,
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
):
    """Creates a Gaussian, naive method state object.

    Define the following quantities:

        - :math:`X_c` as :math:`X` if ``intercept`` is ``False`` and otherwise 
          the column-centered version.
        - :math:`y_c` as :math:`y - \\eta^0` if ``intercept`` is ``False`` and otherwise
          the centered version.

    Parameters
    ----------
    X : (n, p) Union[MatrixNaiveBase32, MatrixNaiveBase64]
        Feature matrix.
        It is typically one of the matrices defined in :mod:`adelie.matrix` submodule.
    y : (n,) ndarray
        Response vector.
        
        .. note::
            This is the original response vector not offsetted!

    X_means : (p,) ndarray
        Column means of ``X`` (weighted by :math:`W`).
    y_mean : float
        Mean of the offsetted response vector :math:`y-\\eta^0` (weighted by :math:`W`),
        i.e. :math:`\\mathbf{1}^\\top W (y-\\eta^0)`.
    y_var : float
        Variance of the offsetted response vector :math:`y-\\eta^0` (weighted by :math:`W`), 
        i.e. :math:`\\|y_c\\|_{W}^2`.
        This is only used for outputting the training :math:`R^2` relative to this value,
        i.e. this quantity is the "null" model MSE.
    resid : (n,) ndarray
        Residual :math:`y_c - X \\beta` where :math:`\\beta` is given by ``screen_beta``.
    resid_sum : float
        Weighted (by :math:`W`) sum of ``resid``.
    constraints : (G,) list[Union[ConstraintBase32, ConstraintBase64]], optional
        List of constraints for each group.
        ``constraints[i]`` is the constraint object corresponding to group ``i``.
        If ``constraints[i]`` is ``None``, then the ``i`` th group is unconstrained.
        If ``None``, every group is unconstrained.
    groups : (G,) ndarray
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
    group_sizes : (G,) ndarray
        List of group sizes corresponding to each element of ``groups``.
        ``group_sizes[i]`` is the size of the ``i`` th group.
    alpha : float
        Elastic net parameter.
        It must be in the range :math:`[0,1]`.
    penalty : (G,) ndarray
        Penalty factor for each group in the same order as ``groups``.
        It must be a non-negative vector.
    weights : (n,) ndarray
        Observation weights :math:`W`.
        The weights must sum to 1.
    offsets : (n,) ndarray
        Observation offsets :math:`\\eta^0`.
    screen_set : (s,) ndarray
        List of indices into ``groups`` that correspond to the screen groups.
        ``screen_set[i]`` is ``i`` th screen group.
        ``screen_set`` must contain at least the true (optimal) active groups
        when the regularization is given by ``lmda``.
    screen_beta : (ws,) ndarray
        Coefficient vector on the screen set.
        ``screen_beta[b:b+p]`` is the coefficient for the ``i`` th screen group 
        where
        ``k = screen_set[i]``,
        ``b = screen_begins[i]``,
        and ``p = group_sizes[k]``.
        The values can be arbitrary but it is recommended to be close to the solution at ``lmda``.
    screen_is_active : (s,) ndarray
        Boolean vector that indicates whether each screen group in ``groups`` is active or not.
        ``screen_is_active[i]`` is ``True`` if and only if ``screen_set[i]`` is active.
    screen_dual : (ds,) ndarray
        Dual vector on the screen set.
        ``screen_dual[b:b+p]`` is the dual for the ``i`` th screen group 
        where
        ``k = screen_set[i]``,
        ``b = screen_dual_begins[i]``,
        and ``p = constraints[k].dual_size``.
    active_set_size : int
        Number of active groups.
        ``active_set[i]`` is only well-defined
        for ``i`` in the range ``[0, active_set_size)``.
    active_set : (G,) ndarray
        List of indices into ``screen_set`` that correspond to active groups.
        ``screen_set[active_set[i]]`` is the ``i`` th active group.
        An active group is one with non-zero coefficient block,
        that is, for every ``i`` th active group, 
        ``screen_beta[b:b+p] == 0`` where 
        ``j = active_set[i]``,
        ``k = screen_set[j]``,
        ``b = screen_begins[j]``,
        and ``p = group_sizes[k]``.
    rsq : float
        The change in unnormalized :math:`R^2` given by 
        :math:`\\|y_c-X_c\\beta_{\\mathrm{old}}\\|_{W}^2 - \\|y_c-X_c\\beta_{\\mathrm{curr}}\\|_{W}^2`.
        Usually, :math:`\\beta_{\\mathrm{old}} = 0` 
        and :math:`\\beta_{\\mathrm{curr}}` is given by ``screen_beta``.
    lmda : float
        The last regularization parameter that was attempted to be solved.
    grad : (p,) ndarray
        The full gradient :math:`X_c^\\top W (y_c - X_c\\beta)` where
        :math:`\\beta` is given by ``screen_beta``.
    lmda_path : (L,) ndarray, optional
        The regularization path to solve for.
        The full path is not considered if ``early_exit`` is ``True``.
        It is recommended that the path is sorted in decreasing order.
        If ``None``, the path will be generated.
        Default is ``None``.
    lmda_max : float, optional
        The smallest :math:`\\lambda` such that the true solution is zero
        for all coefficients that have a non-vanishing group lasso penalty (:math:`\\ell_2`-norm).
        If ``None``, it will be computed.
        Default is ``None``.
    max_iters : int, optional
        Maximum number of coordinate descents.
        Default is ``int(1e5)``.
    tol : float, optional
        Coordinate descent convergence tolerance.
        Default is ``1e-7``.
    adev_tol : float, optional
        Percent deviance explained tolerance.
        If the training percent deviance explained exceeds this quantity
        and ``early_exit`` is ``True``, then the solver terminates.
        Default is ``0.9``.
    ddev_tol : float, optional
        Difference in percent deviance explained tolerance.
        If the difference of the last two training percent deviance explained exceeds this quantity
        and ``early_exit`` is ``True``, then the solver terminates.
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
        ``True`` if the function should early exit based on training percent deviance explained.
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

    Returns
    -------
    wrap
        Wrapper state object.

    See Also
    --------
    adelie.adelie_core.state.StateGaussianNaive32
    adelie.adelie_core.state.StateGaussianNaive64
    """
    (
        max_screen_size,
        max_active_size,
        lmda_path_size,
        setup_lmda_max,
        setup_lmda_path,
        lmda_max,
        lmda_path,
        dtype,
    ) = _render_gaussian_naive_inputs(
        X=X,
        groups=groups,
        lmda_max=lmda_max,
        lmda_path=lmda_path,
        lmda_path_size=lmda_path_size,
        max_screen_size=max_screen_size,
        max_active_size=max_active_size,
    )

    dispatcher = {
        np.float64: core.state.StateGaussianNaive64,
        np.float32: core.state.StateGaussianNaive32,
    }
    core_base = dispatcher[dtype]

    if isinstance(X, np.ndarray):
        X = matrix.dense(X, method="naive", n_threads=n_threads)

    class _gaussian_naive(gaussian_naive_base, core_base):
        def __init__(self):
            self._core_type = core_base
            ## save inputs due to lifetime issues
            # static inputs require a reference to input
            # or copy if it must be made
            self._glm = glm.gaussian(y=y, weights=weights, dtype=dtype)
            self._X = X
            self._X_means = np.array(X_means, copy=True, dtype=dtype)
            self._constraints = render_constraints(groups.shape[0], constraints, dtype)
            self._groups = np.array(groups, copy=True, dtype=int)
            self._group_sizes = np.array(group_sizes, copy=True, dtype=int)
            self._dual_groups = render_dual_groups(self._constraints)
            self._penalty = np.array(penalty, copy=True, dtype=dtype)
            self._offsets = np.array(offsets, copy=True, dtype=dtype)
            self._lmda_path = np.array(lmda_path, copy=False, dtype=dtype)
            self._screen_set = np.array(screen_set, copy=False, dtype=int)
            self._screen_beta = np.array(screen_beta, copy=False, dtype=dtype)
            self._screen_is_active = np.array(screen_is_active, copy=False, dtype=bool)
            self._screen_dual = np.array(screen_dual, copy=False, dtype=dtype)
            self._active_set = np.array(active_set, copy=False, dtype=int)
            self._grad = np.array(grad, copy=False, dtype=dtype)
            self._resid = np.array(resid, copy=False, dtype=dtype)

            # MUST call constructor directly and not use super()!
            # https://pybind11.readthedocs.io/en/stable/advanced/classes.html#forced-trampoline-class-initialisation
            core_base.__init__(
                self,
                X=self._X,
                X_means=self._X_means,
                y_mean=y_mean,
                y_var=y_var,
                resid=self._resid,
                resid_sum=resid_sum,
                constraints=self._constraints,
                groups=self._groups,
                group_sizes=self._group_sizes,
                dual_groups=self._dual_groups,
                alpha=alpha,
                penalty=self._penalty,
                weights=self._glm.weights,
                lmda_path=self._lmda_path,
                lmda_max=lmda_max,
                min_ratio=min_ratio,
                lmda_path_size=lmda_path_size,
                max_screen_size=max_screen_size,
                max_active_size=max_active_size,
                pivot_subset_ratio=pivot_subset_ratio,
                pivot_subset_min=pivot_subset_min,
                pivot_slack_ratio=pivot_slack_ratio,
                screen_rule=screen_rule,
                max_iters=max_iters,
                tol=tol,
                adev_tol=adev_tol,
                ddev_tol=ddev_tol,
                newton_tol=newton_tol,
                newton_max_iters=newton_max_iters,
                early_exit=early_exit,
                setup_lmda_max=setup_lmda_max,
                setup_lmda_path=setup_lmda_path,
                intercept=intercept,
                n_threads=n_threads,
                screen_set=self._screen_set,
                screen_beta=self._screen_beta,
                screen_is_active=self._screen_is_active,
                screen_dual=self._screen_dual,
                active_set_size=active_set_size,
                active_set=self._active_set,
                rsq=rsq,
                lmda=lmda,
                grad=self._grad,
            )

        @classmethod
        def create_from_core(cls, state, core_state):
            obj = base.create_from_core(
                cls, state, core_state, _gaussian_naive, core_base,
            )
            gaussian_naive_base.__init__(obj)
            return obj

    return _gaussian_naive()


def multigaussian_naive(
    *,
    X: Union[MatrixNaiveBase32, MatrixNaiveBase64],
    y: np.ndarray,
    X_means: np.ndarray,
    y_var: float,
    resid: np.ndarray,
    resid_sum: float,
    constraints: list[Union[ConstraintBase32, ConstraintBase64]],
    groups: np.ndarray,
    group_sizes: np.ndarray,
    alpha: float,
    penalty: np.ndarray,
    weights: np.ndarray,
    offsets: np.ndarray,
    screen_set: np.ndarray,
    screen_beta: np.ndarray,
    screen_is_active: np.ndarray,
    screen_dual: np.ndarray,
    active_set_size: int,
    active_set: np.ndarray,
    rsq: float,
    lmda: float,
    grad: np.ndarray,
    lmda_path: np.ndarray =None,
    lmda_max: float =None,
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
):
    """Creates a MultiGaussian, naive method state object.

    Define the following quantities: 

        - :math:`\\tilde{X} = X\\otimes I_K` if ``intercept`` is ``False``, 
          and otherwise :math:`[1 \\otimes I_K, X \\otimes I_K]`.
        - :math:`\\tilde{y}` as the flattened version of :math:`y-\\eta^0` as row-major.
        - :math:`\\tilde{W} = K^{-1} (W \\otimes I_K)`. 

    Parameters
    ----------
    X : (n, p) Union[MatrixNaiveBase32, MatrixNaiveBase64]
        Feature matrix.
        It is typically one of the matrices defined in :mod:`adelie.matrix` submodule.
    y : (n, K) ndarray
        Response matrix.
        
        .. note::
            This is the original response vector not offsetted!

    X_means : ((p+intercept)*K,) ndarray
        Column means (weighted by :math:`\\tilde{W}`) of :math:`\\tilde{X}`.
    y_var : float
        The average of the variance for each response vector
        where variance is given by :math:`\\|y_{k,c} - \\eta_{k,c}^0\\|_W^2` and 
        :math:`z_{k,c}` is the ``k`` th column of :math:`z`, centered if ``intercept`` is ``True``.
        This is only used for outputting the training :math:`R^2` relative to this value,
        i.e. this quantity is the "null" model MSE.
    resid : (n*K,) ndarray
        Residual :math:`\\tilde{y} - \\tilde{X} \\beta` 
        where :math:`\\beta` is given by ``screen_beta``.
    resid_sum : float
        Weighted (by :math:`\\tilde{W}`) sum of ``resid``.
    constraints : (G,) list[Union[ConstraintBase32, ConstraintBase64]], optional
        List of constraints for each group.
        ``constraints[i]`` is the constraint object corresponding to group ``i``.
        If ``constraints[i]`` is ``None``, then the ``i`` th group is unconstrained.
        If ``None``, every group is unconstrained.
    groups : (G,) ndarray
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
    group_sizes : (G,) ndarray
        List of group sizes corresponding to each element of ``groups``.
        ``group_sizes[i]`` is the size of the ``i`` th group.
    alpha : float
        Elastic net parameter.
        It must be in the range :math:`[0,1]`.
    penalty : (G,) ndarray
        Penalty factor for each group in the same order as ``groups``.
        It must be a non-negative vector.
    weights : (n,) ndarray
        Observation weights :math:`W`.
        The weights must sum to 1.
    offsets : (n, K) ndarray
        Observation offsets :math:`\\eta^0`.
    screen_set : (s,) ndarray
        List of indices into ``groups`` that correspond to the screen groups.
        ``screen_set[i]`` is ``i`` th screen group.
        ``screen_set`` must contain at least the true (optimal) active groups
        when the regularization is given by ``lmda``.
    screen_beta : (ws,) ndarray
        Coefficient vector on the screen set.
        ``screen_beta[b:b+p]`` is the coefficient for the ``i`` th screen group 
        where
        ``k = screen_set[i]``,
        ``b = screen_begins[i]``,
        and ``p = group_sizes[k]``.
        The values can be arbitrary but it is recommended to be close to the solution at ``lmda``.
    screen_is_active : (s,) ndarray
        Boolean vector that indicates whether each screen group in ``groups`` is active or not.
        ``screen_is_active[i]`` is ``True`` if and only if ``screen_set[i]`` is active.
    screen_dual : (ds,) ndarray
        Dual vector on the screen set.
        ``screen_dual[b:b+p]`` is the dual for the ``i`` th screen group 
        where
        ``k = screen_set[i]``,
        ``b = screen_dual_begins[i]``,
        and ``p = constraints[k].dual_size``.
    active_set_size : int
        Number of active groups.
        ``active_set[i]`` is only well-defined
        for ``i`` in the range ``[0, active_set_size)``.
    active_set : (G,) ndarray
        List of indices into ``screen_set`` that correspond to active groups.
        ``screen_set[active_set[i]]`` is the ``i`` th active group.
        An active group is one with non-zero coefficient block,
        that is, for every ``i`` th active group, 
        ``screen_beta[b:b+p] == 0`` where 
        ``j = active_set[i]``,
        ``k = screen_set[j]``,
        ``b = screen_begins[j]``,
        and ``p = group_sizes[k]``.
    rsq : float
        The change in unnormalized :math:`R^2` given by 
        :math:`\\|\\tilde{y}-\\tilde{X}\\beta_{\\mathrm{old}}\\|_{\\tilde{W}}^2 - \\|\\tilde{y}-\\tilde{X}\\beta_{\\mathrm{curr}}\\|_{\\tilde{W}}^2`.
        Usually, :math:`\\beta_{\\mathrm{old}} = 0` 
        and :math:`\\beta_{\\mathrm{curr}}` is given by ``screen_beta``.
    lmda : float
        The last regularization parameter that was attempted to be solved.
    grad : ((p+intercept)*K,) ndarray
        The full gradient :math:`\\tilde{X}^\\top \\tilde{W} (\\tilde{y} - \\tilde{X}\\beta)` where
        :math:`\\beta` is given by ``screen_beta``.
    lmda_path : (L,) ndarray, optional
        The regularization path to solve for.
        The full path is not considered if ``early_exit`` is ``True``.
        It is recommended that the path is sorted in decreasing order.
        If ``None``, the path will be generated.
        Default is ``None``.
    lmda_max : float, optional
        The smallest :math:`\\lambda` such that the true solution is zero
        for all coefficients that have a non-vanishing group lasso penalty (:math:`\\ell_2`-norm).
        If ``None``, it will be computed.
        Default is ``None``.
    max_iters : int, optional
        Maximum number of coordinate descents.
        Default is ``int(1e5)``.
    tol : float, optional
        Coordinate descent convergence tolerance.
        Default is ``1e-7``.
    adev_tol : float, optional
        Percent deviance explained tolerance.
        If the training percent deviance explained exceeds this quantity
        and ``early_exit`` is ``True``, then the solver terminates.
        Default is ``0.9``.
    ddev_tol : float, optional
        Difference in percent deviance explained tolerance.
        If the difference of the last two training percent deviance explained exceeds this quantity
        and ``early_exit`` is ``True``, then the solver terminates.
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
        ``True`` if the function should early exit based on training percent deviance explained.
        Default is ``True``.
    min_ratio : float, optional
        The ratio between the largest and smallest :math:`\\lambda` in the regularization sequence
        if it is to be generated.
        Default is ``1e-2``.
    lmda_path_size : int, optional
        Number of regularizations in the path if it is to be generated.
        Default is ``100``.
    intercept : bool, optional 
        ``True`` if the function should fit with intercept for each class.
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

    Returns
    -------
    wrap
        Wrapper state object.

    See Also
    --------
    adelie.adelie_core.state.StateMultiGaussianNaive32
    adelie.adelie_core.state.StateMultiGaussianNaive64
    """
    (
        max_screen_size,
        max_active_size,
        lmda_path_size,
        setup_lmda_max,
        setup_lmda_path,
        lmda_max,
        lmda_path,
        dtype,
    ) = _render_gaussian_naive_inputs(
        X=X,
        groups=groups,
        lmda_max=lmda_max,
        lmda_path=lmda_path,
        lmda_path_size=lmda_path_size,
        max_screen_size=max_screen_size,
        max_active_size=max_active_size,
    )
        
    dispatcher = {
        np.float64: core.state.StateMultiGaussianNaive64,
        np.float32: core.state.StateMultiGaussianNaive32,
    }
    core_base = dispatcher[dtype]

    X_raw = X
    n_classes = y.shape[-1]
    (
        X,
        offsets,
        group_type,
    ) = _render_multi_inputs(
        X=X,
        groups=groups,
        offsets=offsets,
        intercept=intercept,
        n_threads=n_threads,
        dtype=dtype,
    )
    assert X_means.shape[0] == X.cols(), "X_means must have the same length as the number of columns of X after reshaping."

    class _multigaussian_naive(gaussian_naive_base, core_base):
        def __init__(self):
            self._core_type = core_base
            ## save inputs due to lifetime issues
            # static inputs require a reference to input
            # or copy if it must be made
            self._glm = glm.multigaussian(y=y, weights=weights, dtype=dtype)
            self._X = X_raw
            self._X_expanded = X
            self._X_means = np.array(X_means, copy=True, dtype=dtype)
            self._constraints = render_constraints(groups.shape[0], constraints, dtype)
            self._groups = np.array(groups, copy=True, dtype=int)
            self._group_sizes = np.array(group_sizes, copy=True, dtype=int)
            self._dual_groups = render_dual_groups(self._constraints)
            self._penalty = np.array(penalty, copy=True, dtype=dtype)
            self._weights_expanded = np.repeat(self._glm.weights, repeats=n_classes) / n_classes
            self._offsets = np.array(offsets, copy=True, dtype=dtype)
            self._lmda_path = np.array(lmda_path, copy=False, dtype=dtype)
            self._screen_set = np.array(screen_set, copy=False, dtype=int)
            self._screen_beta = np.array(screen_beta, copy=False, dtype=dtype)
            self._screen_is_active = np.array(screen_is_active, copy=False, dtype=bool)
            self._screen_dual = np.array(screen_dual, copy=False, dtype=dtype)
            self._active_set = np.array(active_set, copy=False, dtype=int)
            self._grad = np.array(grad, copy=False, dtype=dtype)
            self._resid = np.array(resid, copy=False, dtype=dtype)

            # MUST call constructor directly and not use super()!
            # https://pybind11.readthedocs.io/en/stable/advanced/classes.html#forced-trampoline-class-initialisation
            core_base.__init__(
                self,
                group_type=group_type,
                n_classes=n_classes,
                multi_intercept=intercept,
                X=self._X_expanded,
                X_means=self._X_means,
                # y_mean is not used in the solver since global intercept is turned off,
                # but it is used to compute loss_null and loss_full.
                # This is not the actual y_mean, but it is a value that will result in correct
                # calculation of loss_null and loss_full.
                y_mean=np.linalg.norm(np.sum(weights[:, None] * (y - offsets), axis=-1) / n_classes),
                y_var=y_var,
                resid=self._resid,
                resid_sum=resid_sum,
                constraints=self._constraints,
                groups=self._groups,
                group_sizes=self._group_sizes,
                dual_groups=self._dual_groups,
                alpha=alpha,
                penalty=self._penalty,
                weights=self._weights_expanded,
                lmda_path=self._lmda_path,
                lmda_max=lmda_max,
                min_ratio=min_ratio,
                lmda_path_size=lmda_path_size,
                max_screen_size=max_screen_size,
                max_active_size=max_active_size,
                pivot_subset_ratio=pivot_subset_ratio,
                pivot_subset_min=pivot_subset_min,
                pivot_slack_ratio=pivot_slack_ratio,
                screen_rule=screen_rule,
                max_iters=max_iters,
                tol=tol,
                adev_tol=adev_tol,
                ddev_tol=ddev_tol,
                newton_tol=newton_tol,
                newton_max_iters=newton_max_iters,
                early_exit=early_exit,
                setup_lmda_max=setup_lmda_max,
                setup_lmda_path=setup_lmda_path,
                intercept=False,
                n_threads=n_threads,
                screen_set=self._screen_set,
                screen_beta=self._screen_beta,
                screen_is_active=self._screen_is_active,
                screen_dual=self._screen_dual,
                active_set_size=active_set_size,
                active_set=self._active_set,
                rsq=rsq,
                lmda=lmda,
                grad=self._grad,
            )

        @classmethod
        def create_from_core(cls, state, core_state):
            obj = base.create_from_core(
                cls, state, core_state, _multigaussian_naive, core_base,
            )
            gaussian_naive_base.__init__(obj)
            return obj

    return _multigaussian_naive()


def _render_glm_naive_inputs(
    *,
    loss_null,
    **kwargs,
):
    out = _render_gaussian_naive_inputs(**kwargs)

    setup_loss_null = loss_null is None
    if setup_loss_null: loss_null = np.inf

    return out + (setup_loss_null, loss_null)


def glm_naive(
    *,
    X: Union[MatrixNaiveBase32, MatrixNaiveBase64],
    glm: Union[GlmBase32, GlmBase64],
    constraints: list[Union[ConstraintBase32, ConstraintBase64]],
    groups: np.ndarray,
    group_sizes: np.ndarray,
    alpha: float,
    penalty: np.ndarray,
    offsets: np.ndarray,
    screen_set: np.ndarray,
    screen_beta: np.ndarray,
    screen_is_active: np.ndarray,
    screen_dual: np.ndarray,
    active_set_size: int,
    active_set: np.ndarray,
    beta0: float,
    lmda: float,
    grad: np.ndarray,
    eta: np.ndarray,
    resid: np.ndarray,
    loss_full: float,
    loss_null: float =None,
    lmda_path: np.ndarray =None,
    lmda_max: float =None,
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
):
    """Creates a GLM, naive method state object.

    Parameters
    ----------
    X : (n, p) Union[MatrixNaiveBase32, MatrixNaiveBase64]
        Feature matrix.
        It is typically one of the matrices defined in :mod:`adelie.matrix` submodule.
    glm : Union[GlmBase32, GlmBase64]
        GLM object.
        It is typically one of the GLM classes defined in :mod:`adelie.glm` submodule.
    constraints : (G,) list[Union[ConstraintBase32, ConstraintBase64]], optional
        List of constraints for each group.
        ``constraints[i]`` is the constraint object corresponding to group ``i``.
        If ``constraints[i]`` is ``None``, then the ``i`` th group is unconstrained.
        If ``None``, every group is unconstrained.
    groups : (G,) ndarray
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
    group_sizes : (G,) ndarray
        List of group sizes corresponding to each element of ``groups``.
        ``group_sizes[i]`` is the size of the ``i`` th group.
    alpha : float
        Elastic net parameter.
        It must be in the range :math:`[0,1]`.
    penalty : (G,) ndarray
        Penalty factor for each group in the same order as ``groups``.
        It must be a non-negative vector.
    offsets : (n,) ndarray
        Observation offsets :math:`\\eta^0`.
    screen_set : (s,) ndarray
        List of indices into ``groups`` that correspond to the screen groups.
        ``screen_set[i]`` is ``i`` th screen group.
        ``screen_set`` must contain at least the true (optimal) active groups
        when the regularization is given by ``lmda``.
    screen_beta : (ws,) ndarray
        Coefficient vector on the screen set.
        ``screen_beta[b:b+p]`` is the coefficient for the ``i`` th screen group 
        where
        ``k = screen_set[i]``,
        ``b = screen_begins[i]``,
        and ``p = group_sizes[k]``.
        The values can be arbitrary but it is recommended to be close to the solution at ``lmda``.
    screen_is_active : (s,) ndarray
        Boolean vector that indicates whether each screen group in ``groups`` is active or not.
        ``screen_is_active[i]`` is ``True`` if and only if ``screen_set[i]`` is active.
    screen_dual : (ds,) ndarray
        Dual vector on the screen set.
        ``screen_dual[b:b+p]`` is the dual for the ``i`` th screen group 
        where
        ``k = screen_set[i]``,
        ``b = screen_dual_begins[i]``,
        and ``p = constraints[k].dual_size``.
    active_set_size : int
        Number of active groups.
        ``active_set[i]`` is only well-defined
        for ``i`` in the range ``[0, active_set_size)``.
    active_set : (G,) ndarray
        List of indices into ``screen_set`` that correspond to active groups.
        ``screen_set[active_set[i]]`` is the ``i`` th active group.
        An active group is one with non-zero coefficient block,
        that is, for every ``i`` th active group, 
        ``screen_beta[b:b+p] == 0`` where 
        ``j = active_set[i]``,
        ``k = screen_set[j]``,
        ``b = screen_begins[j]``,
        and ``p = group_sizes[k]``.
    beta0 : float
        The current intercept value.
        The value can be arbitrary but it is recommended to be close to the solution at ``lmda``.
    lmda : float
        The last regularization parameter that was attempted to be solved.
    grad : (p,) ndarray
        The full gradient :math:`-X^\\top \\nabla \\ell(\\eta)` where
        :math:`\\eta` is given by ``eta``.
    eta : (n,) ndarray
        The natural parameter :math:`\\eta = X\\beta + \\beta_0 \\mathbf{1} + \\eta^0`
        where 
        :math:`\\beta`
        and :math:`\\beta_0` are given by
        ``screen_beta`` and ``beta0``.
    resid : (n,) ndarray
        Residual :math:`-\\nabla \\ell(\\eta)`
        where :math:`\\eta` is given by ``eta``.
    loss_full : float
        Full loss :math:`\\ell(\\eta^\\star)`
        where :math:`\\eta^\\star` is the minimizer.
    loss_null : float, optional
        Null loss :math:`\\ell(\\beta_0^\\star \\mathbf{1} + \\eta^0)`
        from fitting an intercept-only model (if ``intercept`` is ``True``)
        and otherwise :math:`\\ell(\\eta^0)`.
        If ``None``, it will be computed.
        Default is ``None``. 
    lmda_path : (L,) ndarray, optional
        The regularization path to solve for.
        The full path is not considered if ``early_exit`` is ``True``.
        It is recommended that the path is sorted in decreasing order.
        If ``None``, the path will be generated.
        Default is ``None``.
    lmda_max : float, optional
        The smallest :math:`\\lambda` such that the true solution is zero
        for all coefficients that have a non-vanishing group lasso penalty (:math:`\\ell_2`-norm).
        If ``None``, it will be computed.
        Default is ``None``.
    irls_max_iters : int, optional
        Maximum number of IRLS iterations.
        Default is ``int(1e4)``.
    irls_tol : float, optional
        IRLS convergence tolerance.
        Default is ``1e-7``.
    max_iters : int, optional
        Maximum number of coordinate descents.
        Default is ``int(1e5)``.
    tol : float, optional
        Coordinate descent convergence tolerance.
        Default is ``1e-7``.
    adev_tol : float, optional
        Percent deviance explained tolerance.
        If the training percent deviance explained exceeds this quantity
        and ``early_exit`` is ``True``, then the solver terminates.
        Default is ``0.9``.
    ddev_tol : float, optional
        Difference in percent deviance explained tolerance.
        If the difference of the last two training percent deviance explained exceeds this quantity
        and ``early_exit`` is ``True``, then the solver terminates.
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
        ``True`` if the function should early exit based on training percent deviance explained.
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

    Returns
    -------
    wrap
        Wrapper state object.

    See Also
    --------
    adelie.adelie_core.state.StateGlmNaive32
    adelie.adelie_core.state.StateGlmNaive64
    """
    (
        max_screen_size,
        max_active_size,
        lmda_path_size,
        setup_lmda_max,
        setup_lmda_path,
        lmda_max,
        lmda_path,
        dtype,
        setup_loss_null,
        loss_null,
    ) = _render_glm_naive_inputs(
        X=X,
        groups=groups,
        lmda_max=lmda_max,
        lmda_path=lmda_path,
        lmda_path_size=lmda_path_size,
        max_screen_size=max_screen_size,
        max_active_size=max_active_size,
        loss_null=loss_null,
    )

    dispatcher = {
        np.float64: core.state.StateGlmNaive64,
        np.float32: core.state.StateGlmNaive32,
    }
    core_base = dispatcher[dtype]

    if isinstance(X, np.ndarray):
        X = matrix.dense(X, method="naive", n_threads=n_threads)

    class _glm_naive(base, core_base):
        def __init__(self):
            self._core_type = core_base
            ## save inputs due to lifetime issues
            # static inputs require a reference to input
            # or copy if it must be made
            self._glm = glm
            self._X = X
            self._constraints = render_constraints(groups.shape[0], constraints, dtype)
            self._groups = np.array(groups, copy=True, dtype=int)
            self._group_sizes = np.array(group_sizes, copy=True, dtype=int)
            self._dual_groups = render_dual_groups(self._constraints)
            self._penalty = np.array(penalty, copy=True, dtype=dtype)
            self._offsets = np.array(offsets, copy=True, dtype=dtype)
            self._lmda_path = np.array(lmda_path, copy=False, dtype=dtype)
            self._screen_set = np.array(screen_set, copy=False, dtype=int)
            self._screen_beta = np.array(screen_beta, copy=False, dtype=dtype)
            self._screen_is_active = np.array(screen_is_active, copy=False, dtype=bool)
            self._screen_dual = np.array(screen_dual, copy=False, dtype=dtype)
            self._active_set = np.array(active_set, copy=False, dtype=int)
            self._grad = np.array(grad, copy=False, dtype=dtype)
            self._eta = np.array(eta, copy=False, dtype=dtype)
            self._resid = np.array(resid, copy=False, dtype=dtype)

            # MUST call constructor directly and not use super()!
            # https://pybind11.readthedocs.io/en/stable/advanced/classes.html#forced-trampoline-class-initialisation
            core_base.__init__(
                self,
                X=self._X,
                constraints=self._constraints,
                groups=self._groups,
                group_sizes=self._group_sizes,
                dual_groups=self._dual_groups,
                alpha=alpha,
                penalty=self._penalty,
                offsets=self._offsets,
                lmda_path=self._lmda_path,
                loss_null=loss_null,
                loss_full=loss_full,
                lmda_max=lmda_max,
                min_ratio=min_ratio,
                lmda_path_size=lmda_path_size,
                max_screen_size=max_screen_size,
                max_active_size=max_active_size,
                pivot_subset_ratio=pivot_subset_ratio,
                pivot_subset_min=pivot_subset_min,
                pivot_slack_ratio=pivot_slack_ratio,
                screen_rule=screen_rule,
                irls_max_iters=irls_max_iters,
                irls_tol=irls_tol,
                max_iters=max_iters,
                tol=tol,
                adev_tol=adev_tol,
                ddev_tol=ddev_tol,
                newton_tol=newton_tol,
                newton_max_iters=newton_max_iters,
                early_exit=early_exit,
                setup_loss_null=setup_loss_null,
                setup_lmda_max=setup_lmda_max,
                setup_lmda_path=setup_lmda_path,
                intercept=intercept,
                n_threads=n_threads,
                screen_set=self._screen_set,
                screen_beta=self._screen_beta,
                screen_is_active=self._screen_is_active,
                screen_dual=self._screen_dual,
                active_set_size=active_set_size,
                active_set=self._active_set,
                beta0=beta0,
                lmda=lmda,
                grad=self._grad,
                eta=self._eta,
                resid=self._resid,
            )

        @classmethod
        def create_from_core(cls, state, core_state):
            obj = base.create_from_core(
                cls, state, core_state, _glm_naive, core_base,
            )
            return obj

    return _glm_naive()


def multiglm_naive(
    *,
    X: Union[MatrixNaiveBase32, MatrixNaiveBase64],
    glm: Union[GlmMultiBase32, GlmMultiBase64],
    constraints: list[Union[ConstraintBase32, ConstraintBase64]],
    groups: np.ndarray,
    group_sizes: np.ndarray,
    alpha: float,
    penalty: np.ndarray,
    offsets: np.ndarray,
    screen_set: np.ndarray,
    screen_beta: np.ndarray,
    screen_is_active: np.ndarray,
    screen_dual: np.ndarray,
    active_set_size: int,
    active_set: np.ndarray,
    lmda: float,
    grad: np.ndarray,
    eta: np.ndarray,
    resid: np.ndarray,
    loss_full: float,
    loss_null: float =None,
    lmda_path: np.ndarray =None,
    lmda_max: float =None,
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
):
    """Creates a multi-response GLM, naive method state object.

    Define the following quantities: 

        - :math:`\\tilde{X} = X\\otimes I_K` if ``intercept`` is ``False``, 
          and otherwise :math:`[1 \\otimes I_K, X \\otimes I_K]`.
        - :math:`\\tilde{y}` as the flattened version of :math:`y` as row-major.
        - :math:`\\tilde{W} = K^{-1} (W \\otimes I_K)`. 
        - :math:`\\tilde{\\eta}` as the flattened version of :math:`\\eta` as row-major
          and similarly for :math:`\\tilde{\\eta}^0`.

    Parameters
    ----------
    X : (n, p) Union[MatrixNaiveBase32, MatrixNaiveBase64]
        Feature matrix.
        It is typically one of the matrices defined in :mod:`adelie.matrix` submodule.
    glm : Union[GlmMultiBase32, GlmMultiBase64]
        Multi-response GLM object.
        It is typically one of the GLM classes defined in :mod:`adelie.glm` submodule.
    constraints : (G,) list[Union[ConstraintBase32, ConstraintBase64]], optional
        List of constraints for each group.
        ``constraints[i]`` is the constraint object corresponding to group ``i``.
        If ``constraints[i]`` is ``None``, then the ``i`` th group is unconstrained.
        If ``None``, every group is unconstrained.
    groups : (G,) ndarray
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
    group_sizes : (G,) ndarray
        List of group sizes corresponding to each element of ``groups``.
        ``group_sizes[i]`` is the size of the ``i`` th group.
    alpha : float
        Elastic net parameter.
        It must be in the range :math:`[0,1]`.
    penalty : (G,) ndarray
        Penalty factor for each group in the same order as ``groups``.
        It must be a non-negative vector.
    offsets : (n, K) ndarray
        Observation offsets :math:`\\eta^0`.
    screen_set : (s,) ndarray
        List of indices into ``groups`` that correspond to the screen groups.
        ``screen_set[i]`` is ``i`` th screen group.
        ``screen_set`` must contain at least the true (optimal) active groups
        when the regularization is given by ``lmda``.
    screen_beta : (ws,) ndarray
        Coefficient vector on the screen set.
        ``screen_beta[b:b+p]`` is the coefficient for the ``i`` th screen group 
        where
        ``k = screen_set[i]``,
        ``b = screen_begins[i]``,
        and ``p = group_sizes[k]``.
        The values can be arbitrary but it is recommended to be close to the solution at ``lmda``.
    screen_is_active : (s,) ndarray
        Boolean vector that indicates whether each screen group in ``groups`` is active or not.
        ``screen_is_active[i]`` is ``True`` if and only if ``screen_set[i]`` is active.
    screen_dual : (ds,) ndarray
        Dual vector on the screen set.
        ``screen_dual[b:b+p]`` is the dual for the ``i`` th screen group 
        where
        ``k = screen_set[i]``,
        ``b = screen_dual_begins[i]``,
        and ``p = constraints[k].dual_size``.
    active_set_size : int
        Number of active groups.
        ``active_set[i]`` is only well-defined
        for ``i`` in the range ``[0, active_set_size)``.
    active_set : (G,) ndarray
        List of indices into ``screen_set`` that correspond to active groups.
        ``screen_set[active_set[i]]`` is the ``i`` th active group.
        An active group is one with non-zero coefficient block,
        that is, for every ``i`` th active group, 
        ``screen_beta[b:b+p] == 0`` where 
        ``j = active_set[i]``,
        ``k = screen_set[j]``,
        ``b = screen_begins[j]``,
        and ``p = group_sizes[k]``.
    lmda : float
        The last regularization parameter that was attempted to be solved.
    grad : ((p+intercept)*K,) ndarray
        The full gradient :math:`-\\tilde{X}^\\top \\nabla \\ell(\\tilde{\\eta})` where
        :math:`\\tilde{\\eta}` is given by ``eta``.
    eta : (n*K,) ndarray
        The natural parameter :math:`\\tilde{\\eta} = \\tilde{X}\\beta + \\tilde{\\eta}^0`
        where 
        :math:`\\beta`,
        and :math:`\\tilde{\\eta}^0` are given by
        ``screen_beta`` and ``offsets``.
    resid : (n*K,) ndarray
        Residual :math:`-\\nabla \\ell(\\tilde{\\eta})`
        where :math:`\\tilde{\\eta}` is given by ``eta``.
    loss_full : float
        Full loss :math:`\\ell(\\eta^\\star)`
        where :math:`\\eta^\\star` is the minimizer.
    loss_null : float, optional
        Null loss :math:`\\ell(\\mathbf{1} \\beta_0^{\\star\\top} + \\eta^0)`
        from fitting an intercept-only model (if ``intercept`` is ``True``)
        where an intercept is given for each class
        and otherwise :math:`\\ell(\\eta^0)`.
        If ``None``, it will be computed.
        Default is ``None``. 
    lmda_path : (L,) ndarray, optional
        The regularization path to solve for.
        The full path is not considered if ``early_exit`` is ``True``.
        It is recommended that the path is sorted in decreasing order.
        If ``None``, the path will be generated.
        Default is ``None``.
    lmda_max : float, optional
        The smallest :math:`\\lambda` such that the true solution is zero
        for all coefficients that have a non-vanishing group lasso penalty (:math:`\\ell_2`-norm).
        If ``None``, it will be computed.
        Default is ``None``.
    irls_max_iters : int, optional
        Maximum number of IRLS iterations.
        Default is ``int(1e4)``.
    irls_tol : float, optional
        IRLS convergence tolerance.
        Default is ``1e-7``.
    max_iters : int, optional
        Maximum number of coordinate descents.
        Default is ``int(1e5)``.
    tol : float, optional
        Coordinate descent convergence tolerance.
        Default is ``1e-7``.
    adev_tol : float, optional
        Percent deviance explained tolerance.
        If the training percent deviance explained exceeds this quantity
        and ``early_exit`` is ``True``, then the solver terminates.
        Default is ``0.9``.
    ddev_tol : float, optional
        Difference in percent deviance explained tolerance.
        If the difference of the last two training percent deviance explained exceeds this quantity
        and ``early_exit`` is ``True``, then the solver terminates.
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
        ``True`` if the function should early exit based on training percent deviance explained.
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

    Returns
    -------
    wrap
        Wrapper state object.

    See Also
    --------
    adelie.adelie_core.state.StateMultiGlmNaive32
    adelie.adelie_core.state.StateMultiGlmNaive64
    """
    (
        max_screen_size,
        max_active_size,
        lmda_path_size,
        setup_lmda_max,
        setup_lmda_path,
        lmda_max,
        lmda_path,
        dtype,
        setup_loss_null,
        loss_null,
    ) = _render_glm_naive_inputs(
        X=X,
        groups=groups,
        lmda_max=lmda_max,
        lmda_path=lmda_path,
        lmda_path_size=lmda_path_size,
        max_screen_size=max_screen_size,
        max_active_size=max_active_size,
        loss_null=loss_null,
    )

    dispatcher = {
        np.float64: core.state.StateMultiGlmNaive64,
        np.float32: core.state.StateMultiGlmNaive32,
    }
    core_base = dispatcher[dtype]

    X_raw = X
    n_classes = glm.y.shape[-1]
    (
        X,
        offsets,
        group_type,
    ) = _render_multi_inputs(
        X=X,
        groups=groups,
        offsets=offsets,
        intercept=intercept,
        n_threads=n_threads,
        dtype=dtype,
    )

    class _multiglm_naive(base, core_base):
        def __init__(self):
            self._core_type = core_base
            ## save inputs due to lifetime issues
            # static inputs require a reference to input
            # or copy if it must be made
            self._glm = glm
            self._X = X_raw
            self._X_expanded = X
            self._constraints = render_constraints(groups.shape[0], constraints, dtype)
            self._groups = np.array(groups, copy=True, dtype=int)
            self._group_sizes = np.array(group_sizes, copy=True, dtype=int)
            self._dual_groups = render_dual_groups(self._constraints)
            self._penalty = np.array(penalty, copy=True, dtype=dtype)
            self._offsets = np.array(offsets, copy=True, dtype=dtype)
            self._lmda_path = np.array(lmda_path, copy=False, dtype=dtype)
            self._screen_set = np.array(screen_set, copy=False, dtype=int)
            self._screen_beta = np.array(screen_beta, copy=False, dtype=dtype)
            self._screen_is_active = np.array(screen_is_active, copy=False, dtype=bool)
            self._screen_dual = np.array(screen_dual, copy=False, dtype=dtype)
            self._active_set = np.array(active_set, copy=False, dtype=int)
            self._grad = np.array(grad, copy=False, dtype=dtype)
            self._eta = np.array(eta, copy=False, dtype=dtype)
            self._resid = np.array(resid, copy=False, dtype=dtype)

            # MUST call constructor directly and not use super()!
            # https://pybind11.readthedocs.io/en/stable/advanced/classes.html#forced-trampoline-class-initialisation
            core_base.__init__(
                self,
                group_type=group_type,
                n_classes=n_classes,
                multi_intercept=intercept,
                X=self._X_expanded,
                constraints=self._constraints,
                groups=self._groups,
                group_sizes=self._group_sizes,
                dual_groups=self._dual_groups,
                alpha=alpha,
                penalty=self._penalty,
                offsets=self._offsets.ravel(),
                lmda_path=self._lmda_path,
                loss_null=loss_null,
                loss_full=loss_full,
                lmda_max=lmda_max,
                min_ratio=min_ratio,
                lmda_path_size=lmda_path_size,
                max_screen_size=max_screen_size,
                max_active_size=max_active_size,
                pivot_subset_ratio=pivot_subset_ratio,
                pivot_subset_min=pivot_subset_min,
                pivot_slack_ratio=pivot_slack_ratio,
                screen_rule=screen_rule,
                irls_max_iters=irls_max_iters,
                irls_tol=irls_tol,
                max_iters=max_iters,
                tol=tol,
                adev_tol=adev_tol,
                ddev_tol=ddev_tol,
                newton_tol=newton_tol,
                newton_max_iters=newton_max_iters,
                early_exit=early_exit,
                setup_loss_null=setup_loss_null,
                setup_lmda_max=setup_lmda_max,
                setup_lmda_path=setup_lmda_path,
                intercept=False,
                n_threads=n_threads,
                screen_set=self._screen_set,
                screen_beta=self._screen_beta,
                screen_is_active=self._screen_is_active,
                screen_dual=self._screen_dual,
                active_set_size=active_set_size,
                active_set=self._active_set,
                beta0=0,
                lmda=lmda,
                grad=self._grad,
                eta=self._eta,
                resid=self._resid,
            )

        @classmethod
        def create_from_core(cls, state, core_state):
            obj = base.create_from_core(
                cls, state, core_state, _multiglm_naive, core_base,
            )
            return obj

    return _multiglm_naive()