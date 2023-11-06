from . import adelie_core as core
from . import matrix
from . import logger
import numpy as np
import scipy
import os


def deduce_states(
    *,
    group_sizes: np.ndarray,
    screen_set: np.ndarray,
):
    """Deduce state variables.

    Parameters
    ----------
    group_sizes : (G,) np.ndarray
        See ``adelie.adelie_core.state.StatePinBase64``.
    screen_set : (s,) np.ndarray
        See ``adelie.adelie_core.state.StatePinBase64``.

    Returns
    -------
    screen_g1 : (s1,) np.ndarray
        See ``adelie.adelie_core.state.StatePinBase64``.
    screen_g2 : (s2,) np.ndarray
        See ``adelie.adelie_core.state.StatePinBase64``.
    screen_begins : (s,) np.ndarray
        See ``adelie.adelie_core.state.StatePinBase64``.

    See Also
    --------
    adelie.adelie_core.state.StatePinBase64
    """
    S = screen_set.shape[0]
    screen_g1 = np.arange(S)[group_sizes[screen_set] == 1]
    screen_g2 = np.arange(S)[group_sizes[screen_set] > 1]
    screen_begins = np.cumsum(
        np.concatenate([[0], group_sizes[screen_set]]),
        dtype=int,
    )[:-1]
    return (
        screen_g1,
        screen_g2,
        screen_begins,
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

    @staticmethod
    def create_from_core(
        cls, 
        state, 
        core_state, 
        pycls, 
        corecls,
    ):
        """Create a new instance of a pin naive state using given state and new core state.

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


class pin_base(base):
    def __init__(self):
        self.solver = "pin"

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

        # ================ screen_g1 check ====================
        self._check(
            np.all((0 <= self.screen_g1) & (self.screen_g1 < S)),
            "check screen_g1 is in [0, S)",
            method, logger,
        )
        self._check(
            len(self.screen_g1) == len(np.unique(self.screen_g1)),
            "check screen_g1 has unique values",
            method, logger,
        )
        self._check(
            np.all(self.group_sizes[self.screen_set[self.screen_g1]] == 1),
            "check screen_g1 has group sizes of 1",
            method, logger,
        )
        self._check(
            self.screen_g1.dtype == np.dtype("int"),
            "check screen_g1 dtype is int",
            method, logger,
        )

        # ================ screen_g2 check ====================
        self._check(
            np.all((0 <= self.screen_g2) & (self.screen_g2 < S)),
            "check screen_g2 is in [0, S)",
            method, logger,
        )
        self._check(
            len(self.screen_g2) == len(np.unique(self.screen_g2)),
            "check screen_g2 has unique values",
            method, logger,
        )
        self._check(
            np.all(self.group_sizes[self.screen_set[self.screen_g2]] > 1),
            "check screen_g2 has group sizes more than 1",
            method, logger,
        )
        self._check(
            self.screen_g2.dtype == np.dtype("int"),
            "check screen_g2 dtype is int",
            method, logger,
        )
        self._check(
            len(self.screen_g1) + len(self.screen_g2) == S,
            "check screen_g1 and screen_g2 combined have length S",
            method, logger,
        )

        # ================ screen_begins check ====================
        expected = np.cumsum(
            np.concatenate([[0], self.group_sizes[self.screen_set]], dtype=int)
        )
        WS = expected[-1]
        expected = expected[:-1]
        self._check(
            np.all(self.screen_begins == expected),
            "check screen_begins is [0, g1, g2, ...] where gi is the group size of (i-1)th strong group.",
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

        # ================ screen_beta check ====================
        self._check(
            len(self.screen_beta) == WS,
            "check screen_beta size",
            method, logger,
        )

        # ================ screen_is_active check ====================
        self._check(
            np.all(np.arange(S)[self.screen_is_active] == np.sort(self.active_set)),
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
            np.all(self.group_sizes[self.screen_set[self.active_g1]] == 1),
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
            np.all(self.group_sizes[self.screen_set[self.active_g2]] > 1),
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
            np.concatenate([[0], self.group_sizes[self.screen_set[self.active_set]]], dtype=int)
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
        actual = self.groups[self.screen_set[self.active_set[self.active_order]]]
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

        # ================ rsqs check ====================
        self._check(
            self.lmdas.shape == (self.betas.shape[0],),
            "check lmdas shape",
            method, logger,
        )

        # ================ screen_is_actives check ====================
        self._check(
            len(self.screen_is_actives) == self.betas.shape[0],
            "check screen_is_actives shape",
            method, logger,
        )

        # ================ screen_betas check ====================
        self._check(
            len(self.screen_betas) == self.betas.shape[0],
            "check screen_betas shape",
            method, logger,
        )


class pin_naive_base(pin_base):
    """State wrapper base class for all pin, naive method."""
    def default_init(
        self, 
        base_type: core.state.StatePinNaive64 | core.state.StatePinNaive32,
        *,
        X: matrix.base | matrix.MatrixNaiveBase64 | matrix.MatrixNaiveBase32,
        y_mean: float,
        y_var: float,
        groups: np.ndarray,
        group_sizes: np.ndarray,
        alpha: float,
        penalty: np.ndarray,
        screen_set: np.ndarray,
        lmda_path: np.ndarray,
        rsq: float,
        resid: np.ndarray,
        screen_beta: np.ndarray,
        screen_is_active: np.ndarray,
        intercept: bool,
        max_iters: int,
        tol: float,
        rsq_tol: float,
        rsq_slope_tol: float,
        rsq_curv_tol: float,
        newton_tol: float,
        newton_max_iters: int,
        n_threads: int,
        dtype: np.float32 | np.float64,
    ):
        """Default initialization method.
        """
        self.method = "naive"

        ## save inputs due to lifetime issues
        # static inputs require a reference to input
        # or copy if it must be made
        self._X = X

        if isinstance(X, matrix.base):
            X = X.internal()

        self._groups = np.array(groups, copy=False, dtype=int)
        self._group_sizes = np.array(group_sizes, copy=False, dtype=int)
        self._penalty = np.array(penalty, copy=False, dtype=dtype)
        self._screen_set = np.array(screen_set, copy=False, dtype=int)
        self._lmda_path = np.array(lmda_path, copy=False, dtype=dtype)

        # dynamic inputs require a copy to not modify user's inputs
        self._resid = np.copy(resid).astype(dtype)
        self._screen_beta = np.copy(screen_beta).astype(dtype)
        self._screen_is_active = np.copy(screen_is_active).astype(bool)

        (
            self._screen_g1,
            self._screen_g2,
            self._screen_begins,
        ) = deduce_states(
            group_sizes=group_sizes,
            screen_set=screen_set,
        )

        self._screen_vars = []
        self._screen_X_means = []
        self._screen_transforms = []
        for i in self._screen_set:
            g, gs = groups[i], group_sizes[i]
            Xi = np.empty((X.rows(), gs), dtype=dtype, order="F")
            X.to_dense(g, gs, Xi)
            Xi_means = np.mean(Xi, axis=0)
            if intercept:
                Xi -= Xi_means[None]
            _, d, vh = np.linalg.svd(Xi, full_matrices=True, compute_uv=True)
            vars = np.zeros(gs)
            vars[:len(d)] = d ** 2
            self._screen_vars.append(vars)
            self._screen_X_means.append(Xi_means)
            self._screen_transforms.append(np.array(vh.T, copy=False, dtype=dtype, order="F"))
        self._screen_vars = np.concatenate(self._screen_vars, dtype=dtype)
        self._screen_X_means = np.concatenate(self._screen_X_means, dtype=dtype)
        vecmat_type = (
            core.VectorMatrix64
            if dtype == np.float64 else
            core.VectorMatrix32
        )
        self._screen_transforms = vecmat_type(self._screen_transforms)

        resid_sum = np.sum(self._resid)

        # MUST call constructor directly and not use super()!
        # https://pybind11.readthedocs.io/en/stable/advanced/classes.html#forced-trampoline-class-initialisation
        base_type.__init__(
            self,
            X=X,
            y_mean=y_mean,
            y_var=y_var,
            groups=self._groups,
            group_sizes=self._group_sizes,
            alpha=alpha,
            penalty=self._penalty,
            screen_set=self._screen_set,
            screen_g1=self._screen_g1,
            screen_g2=self._screen_g2,
            screen_begins=self._screen_begins,
            screen_vars=self._screen_vars,
            screen_X_means=self._screen_X_means,
            screen_transforms=self._screen_transforms,
            lmda_path=self._lmda_path,
            intercept=intercept,
            max_iters=max_iters,
            tol=tol,
            rsq_tol=rsq_tol,
            rsq_slope_tol=rsq_slope_tol,
            rsq_curv_tol=rsq_curv_tol,
            newton_tol=newton_tol,
            newton_max_iters=newton_max_iters,
            n_threads=n_threads,
            rsq=rsq,
            resid=self._resid,
            resid_sum=resid_sum,
            screen_beta=self._screen_beta,
            screen_is_active=self._screen_is_active,
        )

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
        self._check(
            self.resids.shape == (self.betas.shape[0], self.X.rows()),
            "check resids shape",
            method, logger,
        )


class pin_naive_64(pin_naive_base, core.state.StatePinNaive64):
    """State class for pin, naive method using 64-bit floating point."""

    def __init__(self, *args, **kwargs):
        pin_naive_base.default_init(
            self,
            core.state.StatePinNaive64,
            *args,
            dtype=np.float64,
            **kwargs,
        )

    @classmethod
    def create_from_core(cls, state, core_state):
        obj = base.create_from_core(
            cls, state, core_state, pin_naive_64, core.state.StatePinNaive64,
        )
        pin_naive_base.__init__(obj)
        return obj


class pin_naive_32(pin_naive_base, core.state.StatePinNaive32):
    """State class for pin, naive method using 32-bit floating point."""

    def __init__(self, *args, **kwargs):
        pin_naive_base.default_init(
            self,
            core.state.StatePinNaive32,
            *args,
            dtype=np.float32,
            **kwargs,
        )

    @classmethod
    def create_from_core(cls, state, core_state):
        obj = base.create_from_core(
            cls, state, core_state, pin_naive_32, core.state.StatePinNaive32,
        )
        pin_naive_base.__init__(obj)
        return obj


def pin_naive(
    *,
    X: matrix.base | matrix.MatrixNaiveBase64 | matrix.MatrixNaiveBase32,
    y_mean: float,
    y_var: float,
    groups: np.ndarray,
    group_sizes: np.ndarray,
    alpha: float,
    penalty: np.ndarray,
    screen_set: np.ndarray,
    lmda_path: np.ndarray,
    rsq: float,
    resid: np.ndarray,
    screen_beta: np.ndarray,
    screen_is_active: np.ndarray,
    intercept: bool =True,
    max_iters: int =int(1e5),
    tol: float =1e-12,
    rsq_tol: float =0.9,
    rsq_slope_tol: float =1e-3,
    rsq_curv_tol: float =1e-3,
    newton_tol: float =1e-12,
    newton_max_iters: int =1000,
    n_threads: int =1,
):
    """Creates a pin, naive method state object.

    Parameters
    ----------
    X : Union[adelie.matrix.base, adelie.matrix.MatrixNaiveBase64, adelie.matrix.MatrixNaiveBase32]
        Feature matrix.
        It is typically one of the matrices defined in ``adelie.matrix`` sub-module.
    y_mean : float
        Mean of :math:`y`.
    y_var : float
        :math:`\\ell_2` norm squared of :math:`y_c`.
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
    screen_set : (s,) np.ndarray
        List of indices into ``groups`` that correspond to the strong groups.
        ``screen_set[i]`` is ``i`` th strong group.
    lmda_path : (l,) np.ndarray
        Regularization sequence to fit on.
    rsq : float
        Unnormalized :math:`R^2` value at ``screen_beta``.
        The unnormalized :math:`R^2` is given by :math:`\\|y_c\\|_2^2 - \\|y_c-X_c\\beta\\|_2^2`.
    resid : np.ndarray
        Residual :math:`y_c-X\\beta` at ``screen_beta``.
    screen_beta : (ws,) np.ndarray
        Coefficient vector on the strong set.
        ``screen_beta[b:b+p]`` is the coefficient for the ``i`` th strong group 
        where
        ``k = screen_set[i]``,
        ``b = screen_begins[i]``,
        and ``p = group_sizes[k]``.
    screen_is_active : (a,) np.ndarray
        Boolean vector that indicates whether each strong group in ``groups`` is active or not.
        ``screen_is_active[i]`` is ``True`` if and only if ``screen_set[i]`` is active.
    intercept : bool, optional
        ``True`` to fit with intercept.
        Default is ``True``.
    max_iters : int, optional
        Maximum number of coordinate descents.
        Default is ``int(1e5)``.
    tol : float, optional
        Convergence tolerance.
        Default is ``1e-12``.
    rsq_tol : float, optional
        Early stopping rule check on :math:`R^2`.
        Default is ``0.9``.
    rsq_slope_tol : float, optional
        Early stopping rule check on slope of :math:`R^2`.
        Default is ``1e-3``.
    rsq_curv_tol : float, optional
        Early stopping rule check on curvature of :math:`R^2`.
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

    See Also
    --------
    adelie.state.pin_naive_64
    adelie.state.pin_naive_32
    """
    if isinstance(X, matrix.base):
        X_intr = X.internal()
    else:
        X_intr = X

    if not (
        isinstance(X_intr, matrix.MatrixNaiveBase64) or
        isinstance(X_intr, matrix.MatrixNaiveBase32)
    ):
        raise ValueError(
            "X must be an instance of matrix.MatrixNaiveBase32 or matrix.MatrixNaiveBase64."
        )

    dtype = (
        np.float64
        if isinstance(X_intr, matrix.MatrixNaiveBase64) else
        np.float32
    )
        
    dispatcher = {
        np.float64: pin_naive_64,
        np.float32: pin_naive_32,
    }
    return dispatcher[dtype](
        X=X,
        y_mean=y_mean,
        y_var=y_var,
        groups=groups,
        group_sizes=group_sizes,
        alpha=alpha,
        penalty=penalty,
        screen_set=screen_set,
        lmda_path=lmda_path,
        rsq=rsq,
        resid=resid,
        screen_beta=screen_beta,
        screen_is_active=screen_is_active,
        intercept=intercept,
        max_iters=max_iters,
        tol=tol,
        rsq_tol=rsq_tol,
        rsq_slope_tol=rsq_slope_tol,
        rsq_curv_tol=rsq_curv_tol,
        newton_tol=newton_tol,
        newton_max_iters=newton_max_iters,
        n_threads=n_threads,
    )


class pin_cov_base(pin_base):
    """State wrapper base class for all pin, covariance method."""
    def default_init(
        self, 
        base_type: core.state.StatePinCov64 | core.state.StatePinCov32,
        *,
        A: matrix.base | matrix.MatrixCovBase64 | matrix.MatrixCovBase32,
        groups: np.ndarray,
        group_sizes: np.ndarray,
        alpha: float,
        penalty: np.ndarray,
        screen_set: np.ndarray,
        lmda_path: np.ndarray,
        rsq: float,
        screen_beta: np.ndarray,
        screen_grad: np.ndarray,
        screen_is_active: np.ndarray,
        max_iters: int,
        tol: float,
        rsq_slope_tol: float,
        rsq_curv_tol: float,
        newton_tol: float,
        newton_max_iters: int,
        n_threads: int,
        dtype: np.float32 | np.float64,
    ):
        """Default initialization method.
        """
        self.method = "cov"

        ## save inputs due to lifetime issues
        # static inputs require a reference to input
        # or copy if it must be made
        self._A = A

        if isinstance(A, matrix.base):
            A = A.internal()

        self._groups = np.array(groups, copy=False, dtype=int)
        self._group_sizes = np.array(group_sizes, copy=False, dtype=int)
        self._penalty = np.array(penalty, copy=False, dtype=dtype)
        self._screen_set = np.array(screen_set, copy=False, dtype=int)
        self._lmda_path = np.array(lmda_path, copy=False, dtype=dtype)

        # dynamic inputs require a copy to not modify user's inputs
        self._screen_beta = np.copy(screen_beta).astype(dtype)
        self._screen_grad = np.copy(screen_grad).astype(dtype)
        self._screen_is_active = np.copy(screen_is_active).astype(bool)

        (
            self._screen_g1,
            self._screen_g2,
            self._screen_begins,
        ) = deduce_states(
            group_sizes=group_sizes,
            screen_set=screen_set,
        )

        self._screen_vars = []
        self._screen_transforms = []
        for i in self._screen_set:
            g, gs = groups[i], group_sizes[i]
            Aii = np.empty((gs, gs), dtype=dtype, order="F")
            A.to_dense(g, g, gs, gs, Aii)  
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

        base_type.__init__(
            self,
            A=A,
            groups=self._groups,
            group_sizes=self._group_sizes,
            alpha=alpha,
            penalty=self._penalty,
            screen_set=self._screen_set,
            screen_g1=self._screen_g1,
            screen_g2=self._screen_g2,
            screen_begins=self._screen_begins,
            screen_vars=self._screen_vars,
            screen_transforms=self._screen_transforms,
            lmda_path=self._lmda_path,
            max_iters=max_iters,
            tol=tol,
            rsq_slope_tol=rsq_slope_tol,
            rsq_curv_tol=rsq_curv_tol,
            newton_tol=newton_tol,
            newton_max_iters=newton_max_iters,
            n_threads=n_threads,
            rsq=rsq,
            screen_beta=self._screen_beta,
            screen_grad=self._screen_grad,
            screen_is_active=self._screen_is_active,
        )

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


class pin_cov_64(pin_cov_base, core.state.StatePinCov64):
    """State class for pin, covariance method using 64-bit floating point."""

    def __init__(self, *args, **kwargs):
        pin_cov_base.default_init(
            self,
            core.state.StatePinCov64,
            *args,
            dtype=np.float64,
            **kwargs,
        )

    @classmethod
    def create_from_core(cls, state, core_state):
        obj = base.create_from_core(
            cls, state, core_state, pin_cov_64, core.state.StatePinCov64,
        )
        pin_cov_base.__init__(obj)
        return obj


class pin_cov_32(pin_cov_base, core.state.StatePinCov32):
    """State class for pin, cov method using 32-bit floating point."""

    def __init__(self, *args, **kwargs):
        pin_cov_base.default_init(
            self,
            core.state.StatePinCov32,
            *args,
            dtype=np.float32,
            **kwargs,
        )

    @classmethod
    def create_from_core(cls, state, core_state):
        obj = base.create_from_core(
            cls, state, core_state, pin_cov_32, core.state.StatePinCov32,
        )
        pin_cov_base.__init__(obj)
        return obj


def pin_cov(
    *,
    A: matrix.base | matrix.MatrixCovBase64 | matrix.MatrixCovBase32,
    groups: np.ndarray,
    group_sizes: np.ndarray,
    alpha: float,
    penalty: np.ndarray,
    screen_set: np.ndarray,
    lmda_path: np.ndarray,
    rsq: float,
    screen_beta: np.ndarray,
    screen_grad: np.ndarray,
    screen_is_active: np.ndarray,
    max_iters: int =int(1e5),
    tol: float =1e-12,
    rsq_slope_tol: float =1e-3,
    rsq_curv_tol: float =1e-3,
    newton_tol: float =1e-12,
    newton_max_iters: int =1000,
    n_threads: int =1,
):
    """Creates a pin, covariance method state object.

    Parameters
    ----------
    A : Union[adelie.matrix.base, adelie.matrix.MatrixCovBase64, adelie.matrix.MatrixCovBase32]
        Covariance matrix :math:`X_c^\\top X_c` where `X_c` is column-centered to fit with intercept.
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
    screen_set : (s,) np.ndarray
        List of indices into ``groups`` that correspond to the strong groups.
        ``screen_set[i]`` is ``i`` th strong group.
    lmda_path : (l,) np.ndarray
        Regularization sequence to fit on.
    rsq : float
        Unnormalized :math:`R^2` value at ``screen_beta``.
        The unnormalized :math:`R^2` is given by :math:`\\|y_c\\|_2^2 - \\|y_c-X_c\\beta\\|_2^2`.
    resid : np.ndarray
        Residual :math:`y_c-X\\beta` at ``screen_beta``.
    screen_beta : (ws,) np.ndarray
        Coefficient vector on the strong set.
        ``screen_beta[b:b+p]`` is the coefficient for the ``i`` th strong group 
        where
        ``k = screen_set[i]``,
        ``b = screen_begins[i]``,
        and ``p = group_sizes[k]``.
    screen_grad : (ws,) np.ndarray
        Gradient :math:`X_{c,k}^\\top (y_c-X_c\\beta)` on the strong groups :math:`k` where :math:`beta` is given by ``screen_beta``.
        ``screen_grad[b:b+p]`` is the gradient for the ``i`` th strong group
        where 
        ``k = screen_set[i]``,
        ``b = screen_begins[i]``,
        and ``p = group_sizes[k]``.
    screen_is_active : (a,) np.ndarray
        Boolean vector that indicates whether each strong group in ``groups`` is active or not.
        ``screen_is_active[i]`` is ``True`` if and only if ``screen_set[i]`` is active.
    max_iters : int, optional
        Maximum number of coordinate descents.
        Default is ``int(1e5)``.
    tol : float, optional
        Convergence tolerance.
        Default is ``1e-12``.
    rsq_slope_tol : float, optional
        Early stopping rule check on slope of :math:`R^2`.
        Default is ``1e-3``.
    rsq_curv_tol : float, optional
        Early stopping rule check on curvature of :math:`R^2`.
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

    See Also
    --------
    adelie.state.pin_cov_64
    adelie.state.pin_cov_32
    """
    if isinstance(A, matrix.base):
        A_intr = A.internal()
    else:
        A_intr = A

    if not (
        isinstance(A_intr, matrix.MatrixCovBase64) or 
        isinstance(A_intr, matrix.MatrixCovBase32)
    ):
        raise ValueError(
            "X must be an instance of matrix.MatrixCovBase32 or matrix.MatrixCovBase64."
        )

    dtype = (
        np.float64
        if isinstance(A_intr, matrix.MatrixCovBase64) else
        np.float32
    )
        
    dispatcher = {
        np.float64: pin_cov_64,
        np.float32: pin_cov_32,
    }
    return dispatcher[dtype](
        A=A,
        groups=groups,
        group_sizes=group_sizes,
        alpha=alpha,
        penalty=penalty,
        screen_set=screen_set,
        lmda_path=lmda_path,
        rsq=rsq,
        screen_beta=screen_beta,
        screen_grad=screen_grad,
        screen_is_active=screen_is_active,
        max_iters=max_iters,
        tol=tol,
        rsq_slope_tol=rsq_slope_tol,
        rsq_curv_tol=rsq_curv_tol,
        newton_tol=newton_tol,
        newton_max_iters=newton_max_iters,
        n_threads=n_threads,
    )


class basil_base(base):
    pass


class basil_naive_base(basil_base):
    """State wrapper base class for all basil, naive method."""
    def default_init(
        self, 
        base_type: core.state.StateBasilNaive64 | core.state.StateBasilNaive32,
        *,
        X: matrix.base | matrix.MatrixNaiveBase64 | matrix.MatrixNaiveBase32,
        X_means: np.ndarray,
        X_group_norms: np.ndarray,
        y_mean: float,
        y_var: float,
        resid: np.ndarray,
        groups: np.ndarray,
        group_sizes: np.ndarray,
        alpha: float,
        penalty: np.ndarray,
        lmda_path: np.ndarray,
        lmda_max: float,
        min_ratio: float,
        lmda_path_size: int,
        max_screen_size: int,
        pivot_subset_ratio: float,
        pivot_subset_min: int,
        pivot_slack_ratio: float,
        screen_rule: str,
        max_iters: int,
        tol: float,
        rsq_tol: float,
        rsq_slope_tol: float,
        rsq_curv_tol: float,
        newton_tol: float,
        newton_max_iters: int,
        early_exit: bool,
        setup_lmda_max: bool,
        setup_lmda_path: bool,
        intercept: bool,
        n_threads: int,
        screen_set: np.ndarray,
        screen_beta: np.ndarray,
        screen_is_active: np.ndarray,
        rsq: float,
        lmda: float,
        grad: np.ndarray,
        dtype: np.float32 | np.float64,
    ):
        """Default initialization method.
        """
        ## save inputs due to lifetime issues
        # static inputs require a reference to input
        # or copy if it must be made
        self._X = X

        if isinstance(X, matrix.base):
            X = X.internal()

        self._X_means = np.array(X_means, copy=False, dtype=dtype)
        self._X_group_norms = np.array(X_group_norms, copy=False, dtype=dtype)
        self._groups = np.array(groups, copy=False, dtype=int)
        self._group_sizes = np.array(group_sizes, copy=False, dtype=int)
        self._penalty = np.array(penalty, copy=False, dtype=dtype)
        self._lmda_path = np.array(lmda_path, copy=False, dtype=dtype)
        screen_set = np.array(screen_set, copy=False, dtype=int)
        screen_beta = np.array(screen_beta, copy=False, dtype=dtype)
        screen_is_active = np.array(screen_is_active, copy=False, dtype=bool)
        grad = np.array(grad, copy=False, dtype=dtype)

        # MUST call constructor directly and not use super()!
        # https://pybind11.readthedocs.io/en/stable/advanced/classes.html#forced-trampoline-class-initialisation
        # TODO:
        base_type.__init__(
            self,
            X=X,
            X_means=self._X_means,
            X_group_norms=self._X_group_norms,
            y_mean=y_mean,
            y_var=y_var,
            resid=resid,
            groups=self._groups,
            group_sizes=self._group_sizes,
            alpha=alpha,
            penalty=self._penalty,
            lmda_path=self._lmda_path,
            lmda_max=lmda_max,
            min_ratio=min_ratio,
            lmda_path_size=lmda_path_size,
            max_screen_size=max_screen_size,
            pivot_subset_ratio=pivot_subset_ratio,
            pivot_subset_min=pivot_subset_min,
            pivot_slack_ratio=pivot_slack_ratio,
            screen_rule=screen_rule,
            max_iters=max_iters,
            tol=tol,
            rsq_tol=rsq_tol,
            rsq_slope_tol=rsq_slope_tol,
            rsq_curv_tol=rsq_curv_tol,
            newton_tol=newton_tol,
            newton_max_iters=newton_max_iters,
            early_exit=early_exit,
            setup_lmda_max=setup_lmda_max,
            setup_lmda_path=setup_lmda_path,
            intercept=intercept,
            n_threads=n_threads,
            screen_set=screen_set,
            screen_beta=screen_beta,
            screen_is_active=screen_is_active,
            rsq=rsq,
            lmda=lmda,
            grad=grad,
        )

    def check(
        self,
        y, 
        method: str =None, 
        logger=logger.logger,
    ):
        n, p = self.X.rows(), self.X.cols()

        yc = y
        if self.intercept:
            yc = yc - np.mean(yc)

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

        # ================ configurations check ====================
        self._check(
            0 < self.min_ratio,
            "check min_ratio is > 0",
            method, logger,
        )
        self._check(
            ~self.setup_lmda_path | (self.lmda_path_size > 0),
            "check either lmda_path is not setup or if it is, then path size is > 0",
            method, logger,
        )
        self._check(
            self.max_screen_size >= 0,
            "check max_screen_size >= 0",
            method, logger,
        )
        self._check(
            self.max_iters >= 0,
            "check max_iters >= 0",
            method, logger,
        )
        self._check(
            self.tol >= 0,
            "check tol >= 0",
            method, logger,
        )
        self._check(
            self.rsq_slope_tol >= 0,
            "check rsq_slope_tol >= 0",
            method, logger,
        )
        self._check(
            self.rsq_curv_tol >= 0,
            "check rsq_curv_tol >= 0",
            method, logger,
        )
        self._check(
            self.newton_tol >= 0,
            "check newton_tol >= 0",
            method, logger,
        )
        self._check(
            self.newton_max_iters >= 0,
            "check newton_max_iters >= 0",
            method, logger,
        )
        self._check(
            self.n_threads > 0,
            "check n_threads > 0",
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

        # ================ screen_g1 check ====================
        self._check(
            np.all((0 <= self.screen_g1) & (self.screen_g1 < S)),
            "check screen_g1 is in [0, S)",
            method, logger,
        )
        self._check(
            len(self.screen_g1) == len(np.unique(self.screen_g1)),
            "check screen_g1 has unique values",
            method, logger,
        )
        self._check(
            np.all(self.group_sizes[self.screen_set[self.screen_g1]] == 1),
            "check screen_g1 has group sizes of 1",
            method, logger,
        )
        self._check(
            self.screen_g1.dtype == np.dtype("int"),
            "check screen_g1 dtype is int",
            method, logger,
        )

        # ================ screen_g2 check ====================
        self._check(
            np.all((0 <= self.screen_g2) & (self.screen_g2 < S)),
            "check screen_g2 is in [0, S)",
            method, logger,
        )
        self._check(
            len(self.screen_g2) == len(np.unique(self.screen_g2)),
            "check screen_g2 has unique values",
            method, logger,
        )
        self._check(
            np.all(self.group_sizes[self.screen_set[self.screen_g2]] > 1),
            "check screen_g2 has group sizes more than 1",
            method, logger,
        )
        self._check(
            self.screen_g2.dtype == np.dtype("int"),
            "check screen_g2 dtype is int",
            method, logger,
        )
        self._check(
            len(self.screen_g1) + len(self.screen_g2) == S,
            "check screen_g1 and screen_g2 combined have length S",
            method, logger,
        )

        # ================ screen_begins check ====================
        expected = np.cumsum(
            np.concatenate([[0], self.group_sizes[self.screen_set]], dtype=int)
        )
        WS = expected[-1]
        expected = expected[:-1]
        self._check(
            np.all(self.screen_begins == expected),
            "check screen_begins is [0, g1, g2, ...] where gi is the group size of (i-1)th strong group.",
            method, logger,
        )
        self._check(
            self.screen_begins.dtype == np.dtype("int"),
            "check screen_begins dtype is int",
            method, logger,
        )

        # ================ screen_order check ====================
        self._check(
            np.all(self.screen_set[self.screen_order] == np.sort(self.screen_set)),
            "check screen_order is correctly sorting screen_set",
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
        nzn_idxs = np.array([
            i 
            for i, sb, gs in zip(
                np.arange(len(self.screen_set)),
                self.screen_begins, 
                self.group_sizes[self.screen_set],
            )
            if np.any(self.screen_beta[sb:sb+gs] != 0)
        ], dtype=int)
        self._check(
            np.all(self.screen_is_active[nzn_idxs]),
            "check screen_is_active is only active on non-zeros of screen_beta",
            method, logger,
        )

        # ================ rsq check ====================
        screen_indices = []
        tmp = np.empty(n)
        Xbeta = np.zeros(n)
        for g, gs, b in zip(
            self.groups[self.screen_set], 
            self.group_sizes[self.screen_set],
            self.screen_begins,
        ):
            screen_indices.append(np.arange(g, g + gs))
            self.X.btmul(g, gs, self.screen_beta[b:b+gs], tmp)
            Xbeta += tmp

        if len(screen_indices) == 0:
            screen_indices = np.array(screen_indices, dtype=int)
        else:
            screen_indices = np.concatenate(screen_indices, dtype=int)

        resid = yc - Xbeta
        grad = np.empty(p)
        self.X.mul(resid, grad)
        if self.intercept:
            grad -= self.X_means * np.sum(resid)
        Xcbeta = Xbeta - (self.screen_X_means @ self.screen_beta)
        expected = 2 * np.sum(yc * Xcbeta) - np.linalg.norm(Xcbeta) ** 2
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
        abs_grad = np.array([
            np.linalg.norm(grad[g:g+gs])
            for g, gs in zip(self.groups, self.group_sizes)
        ])
        self._check(
            np.allclose(self.abs_grad, abs_grad),
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
            np.allclose(self.resid_sum, np.sum(resid)),
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
        for ss_idx in range(len(self.screen_set)):
            i = self.screen_set[ss_idx]
            g, gs = self.groups[i], self.group_sizes[i]
            Xi = np.empty((n, gs), order="F")
            self.X.to_dense(g, gs, Xi)
            if self.intercept:
                Xi -= self.X_means[g:g+gs][None]
            V = self.screen_transforms[ss_idx]
            XiV = Xi @ V
            Dsq = XiV.T @ XiV
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

    def update_path(self, path):
        return basil_naive(
            X=self.X,
            X_means=self.X_means,
            X_group_norms=self.X_group_norms,
            y_mean=self.y_mean,
            y_var=self.y_var,
            resid=self.resid,
            groups=self.groups,
            group_sizes=self.group_sizes,
            alpha=self.alpha,
            penalty=self.penalty,
            screen_set=self.screen_set,
            screen_beta=self.screen_beta,
            screen_is_active=self.screen_is_active,
            rsq=self.rsq,
            lmda=self.lmda,
            grad=self.grad,
            lmda_path=path,
            lmda_max=None if self.lmda_max == -1 else self.lmda_max,
            max_iters=self.max_iters,
            tol=self.tol,
            rsq_tol=self.rsq_tol,
            rsq_slope_tol=self.rsq_slope_tol,
            rsq_curv_tol=self.rsq_curv_tol,
            newton_tol=self.newton_tol,
            newton_max_iters=self.newton_max_iters,
            n_threads=self.n_threads,
            early_exit=self.early_exit,
            intercept=self.intercept,
            screen_rule=self.screen_rule,
            min_ratio=self.min_ratio,
            lmda_path_size=self.lmda_path_size,
            max_screen_size=self.max_screen_size,
            pivot_subset_ratio=self.pivot_subset_ratio,
            pivot_subset_min=self.pivot_subset_min,
            pivot_slack_ratio=self.pivot_slack_ratio,
        )


class basil_naive_64(basil_naive_base, core.state.StateBasilNaive64):
    """State class for basil, naive method using 64-bit floating point."""

    def __init__(self, *args, **kwargs):
        basil_naive_base.default_init(
            self,
            core.state.StateBasilNaive64,
            *args,
            dtype=np.float64,
            **kwargs,
        )

    @classmethod
    def create_from_core(cls, state, core_state):
        obj = base.create_from_core(
            cls, state, core_state, basil_naive_64, core.state.StateBasilNaive64,
        )
        basil_naive_base.__init__(obj)
        return obj


class basil_naive_32(basil_naive_base, core.state.StateBasilNaive32):
    """State class for basil, naive method using 32-bit floating point."""

    def __init__(self, *args, **kwargs):
        basil_naive_base.default_init(
            self,
            core.state.StateBasilNaive32,
            *args,
            dtype=np.float32,
            **kwargs,
        )

    @classmethod
    def create_from_core(cls, state, core_state):
        obj = base.create_from_core(
            cls, state, core_state, basil_naive_32, core.state.StateBasilNaive32,
        )
        basil_naive_base.__init__(obj)
        return obj


def basil_naive(
    *,
    X: matrix.base | matrix.MatrixNaiveBase64 | matrix.MatrixNaiveBase32,
    X_means: np.ndarray,
    X_group_norms: np.ndarray,
    y_mean: float,
    y_var: float,
    resid: np.ndarray,
    groups: np.ndarray,
    group_sizes: np.ndarray,
    alpha: float,
    penalty: np.ndarray,
    screen_set: np.ndarray,
    screen_beta: np.ndarray,
    screen_is_active: np.ndarray,
    rsq: float,
    lmda: float,
    grad: np.ndarray,
    lmda_path: np.ndarray =None,
    lmda_max: float =None,
    max_iters: int =int(1e5),
    tol: float =1e-12,
    rsq_tol: float =0.9,
    rsq_slope_tol: float =1e-3,
    rsq_curv_tol: float =1e-3,
    newton_tol: float =1e-12,
    newton_max_iters: int =1000,
    n_threads: int =1,
    early_exit: bool =True,
    intercept: bool =True,
    screen_rule: str ="pivot",
    min_ratio: float =1e-2,
    lmda_path_size: int =100,
    max_screen_size: int =None,
    pivot_subset_ratio: float =0.1,
    pivot_subset_min: int =1,
    pivot_slack_ratio: float =1.25,
):
    """Creates a basil, naive method state object.

    Parameters
    ----------
    X : Union[adelie.matrix.base, adelie.matrix.MatrixNaiveBase64, adelie.matrix.MatrixNaiveBase32]
        Feature matrix.
        It is typically one of the matrices defined in ``adelie.matrix`` sub-module.
    X_means : (p,) np.ndarray
        Column means of ``X``.
    X_group_norms : (G,) np.ndarray
        Group Frobenius norm of ``X``.
        ``X_group_norms[i]`` is :math:`\\|X_{c, g}\\|_F`
        where :math:`g` corresponds to the group index ``i``.
    y_mean : float
        The mean of the response vector :math:`y`.
    y_var : float
        The variance of the response vector :math:`y`, i.e. 
        :math:`\\|y_c\\|_2^2`.
    resid : (n,) np.ndarray
        Residual :math:`y_c - X \\beta` where :math:`\\beta` is given by ``screen_beta``.
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
    screen_set : (s,) np.ndarray
        List of indices into ``groups`` that correspond to the strong groups.
        ``screen_set[i]`` is ``i`` th strong group.
        ``screen_set`` must contain at least the true (optimal) active groups
        when the regularization is given by ``lmda``.
    screen_beta : (ws,) np.ndarray
        Coefficient vector on the strong set.
        ``screen_beta[b:b+p]`` is the coefficient for the ``i`` th strong group 
        where
        ``k = screen_set[i]``,
        ``b = screen_begins[i]``,
        and ``p = group_sizes[k]``.
        This must contain the true solution values for the strong groups.
    screen_is_active : (a,) np.ndarray
        Boolean vector that indicates whether each strong group in ``groups`` is active or not.
        ``screen_is_active[i]`` is ``True`` if and only if ``screen_set[i]`` is active.
    rsq : float
        The true unnormalized :math:`R^2` given by :math:`\\|y_c\\|_2^2 - \\|y_c-X_c\\beta\\|_2^2`
        where :math:`\\beta` is given by ``screen_beta``.
    lmda: float,
        The regularization parameter at which the true solution is given by ``screen_beta``
        (in the transformed space).
    grad: np.ndarray,
        The true full gradient :math:`X_c^\\top (y_c - X_c\\beta)` in the original space where
        :math:`\\beta` is given by ``screen_beta``.
    lmda_path : (l,) np.ndarray, optional
        The regularization path to solve for.
        The full path is not considered if ``early_exit`` is ``True``.
        It is recommended that the path is sorted in decreasing order.
        If ``None``, the path will be generated.
        Default is ``None``.
    lmda_max : float
        The smallest :math:`\\lambda` such that the true solution is zero
        for all coefficients that have a non-vanishing group lasso penalty (:math:`\\ell_2`-norm).
        If ``None``, it will be computed.
        Default is ``None``.
    max_iters : int, optional
        Maximum number of coordinate descents.
        Default is ``int(1e5)``.
    tol : float, optional
        Convergence tolerance.
        Default is ``1e-12``.
    rsq_tol : float, optional
        Early stopping rule check on :math:`R^2`.
        Default is ``0.9``.
    rsq_slope_tol : float, optional
        Early stopping rule check on slope of :math:`R^2`.
        Default is ``1e-3``.
    rsq_curv_tol : float, optional
        Early stopping rule check on curvature of :math:`R^2`.
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
        ``True`` if the function should early exit based on training :math:`R^2`.
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
        The type of strong rule to use. It must be one of the following options:

            - ``"strong"``: discards variables from the safe set based on simple strong rule.
            - ``"fixed_greedy"``: adds variables based on a fixed number of groups with the largest gradient norm.
            - ``"safe"``: adds all safe variables to the strong set.
            - ``"pivot"``: adds all variables whose gradient norms are largest, which is determined
                by searching for a pivot point in the gradient norms.

        Default is ``"pivot"``.
    max_screen_size: int, optional
        Maximum number of strong groups allowed.
        The function will return a valid state and guaranteed to have strong set size
        less than or equal to ``max_screen_size``.
        If ``None``, it will be set to the total number of groups.
        Default is ``None``.
    pivot_subset_ratio : float, optional
        If screening takes place, then the ``(1 + pivot_subset_ratio) * s``
        largest gradient norms are used to determine the pivot point
        where ``s`` is the current strong set size.
        It is only used if ``screen_rule == "pivot"``.
        Default is ``0.1``.
    pivot_subset_min : int, optional
        If screening takes place, then at least ``pivot_subset_min``
        number of gradient norms are used to determine the pivot point.
        It is only used if ``screen_rule == "pivot"``.
        Default is ``1``.
    pivot_slack_ratio : float, optional
        If screening takes place, then ``pivot_slack_ratio``
        number of groups with next smallest (new) active scores 
        below the pivot point are also added to the strong set as slack.
        It is only used if ``screen_rule == "pivot"``.
        Default is ``1.25``.

    See Also
    --------
    adelie.state.basil_naive_64
    adelie.state.basil_naive_32
    """
    if isinstance(X, matrix.base):
        X_intr = X.internal()
    else:
        X_intr = X

    if not (
        isinstance(X_intr, matrix.MatrixNaiveBase64) or 
        isinstance(X_intr, matrix.MatrixNaiveBase32)
    ):
        raise ValueError(
            "X must be an instance of matrix.MatrixNaiveBase32 or matrix.MatrixNaiveBase64."
        )

    if max_screen_size is None:
        max_screen_size = len(groups)

    if max_iters < 0:
        raise ValueError("max_iters must be >= 0.")
    if tol <= 0:
        raise ValueError("tol must be > 0.")
    if rsq_tol < 0 or rsq_tol > 1:
        raise ValueError("rsq_tol must be in [0,1].")
    if rsq_slope_tol < 0:
        raise ValueError("rsq_slope_tol must be >= 0.")
    if rsq_curv_tol < 0:
        raise ValueError("rsq_curv_tol must be >= 0.")
    if newton_tol < 0:
        raise ValueError("newton_tol must be >= 0.")
    if newton_max_iters < 0:
        raise ValueError("newton_max_iters must be >= 0.")
    if n_threads < 1:
        raise ValueError("n_threads must be >= 1.")
    if min_ratio <= 0:
        raise ValueError("min_ratio must be > 0.")
    if lmda_path_size < 0:
        raise ValueError("lmda_path_size must be >= 0.")
    if max_screen_size < 0:
        raise ValueError("max_screen_size must be >= 0.")
    if pivot_subset_ratio <= 0 or pivot_subset_ratio > 1:
        raise ValueError("pivot_subset_ratio must be in (0, 1].")
    if pivot_subset_min < 1:
        raise ValueError("pivot_subset_min must be >= 1.")
    if pivot_slack_ratio < 0:
        raise ValueError("pivot_slack_ratio must be >= 0.")

    actual_lmda_path_size = (
        lmda_path_size
        if lmda_path is None else
        len(lmda_path)
    )

    max_screen_size = np.minimum(max_screen_size, len(groups))

    dtype = (
        np.float64
        if isinstance(X_intr, matrix.MatrixNaiveBase64) else
        np.float32
    )
        
    dispatcher = {
        np.float64: basil_naive_64,
        np.float32: basil_naive_32,
    }

    setup_lmda_max = lmda_max is None
    setup_lmda_path = lmda_path is None

    if setup_lmda_max: lmda_max = -1
    if setup_lmda_path: lmda_path = np.empty(0, dtype=dtype)

    return dispatcher[dtype](
        X=X,
        X_means=X_means,
        X_group_norms=X_group_norms,
        y_mean=y_mean,
        y_var=y_var,
        resid=resid,
        groups=groups,
        group_sizes=group_sizes,
        alpha=alpha,
        penalty=penalty,
        lmda_path=lmda_path,
        lmda_max=lmda_max,
        setup_lmda_max=setup_lmda_max,
        setup_lmda_path=setup_lmda_path,
        max_screen_size=max_screen_size,
        pivot_subset_ratio=pivot_subset_ratio,
        pivot_subset_min=pivot_subset_min,
        pivot_slack_ratio=pivot_slack_ratio,
        screen_rule=screen_rule,
        max_iters=max_iters,
        tol=tol,
        rsq_tol=rsq_tol,
        rsq_slope_tol=rsq_slope_tol,
        rsq_curv_tol=rsq_curv_tol,
        newton_tol=newton_tol,
        newton_max_iters=newton_max_iters,
        early_exit=early_exit,
        min_ratio=min_ratio,
        lmda_path_size=actual_lmda_path_size,
        intercept=intercept,
        n_threads=n_threads,
        screen_set=screen_set,
        screen_beta=screen_beta,
        screen_is_active=screen_is_active,
        rsq=rsq,
        lmda=lmda,
        grad=grad,
    )
