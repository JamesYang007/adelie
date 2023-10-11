from . import adelie_core as core
from . import matrix
from . import logger
import numpy as np
import scipy
import os


def deduce_states(
    *,
    group_sizes: np.ndarray,
    strong_set: np.ndarray,
):
    """Deduce state variables.

    Parameters
    ----------
    group_sizes : (G,) np.ndarray
        See ``adelie.adelie_core.state.StatePinBase64``.
    strong_set : (s,) np.ndarray
        See ``adelie.adelie_core.state.StatePinBase64``.

    Returns
    -------
    strong_g1 : (s1,) np.ndarray
        See ``adelie.adelie_core.state.StatePinBase64``.
    strong_g2 : (s2,) np.ndarray
        See ``adelie.adelie_core.state.StatePinBase64``.
    strong_begins : (s,) np.ndarray
        See ``adelie.adelie_core.state.StatePinBase64``.

    See Also
    --------
    adelie.adelie_core.state.StatePinBase64
    """
    S = strong_set.shape[0]
    strong_g1 = np.arange(S)[group_sizes[strong_set] == 1]
    strong_g2 = np.arange(S)[group_sizes[strong_set] > 1]
    strong_begins = np.cumsum(
        np.concatenate([[0], group_sizes[strong_set]]),
        dtype=int,
    )[:-1]
    return (
        strong_g1,
        strong_g2,
        strong_begins,
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

        # ================ strong_vars check ====================
        self._check(
            len(self.strong_vars) == WS,
            "check strong_vars size",
            method, logger,
        )
        self._check(
            np.all(0 <= self.strong_vars),
            "check strong_vars is non-negative",
            method, logger,
        )

        # ================ strong_transforms check ====================
        self._check(
            len(self.strong_transforms) == S,
            "check strong_transforms size",
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

        # ================ strong_beta check ====================
        self._check(
            len(self.strong_beta) == WS,
            "check strong_beta size",
            method, logger,
        )

        # ================ strong_is_active check ====================
        self._check(
            np.all(np.arange(S)[self.strong_is_active] == np.sort(self.active_set)),
            "check strong_is_active is consistent with active_set",
            method, logger,
        )
        self._check(
            self.strong_is_active.dtype == np.dtype("bool"),
            "check strong_is_active dtype is bool",
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

        # ================ strong_is_actives check ====================
        self._check(
            len(self.strong_is_actives) == self.betas.shape[0],
            "check strong_is_actives shape",
            method, logger,
        )

        # ================ strong_betas check ====================
        self._check(
            len(self.strong_betas) == self.betas.shape[0],
            "check strong_betas shape",
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
        groups: np.ndarray,
        group_sizes: np.ndarray,
        alpha: float,
        penalty: np.ndarray,
        strong_set: np.ndarray,
        lmda_path: np.ndarray,
        rsq: float,
        resid: np.ndarray,
        strong_beta: np.ndarray,
        strong_is_active: np.ndarray,
        intercept: bool,
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
        ## save inputs due to lifetime issues
        # static inputs require a reference to input
        # or copy if it must be made
        self._X = X

        if isinstance(X, matrix.base):
            X = X.internal()

        self._groups = np.array(groups, copy=False, dtype=int)
        self._group_sizes = np.array(group_sizes, copy=False, dtype=int)
        self._penalty = np.array(penalty, copy=False, dtype=dtype)
        self._strong_set = np.array(strong_set, copy=False, dtype=int)
        self._lmda_path = np.array(lmda_path, copy=False, dtype=dtype)

        # dynamic inputs require a copy to not modify user's inputs
        self._resid = np.copy(resid).astype(dtype)
        self._strong_beta = np.copy(strong_beta).astype(dtype)
        self._strong_is_active = np.copy(strong_is_active).astype(bool)

        (
            self._strong_g1,
            self._strong_g2,
            self._strong_begins,
        ) = deduce_states(
            group_sizes=group_sizes,
            strong_set=strong_set,
        )

        self._strong_vars = []
        self._strong_X_means = []
        self._strong_transforms = []
        for i in self._strong_set:
            g, gs = groups[i], group_sizes[i]
            Xi = np.empty((X.rows(), gs), dtype=dtype, order="F")
            X.to_dense(g, gs, Xi)
            _, d, vh = np.linalg.svd(Xi, full_matrices=True, compute_uv=True)
            vars = np.zeros(gs)
            vars[:len(d)] = d ** 2
            self._strong_vars.append(vars)
            self._strong_X_means.append(np.mean(Xi, axis=0))
            self._strong_transforms.append(np.array(vh.T, copy=False, dtype=dtype, order="F"))
        self._strong_vars = np.concatenate(self._strong_vars, dtype=dtype)
        self._strong_X_means = np.concatenate(self._strong_X_means, dtype=dtype)
        vecmat_type = (
            core.VectorMatrix64
            if dtype == np.float64 else
            core.VectorMatrix32
        )
        self._strong_transforms = vecmat_type(self._strong_transforms)

        resid_sum = np.sum(self._resid)

        # MUST call constructor directly and not use super()!
        # https://pybind11.readthedocs.io/en/stable/advanced/classes.html#forced-trampoline-class-initialisation
        base_type.__init__(
            self,
            X=X,
            y_mean=y_mean,
            groups=self._groups,
            group_sizes=self._group_sizes,
            alpha=alpha,
            penalty=self._penalty,
            strong_set=self._strong_set,
            strong_g1=self._strong_g1,
            strong_g2=self._strong_g2,
            strong_begins=self._strong_begins,
            strong_vars=self._strong_vars,
            strong_X_means=self._strong_X_means,
            strong_transforms=self._strong_transforms,
            lmda_path=self._lmda_path,
            intercept=intercept,
            max_iters=max_iters,
            tol=tol,
            rsq_slope_tol=rsq_slope_tol,
            rsq_curv_tol=rsq_curv_tol,
            newton_tol=newton_tol,
            newton_max_iters=newton_max_iters,
            n_threads=n_threads,
            rsq=rsq,
            resid=self._resid,
            resid_sum=resid_sum,
            strong_beta=self._strong_beta,
            strong_is_active=self._strong_is_active,
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
    groups: np.ndarray,
    group_sizes: np.ndarray,
    alpha: float,
    penalty: np.ndarray,
    strong_set: np.ndarray,
    lmda_path: np.ndarray,
    rsq: float,
    resid: np.ndarray,
    strong_beta: np.ndarray,
    strong_is_active: np.ndarray,
    intercept: bool =True,
    max_iters: int =int(1e5),
    tol: float =1e-12,
    rsq_slope_tol: float =1e-2,
    rsq_curv_tol: float =1e-2,
    newton_tol: float =1e-12,
    newton_max_iters: int =1000,
    n_threads: int =os.cpu_count(),
):
    """Creates a pin, naive method state object.

    Parameters
    ----------
    X : Union[adelie.matrix.base, adelie.matrix.MatrixNaiveBase64, adelie.matrix.MatrixNaiveBase32]
        Feature matrix.
        It is typically one of the matrices defined in ``adelie.matrix`` sub-module.
    y_mean : float
        Mean of :math:`y`.
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
    lmda_path : (l,) np.ndarray
        Regularization sequence to fit on.
    rsq : float
        Unnormalized :math:`R^2` value at ``strong_beta``.
        The unnormalized :math:`R^2` is given by :math:`\\|y_c\\|_2^2 - \\|y_c-X_c\\beta\\|_2^2`.
    resid : np.ndarray
        Residual :math:`y_c-X\\beta` at ``strong_beta``.
    strong_beta : (ws,) np.ndarray
        Coefficient vector on the strong set.
        ``strong_beta[b:b+p]`` is the coefficient for the ``i`` th strong group 
        where
        ``k = strong_set[i]``,
        ``b = strong_begins[i]``,
        and ``p = group_sizes[k]``.
    strong_is_active : (a,) np.ndarray
        Boolean vector that indicates whether each strong group in ``groups`` is active or not.
        ``strong_is_active[i]`` is ``True`` if and only if ``strong_set[i]`` is active.
    intercept : bool, optional
        ``True`` to fit with intercept.
        Default is ``True``.
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
        groups=groups,
        group_sizes=group_sizes,
        alpha=alpha,
        penalty=penalty,
        strong_set=strong_set,
        lmda_path=lmda_path,
        rsq=rsq,
        resid=resid,
        strong_beta=strong_beta,
        strong_is_active=strong_is_active,
        intercept=intercept,
        max_iters=max_iters,
        tol=tol,
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
        strong_set: np.ndarray,
        lmda_path: np.ndarray,
        rsq: float,
        strong_beta: np.ndarray,
        strong_grad: np.ndarray,
        strong_is_active: np.ndarray,
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
        ## save inputs due to lifetime issues
        # static inputs require a reference to input
        # or copy if it must be made
        self._A = A

        if isinstance(A, matrix.base):
            A = A.internal()

        self._groups = np.array(groups, copy=False, dtype=int)
        self._group_sizes = np.array(group_sizes, copy=False, dtype=int)
        self._penalty = np.array(penalty, copy=False, dtype=dtype)
        self._strong_set = np.array(strong_set, copy=False, dtype=int)
        self._lmda_path = np.array(lmda_path, copy=False, dtype=dtype)

        # dynamic inputs require a copy to not modify user's inputs
        self._strong_beta = np.copy(strong_beta).astype(dtype)
        self._strong_grad = np.copy(strong_grad).astype(dtype)
        self._strong_is_active = np.copy(strong_is_active).astype(bool)

        (
            self._strong_g1,
            self._strong_g2,
            self._strong_begins,
        ) = deduce_states(
            group_sizes=group_sizes,
            strong_set=strong_set,
        )

        self._strong_vars = []
        self._strong_transforms = []
        for i in self._strong_set:
            g, gs = groups[i], group_sizes[i]
            Aii = np.empty((gs, gs), dtype=dtype, order="F")
            A.to_dense(g, g, gs, gs, Aii)  
            dsq, v = np.linalg.eigh(Aii)
            self._strong_vars.append(dsq)
            self._strong_transforms.append(v)
        self._strong_vars = np.concatenate(self._strong_vars, dtype=dtype)
        vecmat_type = (
            core.VectorMatrix64
            if dtype == np.float64 else
            core.VectorMatrix32
        )
        self._strong_transforms = vecmat_type(self._strong_transforms)

        base_type.__init__(
            self,
            A=A,
            groups=self._groups,
            group_sizes=self._group_sizes,
            alpha=alpha,
            penalty=self._penalty,
            strong_set=self._strong_set,
            strong_g1=self._strong_g1,
            strong_g2=self._strong_g2,
            strong_begins=self._strong_begins,
            strong_vars=self._strong_vars,
            strong_transforms=self._strong_transforms,
            lmda_path=self._lmda_path,
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
            strong_is_active=self._strong_is_active,
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
    strong_set: np.ndarray,
    lmda_path: np.ndarray,
    rsq: float,
    strong_beta: np.ndarray,
    strong_grad: np.ndarray,
    strong_is_active: np.ndarray,
    max_iters: int =int(1e5),
    tol: float =1e-12,
    rsq_slope_tol: float =1e-2,
    rsq_curv_tol: float =1e-2,
    newton_tol: float =1e-12,
    newton_max_iters: int =1000,
    n_threads: int =os.cpu_count(),
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
    strong_set : (s,) np.ndarray
        List of indices into ``groups`` that correspond to the strong groups.
        ``strong_set[i]`` is ``i`` th strong group.
    lmda_path : (l,) np.ndarray
        Regularization sequence to fit on.
    rsq : float
        Unnormalized :math:`R^2` value at ``strong_beta``.
        The unnormalized :math:`R^2` is given by :math:`\\|y_c\\|_2^2 - \\|y_c-X_c\\beta\\|_2^2`.
    resid : np.ndarray
        Residual :math:`y_c-X\\beta` at ``strong_beta``.
    strong_beta : (ws,) np.ndarray
        Coefficient vector on the strong set.
        ``strong_beta[b:b+p]`` is the coefficient for the ``i`` th strong group 
        where
        ``k = strong_set[i]``,
        ``b = strong_begins[i]``,
        and ``p = group_sizes[k]``.
    strong_grad : (ws,) np.ndarray
        Gradient :math:`X_{c,k}^\\top (y_c-X_c\\beta)` on the strong groups :math:`k` where :math:`beta` is given by ``strong_beta``.
        ``strong_grad[b:b+p]`` is the gradient for the ``i`` th strong group
        where 
        ``k = strong_set[i]``,
        ``b = strong_begins[i]``,
        and ``p = group_sizes[k]``.
    strong_is_active : (a,) np.ndarray
        Boolean vector that indicates whether each strong group in ``groups`` is active or not.
        ``strong_is_active[i]`` is ``True`` if and only if ``strong_set[i]`` is active.
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
        strong_set=strong_set,
        lmda_path=lmda_path,
        rsq=rsq,
        strong_beta=strong_beta,
        strong_grad=strong_grad,
        strong_is_active=strong_is_active,
        max_iters=max_iters,
        tol=tol,
        rsq_slope_tol=rsq_slope_tol,
        rsq_curv_tol=rsq_curv_tol,
        newton_tol=newton_tol,
        newton_max_iters=newton_max_iters,
        n_threads=n_threads,
    )


class basil_base(base):
    def check(
        self, 
        method: str =None, 
        logger=logger.logger,
    ):
        # TODO: fill
        return


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
        setup_edpp: bool,
        resid: np.ndarray,
        edpp_safe_set: np.ndarray,
        edpp_v1_0: np.ndarray,
        edpp_resid_0: np.ndarray,
        groups: np.ndarray,
        group_sizes: np.ndarray,
        alpha: float,
        penalty: np.ndarray,
        lmda_path: np.ndarray,
        lmda_max: float,
        delta_lmda_path_size: int,
        delta_strong_size: int,
        max_strong_size: int,
        strong_rule: bool,
        max_iters: int,
        tol: float,
        rsq_slope_tol: float,
        rsq_curv_tol: float,
        newton_tol: float,
        newton_max_iters: int,
        early_exit: bool,
        intercept: bool,
        n_threads: int,
        strong_set: np.ndarray,
        strong_beta: np.ndarray,
        strong_is_active: np.ndarray,
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
        edpp_safe_set = np.array(edpp_safe_set, copy=False, dtype=int)
        edpp_v1_0 = np.array(edpp_v1_0, copy=False, dtype=dtype)
        edpp_resid_0 = np.array(edpp_resid_0, copy=False, dtype=dtype)
        strong_set = np.array(strong_set, copy=False, dtype=int)
        strong_beta = np.array(strong_beta, copy=False, dtype=dtype)
        strong_is_active = np.array(strong_is_active, copy=False, dtype=bool)
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
            setup_edpp=setup_edpp,
            resid=resid,
            edpp_safe_set=edpp_safe_set,
            edpp_v1_0=edpp_v1_0,
            edpp_resid_0=edpp_resid_0,
            groups=self._groups,
            group_sizes=self._group_sizes,
            alpha=alpha,
            penalty=self._penalty,
            lmda_path=self._lmda_path,
            lmda_max=lmda_max,
            delta_lmda_path_size=delta_lmda_path_size,
            delta_strong_size=delta_strong_size,
            max_strong_size=max_strong_size,
            strong_rule=strong_rule,
            max_iters=max_iters,
            tol=tol,
            rsq_slope_tol=rsq_slope_tol,
            rsq_curv_tol=rsq_curv_tol,
            newton_tol=newton_tol,
            newton_max_iters=newton_max_iters,
            early_exit=early_exit,
            intercept=intercept,
            n_threads=n_threads,
            strong_set=strong_set,
            strong_beta=strong_beta,
            strong_is_active=strong_is_active,
            rsq=rsq,
            lmda=lmda,
            grad=grad,
        )

    def check(
        self, 
        method: str =None, 
        logger=logger.logger,
    ):
        # TODO: fill
        return


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
    setup_edpp: bool,
    resid: np.ndarray,
    edpp_safe_set: np.ndarray,
    edpp_v1_0: np.ndarray,
    edpp_resid_0: np.ndarray,
    groups: np.ndarray,
    group_sizes: np.ndarray,
    alpha: float,
    penalty: np.ndarray,
    lmda_path: np.ndarray,
    lmda_max: float,
    strong_set: np.ndarray,
    strong_beta: np.ndarray,
    strong_is_active: np.ndarray,
    rsq: float,
    lmda: float,
    grad: np.ndarray,
    max_iters: int =int(1e5),
    tol: float =1e-12,
    rsq_slope_tol: float =1e-2,
    rsq_curv_tol: float =1e-2,
    newton_tol: float =1e-12,
    newton_max_iters: int =1000,
    n_threads: int =os.cpu_count(),
    early_exit: bool =True,
    intercept: bool =True,
    strong_rule: bool =True,
    delta_lmda_path_size: int =5,
    delta_strong_size: int =5,
    max_strong_size: int =1000,
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
        ``X_group_norms[i]`` is :math:`\\|X[:, g:g+gs]\\|_F`` 
        where ``g = groups[i]`` and ``gs = group_sizes[i]``.
    y_mean : float
        The mean of the response vector :math:`y`.
    y_var : float
        The variance of the response vector :math:`y`, i.e. :math:`\\|y - \overline{y} 1\\|_2^2`.
    setup_edpp : bool
        ``True`` if EDPP setup is required,
        in which case, the solver will always solve at :math:`\lambda_\max`.
        See ``edpp_v1_0`` and ``edpp_resid_0``.
    resid : (n,) np.ndarray
        Residual :math:`y_c - X_c \\beta` where :math:`\\beta` is given by ``strong_beta``
        *inverse-transformed*.
    edpp_safe_set : (E,) np.ndarray
        A list of EDPP safe groups.
    edpp_v1_0: (n,) np.ndarray
        The :math:`v_1` vector in EDPP rule at :math:`\\lambda_\\max`.
    edpp_resid_0: (n,) np.ndarray
        The residual at :math:`\\lambda_\\max`.
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
    lmda_path : (l,) np.ndarray
        The regularization path to solve for.
        The full path is not considered if ``early_exit`` is ``True``.
        It is recommended that the path is sorted in decreasing order.
    lmda_max : float
        The smallest :math:`\\lambda` such that the true solution is zero
        for all coefficients that have a non-vanishing group lasso penalty (:math:`\\ell_2`-norm).
    strong_set : (s,) np.ndarray
        List of indices into ``groups`` that correspond to the strong groups.
        ``strong_set[i]`` is ``i`` th strong group.
        ``strong_set`` must contain at least the true (optimal) active groups
        when the regularization is given by ``lmda``.
    strong_beta : (ws,) np.ndarray
        Transformed coefficient vector on the strong set.
        Note that the coefficient is in the transformed space of :math:`X_c`
        where :math:`X_c` is column-mean centered version of :math:`X` if ``intercept`` is ``True``
        and :math:`X` otherwise.
        ``strong_beta[b:b+p]`` is the coefficient for the ``i`` th strong group 
        where
        ``k = strong_set[i]``,
        ``b = strong_begins[i]``,
        and ``p = group_sizes[k]``.
        This must contain the true solution values for the strong groups.
    strong_is_active : (a,) np.ndarray
        Boolean vector that indicates whether each strong group in ``groups`` is active or not.
        ``strong_is_active[i]`` is ``True`` if and only if ``strong_set[i]`` is active.
    rsq : float
        The true unnormalized :math:`R^2` given by :math:`\\|y_c\\|_2^2 - \\|y_c-X_c\\beta\\|_2^2`
        where :math:`\\beta` is given by ``strong_beta`` *inverse-transformed*.
    lmda: float,
        The regularization parameter at which the true solution is given by ``strong_beta``
        (in the transformed space).
    grad: np.ndarray,
        The true full gradient :math:`X_c^\\top (y_c - X_c\\beta)` in the original space where
        :math:`\\beta` is given by ``strong_beta`` *inverse-transformed*.
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
    early_exit : bool, optional
        ``True`` if BASIL should early exit based on training :math:`R^2`.
        Default is ``True``.
    intercept : bool, optional 
        ``True`` if BASIL should fit with intercept.
        Default is ``True``.
    strong_rule : bool, optional
        ``True`` if strong rule should be used (only a heuristic!).
        Default is ``True``.
    delta_lmda_path_size : int, optional 
        Number of regularizations to batch per BASIL iteration.
        Default is ``5``.
    delta_strong_size : int, optional
        Number of strong groups to include per BASIL iteration 
        if strong rule does not include new groups but optimality is not reached.
        Default is ``5``.
    max_strong_size: int, optional
        Maximum number of strong groups allowed.
        The function will return a valid state and guaranteed to have strong set size
        less than or equal to ``max_strong_size``.
        Default is ``1000``.

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
        isinstance(X_intr, matrix.MatrixBasilNaiveBase64) or 
        isinstance(X_intr, matrix.MatrixBasilNaiveBase32)
    ):
        raise ValueError(
            "X must be an instance of matrix.MatrixBasilNaiveBase32 or matrix.MatrixBasilNaiveBase64."
        )

    dtype = (
        np.float64
        if isinstance(X_intr, matrix.MatrixBasilNaiveBase64) else
        np.float32
    )
        
    dispatcher = {
        np.float64: basil_naive_64,
        np.float32: basil_naive_32,
    }
    return dispatcher[dtype](
        X=X,
        X_means=X_means,
        X_group_norms=X_group_norms,
        y_mean=y_mean,
        y_var=y_var,
        setup_edpp=setup_edpp,
        resid=resid,
        edpp_safe_set=edpp_safe_set,
        edpp_v1_0=edpp_v1_0,
        edpp_resid_0=edpp_resid_0,
        groups=groups,
        group_sizes=group_sizes,
        alpha=alpha,
        penalty=penalty,
        lmda_path=lmda_path,
        lmda_max=lmda_max,
        delta_lmda_path_size=delta_lmda_path_size,
        delta_strong_size=delta_strong_size,
        max_strong_size=max_strong_size,
        strong_rule=strong_rule,
        max_iters=max_iters,
        tol=tol,
        rsq_slope_tol=rsq_slope_tol,
        rsq_curv_tol=rsq_curv_tol,
        newton_tol=newton_tol,
        newton_max_iters=newton_max_iters,
        early_exit=early_exit,
        intercept=intercept,
        n_threads=n_threads,
        strong_set=strong_set,
        strong_beta=strong_beta,
        strong_is_active=strong_is_active,
        rsq=rsq,
        lmda=lmda,
        grad=grad,
    )
