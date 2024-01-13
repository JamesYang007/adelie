from typing import Union
from . import adelie_core as core
from . import matrix
from . import glm
from . import logger
import numpy as np
import scipy


def deduce_states(
    *,
    group_sizes: np.ndarray,
    screen_set: np.ndarray,
):
    """Deduce state variables.

    Parameters
    ----------
    group_sizes : (G,) np.ndarray
        See ``adelie.adelie_core.state.StateGaussianPinBase64``.
    screen_set : (s,) np.ndarray
        See ``adelie.adelie_core.state.StateGaussianPinBase64``.

    Returns
    -------
    screen_g1 : (s1,) np.ndarray
        See ``adelie.adelie_core.state.StateGaussianPinBase64``.
    screen_g2 : (s2,) np.ndarray
        See ``adelie.adelie_core.state.StateGaussianPinBase64``.
    screen_begins : (s,) np.ndarray
        See ``adelie.adelie_core.state.StateGaussianPinBase64``.

    See Also
    --------
    adelie.adelie_core.state.StateGaussianPinBase64
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

        # ================ lmdas check ====================
        self._check(
            self.lmdas.shape == (self.betas.shape[0],),
            "check lmdas shape",
            method, logger,
        )


class gaussian_pin_naive_base(gaussian_pin_base):
    """State wrapper base class for all gaussian pin naive method."""
    def default_init(
        self, 
        base_type: Union[core.state.StateGaussianPinNaive64, core.state.StateGaussianPinNaive32],
        *,
        X: Union[matrix.MatrixNaiveBase64, matrix.MatrixNaiveBase32],
        y_mean: float,
        y_var: float,
        groups: np.ndarray,
        group_sizes: np.ndarray,
        alpha: float,
        penalty: np.ndarray,
        weights: np.ndarray,
        screen_set: np.ndarray,
        lmda_path: np.ndarray,
        rsq: float,
        resid: np.ndarray,
        screen_beta: np.ndarray,
        screen_is_active: np.ndarray,
        intercept: bool,
        max_iters: int,
        tol: float,
        adev_tol: float,
        ddev_tol: float,
        newton_tol: float,
        newton_max_iters: int,
        n_threads: int,
        dtype: Union[np.float32, np.float64],
    ):
        """Default initialization method.
        """
        ## save inputs due to lifetime issues
        # static inputs require a reference to input
        # or copy if it must be made
        self._X = X

        self._groups = np.array(groups, copy=False, dtype=int)
        self._group_sizes = np.array(group_sizes, copy=False, dtype=int)
        self._penalty = np.array(penalty, copy=False, dtype=dtype)
        self._weights = np.array(weights, copy=False, dtype=dtype)
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

        n, p = X.rows(), X.cols()
        sqrt_weights = np.sqrt(self._weights)
        X_means = np.empty(p, dtype=dtype)
        X.means(self._weights, X_means)

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
            vars, v = np.linalg.eigh(XiTXi)
            self._screen_vars.append(np.maximum(vars, 0))
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
            weights=self._weights,
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


def gaussian_pin_naive(
    *,
    X: Union[matrix.MatrixNaiveBase64, matrix.MatrixNaiveBase32],
    y_mean: float,
    y_var: float,
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
    intercept: bool =True,
    max_iters: int =int(1e5),
    tol: float =1e-7,
    adev_tol: float =0.9,
    ddev_tol: float =1e-4,
    newton_tol: float =1e-12,
    newton_max_iters: int =1000,
    n_threads: int =1,
):
    """Creates a gaussian pin naive method state object.

    Parameters
    ----------
    X : Union[adelie.matrix.MatrixNaiveBase64, adelie.matrix.MatrixNaiveBase32]
        Feature matrix.
        It is typically one of the matrices defined in ``adelie.matrix`` sub-module.
    y_mean : float
        Mean (weighted by :math:`W`) of :math:`y`.
    y_var : float
        :math:`\\ell_2` norm squared (weighted by :math:`W`) of :math:`y_c`, i.e. :math:`\\|y_c\\|_{W}^2`.
    groups : (G,) np.ndarray
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
    alpha : float
        Elastic net parameter.
        It must be in the range :math:`[0,1]`.
    penalty : (G,) np.ndarray
        Penalty factor for each group in the same order as ``groups``.
        It must be a non-negative vector.
    weights : (n,) np.ndarray
        Observation weights.
        Internally, it is normalized to sum to one.
    screen_set : (s,) np.ndarray
        List of indices into ``groups`` that correspond to the strong groups.
        ``screen_set[i]`` is ``i`` th strong group.
    lmda_path : (l,) np.ndarray
        Regularization sequence to fit on.
    rsq : float
        Unnormalized :math:`R^2` value at ``screen_beta``.
        The unnormalized :math:`R^2` is given by :math:`\\|y_c\\|_{W}^2 - \\|y_c-X_c\\beta\\|_{W}^2`.
    resid : (n,) np.ndarray
        Residual :math:`W(y_c-X\\beta)` at ``screen_beta``.
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
        Default is ``1e-7``.
    adev_tol : float, optional
        Percent deviance explained tolerance.
        Default is ``0.9``.
    ddev_tol : float, optional
        Difference in percent deviance explained tolerance.
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
    adelie.adelie_core.state.StateGaussianPinNaive64
    """
    if not (
        isinstance(X, matrix.MatrixNaiveBase64) or
        isinstance(X, matrix.MatrixNaiveBase32)
    ):
        raise ValueError(
            "X must be an instance of matrix.MatrixNaiveBase32 or matrix.MatrixNaiveBase64."
        )

    p = X.cols()
    group_sizes = np.concatenate([groups, [p]], dtype=int)
    group_sizes = group_sizes[1:] - group_sizes[:-1]

    weights = weights / np.sum(weights)

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
        def __init__(self, *args, **kwargs):
            self._core_type = core_base
            gaussian_pin_naive_base.default_init(
                self,
                core_base,
                *args,
                dtype=dtype,
                **kwargs,
            )

        @classmethod
        def create_from_core(cls, state, core_state):
            obj = base.create_from_core(
                cls, state, core_state, _gaussian_pin_naive, core_base,
            )
            obj._core_type = core_base
            gaussian_pin_naive_base.__init__(obj)
            return obj


    return _gaussian_pin_naive(
        X=X,
        y_mean=y_mean,
        y_var=y_var,
        groups=groups,
        group_sizes=group_sizes,
        alpha=alpha,
        penalty=penalty,
        weights=weights,
        screen_set=screen_set,
        lmda_path=lmda_path,
        rsq=rsq,
        resid=resid,
        screen_beta=screen_beta,
        screen_is_active=screen_is_active,
        intercept=intercept,
        max_iters=max_iters,
        tol=tol,
        adev_tol=adev_tol,
        ddev_tol=ddev_tol,
        newton_tol=newton_tol,
        newton_max_iters=newton_max_iters,
        n_threads=n_threads,
    )


class gaussian_pin_cov_base(gaussian_pin_base):
    """State wrapper base class for all pin, covariance method."""
    def default_init(
        self, 
        base_type: Union[core.state.StateGaussianPinCov64, core.state.StateGaussianPinCov32],
        *,
        A: Union[matrix.MatrixCovBase64, matrix.MatrixCovBase32],
        y_var: float,
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
        adev_tol: float,
        ddev_tol: float,
        newton_tol: float,
        newton_max_iters: int,
        n_threads: int,
        dtype: Union[np.float32, np.float64],
    ):
        """Default initialization method.
        """
        ## save inputs due to lifetime issues
        # static inputs require a reference to input
        # or copy if it must be made
        self._A = A

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
            screen_transforms=self._screen_transforms,
            lmda_path=self._lmda_path,
            max_iters=max_iters,
            tol=tol,
            adev_tol=adev_tol,
            ddev_tol=ddev_tol,
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


def gaussian_pin_cov(
    *,
    A: Union[matrix.MatrixCovBase64, matrix.MatrixCovBase32],
    groups: np.ndarray,
    alpha: float,
    penalty: np.ndarray,
    screen_set: np.ndarray,
    lmda_path: np.ndarray,
    rsq: float,
    screen_beta: np.ndarray,
    screen_grad: np.ndarray,
    screen_is_active: np.ndarray,
    y_var: float =-1,
    max_iters: int =int(1e5),
    tol: float =1e-7,
    adev_tol: float =0.9,
    ddev_tol: float =1e-4,
    newton_tol: float =1e-12,
    newton_max_iters: int =1000,
    n_threads: int =1,
):
    """Creates a gaussian pin covariance method state object.

    Parameters
    ----------
    A : Union[adelie.matrix.MatrixCovBase64, adelie.matrix.MatrixCovBase32]
        Covariance matrix :math:`X_c^\\top W X_c` where :math:`X_c` 
        is column-centered (weighted by :math:`W`) if fitting with intercept
        and :math:`W` is the observation weights.
        It is typically one of the matrices defined in ``adelie.matrix`` sub-module.
    groups : (G,) np.ndarray
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
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
        The unnormalized :math:`R^2` is given by :math:`\\|y_c\\|_{W}^2 - \\|y_c-X_c\\beta\\|_{W}^2`.
    resid : (n,) np.ndarray
        Residual :math:`W(y_c-X\\beta)` at ``screen_beta``.
    screen_beta : (ws,) np.ndarray
        Coefficient vector on the strong set.
        ``screen_beta[b:b+p]`` is the coefficient for the ``i`` th strong group 
        where
        ``k = screen_set[i]``,
        ``b = screen_begins[i]``,
        and ``p = group_sizes[k]``.
    screen_grad : (ws,) np.ndarray
        Gradient :math:`X_{c,k}^\\top W (y_c-X_c\\beta)` on the strong groups :math:`k` where :math:`\\beta` is given by ``screen_beta``.
        ``screen_grad[b:b+p]`` is the gradient for the ``i`` th strong group
        where 
        ``k = screen_set[i]``,
        ``b = screen_begins[i]``,
        and ``p = group_sizes[k]``.
    screen_is_active : (a,) np.ndarray
        Boolean vector that indicates whether each strong group in ``groups`` is active or not.
        ``screen_is_active[i]`` is ``True`` if and only if ``screen_set[i]`` is active.
    y_var : float, optional
        :math:`\ell_2` norm squared of :math:`y_c` (weighted by :math:`W`).
        If the user does not have access to this quantity, 
        they may be set it to ``-1``.
        The only effect this variable has on the algorithm is early stopping rule.
        Hence, with ``-1``, the early stopping rule is effectively disabled.
        Default is ``-1``.
    max_iters : int, optional
        Maximum number of coordinate descents.
        Default is ``int(1e5)``.
    tol : float, optional
        Convergence tolerance.
        Default is ``1e-7``.
    adev_tol : float, optional
        Percent deviance explained tolerance.
        Default is ``0.9``.
    ddev_tol : float, optional
        Difference in percent deviance explained tolerance.
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
    adelie.adelie_core.state.StateGaussianPinCov64
    """
    if not (
        isinstance(A, matrix.MatrixCovBase64) or 
        isinstance(A, matrix.MatrixCovBase32)
    ):
        raise ValueError(
            "A must be an instance of matrix.MatrixCovBase32 or matrix.MatrixCovBase64."
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
        def __init__(self, *args, **kwargs):
            self._core_type = core_base
            gaussian_pin_cov_base.default_init(
                self,
                core_base,
                *args,
                dtype=dtype,
                **kwargs,
            )

        @classmethod
        def create_from_core(cls, state, core_state):
            obj = base.create_from_core(
                cls, state, core_state, _gaussian_pin_cov, core_base,
            )
            obj._core_type = core_base
            gaussian_pin_cov_base.__init__(obj)
            return obj

    return _gaussian_pin_cov(
        A=A,
        y_var=y_var,
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
        adev_tol=adev_tol,
        ddev_tol=ddev_tol,
        newton_tol=newton_tol,
        newton_max_iters=newton_max_iters,
        n_threads=n_threads,
    )


class gaussian_base(base):
    pass


class gaussian_naive_base(gaussian_base):
    """State wrapper base class for all gaussian naive method."""
    def default_init(
        self, 
        base_type: Union[core.state.StateGaussianNaive64, core.state.StateGaussianNaive32],
        *,
        X: Union[matrix.MatrixNaiveBase64, matrix.MatrixNaiveBase32],
        y: np.ndarray,
        X_means: np.ndarray,
        y_mean: float,
        y_var: float,
        resid: np.ndarray,
        groups: np.ndarray,
        group_sizes: np.ndarray,
        alpha: float,
        penalty: np.ndarray,
        weights: np.ndarray,
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
        adev_tol: float,
        ddev_tol: float,
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
        dtype: Union[np.float32, np.float64],
    ):
        """Default initialization method.
        """
        ## save inputs due to lifetime issues
        # static inputs require a reference to input
        # or copy if it must be made
        self._X = X
        # this is only needed for check()
        self._y = y

        self._X_means = np.array(X_means, copy=False, dtype=dtype)
        self._groups = np.array(groups, copy=False, dtype=int)
        self._group_sizes = np.array(group_sizes, copy=False, dtype=int)
        self._penalty = np.array(penalty, copy=False, dtype=dtype)
        self._weights = np.array(weights, copy=False, dtype=dtype)
        self._lmda_path = np.array(lmda_path, copy=False, dtype=dtype)
        screen_set = np.array(screen_set, copy=False, dtype=int)
        screen_beta = np.array(screen_beta, copy=False, dtype=dtype)
        screen_is_active = np.array(screen_is_active, copy=False, dtype=bool)
        grad = np.array(grad, copy=False, dtype=dtype)

        # MUST call constructor directly and not use super()!
        # https://pybind11.readthedocs.io/en/stable/advanced/classes.html#forced-trampoline-class-initialisation
        base_type.__init__(
            self,
            X=X,
            X_means=self._X_means,
            y_mean=y_mean,
            y_var=y_var,
            resid=resid,
            groups=self._groups,
            group_sizes=self._group_sizes,
            alpha=alpha,
            penalty=self._penalty,
            weights=self._weights,
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
            adev_tol=adev_tol,
            ddev_tol=ddev_tol,
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
        method: str =None, 
        logger=logger.logger,
    ):
        n, p = self.X.rows(), self.X.cols()

        yc = self._y
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

        # ================ penalty check ====================
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
        tmp = np.empty(n)
        WXbeta = np.zeros(n)
        for g, gs, b in zip(
            self.groups[self.screen_set], 
            self.group_sizes[self.screen_set],
            self.screen_begins,
        ):
            screen_indices.append(np.arange(g, g + gs))
            self.X.btmul(g, gs, self.screen_beta[b:b+gs], self.weights, tmp)
            WXbeta += tmp

        if len(screen_indices) == 0:
            screen_indices = np.array(screen_indices, dtype=int)
        else:
            screen_indices = np.concatenate(screen_indices, dtype=int)

        resid = self.weights * yc - WXbeta
        grad = np.empty(p)
        self.X.mul(resid, grad)
        if self.intercept:
            grad -= self.X_means * np.sum(resid)
        WXcbeta = WXbeta - self.weights * (self.screen_X_means @ self.screen_beta)
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
        for i, b, g, gs in zip(
            self.screen_set,
            self.screen_begins,
            self.groups[self.screen_set],
            self.group_sizes[self.screen_set],
        ):
            lmda = 1e35 if np.isinf(self.lmda) else self.lmda
            grad_corr[g:g+gs] -= lmda * (1-self.alpha) * self.penalty[i] * self.screen_beta[b:b+gs]
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
    X: Union[matrix.MatrixNaiveBase64, matrix.MatrixNaiveBase32],
    y: np.ndarray,
    X_means: np.ndarray,
    y_mean: float,
    y_var: float,
    resid: np.ndarray,
    groups: np.ndarray,
    group_sizes: np.ndarray,
    alpha: float,
    penalty: np.ndarray,
    weights: np.ndarray,
    screen_set: np.ndarray,
    screen_beta: np.ndarray,
    screen_is_active: np.ndarray,
    rsq: float,
    lmda: float,
    grad: np.ndarray,
    lmda_path: np.ndarray =None,
    lmda_max: float =None,
    max_iters: int =int(1e5),
    tol: float =1e-7,
    adev_tol: float =0.9,
    ddev_tol: float =1e-4,
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
    """Creates a gaussian, naive method state object.

    Parameters
    ----------
    X : Union[adelie.matrix.MatrixNaiveBase64, adelie.matrix.MatrixNaiveBase32]
        Feature matrix.
        It is typically one of the matrices defined in ``adelie.matrix`` sub-module.
    y : (n,) np.ndarray
        Response vector.
    X_means : (p,) np.ndarray
        Column means (weighted by :math:`W`) of ``X``.
    y_mean : float
        Mean (weighted by :math:`W`) of the response vector :math:`y`.
    y_var : float
        :math:`\\ell_2` norm squared (weighted by :math:`W`) of :math:`y`, i.e. 
        :math:`\\|y_c\\|_{W}^2`.
    resid : (n,) np.ndarray
        Residual :math:`W(y_c - X \\beta)` where :math:`\\beta` is given by ``screen_beta``.
    groups : (G,) np.ndarray
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
    group_sizes : (G,) np.ndarray
        List of group sizes corresponding to each element of ``groups``.
        ``group_sizes[i]`` is the size of the ``i`` th group.
    alpha : float
        Elastic net parameter.
        It must be in the range :math:`[0,1]`.
    penalty : (G,) np.ndarray
        Penalty factor for each group in the same order as ``groups``.
        It must be a non-negative vector.
    weights : (n,) np.ndarray
        Observation weights.
        Internally, it is normalized to sum to one.
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
        The values can be arbitrary but it is recommended to be close to the solution at ``lmda``.
    screen_is_active : (a,) np.ndarray
        Boolean vector that indicates whether each strong group in ``groups`` is active or not.
        ``screen_is_active[i]`` is ``True`` if and only if ``screen_set[i]`` is active.
    rsq : float
        The unnormalized :math:`R^2` given by :math:`\\|y_c\\|_{W}^2 - \\|y_c-X_c\\beta\\|_{W}^2`
        where :math:`\\beta` is given by ``screen_beta``.
    lmda : float
        The regularization parameter at which the solution is given by ``screen_beta``
        (in the transformed space).
    grad : np.ndarray
        The full gradient :math:`X_c^\\top W (y_c - X_c\\beta)` in the original space where
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
        Default is ``1e-7``.
    adev_tol : float, optional
        Percent deviance explained tolerance.
        Default is ``0.9``.
    ddev_tol : float, optional
        Difference in percent deviance explained tolerance.
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
        The type of screening rule to use. It must be one of the following options:

            - ``"strong"``: adds groups whose active scores are above the strong threshold.
            - ``"pivot"``: adds groups whose active scores are above the pivot cutoff with slack.

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

    Returns
    -------
    wrap
        Wrapper state object.

    See Also
    --------
    adelie.adelie_core.state.StateGaussianNaive64
    """
    if not (
        isinstance(X, matrix.MatrixNaiveBase64) or 
        isinstance(X, matrix.MatrixNaiveBase32)
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
    if adev_tol < 0 or adev_tol > 1:
        raise ValueError("adev_tol must be in [0,1].")
    if ddev_tol < 0 or ddev_tol > 1:
        raise ValueError("ddev_tol must be in [0,1].")
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

    lmda_path_size = (
        lmda_path_size
        if lmda_path is None else
        len(lmda_path)
    )

    max_screen_size = np.minimum(max_screen_size, len(groups))

    dtype = (
        np.float64
        if isinstance(X, matrix.MatrixNaiveBase64) else
        np.float32
    )
        
    dispatcher = {
        np.float64: core.state.StateGaussianNaive64,
        np.float32: core.state.StateGaussianNaive32,
    }

    core_base = dispatcher[dtype]

    setup_lmda_max = lmda_max is None
    setup_lmda_path = lmda_path is None

    if setup_lmda_max: lmda_max = -1
    if setup_lmda_path: lmda_path = np.empty(0, dtype=dtype)

    class _gaussian_naive(gaussian_naive_base, core_base):
        def __init__(self, *args, **kwargs):
            self._core_type = core_base
            # this is to keep the API consistent with grpnet with non-trivial GLM object
            self.glm = None
            gaussian_naive_base.default_init(
                self,
                core_base,
                *args,
                dtype=dtype,
                **kwargs,
            )

        @classmethod
        def create_from_core(cls, state, core_state):
            obj = base.create_from_core(
                cls, state, core_state, _gaussian_naive, core_base,
            )
            obj._core_type = core_base
            obj.glm = None
            gaussian_naive_base.__init__(obj)
            return obj

    return _gaussian_naive(
        X=X,
        y=y,
        X_means=X_means,
        y_mean=y_mean,
        y_var=y_var,
        resid=resid,
        groups=groups,
        group_sizes=group_sizes,
        alpha=alpha,
        penalty=penalty,
        weights=weights,
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
        adev_tol=adev_tol,
        ddev_tol=ddev_tol,
        newton_tol=newton_tol,
        newton_max_iters=newton_max_iters,
        early_exit=early_exit,
        min_ratio=min_ratio,
        lmda_path_size=lmda_path_size,
        intercept=intercept,
        n_threads=n_threads,
        screen_set=screen_set,
        screen_beta=screen_beta,
        screen_is_active=screen_is_active,
        rsq=rsq,
        lmda=lmda,
        grad=grad,
    )


class glm_naive_base:
    """State wrapper base class for all glm naive method."""
    def default_init(
        self, 
        base_type: Union[core.state.StateGlmNaive64, core.state.StateGlmNaive32],
        *,
        glm: Union[glm.GlmBase64, glm.GlmBase32],
        X: Union[matrix.MatrixNaiveBase64, matrix.MatrixNaiveBase32],
        y: np.ndarray,
        groups: np.ndarray,
        group_sizes: np.ndarray,
        alpha: float,
        penalty: np.ndarray,
        weights: np.ndarray,
        lmda_path: np.ndarray,
        dev_null: float,
        dev_full: float,
        lmda_max: float,
        min_ratio: float,
        lmda_path_size: int,
        max_screen_size: int,
        pivot_subset_ratio: float,
        pivot_subset_min: int,
        pivot_slack_ratio: float,
        screen_rule: str,
        irls_max_iters: int,
        irls_tol: float,
        max_iters: int,
        tol: float,
        adev_tol: float,
        ddev_tol: float,
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
        beta0: float,
        lmda: float,
        grad: np.ndarray,
        eta: np.ndarray,
        mu: np.ndarray,
        dtype: Union[np.float32, np.float64],
    ):
        """Default initialization method.
        """
        ## save inputs due to lifetime issues
        # static inputs require a reference to input
        # or copy if it must be made
        self._X = X
        # this is only needed for check()
        self._y = y

        self._groups = np.array(groups, copy=False, dtype=int)
        self._group_sizes = np.array(group_sizes, copy=False, dtype=int)
        self._penalty = np.array(penalty, copy=False, dtype=dtype)
        self._weights = np.array(weights, copy=False, dtype=dtype)
        self._lmda_path = np.array(lmda_path, copy=False, dtype=dtype)
        screen_set = np.array(screen_set, copy=False, dtype=int)
        screen_beta = np.array(screen_beta, copy=False, dtype=dtype)
        screen_is_active = np.array(screen_is_active, copy=False, dtype=bool)
        grad = np.array(grad, copy=False, dtype=dtype)

        # MUST call constructor directly and not use super()!
        # https://pybind11.readthedocs.io/en/stable/advanced/classes.html#forced-trampoline-class-initialisation
        base_type.__init__(
            self,
            glm=glm,
            X=X,
            y=y,
            groups=self._groups,
            group_sizes=self._group_sizes,
            alpha=alpha,
            penalty=self._penalty,
            weights=self._weights,
            lmda_path=self._lmda_path,
            dev_null=dev_null,
            dev_full=dev_full,
            lmda_max=lmda_max,
            min_ratio=min_ratio,
            lmda_path_size=lmda_path_size,
            max_screen_size=max_screen_size,
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
            setup_lmda_max=setup_lmda_max,
            setup_lmda_path=setup_lmda_path,
            intercept=intercept,
            n_threads=n_threads,
            screen_set=screen_set,
            screen_beta=screen_beta,
            screen_is_active=screen_is_active,
            beta0=beta0,
            lmda=lmda,
            grad=grad,
            eta=eta,
            mu=mu,
        )

    # TODO: implement check()?


def glm_naive(
    *,
    glm: Union[glm.GlmBase64, glm.GlmBase32],
    X: Union[matrix.MatrixNaiveBase64, matrix.MatrixNaiveBase32],
    y: np.ndarray,
    groups: np.ndarray,
    group_sizes: np.ndarray,
    alpha: float,
    penalty: np.ndarray,
    weights: np.ndarray,
    screen_set: np.ndarray,
    screen_beta: np.ndarray,
    screen_is_active: np.ndarray,
    beta0: float,
    lmda: float,
    grad: np.ndarray,
    eta: np.ndarray,
    mu: np.ndarray,
    dev_null: float,
    dev_full: float,
    lmda_path: np.ndarray =None,
    lmda_max: float =None,
    irls_max_iters: int =int(1e2),
    irls_tol: float =1e-7,
    max_iters: int =int(1e5),
    tol: float =1e-7,
    adev_tol: float =0.9,
    ddev_tol: float =1e-4,
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
    """Creates a gaussian, naive method state object.

    Parameters
    ----------
    glm : Union[adelie.glm.GlmBase64, adelie.glm.GlmBase32]
        GLM object.
    X : Union[adelie.matrix.MatrixNaiveBase64, adelie.matrix.MatrixNaiveBase32]
        Feature matrix.
        It is typically one of the matrices defined in ``adelie.matrix`` sub-module.
    y : (n,) np.ndarray
        Response vector.
    groups : (G,) np.ndarray
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
    group_sizes : (G,) np.ndarray
        List of group sizes corresponding to each element of ``groups``.
        ``group_sizes[i]`` is the size of the ``i`` th group.
    alpha : float
        Elastic net parameter.
        It must be in the range :math:`[0,1]`.
    penalty : (G,) np.ndarray
        Penalty factor for each group in the same order as ``groups``.
        It must be a non-negative vector.
    weights : (n,) np.ndarray
        Observation weights.
        Internally, it is normalized to sum to one.
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
        The values can be arbitrary but it is recommended to be close to the solution at ``lmda``.
    screen_is_active : (a,) np.ndarray
        Boolean vector that indicates whether each strong group in ``groups`` is active or not.
        ``screen_is_active[i]`` is ``True`` if and only if ``screen_set[i]`` is active.
    beta0 : float
        The current intercept value.
        The value can be arbitrary but it is recommended to be close to the solution at ``lmda``.
    lmda : float
        The last regularization parameter that was attempted to be solved.
    grad : (p,) np.ndarray
        The full gradient :math:`X^\\top W (y - \\nabla \\underline{A}(X\\beta + \\beta_0 \\mathbf{1}))` where
        :math:`\\beta` is given by ``screen_beta``
        and :math:`\\beta_0` is given by ``beta0``.
    eta : (n,) np.ndarray
        The natural parameter :math:`\\eta = X\\beta + \\beta_0 \\mathbf{1}`
        where :math:`\\beta` and :math:`\\beta_0` are given by
        ``screen_beta`` and ``beta0``.
    mu : (n,) np.ndarray
        The mean parameter :math:`\\mu \\equiv \\nabla \\underline{A}(\\eta)`
        where :math:`\\eta` is given by ``eta``.
    dev_null : float 
        Null deviance :math:`D(\\eta_0)`
        where :math:`\\eta_0 = \\beta_0 1` is the intercept-only model fit.
    dev_full : float
        Full deviance :math:`D(\\eta^\\star)`
        where :math:`\\eta^\\star = (\\nabla \\underline{A})^{-1}(y)` is the saturated model fit.
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
    irls_max_iters : int, optional
        Maximum number of IRLS iterations.
        Default is ``100``.
    irls_tol : float, optional
        IRLS convergence tolerance.
        Default is ``1e-7``.
    max_iters : int, optional
        Maximum number of coordinate descents.
        Default is ``int(1e5)``.
    tol : float, optional
        Convergence tolerance.
        Default is ``1e-7``.
    adev_tol : float, optional
        Percent deviance explained tolerance.
        Default is ``0.9``.
    ddev_tol : float, optional
        Difference in percent deviance explained tolerance.
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
    intercept : bool, optional 
        ``True`` if the function should fit with intercept.
        Default is ``True``.
    screen_rule : str, optional
        The type of screening rule to use. It must be one of the following options:

            - ``"strong"``: adds groups whose active scores are above the strong threshold.
            - ``"pivot"``: adds groups whose active scores are above the pivot cutoff with slack.

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

    Returns
    -------
    wrap
        Wrapper state object.

    See Also
    --------
    adelie.adelie_core.state.StateGlmNaive64
    """
    if not (
        isinstance(X, matrix.MatrixNaiveBase64) or 
        isinstance(X, matrix.MatrixNaiveBase32)
    ):
        raise ValueError(
            "X must be an instance of matrix.MatrixNaiveBase32 or matrix.MatrixNaiveBase64."
        )

    if max_screen_size is None:
        max_screen_size = len(groups)

    if irls_max_iters < 0:
        raise ValueError("irls_max_iters must be >= 0.")
    if irls_tol <= 0:
        raise ValueError("irls_tol must be > 0.")
    if max_iters < 0:
        raise ValueError("max_iters must be >= 0.")
    if tol <= 0:
        raise ValueError("tol must be > 0.")
    if adev_tol < 0 or adev_tol > 1:
        raise ValueError("adev_tol must be in [0,1]")
    if ddev_tol < 0 or ddev_tol > 1:
        raise ValueError("ddev_tol must be in [0,1]")
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

    lmda_path_size = (
        lmda_path_size
        if lmda_path is None else
        len(lmda_path)
    )

    max_screen_size = np.minimum(max_screen_size, len(groups))

    dtype = (
        np.float64
        if isinstance(X, matrix.MatrixNaiveBase64) else
        np.float32
    )
        
    dispatcher = {
        np.float64: core.state.StateGlmNaive64,
        np.float32: core.state.StateGlmNaive32,
    }

    core_base = dispatcher[dtype]

    setup_lmda_max = lmda_max is None
    setup_lmda_path = lmda_path is None

    if setup_lmda_max: lmda_max = -1
    if setup_lmda_path: lmda_path = np.empty(0, dtype=dtype)

    class _glm_naive(glm_naive_base, core_base):
        def __init__(self, *args, **kwargs):
            self._core_type = core_base
            glm_naive_base.default_init(
                self,
                core_base,
                *args,
                dtype=dtype,
                **kwargs,
            )

        @classmethod
        def create_from_core(cls, state, core_state):
            obj = base.create_from_core(
                cls, state, core_state, _glm_naive, core_base,
            )
            obj._core_type = core_base
            glm_naive_base.__init__(obj)
            return obj

    return _glm_naive(
        glm=glm,
        X=X,
        y=y,
        groups=groups,
        group_sizes=group_sizes,
        alpha=alpha,
        penalty=penalty,
        weights=weights,
        lmda_path=lmda_path,
        dev_null=dev_null,
        dev_full=dev_full,
        lmda_max=lmda_max,
        setup_lmda_max=setup_lmda_max,
        setup_lmda_path=setup_lmda_path,
        max_screen_size=max_screen_size,
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
        min_ratio=min_ratio,
        lmda_path_size=lmda_path_size,
        intercept=intercept,
        n_threads=n_threads,
        screen_set=screen_set,
        screen_beta=screen_beta,
        screen_is_active=screen_is_active,
        beta0=beta0,
        lmda=lmda,
        grad=grad,
        eta=eta,
        mu=mu,
    )