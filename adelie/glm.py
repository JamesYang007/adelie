from . import adelie_core as core
from .adelie_core.glm import (
    GlmBase64,
    GlmBase32,
    GlmMultiBase64,
    GlmMultiBase32,
)
from typing import Union
import numpy as np
import warnings


def _coerce_dtype(y, dtype):
    dtype_map = {
        np.dtype("float32"): np.float32,
        np.dtype("float64"): np.float64,
    }
    valid_dtypes = list(dtype_map.keys())
    y = np.array(y, order="C")
    if dtype is None:
        if not (y.dtype in valid_dtypes):
            raise RuntimeError(
                "y must have an underlying type of np.float32 or np.float64, "
                "or dtype must be explicitly specified."
            )
        dtype = dtype_map[y.dtype]
    else:
        if not (dtype in valid_dtypes):
            raise RuntimeError("dtype must be either np.float32 or np.float64.")
    y = y.astype(dtype)
    return y, dtype


class glm_base:
    """Base wrapper GLM class.

    All Python wrapper classes for core GLM classes must inherit from this class.
    The purpose of this class is to expose extra member interface
    that are more easily written in Python (and not speed-critical).
    """
    def __init__(self, y, weights, core_base, dtype):
        self.core_base = core_base
        self.y = np.array(y, copy=True, dtype=dtype)
        self.dtype = dtype
        if len(y.shape) != 1:
            raise RuntimeError("y must be 1-dimensional.")
        n = y.shape[0]
        if not (weights is None):
            if weights.shape != (n,):
                raise RuntimeError("y and weights must have same length.")
            weights_sum = np.sum(weights)
            if not np.allclose(weights_sum, 1):
                warnings.warn("Normalizing weights to sum to 1.")
                weights = weights / weights_sum
        else:
            weights = np.full(n, 1/n, dtype=dtype)
        self.weights = np.array(weights, copy=True, dtype=dtype)


class multiglm_base:
    """Base wrapper multi-response GLM class.

    All Python wrapper classes for core multi-response GLM classes must inherit from this class.
    The purpose of this class is to expose extra member interface
    that are more easily written in Python (and not speed-critical).
    """
    def __init__(self, y, weights, core_base, dtype):
        self.core_base = core_base
        self.y = np.array(y, copy=True, dtype=dtype)
        self.dtype = dtype
        if len(y.shape) != 2:
            raise RuntimeError("y must be 2-dimensional.")
        n = y.shape[0]
        if not (weights is None):
            if weights.shape != (n,):
                raise RuntimeError("y rows and weights must have same length.")
            weights_sum = np.sum(weights)
            if not np.allclose(weights_sum, 1):
                warnings.warn("Normalizing weights to sum to 1.")
                weights = weights / np.sum(weights)
        else:
            weights = np.full(n, 1/n, dtype=dtype)
        self.weights = np.array(weights, copy=True, dtype=dtype)


def binomial(
    y: np.ndarray,
    *,
    weights: np.ndarray =None,
    link: str ="logit",
    dtype: Union[np.float32, np.float64] =None,
):
    """Creates a Binomial GLM family object.

    The Binomial GLM family with the logit link function 
    specifies the loss function as:

    .. math::
        \\begin{align*}
            \\ell(\\eta)
            =
            \\sum\\limits_{i=1}^n w_i \\left(
                -y_i \\eta_i + \\log(1 + e^{\\eta_i})
            \\right)
        \\end{align*}

    The Binomial GLM family with the probit link function 
    specifies the loss function as:

    .. math::
        \\begin{align*}
            \\ell(\\eta)
            =
            -\\sum\\limits_{i=1}^n w_i \\left(
                y_i \\log(\\Phi(\\eta_i)) + (1-y_i) \\log(1-\\Phi(\\eta_i))
            \\right)
        \\end{align*}

    where :math:`\\Phi` is the standard normal CDF.

    We assume that :math:`y_i \\in [0,1]`.

    Parameters
    ----------
    y : (n,) np.ndarray 
        Response vector :math:`y`.
    weights : (n,) np.ndarray, optional
        Observation weights :math:`W`.
        Weights are normalized such that they sum to ``1``.
        Default is ``None``, in which case, it is set to ``np.full(n, 1/n)``.
    link : str, optional
        The link function type.
        It must be one of the following:

            - ``"logit"``: the logit link function.
            - ``"probit"``: the probit link function.

        Default is ``"logit"``.
    dtype : Union[np.float32, np.float64], optional
        The underlying data type.
        If ``None``, it is inferred from ``y``,
        in which case ``y`` must have an underlying data type of
        ``np.float32`` or ``np.float64``.
        Default is ``None``.

    Returns
    -------
    glm
        Binomial GLM object.

    See Also
    --------
    adelie.adelie_core.glm.GlmBinomialLogit64
    adelie.adelie_core.glm.GlmBinomialProbit64
    """
    dispatcher = {
        "logit": {
            np.float64: core.glm.GlmBinomialLogit64,
            np.float32: core.glm.GlmBinomialLogit32,
        },
        "probit": {
            np.float64: core.glm.GlmBinomialProbit64,
            np.float32: core.glm.GlmBinomialProbit32,
        },
    }

    y, dtype = _coerce_dtype(y, dtype)

    core_base = dispatcher[link][dtype]

    class _binomial(glm_base, core_base):
        def __init__(self):
            glm_base.__init__(self, y, weights, core_base, dtype)
            core_base.__init__(self, self.y, self.weights)

        def reweight(self, weights=None):
            weights = self.weights if weights is None else weights
            return binomial(y=y, weights=weights, dtype=dtype)

    return _binomial()


def cox(
    start: np.ndarray,
    stop: np.ndarray,
    status: np.ndarray,
    *,
    weights: np.ndarray =None,
    tie_method: str ="efron",
    dtype: Union[np.float32, np.float64] =None,
):
    """Creates a Cox GLM family object.

    The Cox GLM family specifies the loss function as:

    .. math::
        \\begin{align*}
            \\ell(\\eta)
            &=
            -\\sum\\limits_{i=1}^n w_i \\delta_i \\eta_i
            +\\sum\\limits_{i=1}^n \\overline{w}_i \\delta_i A_i(\\eta)
            \\\\
            A_i(\\eta)
            &=
            \\log\\left(
                \\sum\\limits_{k \\in R(t_i)} w_k e^{\\eta_k}
                -
                \\sigma_i \\sum\\limits_{k \\in H(t_i)} w_k e^{\\eta_k}
            \\right)
        \\end{align*}

    where
    
    .. math::
        \\begin{align*}
            R(u)
            &=
            \\{i : u \\in (s_i, t_i]\\}
            \\\\
            H(u)
            &=
            \\{i : t_i = u, \\delta_i = 1\\}
            \\\\
            \\overline{w}_i 
            &= 
            \\frac{\\sum_{k \\in H(t_i)} w_k}{\\sum_{k \\in H(t_i)} 1_{w_k > 0}} 
            1_{\\delta_i = 1, w_i > 0}
        \\end{align*}

    Here,
    :math:`\\delta` is the status (``1`` for event, ``0`` for censored) vector,
    :math:`s` is the vector of start times,
    :math:`t` is the vector of stop times,
    :math:`R(u)` is the at-risk set at time :math:`u`,
    :math:`H(u)` is the set of ties at event time :math:`u`,
    :math:`\\overline{w}` is the vector of average weights within ties with positive weights,
    :math:`\\sigma` is the correction scale for tie-breaks, 
    which is determined by the type of correction method (Breslow or Efron).
    Note that :math:`\\overline{w}_i` and :math:`A_i(\\eta)` are only well-defined 
    whenever :math:`\\delta_i=1`, which is not an issue in the computation of :math:`\\ell(\\eta)`.

    Parameters
    ----------
    start : (n,) np.ndarray
        Start time vector :math:`s`.
    stop : (n,) np.ndarray
        Stop time vector :math:`t`.
    status : (n,) np.ndarray 
        Status vector :math:`\\delta`.
    weights : (n,) np.ndarray, optional
        Observation weights :math:`W`.
        Weights are normalized such that they sum to ``1``.
        Default is ``None``, in which case, it is set to ``np.full(n, 1/n)``.
    tie_method : str, optional
        The tie-breaking method that determines the scales :math:`\\sigma`.
        It must be one of the following:

            - ``"efron"``
            - ``"breslow"``

        Default is ``"efron"``.
    dtype : Union[np.float32, np.float64], optional
        The underlying data type.
        If ``None``, it is inferred from ``status``,
        in which case ``status`` must have an underlying data type of
        ``np.float32`` or ``np.float64``.
        Default is ``None``.

    Returns
    -------
    glm
        Cox GLM object.

    See Also
    --------
    adelie.adelie_core.glm.GlmCox64
    """
    dispatcher = {
        np.float64: core.glm.GlmCox64,
        np.float32: core.glm.GlmCox32,
    }

    status, dtype = _coerce_dtype(status, dtype)

    core_base = dispatcher[dtype]

    class _cox(glm_base, core_base):
        def __init__(self):
            self.start = np.array(start, copy=True, dtype=dtype)
            self.stop = np.array(stop, copy=True, dtype=dtype)
            glm_base.__init__(self, status, weights, core_base, dtype)
            self.status = self.y
            self.tie_method = tie_method
            core_base.__init__(
                self, 
                self.start, 
                self.stop, 
                self.status, 
                self.weights, 
                self.tie_method,
            )

        def reweight(self, weights=None):
            weights = self.weights if weights is None else weights
            return cox(
                start=start, 
                stop=stop,
                status=status,
                weights=weights, 
                tie_method=tie_method,
                dtype=dtype,
            )

    return _cox()


def gaussian(
    y: np.ndarray,
    *,
    weights: np.ndarray =None,
    dtype: Union[np.float32, np.float64] =None,
    opt: bool =True,
):
    """Creates a Gaussian GLM family object.

    The Gaussian GLM family specifies the loss function as:

    .. math::
        \\begin{align*}
            \\ell(\\eta)
            =
            \\sum\\limits_{i=1}^n w_i \\left(
                -y_i \\eta_i + \\frac{\\eta_i^2}{2}
            \\right) 
        \\end{align*}

    Parameters
    ----------
    y : (n,) np.ndarray 
        Response vector :math:`y`.
    weights : (n,) np.ndarray, optional
        Observation weights :math:`W`.
        Weights are normalized such that they sum to ``1``.
        Default is ``None``, in which case, it is set to ``np.full(n, 1/n)``.
    dtype : Union[np.float32, np.float64], optional
        The underlying data type.
        If ``None``, it is inferred from ``y``,
        in which case ``y`` must have an underlying data type of
        ``np.float32`` or ``np.float64``.
        Default is ``None``.
    opt : bool, optional
        If ``True``, an optimized routine is used when passed into ``adelie.grpnet``.
        Otherwise, a general routine with IRLS is used.
        This flag is mainly for developers for testing purposes.
        We advise users to use the default value.
        Default is ``True``.

    Returns
    -------
    glm
        Gaussian GLM object.

    See Also
    --------
    adelie.adelie_core.glm.GlmGaussian64
    """
    dispatcher = {
        np.float64: core.glm.GlmGaussian64,
        np.float32: core.glm.GlmGaussian32,
    }

    y, dtype = _coerce_dtype(y, dtype)

    core_base = dispatcher[dtype]

    class _gaussian(glm_base, core_base):
        def __init__(self):
            self.opt = opt
            glm_base.__init__(self, y, weights, core_base, dtype)
            core_base.__init__(self, self.y, self.weights)

        def reweight(self, weights=None):
            weights = self.weights if weights is None else weights
            return gaussian(y=y, weights=weights, dtype=dtype, opt=opt)

    return _gaussian()


def multigaussian(
    y: np.ndarray,
    *,
    weights: np.ndarray =None,
    dtype: Union[np.float32, np.float64] =None,
    opt: bool =True,
):
    """Creates a MultiGaussian GLM family object.

    The MultiGaussian GLM family specifies the loss function as:

    .. math::
        \\begin{align*}
            \\ell(\\eta)
            =
            \\frac{1}{K}
            \\sum\\limits_{i=1}^n 
            w_{i} \\left(
                -\\sum\\limits_{k=1}^K y_{ik} \\eta_{ik} 
                +\\frac{\\|\\eta_{i\\cdot}\\|^2}{2}
            \\right)
        \\end{align*}

    Parameters
    ----------
    y : (n, K) np.ndarray 
        Response matrix :math:`y`.
    weights : (n,) np.ndarray, optional
        Observation weights :math:`W`.
        Weights are normalized such that they sum to ``1``.
        Default is ``None``, in which case, it is set to ``np.full(n, 1/n)``.
    dtype : Union[np.float32, np.float64], optional
        The underlying data type.
        If ``None``, it is inferred from ``y``,
        in which case ``y`` must have an underlying data type of
        ``np.float32`` or ``np.float64``.
        Default is ``None``.
    opt : bool, optional
        If ``True``, an optimized routine is used when passed into ``adelie.grpnet``.
        Otherwise, a general routine with IRLS is used.
        This flag is mainly for developers for testing purposes.
        We advise users to use the default value.
        Default is ``True``.

    Returns
    -------
    glm
        MultiGaussian GLM object.

    See Also
    --------
    adelie.adelie_core.glm.GlmMultiGaussian64
    """
    dispatcher = {
        np.float64: core.glm.GlmMultiGaussian64,
        np.float32: core.glm.GlmMultiGaussian32,
    }

    y, dtype = _coerce_dtype(y, dtype)

    core_base = dispatcher[dtype]

    class _multigaussian(multiglm_base, core_base):
        def __init__(self):
            self.opt = opt
            multiglm_base.__init__(self, y, weights, core_base, dtype)
            core_base.__init__(self, self.y, self.weights)

        def reweight(self, weights=None):
            weights = self.weights if weights is None else weights
            return multigaussian(y=y, weights=weights, dtype=dtype, opt=opt)

    return _multigaussian()


def multinomial(
    y: np.ndarray,
    *,
    weights: np.ndarray =None,
    dtype: Union[np.float32, np.float64] =None,
):
    """Creates a Multinomial GLM family object.

    The Multinomial GLM family specifies the loss function as:

    .. math::
        \\begin{align*}
            \\ell(\\eta)
            =
            \\frac{1}{K}
            \\sum\\limits_{i=1}^n 
            w_i
            \\left(
            -\\sum\\limits_{k=1}^{K} y_{ik} \\eta_{ik} 
            + \\log\\left(
                \\sum\\limits_{\\ell=1}^{K} e^{\\eta_{i\\ell}}
            \\right)
            \\right)
        \\end{align*}

    We assume that every :math:`y_{ik} \\in [0,1]` and
    for each fixed :math:`i`, :math:`\\sum_{k=1}^K y_{ik} = 1`.

    .. note::
        The ``hessian`` method computes :math:`2 \\mathrm{diag}(\\nabla^2 \\ell(\\eta))`
        as the diagonal majorization.

    Parameters
    ----------
    y : (n, K) np.ndarray 
        Response matrix :math:`y`.
    weights : (n,) np.ndarray, optional
        Observation weights :math:`W`.
        Weights are normalized such that they sum to ``1``.
        Default is ``None``, in which case, it is set to ``np.full(n, 1/n)``.
    dtype : Union[np.float32, np.float64], optional
        The underlying data type.
        If ``None``, it is inferred from ``y``,
        in which case ``y`` must have an underlying data type of
        ``np.float32`` or ``np.float64``.
        Default is ``None``.

    Returns
    -------
    glm
        Multinomial GLM object.

    See Also
    --------
    adelie.adelie_core.glm.GlmMultinomial64
    """
    dispatcher = {
        np.float64: core.glm.GlmMultinomial64,
        np.float32: core.glm.GlmMultinomial32,
    }

    y, dtype = _coerce_dtype(y, dtype)

    core_base = dispatcher[dtype]

    class _multinomial(multiglm_base, core_base):
        def __init__(self):
            multiglm_base.__init__(self, y, weights, core_base, dtype)
            core_base.__init__(self, self.y, self.weights)

        def reweight(self, weights=None):
            weights = self.weights if weights is None else weights
            return multinomial(y=y, weights=weights, dtype=dtype)

    return _multinomial()


def poisson(
    y: np.ndarray,
    *,
    weights: np.ndarray =None,
    dtype: Union[np.float32, np.float64] =None,
):
    """Creates a Poisson GLM family object.

    The Poisson GLM family specifies the loss function as:

    .. math::
        \\begin{align*}
            \\ell(\\eta)
            =
            \\sum\\limits_{i=1}^n w_i \\left(
                -y_i \\eta_i + e^{\\eta_i}
            \\right) 
        \\end{align*}

    We assume that :math:`y_i \\in \\mathbb{N}_0`.

    Parameters
    ----------
    y : (n,) np.ndarray 
        Response vector :math:`y`.
    weights : (n,) np.ndarray, optional
        Observation weights :math:`W`.
        Weights are normalized such that they sum to ``1``.
        Default is ``None``, in which case, it is set to ``np.full(n, 1/n)``.
    dtype : Union[np.float32, np.float64], optional
        The underlying data type.
        If ``None``, it is inferred from ``y``,
        in which case ``y`` must have an underlying data type of
        ``np.float32`` or ``np.float64``.
        Default is ``None``.

    Returns
    -------
    glm
        Poisson GLM object.

    See Also
    --------
    adelie.adelie_core.glm.GlmPoisson64
    """
    dispatcher = {
        np.float64: core.glm.GlmPoisson64,
        np.float32: core.glm.GlmPoisson32,
    }

    y, dtype = _coerce_dtype(y, dtype)

    core_base = dispatcher[dtype]

    class _poisson(glm_base, core_base):
        def __init__(self):
            glm_base.__init__(self, y, weights, core_base, dtype)
            core_base.__init__(self, self.y, self.weights)

        def reweight(self, weights=None):
            weights = self.weights if weights is None else weights
            return poisson(y=y, weights=weights, dtype=dtype)

    return _poisson()
