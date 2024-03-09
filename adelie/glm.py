from typing import Union
from . import adelie_core as core
from .adelie_core.glm import (
    GlmBase64,
    GlmBase32,
    GlmMultiBase64,
    GlmMultiBase32,
)
import numpy as np
import warnings


class glm_base:
    """Base wrapper GLM class.

    All Python wrapper classes for core GLM classes must inherit from this class.
    The purpose of this class is to expose extra member interface
    that are more easily written in Python (and not speed-critical).
    """
    def __init__(self, y, weights, core_base, dtype):
        self.core_base = core_base
        self.dtype = dtype
        self.y = np.array(y, order="C", dtype=dtype)
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
        self.weights = np.array(weights, order="C", dtype=dtype)


class multiglm_base:
    """Base wrapper Multi-response GLM class.

    All Python wrapper classes for core multi-response GLM classes must inherit from this class.
    The purpose of this class is to expose extra member interface
    that are more easily written in Python (and not speed-critical).
    """
    def __init__(self, y, weights, core_base, dtype):
        self.core_base = core_base
        self.dtype = dtype
        self.y = np.array(y, order="C", dtype=dtype)
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
        self.weights = np.array(weights, order="C", dtype=dtype)


def gaussian(
    *,
    y: np.ndarray,
    weights: np.ndarray =None,
    dtype: Union[np.float32, np.float64] =np.float64,
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
        Default is ``np.float64``.
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
    adelie.glm.GlmBase64
    """
    dispatcher = {
        np.float64: core.glm.GlmGaussian64,
        np.float32: core.glm.GlmGaussian32,
    }

    core_base = dispatcher[dtype]

    class _gaussian(glm_base, core_base):
        def __init__(
            self,
        ):
            self.opt = opt
            glm_base.__init__(self, y, weights, core_base, dtype)
            core_base.__init__(self, self.y, self.weights)

        def reweight(self, weights=None):
            weights = self.weights if weights is None else weights
            return gaussian(y=y, weights=weights, dtype=dtype, opt=opt)

    return _gaussian()


def multigaussian(
    *,
    y: np.ndarray,
    weights: np.ndarray =None,
    dtype: Union[np.float32, np.float64] =np.float64,
    opt: bool =True,
):
    """Creates a Multi-response Gaussian GLM family object.

    The Multi-Response Gaussian GLM family specifies the loss function as:

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
        Default is ``np.float64``.
    opt : bool, optional
        If ``True``, an optimized routine is used when passed into ``adelie.grpnet``.
        Otherwise, a general routine with IRLS is used.
        This flag is mainly for developers for testing purposes.
        We advise users to use the default value.
        Default is ``True``.

    Returns
    -------
    glm
        Multi-response Gaussian GLM object.

    See Also
    --------
    adelie.glm.GlmBase64
    """
    dispatcher = {
        np.float64: core.glm.GlmMultiGaussian64,
        np.float32: core.glm.GlmMultiGaussian32,
    }

    core_base = dispatcher[dtype]

    class _multigaussian(multiglm_base, core_base):
        def __init__(
            self,
        ):
            self.opt = opt
            multiglm_base.__init__(self, y, weights, core_base, dtype)
            core_base.__init__(self, self.y, self.weights)

        def reweight(self, weights=None):
            weights = self.weights if weights is None else weights
            return multigaussian(y=y, weights=weights, dtype=dtype, opt=opt)

    return _multigaussian()


def binomial(
    *,
    y: np.ndarray,
    weights: np.ndarray =None,
    dtype: Union[np.float32, np.float64] =np.float64,
):
    """Creates a Binomial GLM family object.

    The Binomial GLM family specifies the loss function as:

    .. math::
        \\begin{align*}
            \\ell(\\eta)
            =
            \\sum\\limits_{i=1}^n w_i \\left(
                -y_i \\eta_i + \\log(1 + e^{\\eta_i})
            \\right)
        \\end{align*}

    We assume that :math:`y_i \\in \\{0,1\\}`.

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
        Default is ``np.float64``.

    Returns
    -------
    glm
        Binomial GLM object.

    See Also
    --------
    adelie.glm.GlmBase64
    """
    dispatcher = {
        np.float64: core.glm.GlmBinomial64,
        np.float32: core.glm.GlmBinomial32,
    }

    core_base = dispatcher[dtype]

    class _binomial(glm_base, core_base):
        def __init__(
            self,
        ):
            glm_base.__init__(self, y, weights, core_base, dtype)
            core_base.__init__(self, self.y, self.weights)

        def reweight(self, weights=None):
            weights = self.weights if weights is None else weights
            return binomial(y=y, weights=weights, dtype=dtype)

    return _binomial()


def cox(
    *,
    start: np.ndarray,
    stop: np.ndarray,
    status: np.ndarray,
    weights: np.ndarray =None,
    tie_method: str ="efron",
    dtype: Union[np.float32, np.float64] =np.float64,
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
        Default is ``np.float64``.

    Returns
    -------
    glm
        Cox GLM object.

    See Also
    --------
    adelie.glm.GlmBase64
    """
    dispatcher = {
        np.float64: core.glm.GlmCox64,
        np.float32: core.glm.GlmCox32,
    }

    core_base = dispatcher[dtype]

    class _cox(glm_base, core_base):
        def __init__(
            self,
        ):
            self.start = start.astype(dtype)
            self.stop = stop.astype(dtype)
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


def multinomial(
    *,
    y: np.ndarray,
    weights: np.ndarray =None,
    dtype: Union[np.float32, np.float64] =np.float64,
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

    We assume that every :math:`y_{ik} \\in \\{0,1\\}` and
    for each fixed :math:`i`, 
    there is exactly one :math:`k` such that :math:`y_{ik} = 1`.

    .. note::
        The ``hessian()`` method computes :math:`2 \\mathrm{diag}(\\nabla^2 \\ell(\\eta))`
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
        Default is ``np.float64``.

    Returns
    -------
    glm
        Multinomial GLM object.

    See Also
    --------
    adelie.glm.GlmMultiBase64
    """
    dispatcher = {
        np.float64: core.glm.GlmMultinomial64,
        np.float32: core.glm.GlmMultinomial32,
    }

    core_base = dispatcher[dtype]

    class _multinomial(multiglm_base, core_base):
        def __init__(
            self,
        ):
            multiglm_base.__init__(self, y, weights, core_base, dtype)
            core_base.__init__(self, self.y, self.weights)

        def reweight(self, weights=None):
            weights = self.weights if weights is None else weights
            return multinomial(y=y, weights=weights, dtype=dtype)

    return _multinomial()


def poisson(
    *,
    y: np.ndarray,
    weights: np.ndarray =None,
    dtype: Union[np.float32, np.float64] =np.float64,
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
        Default is ``np.float64``.

    Returns
    -------
    glm
        Poisson GLM object.

    See Also
    --------
    adelie.glm.GlmBase64
    """
    dispatcher = {
        np.float64: core.glm.GlmPoisson64,
        np.float32: core.glm.GlmPoisson32,
    }

    core_base = dispatcher[dtype]

    class _poisson(glm_base, core_base):
        def __init__(
            self,
        ):
            glm_base.__init__(self, y, weights, core_base, dtype)
            core_base.__init__(self, self.y, self.weights)

        def reweight(self, weights=None):
            weights = self.weights if weights is None else weights
            return poisson(y=y, weights=weights, dtype=dtype)

    return _poisson()
