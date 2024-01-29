from typing import Union
from . import adelie_core as core
from .adelie_core.glm import (
    GlmBase64,
    GlmBase32,
)
import numpy as np


class glm_base:
    """Base wrapper GLM class.

    All Python wrapper classes for core GLM classes must inherit from this class.
    The purpose of this class is to expose extra member interface
    that are more easily written in Python (and not speed-critical).
    """

    def sample(self, mu: np.ndarray):
        """Samples from the GLM distribution.

        This function samples the response from the distribution determined by the GLM.
        
        Parameters
        ----------
        mu : (n,) np.ndarray
            The mean parameter of the GLM.

        Returns
        -------
        y : (n,) np.ndarray
            Response samples.
        """
        pass


def gaussian(
    *,
    dtype: Union[np.float32, np.float64] =np.float64,
    opt: bool =True,
):
    """Creates a Gaussian GLM family object.

    The Gaussian GLM family specifies the loss function as:

    .. math::
        \\begin{align*}
            \\sum\\limits_{i=1}^n w_i \\left(
                -y_i \\eta_i + \\frac{\\eta_i^2}{2}
            \\right) 
        \\end{align*}

    Parameters
    ----------
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

    class _gaussian(core_base, glm_base):
        def __init__(
            self,
        ):
            if opt:
                self._type = "gaussian"
            else:
                self._type = "gaussian-irls"
            self._is_multi = False
            core_base.__init__(self)

        def sample(self, mu):
            return np.random.normal(mu)

    return _gaussian()


def multigaussian(
    *,
    dtype: Union[np.float32, np.float64] =np.float64,
    opt: bool =True,
):
    """Creates a Multi-Response Gaussian GLM family object.

    The Multi-Response Gaussian GLM family specifies the loss function as:

    .. math::
        \\begin{align*}
            \\sum\\limits_{i=1}^n 
            \\sum\\limits_{k=1}^K 
            w_{ik} \\left(
                -y_{ik} \\eta_{ik} + \\frac{\\eta_{ik}^2}{2}
            \\right)
        \\end{align*}

    Implementation-wise, it is no different from ``adelie.glm.gaussian``,
    however, it allows ``adelie.grpnet`` to interpret the inputs such as ``X``, ``y``, and ``w``
    differently so that they are reshaped properly.

    Parameters
    ----------
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
        Multi-Response Gaussian GLM object.

    See Also
    --------
    adelie.glm.GlmBase64
    """
    out = gaussian(dtype=dtype, opt=opt)
    out._is_multi = True
    return out


def binomial(
    *,
    dtype: Union[np.float32, np.float64] =np.float64,
):
    """Creates a Binomial GLM family object.

    The Binomial GLM family specifies the loss function as:

    .. math::
        \\begin{align*}
            \\sum\\limits_{i=1}^n w_i \\left(
                -y_i \\eta_i + \\log(1 + e^{\\eta_i})
            \\right)
        \\end{align*}

    We assume that :math:`y_i \\in \\{0,1\\}`.

    Parameters
    ----------
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

    class _binomial(core_base, glm_base):
        def __init__(
            self,
        ):
            self._type = "binomial"
            self._is_multi = False
            core_base.__init__(self)

        def sample(self, mu):
            return np.random.binomial(1, mu)

    return _binomial()


def multinomial(
    K: int,
    *,
    dtype: Union[np.float32, np.float64] =np.float64,
):
    """Creates a Multinomial GLM family object.

    The Multinomial GLM family specifies the loss function as:

    .. math::
        \\begin{align*}
            \\sum\\limits_{i=1}^n 
            w_i
            \\left(
            -\\sum\\limits_{k=1}^{K} y_{ik} \\eta_{ik} 
            + \\log\\left(
                1 + \\sum\\limits_{\\ell=1}^{K} e^{\\eta_{i\\ell}}
            \\right)
            \\right)
        \\end{align*}

    We assume that every :math:`y_{ik} \\in \\{0,1\\}` and
    for each fixed :math:`i`, 
    there is at most one :math:`k` such that :math:`y_{ik} = 1`.
    Here, :math:`K+1` is the total number of classes,
    however only the first :math:`K` classes are needed since 
    the probability estimate for the last class is fully determined by them.

    .. note::
        We may think of the weights as :math:`w_{ik} = w_i`
        with log-partition function as 
        
        .. math::
            \\begin{align*}
                A(\\eta) 
                &= 
                \\sum\\limits_{i=1}^n 
                \\sum\\limits_{k=1}^{K} 
                w_{ik} A_{ik}(\\eta)
                \\\\
                A_{ik}(\\eta)
                &=
                \\frac{1}{K}
                \\log\\left(1 + \\sum\\limits_{\\ell=1}^{K} e^{\\eta_{i\\ell}} \\right)
            \\end{align*}

        Hence, all weights will be of length :math:`nK` and it can be assumed that
        for each :math:`i`, :math:`w_{ik}` is identical across :math:`k`.
    
    Parameters
    ----------
    K : int
        Number of effective classes.
    dtype : Union[np.float32, np.float64], optional
        The underlying data type.
        Default is ``np.float64``.

    Returns
    -------
    glm
        Multinomial GLM object.

    See Also
    --------
    adelie.glm.GlmBase64
    """
    dispatcher = {
        np.float64: core.glm.GlmMultinomial64,
        np.float32: core.glm.GlmMultinomial32,
    }

    core_base = dispatcher[dtype]

    class _multinomial(core_base, glm_base):
        def __init__(
            self,
            K,
        ):
            self._type = "multinomial"
            self._is_multi = True
            core_base.__init__(self, K)

        def sample(self, mu):
            return np.random.multinomial(1, mu)

    return _multinomial(K)


def poisson(
    *,
    dtype: Union[np.float32, np.float64] =np.float64,
):
    """Creates a Poisson GLM family object.

    The Poisson GLM family specifies the loss function as:

    .. math::
        \\begin{align*}
            \\sum\\limits_{i=1}^n w_i \\left(
                -y_i \\eta_i + e^{\\eta_i}
            \\right) 
        \\end{align*}

    We assume that :math:`y_i \\in \\mathbb{N}_0`.

    Parameters
    ----------
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

    class _poisson(core_base, glm_base):
        def __init__(
            self,
        ):
            self._type = "poisson"
            self._is_multi = False
            core_base.__init__(self)

        def sample(self, mu):
            return np.random.poisson(mu)

    return _poisson()
