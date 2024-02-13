from typing import Union
from . import adelie_core as core
from .adelie_core.glm import (
    GlmBase64,
    GlmBase32,
    GlmMultiBase64,
    GlmMultiBase32,
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
            \\ell(\\eta)
            =
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
            self.opt = opt
            core_base.__init__(self)

        def sample(self, mu):
            return np.random.normal(mu)

    return _gaussian()


def multigaussian(
    *,
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

    class _multigaussian(core_base, glm_base):
        def __init__(
            self,
        ):
            self.opt = opt
            core_base.__init__(self)

        def sample(self, mu):
            return np.random.normal(mu)

    return _multigaussian()


def binomial(
    *,
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
            core_base.__init__(self)

        def sample(self, mu):
            return np.random.binomial(1, mu)

    return _binomial()


def multinomial(
    *,
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
    there is excatly one :math:`k` such that :math:`y_{ik} = 1`.

    Parameters
    ----------
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

    class _multinomial(core_base, glm_base):
        def __init__(
            self,
        ):
            core_base.__init__(self)

        def sample(self, mu):
            return np.random.multinomial(1, mu)

    return _multinomial()


def poisson(
    *,
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
            core_base.__init__(self)

        def sample(self, mu):
            return np.random.poisson(mu)

    return _poisson()
