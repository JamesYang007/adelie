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
):
    """Creates a Gaussian GLM family object.

    The Gaussian GLM family specifies the log-partition function as:

    .. math::
        \\begin{align*}
            A(\\eta) = \\frac{\\eta^2}{2}
        \\end{align*}

    Parameters
    ----------
    dtype : Union[np.float32, np.float64], optional
        The underlying data type.
        Default is ``np.float64``.

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
            core_base.__init__(self)

        def sample(self, mu):
            return np.random.normal(mu)

    return _gaussian()


def binomial(
    *,
    dtype: Union[np.float32, np.float64] =np.float64,
):
    """Creates a Binomial GLM family object.

    The Binomial GLM family specifies the log-partition function as:

    .. math::
        \\begin{align*}
            A(\\eta) = \\log(1 + e^{\\eta})
        \\end{align*}


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


def poisson(
    *,
    dtype: Union[np.float32, np.float64] =np.float64,
):
    """Creates a Poisson GLM family object.

    The Poisson GLM family specifies the log-partition function as:

    .. math::
        \\begin{align*}
            A(\\eta) = e^{\eta}
        \\end{align*}


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
