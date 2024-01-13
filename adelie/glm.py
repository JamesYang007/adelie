from typing import Union
from . import adelie_core as core
from .adelie_core.glm import (
    GlmBase64,
    GlmBase32,
)
import numpy as np


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

    class _gaussian(core_base):
        def __init__(
            self,
        ):
            core_base.__init__(self)

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

    class _binomial(core_base):
        def __init__(
            self,
        ):
            core_base.__init__(self)

    return _binomial()
