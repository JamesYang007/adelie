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
