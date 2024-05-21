# Only happy guins may read this code.
# Examples of happy guins are:
#   - Anav Guin (a.k.a. Kumar)
#   - Leda Guin
#   - Ginnie Guin

# Set environment flags before loading adelie_core
import os

# ---------------------------------------------------------------
# OMP Flags
# ---------------------------------------------------------------
# Set thread block time (polling) to be sufficiently large.
# It is very costly when OMP puts threads to sleep.
if not ("KMP_BLOCKTIME" in os.environ):
    os.environ["KMP_BLOCKTIME"] = "1ms"
if not ("GOMP_SPINCOUNT" in os.environ):
    os.environ["GOMP_SPINCOUNT"] = "1000000"

# Bind each thread to a single core to localize memory.
if not ("OMP_PROC_BIND" in os.environ):
    os.environ["OMP_PROC_BIND"] = "TRUE"

from . import adelie_core
from . import bcd
from . import configs
from . import cv
from . import data
from . import diagnostic
from . import glm
from . import io
from . import logger
from . import matrix
from . import optimization
from . import solver
from . import state
from .cv import (
    cv_grpnet,
)
from .solver import (
    gaussian_cov, 
    grpnet,
)