# Only happy guins may read this code.
# Examples of happy guins are:
#   - Anav Guin (a.k.a. Kumar)
#   - Leda Guin
#   - Ginnie Guin

__version__ = "1.1.35"

try:
    # See setup.py where it is injected into __builtins__.
    # mypy error: Cannot determine type of `__ADELIE_SETUP__`
    __ADELIE_SETUP__ # type: ignore
except NameError:
    __ADELIE_SETUP__ = False

if __ADELIE_SETUP__:
    # Partial import of the package for build process
    import sys
    sys.stderr.write("Partial import of adelie during the build process.\n")

else:
    # Set environment flags before loading adelie_core
    import os

    # ---------------------------------------------------------------
    # OMP Flags
    # ---------------------------------------------------------------
    # Set thread block time (polling) to be sufficiently large.
    # It is very costly when OMP puts threads to sleep.
    os.environ.setdefault("KMP_BLOCKTIME", "1") # in ms
    os.environ.setdefault("GOMP_SPINCOUNT", "1000000")

    # Allow duplicate libs by default.
    # See https://github.com/scikit-learn/scikit-learn/blob/09781c540077a7f1f4f2392c9287e08e479c4f29/sklearn/__init__.py#L53
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

    # Bind each thread to a single core to localize memory.
    os.environ.setdefault("OMP_PROC_BIND", "TRUE")

    from . import adelie_core
    from . import bcd
    from . import configs
    from . import constraint
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
