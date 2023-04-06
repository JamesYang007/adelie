from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension 
import sysconfig
import os

__version__ = "0.0.9"

ENVPATH = os.getenv("CONDA_PREFIX")

extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += ["-Wall", "-Wextra", "-DNDEBUG", "-O3", "-fopenmp"]

ext_modules = [
    Pybind11Extension(
        "pyglstudy.pyglstudy_ext",
        sorted(glob("pyglstudy/src/*.cpp")),  # Sort source files for reproducibility
        define_macros = [
            ('EIGEN_MATRIXBASE_PLUGIN', '\"ghostbasil/util/eigen/matrixbase_plugin.hpp\"'),
        ],
        include_dirs=[
            'src/include',
            os.path.join(ENVPATH, 'include'),
            os.path.join(ENVPATH, 'include/eigen3'),
        ],
        extra_compile_args=extra_compile_args,
        libraries=['gomp'],
        cxx_std=14,
    ),
]

setup(
    name='pyglstudy', 
    version=__version__,
    description='A comprehensive test-bed library for group lasso.',
    long_description='',
    author='James Yang',
    author_email='jamesyang916@gmail.com',
    maintainer='James Yang',
    maintainer_email='jamesyang916@gmail.com',
    ext_modules=ext_modules,
    zip_safe=False,
)