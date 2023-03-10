from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension 
import os

__version__ = "0.0.9"

ENVPATH = os.getenv("CONDA_PREFIX")

ext_modules = [
    Pybind11Extension(
        "pyglstudy.pyglstudy_ext",
        sorted(glob("pyglstudy/src/*.cpp")),  # Sort source files for reproducibility
        include_dirs=[
            'src/include',
            os.path.join(ENVPATH, 'include'),
            os.path.join(ENVPATH, 'include/eigen3'),
        ],
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