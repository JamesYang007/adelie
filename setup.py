from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension 
from pybind11.setup_helpers import ParallelCompile
import sysconfig
import os
import platform

# Optional multithreaded build
ParallelCompile("NPY_NUM_BUILD_JOBS").install()

__version__ = "1.1.0"

ENVPATH = os.getenv("CONDA_PREFIX")

extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += ["-Wall", "-Wextra", "-DNDEBUG", "-O3"]
libraries = []
extra_link_args = []

system_name = platform.system()
if (system_name == "Darwin"):
    extra_compile_args += [
        "-I/opt/homebrew/opt/libomp/include", 
        "-Xclang",
        "-fopenmp",
    ]
    extra_link_args += ['-L/opt/homebrew/opt/libomp/lib']
    libraries = ['omp']
    
if (system_name == "Linux"):
    extra_compile_args += ["-fopenmp"]
    libraries = ['gomp']

ext_modules = [
    Pybind11Extension(
        "adelie.adelie_core",
        sorted(glob("adelie/src/*.cpp")),  # Sort source files for reproducibility
        define_macros = [
            ('EIGEN_MATRIXBASE_PLUGIN', '\"adelie_core/util/eigen/matrixbase_plugin.hpp\"'),
        ],
        include_dirs=[
            'adelie_core/include',
            os.path.join(ENVPATH, 'include'),
            os.path.join(ENVPATH, 'include/eigen3'),
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=libraries,
        cxx_std=14,
    ),
]

setup(
    name='adelie', 
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