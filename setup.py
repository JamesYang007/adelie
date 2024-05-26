from glob import glob
from setuptools import setup, find_packages
from distutils.dir_util import copy_tree
from pybind11.setup_helpers import Pybind11Extension 
from pybind11.setup_helpers import ParallelCompile
import os
import pathlib
import platform
import shutil
import subprocess
import sysconfig


def run_cmd(cmd):
    try:
        output = subprocess.check_output(
            cmd.split(" "), stderr=subprocess.STDOUT
        ).decode()
    except subprocess.CalledProcessError as e:
        output = e.output.decode()
        raise RuntimeError(output)
    return output.rstrip()


# Optional multithreaded build
ParallelCompile("NPY_NUM_BUILD_JOBS").install()

__version__ = open("VERSION", "r").read().strip()

extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += [
    "-g0",
    "-Wall", 
    "-Wextra", 
    "-DNDEBUG", 
    "-O3",
]
include_dirs = [
    "adelie/src",
    "adelie/src/include",
    "adelie/src/third_party/eigen3",
]
libraries = []
library_dirs = []
runtime_library_dirs = []

system_name = platform.system()
if (system_name == "Darwin"):
    conda_path = run_cmd("conda info --base")
    conda_env_path = os.path.join(conda_path, "envs/adelie")

    # if user provides OpenMP install prefix (containing lib/ and include/)
    if "OPENMP_PREFIX" in os.environ and os.environ["OPENMP_PREFIX"] != "":
        omp_prefix = os.environ["OPENMP_PREFIX"]

    # else if conda environment is activated
    elif os.path.isdir(conda_env_path):
        omp_prefix = conda_env_path
    
    # otherwise check brew installation
    else:
        # check if OpenMP is installed
        no_omp_msg = (
            "OpenMP is not detected. "
            "MacOS users should install Homebrew and run 'brew install libomp' "
            "to install OpenMP. "
        )
        try:
            libomp_info = run_cmd("brew info libomp")
        except:
            raise RuntimeError(no_omp_msg)
        if "Not installed" in libomp_info:
            raise RuntimeError(no_omp_msg)

        # grab include and lib directory
        omp_prefix = run_cmd("brew --prefix libomp")

    omp_include = os.path.join(omp_prefix, "include")
    omp_lib = os.path.join(omp_prefix, "lib")

    # augment arguments
    include_dirs += [f"{omp_include}"]
    extra_compile_args += [
        "-Xpreprocessor",
        "-fopenmp",
    ]
    runtime_library_dirs += [f"{omp_lib}"]
    library_dirs += [f"{omp_lib}"]
    libraries += ['omp']
    
if (system_name == "Linux"):
    extra_compile_args += [
        "-fopenmp", 
        "-march=native",
    ]
    libraries += ['gomp']

ext_modules = [
    Pybind11Extension(
        "adelie.adelie_core",
        sorted(glob("adelie/src/*.cpp")),  # Sort source files for reproducibility
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        runtime_library_dirs=runtime_library_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        cxx_std=17,
    ),
]

setup(
    name='adelie', 
    version=__version__,
    description='A fast, flexible package for group elastic net.',
    long_description='',
    author='James Yang',
    author_email='jamesyang916@gmail.com',
    maintainer='James Yang',
    maintainer_email='jamesyang916@gmail.com',
    packages=find_packages(include=["adelie", "adelie.*"]), 
    package_data={
        "adelie": [
            "src/**/*.hpp", 
            "src/third_party/**/*",
            "adelie_core.cpython*",
        ],
    },
    ext_modules=ext_modules,
    zip_safe=False,
)