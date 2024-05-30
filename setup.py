from glob import glob
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension 
from pybind11.setup_helpers import ParallelCompile
import os
import platform
import subprocess
import sysconfig


# Partial import of adelie
import builtins
builtins.__ADELIE_SETUP__ = True
from adelie import __version__


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
]
libraries = []
library_dirs = []
runtime_library_dirs = []

# check if conda environment activated
if "CONDA_PREFIX" in os.environ:
    conda_prefix = os.environ["CONDA_PREFIX"]
# check if micromamba environment activated (CI)
elif "MAMBA_ROOT_PREFIX" in os.environ:
    conda_prefix = os.path.join(os.environ["MAMBA_ROOT_PREFIX"], "envs/adelie")
else:
    conda_prefix = None

# add include and include/eigen3
if not (conda_prefix is None):
    conda_include_path = os.path.join(conda_prefix, "include")
    eigen_include_path = os.path.join(conda_include_path, "eigen3")
    include_dirs += [
        conda_include_path,
        eigen_include_path,
    ]

system_name = platform.system()
if system_name == "Darwin":
    # if user provides OpenMP install prefix (containing include/ and lib/)
    if "OPENMP_PREFIX" in os.environ and os.environ["OPENMP_PREFIX"] != "":
        omp_prefix = os.environ["OPENMP_PREFIX"]

    # else if conda environment is activated
    elif not (conda_prefix is None):
        omp_prefix = conda_prefix
    
    # otherwise check brew installation
    else:
        # check if OpenMP is installed
        no_omp_msg = (
            "OpenMP is not detected. "
            "MacOS users should either provide the OpenMP path via the environment variable OPENMP_PREFIX, "
            "create a conda environment containing llvm-openmp, "
            "or install Homebrew and run 'brew install libomp'. "
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
    include_dirs += [omp_include]
    extra_compile_args += [
        "-Xpreprocessor",
        "-fopenmp",
    ]
    runtime_library_dirs += [omp_lib]
    library_dirs += [omp_lib]
    libraries += ['omp']
    
elif system_name == "Linux":
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
            "src/**/*.cpp", 
            "adelie_core.cpython*",
        ],
    },
    ext_modules=ext_modules,
    zip_safe=False,
)