from glob import glob
from setuptools import setup, find_namespace_packages
from pybind11.setup_helpers import Pybind11Extension 
from pybind11.setup_helpers import ParallelCompile
import os
import platform
import subprocess


# Hack to get __version__ from adelie/__init__.py
with open("adelie/__init__.py") as f:
    for line in f:
        if line.startswith("__version__ = "):
            __version__ = line.split('"')[1]


def run_cmd(cmd):
    try:
        output = subprocess.check_output(
            cmd.split(" "), stderr=subprocess.STDOUT
        ).decode()
    except subprocess.CalledProcessError as e:
        output = e.output.decode()
        raise RuntimeError(output)
    return output.rstrip()


def maybe_run_cmd(cmd):
    try:
        return run_cmd(cmd)
    except RuntimeError:
        return None


def append_unique_dirs(dirs, new_dirs):
    for new_dir in new_dirs:
        if new_dir and os.path.isdir(new_dir) and new_dir not in dirs:
            dirs.append(new_dir)


def has_eigen_headers(include_dir):
    return os.path.isfile(os.path.join(include_dir, "Eigen", "Core"))


def resolve_eigen_include_dirs(conda_prefix, system_name):
    include_candidates = []

    for env_var in ["EIGEN3_INCLUDE_DIR", "EIGEN_INCLUDE_DIR"]:
        include_dir = os.environ.get(env_var)
        if include_dir:
            include_candidates.append(include_dir)

    for env_var in ["EIGEN3_PREFIX", "EIGEN_PREFIX"]:
        prefix = os.environ.get(env_var)
        if prefix:
            include_candidates += [
                prefix,
                os.path.join(prefix, "include"),
                os.path.join(prefix, "include", "eigen3"),
            ]

    if not (conda_prefix is None):
        if system_name in ["Darwin", "Linux"]:
            conda_include_path = os.path.join(conda_prefix, "include")
        else:
            conda_include_path = os.path.join(conda_prefix, "Library", "include")
        include_candidates += [
            conda_include_path,
            os.path.join(conda_include_path, "eigen3"),
        ]

    if system_name == "Darwin":
        brew_eigen_prefix = maybe_run_cmd("brew --prefix eigen")
        if brew_eigen_prefix:
            include_candidates += [
                os.path.join(brew_eigen_prefix, "include"),
                os.path.join(brew_eigen_prefix, "include", "eigen3"),
            ]

    if system_name in ["Darwin", "Linux"]:
        include_candidates += [
            "/opt/homebrew/include/eigen3",
            "/usr/local/include/eigen3",
            "/usr/include/eigen3",
        ]

    eigen_include_dirs = []
    for include_dir in include_candidates:
        if has_eigen_headers(include_dir):
            append_unique_dirs(eigen_include_dirs, [include_dir])

    if not eigen_include_dirs:
        raise RuntimeError(
            "Eigen headers are not detected. "
            "Set EIGEN3_INCLUDE_DIR to the directory containing 'Eigen/Core', "
            "activate a conda environment containing eigen, "
            "or install Homebrew and run 'brew install eigen'."
        )

    return eigen_include_dirs


ParallelCompile("NPY_NUM_BUILD_JOBS").install()


if os.name == "posix":
    # GCC + Clang options to be extra stringent with warnings.
    extra_compile_args = [
        "-g0",
        "-Wall", 
        "-Wextra", 
        "-Werror",
        "-DNDEBUG", 
        "-O3",
    ]
elif os.name == "nt":
    extra_compile_args = [
        "/W3",
        "/WX",
        "/wd4566", # unicode not representable
        "/wd4244", # 'conversion' conversion from 'type1' to 'type2', possible loss of data
        "/wd4305", # 'conversion': truncation from 'type1' to 'type2'
        "/wd4267", # 'var' : conversion from 'size_t' to 'type', possible loss of data
        "/wd4849", # OpenMP 'clause' clause ignored in 'directive' directive
        "/wd4506", # no definition for inline function (I know what I'm doing Microsoft...)
        "/O2",
    ]
include_dirs = [
    os.path.join("adelie", "src"),
    os.path.join("adelie", "src", "include"),
    os.path.join("adelie", "src", "src"),
]
extra_link_args = []
libraries = []
library_dirs = []
runtime_library_dirs = []

# check if conda environment activated
if "CONDA_PREFIX" in os.environ:
    conda_prefix = os.environ["CONDA_PREFIX"]
# check if micromamba environment activated (CI)
elif "MAMBA_ROOT_PREFIX" in os.environ:
    conda_prefix = os.path.join(os.environ["MAMBA_ROOT_PREFIX"], "envs", "adelie")
else:
    conda_prefix = None

system_name = platform.system()

include_dirs += resolve_eigen_include_dirs(conda_prefix, system_name)

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
    extra_link_args += [
        "-framework",
        "Accelerate",
    ]
    runtime_library_dirs += [omp_lib]
    library_dirs += [omp_lib]
    libraries += ['omp']
    
elif system_name == "Linux":
    extra_compile_args += [
        "-fopenmp", 
        "-march=native",
    ]
    libraries += [
        "gomp",
    ]

else:
    extra_compile_args += [
        "/openmp",
    ]

ext_modules = [
    Pybind11Extension(
        "adelie.adelie_core",
        sorted(
            glob("adelie/src/src/constraint/*.cpp") +
            glob("adelie/src/src/glm/*.cpp") +
            glob("adelie/src/src/io/*.cpp") +
            glob("adelie/src/src/matrix/*.cpp") +
            glob("adelie/src/src/state/*.cpp") +
            glob("adelie/src/*.cpp")
        ),  # Sort source files for reproducibility
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        runtime_library_dirs=runtime_library_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        cxx_std=17,
    ),
]

# Include adelie and all submodules.
# This removes setuptools warning about missing namespace packages.
packages = ["adelie"] + [
    f"adelie.{submod}"
    for submod in find_namespace_packages("adelie")
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
    packages=packages,
    package_data={
        "adelie": [
            "src/**/*.hpp",
            "src/**/*.cpp",
        ],
    },
    ext_modules=ext_modules,
    zip_safe=False,
)
