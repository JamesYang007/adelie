from glob import glob
from setuptools import setup, find_packages
from distutils.dir_util import copy_tree
from pybind11.setup_helpers import Pybind11Extension 
from pybind11.setup_helpers import ParallelCompile
import sysconfig
import os
import platform
import subprocess


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

ENVPATH = os.getenv("CONDA_PREFIX")

extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += [
    "-Wall", 
    "-Wextra", 
    "-DNDEBUG", 
    "-O3",
]
libraries = []
extra_link_args = []

system_name = platform.system()
if (system_name == "Darwin"):
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
    omp_prefix = run_cmd("brew --prefix libomp")
    omp_include = os.path.join(omp_prefix, "include")
    omp_lib = os.path.join(omp_prefix, "lib")
    extra_compile_args += [
        f"-I{omp_include}",
        "-Xclang",
        "-fopenmp",
    ]
    extra_link_args += [f'-L{omp_lib}']
    libraries = ['omp']
    
if (system_name == "Linux"):
    extra_compile_args += ["-fopenmp"]
    libraries = ['gomp']

ext_modules = [
    Pybind11Extension(
        "adelie.adelie_core",
        sorted(glob("adelie/src/*.cpp")),  # Sort source files for reproducibility
        include_dirs=[
            "adelie/src",
            "adelie/src/include",
            "adelie/src/third_party/eigen3",
            os.path.join(ENVPATH, 'include'),
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=libraries,
        cxx_std=17, # TODO: changed from 14 to 17 just because of tqdm
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