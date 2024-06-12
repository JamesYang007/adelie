#pragma once
#if defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#pragma warning( push, 1 )
#endif
#include <pybind11/pybind11.h>
// Ignore all warnings for Eigen on GCC and Clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall" 
#pragma GCC diagnostic ignored "-Wextra" 
#include <pybind11/eigen.h>
#pragma GCC diagnostic pop
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#if defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#pragma warning( pop )
#endif
#include <adelie_core/util/types.hpp>

PYBIND11_MAKE_OPAQUE(std::vector<adelie_core::util::rowmat_type<double>>);
PYBIND11_MAKE_OPAQUE(std::vector<adelie_core::util::rowmat_type<float>>);

namespace py = pybind11;

void register_bcd(py::module_&);
void register_configs(py::module_&);
void register_matrix(py::module_&);
void register_matrix_utils(py::module_&);
void register_optimization(py::module_&);
void register_state(py::module_&);
void register_solver(py::module_&);
void register_io(py::module_&);
void register_glm(py::module_&);