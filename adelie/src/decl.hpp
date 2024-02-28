#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/iostream.h>
#include <adelie_core/util/types.hpp>

PYBIND11_MAKE_OPAQUE(std::vector<adelie_core::util::rowmat_type<double>>);
PYBIND11_MAKE_OPAQUE(std::vector<adelie_core::util::rowmat_type<float>>);

namespace py = pybind11;

void register_bcd(py::module_&);
void register_configs(py::module_&);
void register_matrix(py::module_&);
void register_optimization(py::module_&);
void register_state(py::module_&);
void register_solver(py::module_&);
void register_io(py::module_&);
void register_glm(py::module_&);