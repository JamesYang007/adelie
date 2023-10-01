#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

void register_bcd(py::module_&);
void register_matrix(py::module_&);
void register_state(py::module_&);
void register_grpnet(py::module_&);