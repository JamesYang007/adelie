#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <ghostbasil/optimization/group_basil.hpp>

namespace py = pybind11;
using namespace pybind11::literals; // to bring in the `_a` literal

static py::dict transform_data__(
    Eigen::Ref<Eigen::MatrixXd>& X,
    const Eigen::Ref<Eigen::VectorXi>& groups,
    const Eigen::Ref<Eigen::VectorXi>& group_sizes,
    size_t n_threads
)
{
    Eigen::VectorXd diag(X.cols());
    std::vector<Eigen::BDCSVD<Eigen::MatrixXd>> decomps;
    ghostbasil::group_lasso::transform_data(X, groups, group_sizes, n_threads, diag, decomps);
    py::dict d(
        "A_diag"_a=diag
    );
    return d;
}