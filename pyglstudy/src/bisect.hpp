#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <bisect.hpp>
#include <objective.hpp>

namespace py = pybind11;
using namespace pybind11::literals; // to bring in the `_a` literal

static py::dict brent_bu(
    Eigen::Ref<Eigen::VectorXd> L,
    Eigen::Ref<Eigen::VectorXd> v,
    double l1,
    double l2,
    double tol,
    size_t max_iters
)
{
    double x = 0;
    const Eigen::VectorXd buffer1 = L.array() + l2;
    const auto a = glstudy::compute_h_min(buffer1, v, l1);
    const auto b = std::get<0>(glstudy::compute_h_max(buffer1, v, l1));
    size_t iters = 0;
    const auto phi = [&](auto h) {
        return glstudy::block_norm_objective(h, buffer1, v, l1);
    };
    glstudy::brent(phi, tol, max_iters, a, b, x, iters);
    py::dict d("x"_a=x, "iters"_a=iters);
    return d;
}
