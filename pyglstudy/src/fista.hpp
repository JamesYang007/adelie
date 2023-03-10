#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <fista.hpp>

namespace py = pybind11;
using namespace pybind11::literals; // to bring in the `_a` literal

py::dict ista_solver(
    Eigen::Ref<Eigen::VectorXd> L,
    Eigen::Ref<Eigen::VectorXd> v,
    double l1,
    double l2,
    double tol,
    size_t max_iters
)
{
    Eigen::VectorXd x(L.size());
    size_t iters = 0;
    glstudy::ista_solver(L, v, l1, l2, tol, max_iters, x, iters);
    py::dict d("beta"_a=x, "iters"_a=iters);
    return d;
}

py::dict fista_solver(
    Eigen::Ref<Eigen::VectorXd> L,
    Eigen::Ref<Eigen::VectorXd> v,
    double l1,
    double l2,
    double tol,
    size_t max_iters
)
{
    Eigen::VectorXd x(L.size());
    size_t iters = 0;
    glstudy::fista_solver(L, v, l1, l2, tol, max_iters, x, iters);
    py::dict d("beta"_a=x, "iters"_a=iters);
    return d;
}

py::dict fista_adares_solver(
    Eigen::Ref<Eigen::VectorXd> L,
    Eigen::Ref<Eigen::VectorXd> v,
    double l1,
    double l2,
    double tol,
    size_t max_iters
)
{
    Eigen::VectorXd x(L.size());
    size_t iters = 0;
    glstudy::fista_adares_solver(L, v, l1, l2, tol, max_iters, x, iters);
    py::dict d("beta"_a=x, "iters"_a=iters);
    return d;
}