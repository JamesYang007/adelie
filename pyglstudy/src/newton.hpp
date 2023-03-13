#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <newton.hpp>

namespace py = pybind11;
using namespace pybind11::literals; // to bring in the `_a` literal

static py::dict newton_solver(
    Eigen::Ref<Eigen::VectorXd> L,
    Eigen::Ref<Eigen::VectorXd> v,
    double l1,
    double l2,
    double tol,
    size_t max_iters
)
{
    Eigen::VectorXd x(L.size());
    Eigen::VectorXd buffer1(L.size());
    Eigen::VectorXd buffer2(L.size());
    size_t iters = 0;
    glstudy::newton_solver(L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2);

    py::dict d("beta"_a=x, "iters"_a=iters);
    return d;
}

static py::dict newton_brent_solver(
    Eigen::Ref<Eigen::VectorXd> L,
    Eigen::Ref<Eigen::VectorXd> v,
    double l1,
    double l2,
    double tol,
    size_t max_iters
)
{
    Eigen::VectorXd x(L.size());
    Eigen::VectorXd buffer1(L.size());
    Eigen::VectorXd buffer2(L.size());
    size_t iters = 0;
    glstudy::newton_brent_solver(L, v, l1, l2, tol, tol, max_iters, x, iters, buffer1, buffer2);

    py::dict d("beta"_a=x, "iters"_a=iters);
    return d;
}

static py::dict newton_abs_solver(
    Eigen::Ref<Eigen::VectorXd> L,
    Eigen::Ref<Eigen::VectorXd> v,
    double l1,
    double l2,
    double tol,
    size_t max_iters
)
{
    Eigen::VectorXd x(L.size());
    Eigen::VectorXd buffer1(L.size());
    Eigen::VectorXd buffer2(L.size());
    size_t iters = 0;
    glstudy::newton_abs_solver(L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2);

    py::dict d("beta"_a=x, "iters"_a=iters);
    return d;
}

static py::dict newton_solver_debug(
    Eigen::Ref<Eigen::VectorXd> L,
    Eigen::Ref<Eigen::VectorXd> v,
    double l1,
    double l2,
    double tol,
    size_t max_iters,
    bool smart_init
)
{
    double h_min, h_max;
    Eigen::VectorXd x(L.size());
    Eigen::VectorXd buffer1(L.size());
    Eigen::VectorXd buffer2(L.size());
    std::vector<double> iters;
    iters.reserve(L.size());
    std::vector<double> smart_iters;
    smart_iters.reserve(L.size());
    glstudy::newton_solver_debug(
        L, v, l1, l2, tol, max_iters, smart_init, 
        h_min, h_max, x, iters, smart_iters, buffer1, buffer2
    );
    
    py::dict d(
        "beta"_a=x,
        "h_min"_a=h_min,
        "h_max"_a=h_max,
        "iters"_a=iters,
        "smart_iters"_a=smart_iters
    );
    return d;
}
