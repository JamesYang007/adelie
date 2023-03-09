// cppimport
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <newton.hpp>
#include <fista.hpp>

namespace py = pybind11;
using namespace pybind11::literals; // to bring in the `_a` literal

py::dict newton_solver(
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

py::dict newton_abs_solver(
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

PYBIND11_MODULE(glstudy, m) {
    m.def("newton_solver", &newton_solver);
    m.def("newton_abs_solver", &newton_abs_solver);
    m.def("ista_solver", &ista_solver);
    m.def("fista_solver", &fista_solver);
    m.def("fista_adares_solver", &fista_adares_solver);
}
/*
<%
setup_pybind11(cfg)
cfg['include_dirs'] += [
    '/opt/homebrew/include/eigen3',
    '../glstudy/inst/include',
]
%>
*/