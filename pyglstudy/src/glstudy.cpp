#include "fista.hpp"
#include "newton.hpp"
#include "bisect.hpp"

PYBIND11_MODULE(pyglstudy_ext, m) {
    m.def("brent_bu", &brent_bu);

    m.def("newton_solver", &newton_solver);
    m.def("newton_brent_solver", &newton_brent_solver);
    m.def("newton_abs_solver", &newton_abs_solver);
    m.def("newton_solver_debug", &newton_solver_debug);

    m.def("ista_solver", &ista_solver);
    m.def("fista_solver", &fista_solver);
    m.def("fista_adares_solver", &fista_adares_solver);
}