#include "fista.hpp"
#include "newton.hpp"
#include "bisect.hpp"
#include "objective.hpp"
#include "group_lasso.hpp"

PYBIND11_MODULE(pyglstudy_ext, m) {
    m.def("brent_bu", &brent_bu);

    m.def("newton_solver", &newton_solver);
    m.def("newton_brent_solver", &newton_brent_solver);
    m.def("newton_abs_solver", &newton_abs_solver);
    m.def("newton_solver_debug", &newton_solver_debug);

    m.def("ista_solver", &ista_solver);
    m.def("fista_solver", &fista_solver);
    m.def("fista_adares_solver", &fista_adares_solver);
    
    m.def("compute_h_min", &compute_h_min);
    m.def("compute_h_max", &compute_h_max);
    m.def("block_norm_objective", &block_norm_objective);
    
    m.def("group_lasso", &group_lasso);
}