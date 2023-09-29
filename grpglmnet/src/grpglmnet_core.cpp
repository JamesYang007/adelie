#include <pybind11/stl_bind.h>
#include "group_elnet.hpp"
//#include "group_basil.hpp"

PYBIND11_MODULE(grpglmnet_core, m) {
    m.def("brent_solver", &brent_solver);
    m.def("newton_solver", &newton_solver);
    m.def("newton_brent_solver", &newton_brent_solver);
    m.def("newton_abs_solver", &newton_abs_solver);
    m.def("newton_abs_debug_solver", &newton_abs_debug_solver);
    m.def("ista_solver", &ista_solver);
    m.def("fista_solver", &fista_solver);
    m.def("fista_adares_solver", &fista_adares_solver);
    m.def("bcd_root_lower_bound", &bcd_root_lower_bound);
    m.def("bcd_root_upper_bound", &bcd_root_upper_bound);
    m.def("bcd_root_function", &bcd_root_function);
    register_group_elnet_state<
        Eigen::Ref<const gg::util::rowmat_type<double>>
    >(m, "GroupElnetStateDense");
    m.def("group_elnet_objective", &group_elnet_objective);
    m.def("group_elnet_naive_dense", &group_elnet_naive_dense);
    //m.def("group_elnet__", &group_elnet__);
    //m.def("group_elnet_data__", &group_elnet_data__);
    //m.def("group_elnet_data_newton__", &group_elnet_data_newton__);
    
    //m.def("transform_data", &transform_data__);
    //m.def("group_basil_cov__", &group_basil_cov__);
    //m.def("group_basil_naive__", &group_basil_naive__);
}