#include "decl.hpp"
//#include "group_basil.hpp"

PYBIND11_MODULE(adelie_core, m) {
    auto m_bcd = m.def_submodule("bcd", "BCD submodule.");
    register_bcd(m_bcd);
    
    auto m_matrix = m.def_submodule("matrix", "Matrix submodule.");
    register_matrix(m_matrix);

    //register_group_elnet_state<
    //    Eigen::Ref<const gg::util::rowmat_type<double>>
    //>(m, "GroupElnetStateDense");
    //m.def("group_elnet_objective", &group_elnet_objective);
    //m.def("group_elnet_naive_dense", &group_elnet_naive_dense);

    //m.def("group_elnet__", &group_elnet__);
    //m.def("group_elnet_data__", &group_elnet_data__);
    //m.def("group_elnet_data_newton__", &group_elnet_data_newton__);
    
    //m.def("transform_data", &transform_data__);
    //m.def("group_basil_cov__", &group_basil_cov__);
    //m.def("group_basil_naive__", &group_basil_naive__);
}