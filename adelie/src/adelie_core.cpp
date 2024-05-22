#include "decl.hpp"
#include <adelie_core/util/types.hpp>

namespace ad = adelie_core;

PYBIND11_MODULE(adelie_core, m) {

    py::bind_vector<std::vector<ad::util::rowmat_type<double>>>(m, "VectorMatrix64");
    py::bind_vector<std::vector<ad::util::rowmat_type<float>>>(m, "VectorMatrix32");

    auto m_bcd = m.def_submodule("bcd", "BCD submodule.");
    register_bcd(m_bcd);

    auto m_configs = m.def_submodule("configs", "Configurations submodule.");
    register_configs(m_configs);

    auto m_glm = m.def_submodule("glm", "GLM submodule.");
    register_glm(m_glm);

    auto m_io = m.def_submodule("io", "IO submodule.");
    register_io(m_io);

    auto m_matrix = m.def_submodule("matrix", "Matrix submodule.");
    register_matrix(m_matrix);

    auto m_matrix_utils = m_matrix.def_submodule("utils", "Matrix utility submodule.");
    register_matrix_utils(m_matrix_utils);

    auto m_optimization = m.def_submodule("optimization", "Optimization submodule.");
    register_optimization(m_optimization);

    auto m_solver = m.def_submodule("solver", "Grpnet submodule.");
    register_solver(m_solver);

    auto m_state = m.def_submodule("state", "State submodule.");
    register_state(m_state);
}