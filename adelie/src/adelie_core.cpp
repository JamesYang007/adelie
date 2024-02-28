#include "decl.hpp"
#include <adelie_core/util/types.hpp>

namespace ad = adelie_core;

PYBIND11_MODULE(adelie_core, m) {

    py::bind_vector<std::vector<ad::util::rowmat_type<double>>>(m, "VectorMatrix64");
    py::bind_vector<std::vector<ad::util::rowmat_type<float>>>(m, "VectorMatrix32");

    register_configs(m);

    auto m_bcd = m.def_submodule("bcd", "BCD submodule.");
    register_bcd(m_bcd);

    auto m_io = m.def_submodule("io", "IO submodule.");
    register_io(m_io);

    auto m_glm = m.def_submodule("glm", "GLM submodule.");
    register_glm(m_glm);

    auto m_matrix = m.def_submodule("matrix", "Matrix submodule.");
    register_matrix(m_matrix);

    auto m_state = m.def_submodule("state", "State submodule.");
    register_state(m_state);

    auto m_solver = m.def_submodule("solver", "Grpnet submodule.");
    register_solver(m_solver);

    auto m_optimization = m.def_submodule("optimization", "Optimization submodule.");
    register_optimization(m_optimization);
}