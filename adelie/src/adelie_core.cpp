#include "decl.hpp"
#include <adelie_core/util/types.hpp>

namespace ad = adelie_core;

PYBIND11_MODULE(adelie_core, m) {

    py::bind_vector<std::vector<ad::util::rowmat_type<double>>>(m, "VectorMatrix64");
    py::bind_vector<std::vector<ad::util::rowmat_type<float>>>(m, "VectorMatrix32");

    auto m_bcd = m.def_submodule("bcd", "BCD submodule.");
    register_bcd(m_bcd);
    
    auto m_matrix = m.def_submodule("matrix", "Matrix submodule.");
    register_matrix(m_matrix);

    auto m_state = m.def_submodule("state", "State submodule.");
    register_state(m_state);

    auto m_solver = m.def_submodule("solver", "Grpnet submodule.");
    register_solver(m_solver);
}