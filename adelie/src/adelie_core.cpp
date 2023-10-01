#include "decl.hpp"

PYBIND11_MODULE(adelie_core, m) {
    auto m_bcd = m.def_submodule("bcd", "BCD submodule.");
    register_bcd(m_bcd);
    
    auto m_matrix = m.def_submodule("matrix", "Matrix submodule.");
    register_matrix(m_matrix);

    auto m_state = m.def_submodule("state", "State submodule.");
    register_state(m_state);

    auto m_grpnet = m.def_submodule("grpnet", "Grpnet submodule.");
    register_grpnet(m_grpnet);
}