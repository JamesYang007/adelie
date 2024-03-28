#include "decl.hpp"
#include <adelie_core/configs.hpp>

namespace ad = adelie_core;

void configs(py::module_& m)
{
    using configs_t = ad::Configs;
    py::class_<configs_t>(m, "Configs")
        .def_readonly_static("hessian_min_def", &configs_t::hessian_min_def)
        .def_readonly_static("pb_symbol_def", &configs_t::pb_symbol_def)
        .def_readonly_static("dbeta_tol_def", &configs_t::dbeta_tol_def)
        .def_readwrite_static("hessian_min", &configs_t::hessian_min)
        .def_readwrite_static("pb_symbol", &configs_t::pb_symbol)
        .def_readwrite_static("dbeta_tol", &configs_t::dbeta_tol)
        ;
}

void register_configs(py::module_& m)
{
    configs(m);
}