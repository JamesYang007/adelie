#include "py_decl.hpp"
#include <adelie_core/configs.hpp>

namespace ad = adelie_core;

void configs(py::module_& m)
{
    using configs_t = ad::Configs;
    py::class_<configs_t>(m, "Configs")
        .def_readonly_static("hessian_min_def", &configs_t::hessian_min_def,
        "Default value for ``hessian_min``.")
        .def_readonly_static("pb_symbol_def", &configs_t::pb_symbol_def,
        "Default value for ``pb_symbol``.")
        .def_readonly_static("dbeta_tol_def", &configs_t::dbeta_tol_def,
        "Default value for ``dbeta_tol``.")
        .def_readonly_static("min_bytes_def", &configs_t::min_bytes_def,
        "Default value for ``min_bytes``.")
        .def_readonly_static("project_def", &configs_t::project_def,
        "Default value for ``project``.")
        .def_readonly_static("max_solver_value", &configs_t::max_solver_value_def,
        "Default value for ``max_solver_value``.")
        .def_readwrite_static("hessian_min", &configs_t::hessian_min,
        "The value at which the diagonal of the hessian is clipped from below. "
        "This ensures that the proximal Newton step is well-defined. "
        "It must be a positive value."
        )
        .def_readwrite_static("pb_symbol", &configs_t::pb_symbol,
        "The progress bar symbol.")
        .def_readwrite_static("dbeta_tol", &configs_t::dbeta_tol, R"delimiter(
        Tolerance level corresponding to :math:`\epsilon` when comparing
        :math:`\|\Delta \beta\|_2 \leq \epsilon` where
        :math:`\Delta \beta` is the change in a group of coefficients after its coordinate update.
        If the change is small, then a sufficiently large tolerance level may save computation.
        However, if it is too large, the solver may not converge properly.
        )delimiter")
        .def_readwrite_static("min_bytes", &configs_t::min_bytes, R"delimiter(
        Minimum number of bytes needed to be processed sequentially before
        parallelization is used instead.
        The smaller the value, the sooner parallelization is used.
        )delimiter")
        .def_readwrite_static("project", &configs_t::project, R"delimiter(
        If ``True``, the solver will perform a final projection for each group of coefficients 
        to their respective feasible sets.
        )delimiter")
        .def_readwrite_static("max_solver_value", &configs_t::max_solver_value,
        "Maximum value to be interpreted as ``INF``. "
        "It is sometimes necessary to clip infinite values to a finite number. ")
        ;
}

void register_configs(py::module_& m)
{
    configs(m);
}