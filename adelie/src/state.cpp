#include "decl.hpp"
#include <adelie_core/matrix/matrix_base.hpp>
#include <adelie_core/state/pin_naive.hpp>

namespace py = pybind11;
namespace ad = adelie_core;

template <class MatrixType>
void pin_naive(py::module_& m, const char* name)
{
    using matrix_t = MatrixType;
    using state_t = ad::state::PinNaive<matrix_t>;
    using index_t = typename state_t::index_t;
    using value_t = typename state_t::value_t;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_bool_t = typename state_t::vec_bool_t;
    using dyn_vec_index_t = typename state_t::dyn_vec_index_t;
    using dyn_vec_value_t = typename state_t::dyn_vec_value_t;
    using dyn_vec_vec_value_t = typename state_t::dyn_vec_vec_value_t;
    using dyn_vec_sp_vec_t = typename state_t::dyn_vec_sp_vec_t;

    py::class_<state_t>(m, name)
        .def(py::init<
            const matrix_t&,
            const Eigen::Ref<const vec_index_t>&, 
            const Eigen::Ref<const vec_index_t>&,
            value_t, 
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_index_t>&, 
            const Eigen::Ref<const vec_index_t>&,
            const Eigen::Ref<const vec_index_t>&,
            const Eigen::Ref<const vec_index_t>&, 
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&, 
            size_t,
            value_t,
            value_t,
            value_t,
            value_t,
            size_t,
            value_t,
            Eigen::Ref<vec_value_t>,
            Eigen::Ref<vec_value_t>, 
            Eigen::Ref<vec_value_t>,
            dyn_vec_index_t,
            dyn_vec_index_t,
            dyn_vec_index_t,
            dyn_vec_index_t,
            dyn_vec_index_t,
            Eigen::Ref<vec_bool_t>,
            dyn_vec_sp_vec_t, 
            dyn_vec_value_t,
            dyn_vec_vec_value_t 
        >(),
            py::arg("X"),
            py::arg("groups").noconvert(),
            py::arg("group_sizes").noconvert(),
            py::arg("alpha"),
            py::arg("penalty").noconvert(),
            py::arg("strong_set").noconvert(),
            py::arg("strong_g1").noconvert(),
            py::arg("strong_g2").noconvert(),
            py::arg("strong_begins").noconvert(),
            py::arg("strong_var").noconvert(),
            py::arg("lmdas").noconvert(),
            py::arg("max_iters"),
            py::arg("tol"),
            py::arg("rsq_slope_tol"),
            py::arg("rsq_curv_tol"),
            py::arg("newton_tol"),
            py::arg("newton_max_iters"),
            py::arg("rsq"),
            py::arg("resid").noconvert(),
            py::arg("strong_beta").noconvert(),
            py::arg("strong_grad").noconvert(),
            py::arg("active_set"),
            py::arg("active_g1"),
            py::arg("active_g2"),
            py::arg("active_begins"),
            py::arg("active_order"),
            py::arg("is_active").noconvert(),
            py::arg("betas"),
            py::arg("rsqs"),
            py::arg("resids")
        )
        .def_readonly("X", &state_t::X)
        .def_readonly("groups", &state_t::groups)
        .def_readonly("group_sizes", &state_t::group_sizes)
        .def_readonly("alpha", &state_t::alpha)
        .def_readonly("penalty", &state_t::penalty)
        .def_readonly("strong_set", &state_t::strong_set)
        .def_readonly("strong_g1", &state_t::strong_g1)
        .def_readonly("strong_g2", &state_t::strong_g2)
        .def_readonly("strong_begins", &state_t::strong_begins)
        .def_readonly("strong_var", &state_t::strong_var)
        .def_readonly("lmdas", &state_t::lmdas)
        .def_readonly("max_iters", &state_t::max_iters)
        .def_readonly("tol", &state_t::tol)
        .def_readonly("rsq_slope_tol", &state_t::rsq_slope_tol)
        .def_readonly("rsq_curv_tol", &state_t::rsq_curv_tol)
        .def_readonly("newton_tol", &state_t::newton_tol)
        .def_readonly("newton_max_iters", &state_t::newton_max_iters)
        .def_readonly("rsq", &state_t::rsq)
        .def_readonly("resid", &state_t::resid)
        .def_readonly("strong_beta", &state_t::strong_beta)
        .def_readonly("strong_grad", &state_t::strong_grad)
        .def_property_readonly("active_set", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<index_t>>(
                s.active_set.data(),
                s.active_set.size()
            );
        })
        .def_property_readonly("active_g1", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<index_t>>(
                s.active_g1.data(),
                s.active_g1.size()
            );
        })
        .def_property_readonly("active_g2", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<index_t>>(
                s.active_g2.data(),
                s.active_g2.size()
            );
        })
        .def_property_readonly("active_begins", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<index_t>>(
                s.active_begins.data(),
                s.active_begins.size()
            );
        })
        .def_property_readonly("active_order", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<index_t>>(
                s.active_order.data(),
                s.active_order.size()
            );
        })
        .def_readonly("is_active", &state_t::is_active)
        .def_property_readonly("betas", [](const state_t& s) {
            Eigen::SparseMatrix<value_t, Eigen::RowMajor> betas(s.betas.size(), s.X->cols());
            for (size_t i = 0; i < s.betas.size(); ++i) {
                const auto& curr = s.betas[i];
                for (int j = 0; j < curr.nonZeros(); ++j) {
                    const auto jj = curr.innerIndexPtr()[j];
                    betas.coeffRef(i, jj) = curr.valuePtr()[j];
                }
            }
            betas.makeCompressed();
            return betas;
        })
        .def_property_readonly("rsqs", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<value_t>>(
                s.rsqs.data(),
                s.rsqs.size()
            );
        })
        .def_property_readonly("resids", [](const state_t& s) {
            ad::util::rowarr_type<value_t> resids(s.resids.size(), s.resid.size());
            for (size_t i = 0; i < s.resids.size(); ++i) {
                resids.row(i) = s.resids[i];
            }
            return resids;
        })
        .def_readonly("n_cds", &state_t::n_cds)
        .def_property_readonly("time_strong_cd", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<double>>(
                s.time_strong_cd.data(),
                s.time_strong_cd.size()
            );
        })
        .def_property_readonly("time_active_cd", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<double>>(
                s.time_active_cd.data(),
                s.time_active_cd.size()
            );
        })
        ;
}

void register_state(py::module_& m)
{
    pin_naive<ad::matrix::MatrixBase<double>>(m, "PinNaive64");
    pin_naive<ad::matrix::MatrixBase<float>>(m, "PinNaive32");
}