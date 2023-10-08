#include "decl.hpp"
#include <adelie_core/matrix/matrix_pin_cov_base.hpp>
#include <adelie_core/matrix/matrix_pin_naive_base.hpp>
#include <adelie_core/state/state_pin_cov.hpp>
#include <adelie_core/state/state_pin_naive.hpp>

namespace py = pybind11;
namespace ad = adelie_core;

/**
 * @brief Registers StatePinBase instantiations.
 * 
 * The purpose for exposing this class is to expose the attributes,
 * and create common docstrings for all derived classes.
 * 
 * @tparam ValueType 
 * @param m 
 * @param name 
 */
template <class ValueType>
void state_pin_base(py::module_& m, const char* name)
{
    using state_t = ad::state::StatePinBase<ValueType>;
    using index_t = typename state_t::index_t;
    using value_t = typename state_t::value_t;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_bool_t = typename state_t::vec_bool_t;
    using dyn_vec_index_t = typename state_t::dyn_vec_index_t;
    using dyn_vec_value_t = typename state_t::dyn_vec_value_t;
    using dyn_vec_sp_vec_t = typename state_t::dyn_vec_sp_vec_t;

    py::class_<state_t>(m, name, R"delimiter(
        Base core state class for all pin methods.
        )delimiter")
        .def(py::init<
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
            size_t,
            value_t,
            Eigen::Ref<vec_value_t>, 
            Eigen::Ref<vec_value_t>,
            dyn_vec_index_t,
            dyn_vec_index_t,
            dyn_vec_index_t,
            dyn_vec_index_t,
            dyn_vec_index_t,
            Eigen::Ref<vec_bool_t>,
            dyn_vec_sp_vec_t, 
            dyn_vec_value_t
        >(),
            py::arg("groups").noconvert(),
            py::arg("group_sizes").noconvert(),
            py::arg("alpha"),
            py::arg("penalty").noconvert(),
            py::arg("strong_set").noconvert(),
            py::arg("strong_g1").noconvert(),
            py::arg("strong_g2").noconvert(),
            py::arg("strong_begins").noconvert(),
            py::arg("strong_vars").noconvert(),
            py::arg("lmdas").noconvert(),
            py::arg("max_iters"),
            py::arg("tol"),
            py::arg("rsq_slope_tol"),
            py::arg("rsq_curv_tol"),
            py::arg("newton_tol"),
            py::arg("newton_max_iters"),
            py::arg("n_threads"),
            py::arg("rsq"),
            py::arg("strong_beta").noconvert(),
            py::arg("strong_grad").noconvert(),
            py::arg("active_set"),
            py::arg("active_g1"),
            py::arg("active_g2"),
            py::arg("active_begins"),
            py::arg("active_order"),
            py::arg("is_active").noconvert(),
            py::arg("betas"),
            py::arg("rsqs")
        )
        .def_readonly("groups", &state_t::groups, R"delimiter(
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
        )delimiter")
        .def_readonly("group_sizes", &state_t::group_sizes, R"delimiter(
        List of group sizes corresponding to each element in ``groups``.
        ``group_sizes[i]`` is the group size of the ``i`` th group. 
        )delimiter")
        .def_readonly("alpha", &state_t::alpha, R"delimiter(
        Elastic net parameter.
        It must be in the range :math:`[0,1]`.
        )delimiter")
        .def_readonly("penalty", &state_t::penalty, R"delimiter(
        Penalty factor for each group in the same order as ``groups``.
        It must be a non-negative vector.
        )delimiter")
        .def_readonly("strong_set", &state_t::strong_set, R"delimiter(
        List of indices into ``groups`` that correspond to the strong groups.
        ``strong_set[i]`` is ``i`` th strong group.
        )delimiter")
        .def_readonly("strong_g1", &state_t::strong_g1, R"delimiter(
        List of indices into ``strong_set`` that correspond to groups of size ``1``.
        ``strong_set[strong_g1[i]]`` is the ``i`` th strong group of size ``1``
        such that ``group_sizes[strong_set[strong_g1[i]]]`` is ``1``.
        )delimiter")
        .def_readonly("strong_g2", &state_t::strong_g2, R"delimiter(
        List of indices into ``strong_set`` that correspond to groups more than size ``1``.
        ``strong_set[strong_g2[i]]`` is the ``i`` th strong group of size more than ``1``
        such that ``group_sizes[strong_set[strong_g2[i]]]`` is more than ``1``.
        )delimiter")
        .def_readonly("strong_begins", &state_t::strong_begins, R"delimiter(
        List of indices that index a corresponding list of values for each strong group.
        ``strong_begins[i]`` is the starting index corresponding to the ``i`` th strong group.
        From this index, reading ``group_sizes[strong_set[i]]`` number of elements
        will grab values corresponding to the full ``i`` th strong group block.
        )delimiter")
        .def_readonly("strong_vars", &state_t::strong_vars, R"delimiter(
        List of the diagonal of :math:`X_k^\top X_k` along the strong groups :math:`k`.
        ``strong_vars[b:b+p]`` is the diagonal of :math:`X_k^\top X_k` for the ``i`` th strong group where
        ``k = strong_set[i]``,
        ``b = strong_begins[i]``,
        and ``p = group_sizes[k]``.
        )delimiter")
        .def_readonly("lmdas", &state_t::lmdas, R"delimiter(
        Regularization sequence to fit on.
        )delimiter")
        .def_readonly("max_iters", &state_t::max_iters, R"delimiter(
        Maximum number of coordinate descents.
        )delimiter")
        .def_readonly("tol", &state_t::tol, R"delimiter(
        Convergence tolerance.
        )delimiter")
        .def_readonly("rsq_slope_tol", &state_t::rsq_slope_tol, R"delimiter(
        Early stopping rule check on slope of :math:`R^2`.
        )delimiter")
        .def_readonly("rsq_curv_tol", &state_t::rsq_curv_tol, R"delimiter(
        Early stopping rule check on curvature of :math:`R^2`.
        )delimiter")
        .def_readonly("newton_tol", &state_t::newton_tol, R"delimiter(
        Convergence tolerance for the BCD update.
        )delimiter")
        .def_readonly("newton_max_iters", &state_t::newton_max_iters, R"delimiter(
        Maximum number of iterations for the BCD update.
        )delimiter")
        .def_readonly("n_threads", &state_t::n_threads, R"delimiter(
        Number of threads.
        )delimiter")
        .def_readonly("rsq", &state_t::rsq, R"delimiter(
        Unnormalized :math:`R^2` value at ``strong_beta``.
        The unnormalized :math:`R^2` is given by :math:`\|y\|_2^2 - \|y-X\beta\|_2^2`.
        )delimiter")
        .def_readonly("strong_beta", &state_t::strong_beta, R"delimiter(
        Coefficient vector on the strong set.
        ``strong_beta[b:b+p]`` is the coefficient for the ``i`` th strong group 
        where
        ``k = strong_set[i]``,
        ``b = strong_begins[i]``,
        and ``p = group_sizes[k]``.
        )delimiter")
        .def_readonly("strong_grad", &state_t::strong_grad, R"delimiter(
        Gradient :math:`X_k^\top (y-X\beta)` on the strong groups :math:`k` where :math:`\beta` is given by ``strong_beta``.
        ``strong_grad[b:b+p]`` is the gradient for the ``i`` th strong group
        where 
        ``k = strong_set[i]``,
        ``b = strong_begins[i]``,
        and ``p = group_sizes[k]``.
        )delimiter")
        .def_property_readonly("active_set", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<index_t>>(
                s.active_set.data(),
                s.active_set.size()
            );
        }, R"delimiter(
        List of indices into ``strong_set`` that correspond to active groups.
        ``strong_set[active_set[i]]`` is the ``i`` th active group.
        An active group is one with non-zero coefficient block,
        that is, for every ``i`` th active group, 
        ``strong_beta[b:b+p] == 0`` where 
        ``j = active_set[i]``,
        ``k = strong_set[j]``,
        ``b = strong_begins[j]``,
        and ``p = group_sizes[k]``.
        )delimiter")
        .def_property_readonly("active_g1", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<index_t>>(
                s.active_g1.data(),
                s.active_g1.size()
            );
        }, R"delimiter(
        Subset of ``active_set`` that correspond to groups of size ``1``.
        )delimiter")
        .def_property_readonly("active_g2", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<index_t>>(
                s.active_g2.data(),
                s.active_g2.size()
            );
        }, R"delimiter(
        Subset of ``active_set`` that correspond to groups of size more than ``1``.
        )delimiter")
        .def_property_readonly("active_begins", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<index_t>>(
                s.active_begins.data(),
                s.active_begins.size()
            );
        }, R"delimiter(
        List of indices that index a corresponding list of values for each active group.
        ``active_begins[i]`` is the starting index corresponding to the ``i`` th active group.
        )delimiter")
        .def_property_readonly("active_order", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<index_t>>(
                s.active_order.data(),
                s.active_order.size()
            );
        }, R"delimiter(
        Ordering such that ``groups`` is sorted in ascending order for the active groups.
        ``groups[strong_set[active_order[i]]]`` is the ``i`` th active group in ascending order.
        )delimiter")
        .def_readonly("is_active", &state_t::is_active, R"delimiter(
        Boolean vector that indicates whether each strong group in ``groups`` is active or not.
        ``is_active[i]`` is ``True`` if and only if ``strong_set[i]`` is active.
        )delimiter")
        .def_property_readonly("betas", [](const state_t& s) {
            const auto p = s.group_sizes.sum();
            Eigen::SparseMatrix<value_t, Eigen::RowMajor> betas(s.betas.size(), p);
            for (size_t i = 0; i < s.betas.size(); ++i) {
                const auto& curr = s.betas[i];
                for (int j = 0; j < curr.nonZeros(); ++j) {
                    const auto jj = curr.innerIndexPtr()[j];
                    betas.coeffRef(i, jj) = curr.valuePtr()[j];
                }
            }
            betas.makeCompressed();
            return betas;
        }, R"delimiter(
        ``betas[i]`` corresponds to the solution corresponding to ``lmdas[i]``.
        )delimiter")
        .def_property_readonly("rsqs", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<value_t>>(
                s.rsqs.data(),
                s.rsqs.size()
            );
        }, R"delimiter(
        ``rsqs[i]`` corresponds to the unnormalized :math:`R^2` at ``betas[i]``.
        )delimiter")
        .def_readonly("iters", &state_t::iters, R"delimiter(
        Number of coordinate descents taken.
        )delimiter")
        .def_property_readonly("time_strong_cd", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<double>>(
                s.time_strong_cd.data(),
                s.time_strong_cd.size()
            );
        }, R"delimiter(
        Benchmark time for performing coordinate-descent on the strong set at every iteration.
        )delimiter")
        .def_property_readonly("time_active_cd", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<double>>(
                s.time_active_cd.data(),
                s.time_active_cd.size()
            );
        }, R"delimiter(
        Benchmark time for performing coordinate-descent on the active set at every iteration.
        )delimiter")
        ;
}

template <class MatrixType>
class PyStatePinNaive : public ad::state::StatePinNaive<MatrixType>
{
    using base_t = ad::state::StatePinNaive<MatrixType>;
public:
    using base_t::base_t;
    PyStatePinNaive(base_t&& base) : base_t(std::move(base)) {}
};

template <class MatrixType>
void state_pin_naive(py::module_& m, const char* name)
{
    using matrix_t = MatrixType;
    using state_t = ad::state::StatePinNaive<matrix_t>;
    using base_t = typename state_t::base_t;
    using value_t = typename state_t::value_t;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_bool_t = typename state_t::vec_bool_t;
    using dyn_vec_index_t = typename state_t::dyn_vec_index_t;
    using dyn_vec_value_t = typename state_t::dyn_vec_value_t;
    using dyn_vec_vec_value_t = typename state_t::dyn_vec_vec_value_t;
    using dyn_vec_sp_vec_t = typename state_t::dyn_vec_sp_vec_t;

    py::class_<state_t, base_t, PyStatePinNaive<matrix_t>>(m, name)
        .def(py::init<
            matrix_t&,
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
            py::arg("strong_vars").noconvert(),
            py::arg("lmdas").noconvert(),
            py::arg("max_iters"),
            py::arg("tol"),
            py::arg("rsq_slope_tol"),
            py::arg("rsq_curv_tol"),
            py::arg("newton_tol"),
            py::arg("newton_max_iters"),
            py::arg("n_threads"),
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
        .def(py::init([](const state_t& s) { return new state_t(s); }))
        .def_readonly("X", &state_t::X, R"delimiter(
        Feature matrix where each column block :math:`X_k` defined by the groups
        is such that :math:`X_k^\top X_k` is diagonal.
        )delimiter")
        .def_readonly("resid", &state_t::resid, R"delimiter(
        Residual :math:`y-X\beta` at ``strong_beta``.
        )delimiter")
        .def_property_readonly("resids", [](const state_t& s) {
            ad::util::rowarr_type<value_t> resids(s.resids.size(), s.resid.size());
            for (size_t i = 0; i < s.resids.size(); ++i) {
                resids.row(i) = s.resids[i];
            }
            return resids;
        }, R"delimiter(
        ``resids[i]`` is the residual at ``betas[i]``.
        )delimiter")
        ;
}

template <class MatrixType>
class PyStatePinCov : public ad::state::StatePinCov<MatrixType>
{
    using base_t = ad::state::StatePinCov<MatrixType>;
public:
    using base_t::base_t;
    PyStatePinCov(base_t&& base) : base_t(std::move(base)) {}
};

template <class MatrixType>
void state_pin_cov(py::module_& m, const char* name)
{
    using matrix_t = MatrixType;
    using state_t = ad::state::StatePinCov<matrix_t>;
    using base_t = typename state_t::base_t;
    using value_t = typename state_t::value_t;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_bool_t = typename state_t::vec_bool_t;
    using dyn_vec_index_t = typename state_t::dyn_vec_index_t;
    using dyn_vec_value_t = typename state_t::dyn_vec_value_t;
    using dyn_vec_sp_vec_t = typename state_t::dyn_vec_sp_vec_t;

    py::class_<state_t, base_t, PyStatePinCov<matrix_t>>(m, name)
        .def(py::init<
            matrix_t&,
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
            size_t,
            value_t,
            Eigen::Ref<vec_value_t>, 
            Eigen::Ref<vec_value_t>,
            dyn_vec_index_t,
            dyn_vec_index_t,
            dyn_vec_index_t,
            dyn_vec_index_t,
            dyn_vec_index_t,
            Eigen::Ref<vec_bool_t>,
            dyn_vec_sp_vec_t, 
            dyn_vec_value_t
        >(),
            py::arg("A"),
            py::arg("groups").noconvert(),
            py::arg("group_sizes").noconvert(),
            py::arg("alpha"),
            py::arg("penalty").noconvert(),
            py::arg("strong_set").noconvert(),
            py::arg("strong_g1").noconvert(),
            py::arg("strong_g2").noconvert(),
            py::arg("strong_begins").noconvert(),
            py::arg("strong_vars").noconvert(),
            py::arg("lmdas").noconvert(),
            py::arg("max_iters"),
            py::arg("tol"),
            py::arg("rsq_slope_tol"),
            py::arg("rsq_curv_tol"),
            py::arg("newton_tol"),
            py::arg("newton_max_iters"),
            py::arg("n_threads"),
            py::arg("rsq"),
            py::arg("strong_beta").noconvert(),
            py::arg("strong_grad").noconvert(),
            py::arg("active_set"),
            py::arg("active_g1"),
            py::arg("active_g2"),
            py::arg("active_begins"),
            py::arg("active_order"),
            py::arg("is_active").noconvert(),
            py::arg("betas"),
            py::arg("rsqs")
        )
        .def(py::init([](const state_t& s) { return new state_t(s); }))
        .def_readonly("A", &state_t::A, R"delimiter(
        Feature covariance matrix :math:`X^\top X` 
        with diagonal blocks :math:`X_k^\top X_k` for each strong group :math:`k`. 
        )delimiter")
        ;
}

void register_state(py::module_& m)
{
    state_pin_base<double>(m, "StatePinBase64");
    state_pin_base<float>(m, "StatePinBase32");
    state_pin_naive<ad::matrix::MatrixPinNaiveBase<double>>(m, "StatePinNaive64");
    state_pin_naive<ad::matrix::MatrixPinNaiveBase<float>>(m, "StatePinNaive32");
    state_pin_cov<ad::matrix::MatrixPinCovBase<double>>(m, "StatePinCov64");
    state_pin_cov<ad::matrix::MatrixPinCovBase<float>>(m, "StatePinCov32");
}