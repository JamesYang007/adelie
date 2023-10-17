#include "decl.hpp"
#include <adelie_core/matrix/matrix_cov_base.hpp>
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/state/state_pin_cov.hpp>
#include <adelie_core/state/state_pin_naive.hpp>
#include <adelie_core/state/state_basil_naive.hpp>

namespace py = pybind11;
namespace ad = adelie_core;

// ========================================================================
// Pin State 
// ========================================================================

template <class ValueType>
void state_pin_base(py::module_& m, const char* name)
{
    using state_t = ad::state::StatePinBase<ValueType>;
    using index_t = typename state_t::index_t;
    using value_t = typename state_t::value_t;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_bool_t = typename state_t::vec_bool_t;
    using dyn_vec_mat_value_t = typename state_t::dyn_vec_mat_value_t;

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
            const dyn_vec_mat_value_t&,
            const Eigen::Ref<const vec_value_t>&, 
            bool,
            size_t,
            value_t,
            value_t,
            value_t,
            value_t,
            size_t,
            size_t,
            value_t,
            Eigen::Ref<vec_value_t>,
            Eigen::Ref<vec_bool_t>
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
            py::arg("strong_transforms").noconvert(),
            py::arg("lmda_path").noconvert(),
            py::arg("intercept"),
            py::arg("max_iters"),
            py::arg("tol"),
            py::arg("rsq_slope_tol"),
            py::arg("rsq_curv_tol"),
            py::arg("newton_tol"),
            py::arg("newton_max_iters"),
            py::arg("n_threads"),
            py::arg("rsq"),
            py::arg("strong_beta").noconvert(),
            py::arg("strong_is_active").noconvert()
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
        List of :math:`D_k^2` where :math:`D_k` is from the SVD of :math:`X_{c,k}` 
        along the strong groups :math:`k` and for possibly column-centered :math:`X_k`.
        ``strong_vars[b:b+p]`` is :math:`D_k` for the ``i`` th strong group where
        ``k = strong_set[i]``,
        ``b = strong_begins[i]``,
        and ``p = group_sizes[k]``.
        )delimiter")
        .def_readonly("strong_transforms", &state_t::strong_transforms, R"delimiter(
        List of :math:`V_k` where :math:`V_k` is from the SVD of :math:`X_{c,k}`
        along the strong groups :math:`k` and for possibly column-centered :math:`X_k`.
        It *only* needs to be properly initialized for groups with size > 1.
        ``strong_transforms[i]`` is :math:`V_k` for the ``i`` th strong group where
        ``k = strong_set[i]``.
        )delimiter")
        .def_readonly("lmda_path", &state_t::lmda_path, R"delimiter(
        Regularization sequence to fit on.
        )delimiter")
        .def_readonly("intercept", &state_t::intercept, R"delimiter(
        ``True`` to fit with intercept.
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
        The unnormalized :math:`R^2` is given by :math:`\|y_c\|_2^2 - \|y_c-X_c\beta\|_2^2`.
        )delimiter")
        .def_readonly("strong_beta", &state_t::strong_beta, R"delimiter(
        Coefficient vector on the strong set.
        ``strong_beta[b:b+p]`` is the coefficient for the ``i`` th strong group 
        where
        ``k = strong_set[i]``,
        ``b = strong_begins[i]``,
        and ``p = group_sizes[k]``.
        )delimiter")
        .def_readonly("strong_is_active", &state_t::strong_is_active, R"delimiter(
        Boolean vector that indicates whether each strong group in ``groups`` is active or not.
        ``strong_is_active[i]`` is ``True`` if and only if ``strong_set[i]`` is active.
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
        .def_property_readonly("intercepts", [](const state_t& s) {
            return Eigen::Map<const vec_value_t>(s.intercepts.data(), s.intercepts.size());
        }, R"delimiter(
        ``intercepts[i]`` is the intercept at ``lmdas[i]``.
        )delimiter")
        .def_property_readonly("rsqs", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<value_t>>(
                s.rsqs.data(),
                s.rsqs.size()
            );
        }, R"delimiter(
        ``rsqs[i]`` corresponds to the unnormalized :math:`R^2` at ``betas[i]``.
        )delimiter")
        .def_property_readonly("lmdas", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<value_t>>(
                s.lmdas.data(),
                s.lmdas.size()
            );
        }, R"delimiter(
        ``lmdas[i]`` corresponds to the regularization :math:`\lambda`
        used for the ``i`` th outputted solution.
        )delimiter")
        .def_readonly("strong_is_actives", &state_t::strong_is_actives, R"delimiter(
        ``strong_is_actives[i]`` is the state of ``strong_is_active`` 
        when the ``i`` th solution is computed.
        )delimiter")
        .def_readonly("strong_betas", &state_t::strong_betas, R"delimiter(
        ``strong_betas[i]`` is the state of ``strong_beta`` 
        when the ``i`` th solution is computed.
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
    using dyn_vec_mat_value_t = typename state_t::dyn_vec_mat_value_t;

    py::class_<state_t, base_t, PyStatePinNaive<matrix_t>>(m, name)
        .def(py::init<
            matrix_t&,
            value_t,
            value_t,
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
            const dyn_vec_mat_value_t&,
            const Eigen::Ref<const vec_value_t>&, 
            bool,
            size_t,
            value_t,
            value_t,
            value_t,
            value_t,
            value_t,
            size_t,
            size_t,
            value_t,
            Eigen::Ref<vec_value_t>,
            value_t,
            Eigen::Ref<vec_value_t>, 
            Eigen::Ref<vec_bool_t>
        >(),
            py::arg("X"),
            py::arg("y_mean"),
            py::arg("y_var"),
            py::arg("groups").noconvert(),
            py::arg("group_sizes").noconvert(),
            py::arg("alpha"),
            py::arg("penalty").noconvert(),
            py::arg("strong_set").noconvert(),
            py::arg("strong_g1").noconvert(),
            py::arg("strong_g2").noconvert(),
            py::arg("strong_begins").noconvert(),
            py::arg("strong_vars").noconvert(),
            py::arg("strong_X_means").noconvert(),
            py::arg("strong_transforms").noconvert(),
            py::arg("lmda_path").noconvert(),
            py::arg("intercept"),
            py::arg("max_iters"),
            py::arg("tol"),
            py::arg("rsq_tol"),
            py::arg("rsq_slope_tol"),
            py::arg("rsq_curv_tol"),
            py::arg("newton_tol"),
            py::arg("newton_max_iters"),
            py::arg("n_threads"),
            py::arg("rsq"),
            py::arg("resid").noconvert(),
            py::arg("resid_sum"),
            py::arg("strong_beta").noconvert(),
            py::arg("strong_is_active").noconvert()
        )
        .def(py::init([](const state_t& s) { return new state_t(s); }))
        .def_readonly("y_mean", &state_t::y_mean, R"delimiter(
        Mean of :math:`y`.
        )delimiter")
        .def_readonly("y_var", &state_t::y_var, R"delimiter(
        :math:`\ell_2` norm squared of :math:`y_c`.
        )delimiter")
        .def_readonly("rsq_tol", &state_t::rsq_tol, R"delimiter(
        Early stopping rule check on :math:`R^2`.
        )delimiter")
        .def_readonly("strong_X_means", &state_t::strong_X_means, R"delimiter(
        Column means of :math:`X` for strong groups.
        )delimiter")
        .def_readonly("X", &state_t::X, R"delimiter(
        Feature matrix.
        )delimiter")
        .def_readonly("resid", &state_t::resid, R"delimiter(
        Residual :math:`y_c-X\beta` at ``strong_beta`` 
        (note that it always uses uncentered :math:`X`!).
        )delimiter")
        .def_readonly("resid_sum", &state_t::resid_sum, R"delimiter(
        Sum of ``resid``.
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
        .def_property_readonly("resid_sums", [](const state_t& s) {
            return Eigen::Map<const vec_value_t>(s.resid_sums.data(), s.resid_sums.size());
        }, R"delimiter(
        ``resid_sums[i]`` is the sum of ``resids[i]``.
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
    using dyn_vec_mat_value_t = typename state_t::dyn_vec_mat_value_t;

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
            const dyn_vec_mat_value_t&,
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
            Eigen::Ref<vec_bool_t>
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
            py::arg("strong_transforms").noconvert(),
            py::arg("lmda_path").noconvert(),
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
            py::arg("strong_is_active").noconvert()
        )
        .def(py::init([](const state_t& s) { return new state_t(s); }))
        .def_readonly("A", &state_t::A, R"delimiter(
        Covariance matrix :math:`X_c^\\top X_c` where `X_c` is column-centered to fit with intercept.
        It is typically one of the matrices defined in ``adelie.matrix`` sub-module.
        )delimiter")
        .def_readonly("strong_grad", &state_t::strong_grad, R"delimiter(
        Gradient :math:`X_{c,k}^\top (y_c-X_c\beta)` on the strong groups :math:`k` where :math:`\beta` is given by ``strong_beta``.
        ``strong_grad[b:b+p]`` is the gradient for the ``i`` th strong group
        where 
        ``k = strong_set[i]``,
        ``b = strong_begins[i]``,
        and ``p = group_sizes[k]``.
        )delimiter")
        .def_readonly("strong_grads", &state_t::strong_grads, R"delimiter(
        ``strong_grads[i]`` is the state of ``strong_grad`` 
        when the ``i`` th solution is computed.
        )delimiter")
        ;
}

// ========================================================================
// Basil State 
// ========================================================================

template <class ValueType>
void state_basil_base(py::module_& m, const char* name)
{
    using state_t = ad::state::StateBasilBase<ValueType>;
    using value_t = typename state_t::value_t;
    using safe_bool_t = typename state_t::safe_bool_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_bool_t = typename state_t::vec_bool_t;
    py::class_<state_t>(m, name, R"delimiter(
        Base core state class for all basil methods.
        )delimiter") 
        .def(py::init<
            const Eigen::Ref<const vec_index_t>&, 
            const Eigen::Ref<const vec_index_t>&,
            value_t, 
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            value_t,
            value_t,
            size_t,
            size_t,
            size_t,
            size_t,
            const std::string&,
            size_t,
            value_t,
            value_t,
            value_t,
            value_t,
            value_t,
            size_t,
            bool,
            bool,
            bool,
            bool,
            size_t,
            const Eigen::Ref<const vec_index_t>&,
            const Eigen::Ref<const vec_value_t>&, 
            const Eigen::Ref<const vec_bool_t>&,
            value_t,
            value_t,
            const Eigen::Ref<const vec_value_t>& 
        >(),
            py::arg("groups").noconvert(),
            py::arg("group_sizes").noconvert(),
            py::arg("alpha"),
            py::arg("penalty").noconvert(),
            py::arg("lmda_path").noconvert(),
            py::arg("lmda_max"),
            py::arg("min_ratio"),
            py::arg("lmda_path_size"),
            py::arg("delta_lmda_path_size"),
            py::arg("delta_strong_size"),
            py::arg("max_strong_size"),
            py::arg("strong_rule"),
            py::arg("max_iters"),
            py::arg("tol"),
            py::arg("rsq_tol"),
            py::arg("rsq_slope_tol"),
            py::arg("rsq_curv_tol"),
            py::arg("newton_tol"),
            py::arg("newton_max_iters"),
            py::arg("early_exit"),
            py::arg("setup_lmda_max"),
            py::arg("setup_lmda_path"),
            py::arg("intercept"),
            py::arg("n_threads"),
            py::arg("strong_set").noconvert(),
            py::arg("strong_beta").noconvert(),
            py::arg("strong_is_active").noconvert(),
            py::arg("rsq"),
            py::arg("lmda"),
            py::arg("grad").noconvert()
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
        .def_readonly("lmda_max", &state_t::lmda_max, R"delimiter(
        The smallest :math:`\lambda` such that the true solution is zero
        for all coefficients that have a non-vanishing group lasso penalty (:math:`\ell_2`-norm).
        )delimiter")
        .def_readonly("min_ratio", &state_t::min_ratio, R"delimiter(
        The ratio between the largest and smallest :math:`\lambda` in the regularization sequence
        if it is to be generated.
        )delimiter")
        .def_readonly("lmda_path_size", &state_t::lmda_path_size, R"delimiter(
        Number of regularizations in the path if it is to be generated.
        )delimiter")
        .def_readonly("delta_lmda_path_size", &state_t::delta_lmda_path_size, R"delimiter(
        Number of regularizations to batch per BASIL iteration.
        )delimiter")
        .def_readonly("delta_strong_size", &state_t::delta_strong_size, R"delimiter(
        Number of strong groups to include per BASIL iteration 
        if strong rule does not include new groups but optimality is not reached.
        )delimiter")
        .def_readonly("max_strong_size", &state_t::max_strong_size, R"delimiter(
        Maximum number of strong groups allowed.
        The function will return a valid state and guaranteed to have strong set size
        less than or equal to ``max_strong_size``.
        )delimiter")
        .def_property_readonly("strong_rule", [](const state_t& s) -> std::string {
            switch (s.strong_rule) {
                case ad::state::strong_rule_type::_default:
                    return "default";
                case ad::state::strong_rule_type::_fixed_greedy:
                    return "fixed_greedy";
                case ad::state::strong_rule_type::_safe:
                    return "safe";
            }
            throw std::runtime_error("Invalid strong rule type!");
        }, R"delimiter(
        ``True`` if strong rule should be used (only a heuristic!).
        )delimiter")
        .def_readonly("max_iters", &state_t::max_iters, R"delimiter(
        Maximum number of coordinate descents.
        )delimiter")
        .def_readonly("tol", &state_t::tol, R"delimiter(
        Convergence tolerance.
        )delimiter")
        .def_readonly("rsq_tol", &state_t::rsq_tol, R"delimiter(
        Early stopping rule check on :math:`R^2`.
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
        .def_readonly("early_exit", &state_t::early_exit, R"delimiter(
        ``True`` if the function should early exit based on training :math:`R^2`.
        )delimiter")
        .def_readonly("setup_lmda_max", &state_t::setup_lmda_max, R"delimiter(
        ``True`` if the function should setup :math:`\lambda_\max`.
        )delimiter")
        .def_readonly("setup_lmda_path", &state_t::setup_lmda_path, R"delimiter(
        ``True`` if the function should setup the regularization path.
        )delimiter")
        .def_readonly("intercept", &state_t::intercept, R"delimiter(
        ``True`` if the function should fit with intercept.
        )delimiter")
        .def_readonly("n_threads", &state_t::n_threads, R"delimiter(
        Number of threads.
        )delimiter")
        .def_readonly("lmda_path", &state_t::lmda_path, R"delimiter(
        The regularization path to solve for.
        The full path is not considered if ``early_exit`` is ``True``.
        It is recommended that the path is sorted in decreasing order.
        )delimiter")
        .def_readonly("strong_hashset", &state_t::strong_hashset, R"delimiter(
        Hashmap containing the same values as ``strong_set``.
        It is used to check if a given group is strong or not.
        )delimiter")
        .def_property_readonly("strong_set", [](const state_t& s) {
            return Eigen::Map<const vec_index_t>(s.strong_set.data(), s.strong_set.size());
        }, R"delimiter(
        List of indices into ``groups`` that correspond to the strong groups.
        ``strong_set[i]`` is ``i`` th strong group.
        ``strong_set`` must contain at least the true (optimal) active groups
        when the regularization is given by ``lmda``.
        )delimiter")
        .def_property_readonly("strong_g1", [](const state_t& s) {
            return Eigen::Map<const vec_index_t>(s.strong_g1.data(), s.strong_g1.size());
        }, R"delimiter(
        List of indices into ``strong_set`` that correspond to groups of size ``1``.
        ``strong_set[strong_g1[i]]`` is the ``i`` th strong group of size ``1``
        such that ``group_sizes[strong_set[strong_g1[i]]]`` is ``1``.
        )delimiter")
        .def_property_readonly("strong_g2", [](const state_t& s) {
            return Eigen::Map<const vec_index_t>(s.strong_g2.data(), s.strong_g2.size());
        }, R"delimiter(
        List of indices into ``strong_set`` that correspond to groups more than size ``1``.
        ``strong_set[strong_g2[i]]`` is the ``i`` th strong group of size more than ``1``
        such that ``group_sizes[strong_set[strong_g2[i]]]`` is more than ``1``.
        )delimiter")
        .def_property_readonly("strong_begins", [](const state_t& s) {
            return Eigen::Map<const vec_index_t>(s.strong_begins.data(), s.strong_begins.size());
        }, R"delimiter(
        List of indices that index a corresponding list of values for each strong group.
        ``strong_begins[i]`` is the starting index corresponding to the ``i`` th strong group.
        From this index, reading ``group_sizes[strong_set[i]]`` number of elements
        will grab values corresponding to the full ``i`` th strong group block.
        )delimiter")
        .def_property_readonly("strong_order", [](const state_t& s) {
            return Eigen::Map<const vec_index_t>(s.strong_order.data(), s.strong_order.size());
        }, R"delimiter(
        Ordering such that ``groups`` is sorted in ascending order for the strong groups.
        ``groups[strong_set[i]]`` is the ``i`` th strong group in ascending order.
        )delimiter")
        .def_property_readonly("strong_beta", [](const state_t& s) {
            return Eigen::Map<const vec_value_t>(s.strong_beta.data(), s.strong_beta.size());
        }, R"delimiter(
        Coefficient vector on the strong set.
        ``strong_beta[b:b+p]`` is the coefficient for the ``i`` th strong group 
        where
        ``k = strong_set[i]``,
        ``b = strong_begins[i]``,
        and ``p = group_sizes[k]``.
        This must contain the true solution values for the strong groups.
        )delimiter")
        .def_property_readonly("strong_is_active", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<safe_bool_t>>(
                s.strong_is_active.data(), 
                s.strong_is_active.size()
            );
        }, R"delimiter(
        Boolean vector that indicates whether each strong group in ``groups`` is active or not.
        ``strong_is_active[i]`` is ``True`` if and only if ``strong_set[i]`` is active.
        )delimiter")
        .def_readonly("rsq", &state_t::rsq, R"delimiter(
        The true unnormalized :math:`R^2` given by :math:`\|y_c\|_2^2 - \|y_c-X_c\beta\|_2^2`
        where :math:`\beta` is given by ``strong_beta``.
        )delimiter")
        .def_readonly("lmda", &state_t::lmda, R"delimiter(
        The regularization parameter at which the true solution is given by ``strong_beta``.
        )delimiter")
        .def_readonly("grad", &state_t::grad, R"delimiter(
        The true full gradient :math:`X_c^\top (y_c - X_c\beta)` where
        :math:`\beta` is given by ``strong_beta``.
        )delimiter")
        .def_readonly("abs_grad", &state_t::abs_grad, R"delimiter(
        The :math:`\ell_2` norms of ``grad`` across each group.
        ``abs_grad[i]`` is given by ``np.linalg.norm(grad[g:g+gs])``
        where ``g = groups[i]`` and ``gs = group_sizes[i]``.
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
        ``betas[i]`` is the (untransformed) solution corresponding to ``lmdas[i]``.
        )delimiter")
        .def_property_readonly("rsqs", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<value_t>>(
                s.rsqs.data(),
                s.rsqs.size()
            );
        }, R"delimiter(
        ``rsqs[i]`` is the (normalized) :math:`R^2` at ``betas[i]``.
        )delimiter")
        .def_property_readonly("lmdas", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<value_t>>(
                s.lmdas.data(),
                s.lmdas.size()
            );
        }, R"delimiter(
        ``lmdas[i]`` is the ``i`` th :math:`\lambda` regularization in the solution set.
        )delimiter")
        .def_property_readonly("intercepts", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<value_t>>(
                s.intercepts.data(),
                s.intercepts.size()
            );
        }, R"delimiter(
        ``intercepts[i]`` is the intercept solution corresponding to ``lmdas[i]``.
        )delimiter")
        .def_property_readonly("benchmark_screen", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<double>>(
                s.benchmark_screen.data(),
                s.benchmark_screen.size()
            );
        }, R"delimiter(
        Screen time for a given BASIL iteration.
        )delimiter")
        .def_property_readonly("benchmark_fit", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<double>>(
                s.benchmark_fit.data(),
                s.benchmark_fit.size()
            );
        }, R"delimiter(
        Fit time for a given BASIL iteration.
        )delimiter")
        .def_property_readonly("benchmark_kkt", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<double>>(
                s.benchmark_kkt.data(),
                s.benchmark_kkt.size()
            );
        }, R"delimiter(
        KKT time for a given BASIL iteration.
        )delimiter")
        .def_property_readonly("benchmark_invariance", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<double>>(
                s.benchmark_invariance.data(),
                s.benchmark_invariance.size()
            );
        }, R"delimiter(
        Invariance time for a given BASIL iteration.
        )delimiter")
        ;
}

template <class MatrixType>
class PyStateBasilNaive : public ad::state::StateBasilNaive<MatrixType>
{
    using base_t = ad::state::StateBasilNaive<MatrixType>;
public:
    using base_t::base_t;
    PyStateBasilNaive(base_t&& base) : base_t(std::move(base)) {}
};

template <class MatrixType>
void state_basil_naive(py::module_& m, const char* name)
{
    using matrix_t = MatrixType;
    using state_t = ad::state::StateBasilNaive<matrix_t>;
    using base_t = typename state_t::base_t;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_bool_t = typename state_t::vec_bool_t;
    py::class_<state_t, base_t, PyStateBasilNaive<matrix_t>>(m, name)
        .def(py::init<
            matrix_t&,
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            value_t,
            value_t,
            bool,
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_index_t>&, 
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_index_t>&,
            const Eigen::Ref<const vec_index_t>&,
            value_t, 
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            value_t,
            value_t,
            size_t,
            size_t,
            size_t,
            size_t,
            const std::string&,
            size_t,
            value_t,
            value_t,
            value_t,
            value_t,
            value_t,
            size_t,
            bool,
            bool,
            bool,
            bool,
            size_t,
            const Eigen::Ref<const vec_index_t>&,
            const Eigen::Ref<const vec_value_t>&, 
            const Eigen::Ref<const vec_bool_t>&,
            value_t,
            value_t,
            const Eigen::Ref<const vec_value_t>& 
        >(),
            py::arg("X"),
            py::arg("X_means").noconvert(),
            py::arg("X_group_norms").noconvert(),
            py::arg("y_mean"),
            py::arg("y_var"),
            py::arg("setup_edpp"),
            py::arg("resid").noconvert(),
            py::arg("edpp_safe_set").noconvert(),
            py::arg("edpp_v1_0").noconvert(),
            py::arg("edpp_resid_0").noconvert(),
            py::arg("groups").noconvert(),
            py::arg("group_sizes").noconvert(),
            py::arg("alpha"),
            py::arg("penalty").noconvert(),
            py::arg("lmda_path").noconvert(),
            py::arg("lmda_max"),
            py::arg("min_ratio"),
            py::arg("lmda_path_size"),
            py::arg("delta_lmda_path_size"),
            py::arg("delta_strong_size"),
            py::arg("max_strong_size"),
            py::arg("strong_rule"),
            py::arg("max_iters"),
            py::arg("tol"),
            py::arg("rsq_tol"),
            py::arg("rsq_slope_tol"),
            py::arg("rsq_curv_tol"),
            py::arg("newton_tol"),
            py::arg("newton_max_iters"),
            py::arg("early_exit"),
            py::arg("setup_lmda_max"),
            py::arg("setup_lmda_path"),
            py::arg("intercept"),
            py::arg("n_threads"),
            py::arg("strong_set").noconvert(),
            py::arg("strong_beta").noconvert(),
            py::arg("strong_is_active").noconvert(),
            py::arg("rsq"),
            py::arg("lmda"),
            py::arg("grad").noconvert()
        )
        .def(py::init([](const state_t& s) { return new state_t(s); }))
        .def_readonly("X_means", &state_t::X_means, R"delimiter(
        Column means of ``X``.
        )delimiter")
        .def_readonly("X_group_norms", &state_t::X_group_norms, R"delimiter(
        Group Frobenius norm of ``X``.
        ``X_group_norms[i]`` is :math:`\|X_{c, g}\|_F`
        where :math:`g` corresponds to the group index ``i``.
        )delimiter")
        .def_readonly("y_mean", &state_t::y_mean, R"delimiter(
        The mean of the response vector :math:`y`.
        )delimiter")
        .def_readonly("y_var", &state_t::y_var, R"delimiter(
        The variance of the response vector :math:`y`, i.e. 
        :math:`\|y_c\|_2^2`.
        )delimiter")
        .def_readonly("use_edpp", &state_t::use_edpp, R"delimiter(
        ``True`` if EDPP should be used.
        )delimiter")
        .def_readonly("setup_edpp", &state_t::setup_edpp, R"delimiter(
        ``True`` if EDPP setup is required,
        in which case, the solver will always solve at :math:`\lambda_\max`.
        See ``edpp_v1_0`` and ``edpp_resid_0``.
        )delimiter")
        .def_readonly("X", &state_t::X, R"delimiter(
        Feature matrix.
        )delimiter")
        .def_readonly("resid", &state_t::resid, R"delimiter(
        Residual :math:`y_c - X \beta` where :math:`\beta` is given by ``strong_beta``.
        )delimiter")
        .def_readonly("resid_sum", &state_t::resid_sum, R"delimiter(
        Sum of ``resid``.
        )delimiter")
        .def_readonly("strong_X_means", &state_t::strong_X_means, R"delimiter(
        Column means of ``X`` on the strong set.
        )delimiter")
        .def_readonly("strong_transforms", &state_t::strong_transforms, R"delimiter(
        The :math:`V` from the SVD of :math:`X_{c,k}` where :math:`X_c` 
        is the possibly centered feature matrix and :math:`k` is an index to ``strong_set``.
        )delimiter")
        .def_property_readonly("strong_vars", [](const state_t& s) {
            return Eigen::Map<const vec_value_t>(s.strong_vars.data(), s.strong_vars.size());
        }, R"delimiter(
        The :math:`D^2` from the SVD of :math:`X_{c,k}` where :math:`X_c` 
        is the possibly centered feature matrix and :math:`k` is an index to ``strong_set``.
        )delimiter")
        .def_readonly("edpp_safe_hashset", &state_t::edpp_safe_hashset, R"delimiter(
        Hashset containing all the safe groups.
        If EDPP is not used, it contains all the variables.
        )delimiter")
        .def_property_readonly("edpp_safe_set", [](const state_t& s) {
            return Eigen::Map<const vec_index_t>(s.edpp_safe_set.data(), s.edpp_safe_set.size());
        }, R"delimiter(
        A list of EDPP safe groups.
        )delimiter")
        .def_readonly("edpp_v1_0", &state_t::edpp_v1_0, R"delimiter(
        The :math:`v_1` vector in EDPP rule at :math:`\lambda_\max`.
        )delimiter")
        .def_readonly("edpp_resid_0", &state_t::edpp_resid_0, R"delimiter(
        The residual :math:`y_c - X_c\beta` where :math:`\beta` is the solution at :math:`\lambda_\max`.
        )delimiter")
        ;
}

void register_state(py::module_& m)
{
    state_pin_base<double>(m, "StatePinBase64");
    state_pin_base<float>(m, "StatePinBase32");
    state_pin_naive<ad::matrix::MatrixNaiveBase<double>>(m, "StatePinNaive64");
    state_pin_naive<ad::matrix::MatrixNaiveBase<float>>(m, "StatePinNaive32");
    state_pin_cov<ad::matrix::MatrixCovBase<double>>(m, "StatePinCov64");
    state_pin_cov<ad::matrix::MatrixCovBase<float>>(m, "StatePinCov32");

    state_basil_base<double>(m, "StateBasilBase64");
    state_basil_base<float>(m, "StateBasilBase32");
    state_basil_naive<ad::matrix::MatrixNaiveBase<double>>(m, "StateBasilNaive64");
    state_basil_naive<ad::matrix::MatrixNaiveBase<float>>(m, "StateBasilNaive32");
}