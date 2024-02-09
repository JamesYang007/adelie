#include "decl.hpp"
#include <adelie_core/glm/glm_base.hpp>
#include <adelie_core/matrix/matrix_cov_base.hpp>
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/state/state_gaussian_pin_cov.hpp>
#include <adelie_core/state/state_gaussian_pin_naive.hpp>
#include <adelie_core/state/state_gaussian_naive.hpp>
#include <adelie_core/state/state_glm_naive.hpp>
#include <adelie_core/state/state_multi_gaussian_naive.hpp>

namespace py = pybind11;
namespace ad = adelie_core;

template <class BetasType>
static auto convert_betas(
    size_t p,
    const BetasType& betas
)
{
    using value_t = typename std::decay_t<BetasType>::value_type::Scalar;
    using vec_value_t = ad::util::rowvec_type<value_t>;
    using vec_index_t = ad::util::rowvec_type<Eigen::Index>;
    using sp_mat_t = Eigen::SparseMatrix<value_t, Eigen::RowMajor, Eigen::Index>;

    const size_t l = betas.size();
    size_t nnz = 0;
    for (const auto& beta : betas) {
        nnz += beta.nonZeros();
    }
    vec_value_t values(nnz);
    vec_index_t inners(nnz); 
    vec_index_t outers(l+1);
    outers[0] = 0;
    int inner_idx = 0;
    for (size_t i = 0; i < l; ++i) {
        const auto& curr = betas[i];
        const auto nnz_curr = curr.nonZeros();
        Eigen::Map<vec_value_t>(
            values.data() + inner_idx,
            nnz_curr
        ) = Eigen::Map<const vec_value_t>(
            curr.valuePtr(),
            nnz_curr
        );
        Eigen::Map<vec_index_t>(
            inners.data() + inner_idx,
            nnz_curr
        ) = Eigen::Map<const vec_index_t>(
            curr.innerIndexPtr(),
            nnz_curr
        );
        outers[i+1] = outers[i] + nnz_curr;
        inner_idx += nnz_curr;
    }
    sp_mat_t out;
    out = Eigen::Map<const sp_mat_t>(
        l, 
        p,
        nnz,
        outers.data(),
        inners.data(),
        values.data()
    );
    return out;
}

// ========================================================================
// Pin State 
// ========================================================================

template <class ValueType>
void state_gaussian_pin_base(py::module_& m, const char* name)
{
    using state_t = ad::state::StateGaussianPinBase<ValueType>;
    using index_t = typename state_t::index_t;
    using value_t = typename state_t::value_t;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_bool_t = typename state_t::vec_bool_t;
    using dyn_vec_mat_value_t = typename state_t::dyn_vec_mat_value_t;

    py::class_<state_t>(m, name, R"delimiter(
        Base core state class for all gaussian pin methods.
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
            py::arg("screen_set").noconvert(),
            py::arg("screen_g1").noconvert(),
            py::arg("screen_g2").noconvert(),
            py::arg("screen_begins").noconvert(),
            py::arg("screen_vars").noconvert(),
            py::arg("screen_transforms").noconvert(),
            py::arg("lmda_path").noconvert(),
            py::arg("intercept"),
            py::arg("max_active_size"),
            py::arg("max_iters"),
            py::arg("tol"),
            py::arg("adev_tol"),
            py::arg("ddev_tol"),
            py::arg("newton_tol"),
            py::arg("newton_max_iters"),
            py::arg("n_threads"),
            py::arg("rsq"),
            py::arg("screen_beta").noconvert(),
            py::arg("screen_is_active").noconvert()
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
        .def_readonly("screen_set", &state_t::screen_set, R"delimiter(
        List of indices into ``groups`` that correspond to the screen groups.
        ``screen_set[i]`` is ``i`` th screen group.
        )delimiter")
        .def_readonly("screen_g1", &state_t::screen_g1, R"delimiter(
        List of indices into ``screen_set`` that correspond to groups of size ``1``.
        ``screen_set[screen_g1[i]]`` is the ``i`` th screen group of size ``1``
        such that ``group_sizes[screen_set[screen_g1[i]]]`` is ``1``.
        )delimiter")
        .def_readonly("screen_g2", &state_t::screen_g2, R"delimiter(
        List of indices into ``screen_set`` that correspond to groups more than size ``1``.
        ``screen_set[screen_g2[i]]`` is the ``i`` th screen group of size more than ``1``
        such that ``group_sizes[screen_set[screen_g2[i]]]`` is more than ``1``.
        )delimiter")
        .def_readonly("screen_begins", &state_t::screen_begins, R"delimiter(
        List of indices that index a corresponding list of values for each screen group.
        ``screen_begins[i]`` is the starting index corresponding to the ``i`` th screen group.
        From this index, reading ``group_sizes[screen_set[i]]`` number of elements
        will grab values corresponding to the full ``i`` th screen group block.
        )delimiter")
        .def_readonly("screen_vars", &state_t::screen_vars, R"delimiter(
        List of :math:`D_k^2` where :math:`D_k` is from the SVD of :math:`\sqrt{W} X_{c,k}` 
        along the screen groups :math:`k` and for possibly column-centered (weighted by :math:`W`) :math:`X_k`.
        ``screen_vars[b:b+p]`` is :math:`D_k` for the ``i`` th screen group where
        ``k = screen_set[i]``,
        ``b = screen_begins[i]``,
        and ``p = group_sizes[k]``.
        )delimiter")
        .def_readonly("screen_transforms", &state_t::screen_transforms, R"delimiter(
        List of :math:`V_k` where :math:`V_k` is from the SVD of :math:`\sqrt{W} X_{c,k}`
        along the screen groups :math:`k` and for possibly column-centered (weighted by :math:`W`) :math:`X_k`.
        It *only* needs to be properly initialized for groups with size > 1.
        ``screen_transforms[i]`` is :math:`V_k` for the ``i`` th screen group where
        ``k = screen_set[i]``.
        )delimiter")
        .def_readonly("lmda_path", &state_t::lmda_path, R"delimiter(
        Regularization sequence to fit on.
        )delimiter")
        .def_readonly("intercept", &state_t::intercept, R"delimiter(
        ``True`` to fit with intercept.
        )delimiter")
        .def_readonly("max_active_size", &state_t::max_active_size, R"delimiter(
        Maximum number of active groups allowed.
        )delimiter")
        .def_readonly("max_iters", &state_t::max_iters, R"delimiter(
        Maximum number of coordinate descents.
        )delimiter")
        .def_readonly("tol", &state_t::tol, R"delimiter(
        Convergence tolerance.
        )delimiter")
        .def_readonly("adev_tol", &state_t::adev_tol, R"delimiter(
        Percent deviance explained tolerance.
        )delimiter")
        .def_readonly("ddev_tol", &state_t::ddev_tol, R"delimiter(
        Difference in percent deviance explained tolerance.
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
        Unnormalized :math:`R^2` value at ``screen_beta``.
        The unnormalized :math:`R^2` is given by :math:`\|y_c\|_{W}^2 - \|y_c-X_c\beta\|_{W}^2`.
        )delimiter")
        .def_readonly("screen_beta", &state_t::screen_beta, R"delimiter(
        Coefficient vector on the screen set.
        ``screen_beta[b:b+p]`` is the coefficient for the ``i`` th screen group 
        where
        ``k = screen_set[i]``,
        ``b = screen_begins[i]``,
        and ``p = group_sizes[k]``.
        )delimiter")
        .def_readonly("screen_is_active", &state_t::screen_is_active, R"delimiter(
        Boolean vector that indicates whether each screen group in ``groups`` is active or not.
        ``screen_is_active[i]`` is ``True`` if and only if ``screen_set[i]`` is active.
        )delimiter")
        .def_property_readonly("active_set", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<index_t>>(
                s.active_set.data(),
                s.active_set.size()
            );
        }, R"delimiter(
        List of indices into ``screen_set`` that correspond to active groups.
        ``screen_set[active_set[i]]`` is the ``i`` th active group.
        An active group is one with non-zero coefficient block,
        that is, for every ``i`` th active group, 
        ``screen_beta[b:b+p] == 0`` where 
        ``j = active_set[i]``,
        ``k = screen_set[j]``,
        ``b = screen_begins[j]``,
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
        ``groups[screen_set[active_order[i]]]`` is the ``i`` th active group in ascending order.
        )delimiter")
        .def_property_readonly("betas", [](const state_t& s) {
            return convert_betas(
                s.group_sizes.sum(),
                s.betas
            );
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
        .def_readonly("iters", &state_t::iters, R"delimiter(
        Number of coordinate descents taken.
        )delimiter")
        .def_property_readonly("benchmark_screen", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<double>>(
                s.benchmark_screen.data(),
                s.benchmark_screen.size()
            );
        }, R"delimiter(
        Benchmark time for performing coordinate-descent on the screen set for each :math:`\lambda`.
        )delimiter")
        .def_property_readonly("benchmark_active", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<double>>(
                s.benchmark_active.data(),
                s.benchmark_active.size()
            );
        }, R"delimiter(
        Benchmark time for performing coordinate-descent on the active set for each :math:`\lambda`.
        )delimiter")
        ;
}

template <class MatrixType>
class PyStateGaussianPinNaive: public ad::state::StateGaussianPinNaive<MatrixType>
{
    using base_t = ad::state::StateGaussianPinNaive<MatrixType>;
public:
    using base_t::base_t;
    PyStateGaussianPinNaive(base_t&& base) : base_t(std::move(base)) {}
};

template <class MatrixType>
void state_gaussian_pin_naive(py::module_& m, const char* name)
{
    using matrix_t = MatrixType;
    using state_t = ad::state::StateGaussianPinNaive<matrix_t>;
    using base_t = typename state_t::base_t;
    using value_t = typename state_t::value_t;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_bool_t = typename state_t::vec_bool_t;
    using dyn_vec_mat_value_t = typename state_t::dyn_vec_mat_value_t;

    py::class_<state_t, base_t, PyStateGaussianPinNaive<matrix_t>>(m, name)
        .def(py::init<
            matrix_t&,
            value_t,
            value_t,
            const Eigen::Ref<const vec_index_t>&, 
            const Eigen::Ref<const vec_index_t>&,
            value_t, 
            const Eigen::Ref<const vec_value_t>&,
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
            size_t,
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
            py::arg("weights").noconvert(),
            py::arg("screen_set").noconvert(),
            py::arg("screen_g1").noconvert(),
            py::arg("screen_g2").noconvert(),
            py::arg("screen_begins").noconvert(),
            py::arg("screen_vars").noconvert(),
            py::arg("screen_X_means").noconvert(),
            py::arg("screen_transforms").noconvert(),
            py::arg("lmda_path").noconvert(),
            py::arg("intercept"),
            py::arg("max_active_size"),
            py::arg("max_iters"),
            py::arg("tol"),
            py::arg("adev_tol"),
            py::arg("ddev_tol"),
            py::arg("newton_tol"),
            py::arg("newton_max_iters"),
            py::arg("n_threads"),
            py::arg("rsq"),
            py::arg("resid").noconvert(),
            py::arg("resid_sum"),
            py::arg("screen_beta").noconvert(),
            py::arg("screen_is_active").noconvert()
        )
        .def(py::init([](const state_t& s) { return new state_t(s); }))
        .def_readonly("weights", &state_t::weights, R"delimiter(
        Observation weights.
        )delimiter")
        .def_readonly("y_mean", &state_t::y_mean, R"delimiter(
        Mean (weighted by :math:`W`) of :math:`y`.
        )delimiter")
        .def_readonly("y_var", &state_t::y_var, R"delimiter(
        :math:`\ell_2` norm squared of :math:`y_c` (weighted by :math:`W`).
        )delimiter")
        .def_readonly("screen_X_means", &state_t::screen_X_means, R"delimiter(
        Column means (weighted by :math:`W`) of :math:`X` for screen groups.
        )delimiter")
        .def_readonly("X", &state_t::X, R"delimiter(
        Feature matrix.
        )delimiter")
        .def_readonly("resid", &state_t::resid, R"delimiter(
        Residual :math:`W(y_c-X\beta)` at ``screen_beta`` 
        (note that it always uses uncentered :math:`X`!).

        .. note:: 
            This definition is unconventional.
            This was done deliberately as the algorithm is most efficient 
            when it updates this quantity compared to 
            the conventional quantity :math:`y_c-X_c \beta`.

        )delimiter")
        .def_readonly("resid_sum", &state_t::resid_sum, R"delimiter(
        Sum of ``resid``.
        )delimiter")
        ;
}

template <class MatrixType>
class PyStateGaussianPinCov : public ad::state::StateGaussianPinCov<MatrixType>
{
    using base_t = ad::state::StateGaussianPinCov<MatrixType>;
public:
    using base_t::base_t;
    PyStateGaussianPinCov(base_t&& base) : base_t(std::move(base)) {}
};

template <class MatrixType>
void state_gaussian_pin_cov(py::module_& m, const char* name)
{
    using matrix_t = MatrixType;
    using state_t = ad::state::StateGaussianPinCov<matrix_t>;
    using base_t = typename state_t::base_t;
    using value_t = typename state_t::value_t;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_bool_t = typename state_t::vec_bool_t;
    using dyn_vec_mat_value_t = typename state_t::dyn_vec_mat_value_t;

    py::class_<state_t, base_t, PyStateGaussianPinCov<matrix_t>>(m, name)
        .def(py::init<
            matrix_t&,
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
            const dyn_vec_mat_value_t&,
            const Eigen::Ref<const vec_value_t>&, 
            size_t,
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
            py::arg("y_var"),
            py::arg("groups").noconvert(),
            py::arg("group_sizes").noconvert(),
            py::arg("alpha"),
            py::arg("penalty").noconvert(),
            py::arg("screen_set").noconvert(),
            py::arg("screen_g1").noconvert(),
            py::arg("screen_g2").noconvert(),
            py::arg("screen_begins").noconvert(),
            py::arg("screen_vars").noconvert(),
            py::arg("screen_transforms").noconvert(),
            py::arg("lmda_path").noconvert(),
            py::arg("max_active_size"),
            py::arg("max_iters"),
            py::arg("tol"),
            py::arg("adev_tol"),
            py::arg("ddev_tol"),
            py::arg("newton_tol"),
            py::arg("newton_max_iters"),
            py::arg("n_threads"),
            py::arg("rsq"),
            py::arg("screen_beta").noconvert(),
            py::arg("screen_grad").noconvert(),
            py::arg("screen_is_active").noconvert()
        )
        .def(py::init([](const state_t& s) { return new state_t(s); }))
        .def_readonly("y_var", &state_t::y_var, R"delimiter(
        :math:`\ell_2` norm squared of :math:`y_c` (weighted by :math:`W`).
        If the user does not have access to this quantity, 
        they may be set it to ``-1``.
        The only effect this variable has on the algorithm is early stopping rule.
        Hence, with ``-1``, the early stopping rule is effectively disabled.
        )delimiter")
        .def_readonly("A", &state_t::A, R"delimiter(
        Covariance matrix :math:`X_c^\top W X_c` where :math:`X_c` is column-centered 
        (weighted by :math:`W`) to fit with intercept.
        It is typically one of the matrices defined in ``adelie.matrix`` sub-module.
        )delimiter")
        .def_readonly("screen_grad", &state_t::screen_grad, R"delimiter(
        Gradient :math:`X_{c,k}^\top W (y_c-X_c\beta)` on the screen groups :math:`k` where :math:`\beta` is given by ``screen_beta``.
        ``screen_grad[b:b+p]`` is the gradient for the ``i`` th screen group
        where 
        ``k = screen_set[i]``,
        ``b = screen_begins[i]``,
        and ``p = group_sizes[k]``.
        )delimiter")
        ;
}

// ========================================================================
// Gaussian State 
// ========================================================================

template <class ValueType>
void state_gaussian_base(py::module_& m, const char* name)
{
    using state_t = ad::state::StateGaussianBase<ValueType>;
    using value_t = typename state_t::value_t;
    using safe_bool_t = typename state_t::safe_bool_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_bool_t = typename state_t::vec_bool_t;
    py::class_<state_t>(m, name, R"delimiter(
        Base core state class for all gaussian methods.
        )delimiter") 
        .def(py::init<
            const Eigen::Ref<const vec_index_t>&, 
            const Eigen::Ref<const vec_index_t>&,
            value_t, 
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            value_t,
            value_t,
            size_t,
            size_t,
            size_t,
            value_t,
            size_t,
            value_t,
            const std::string&,
            size_t,
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
            py::arg("weights").noconvert(),
            py::arg("lmda_path").noconvert(),
            py::arg("lmda_max"),
            py::arg("min_ratio"),
            py::arg("lmda_path_size"),
            py::arg("max_screen_size"),
            py::arg("max_active_size"),
            py::arg("pivot_subset_ratio"),
            py::arg("pivot_subset_min"),
            py::arg("pivot_slack_ratio"),
            py::arg("screen_rule"),
            py::arg("max_iters"),
            py::arg("tol"),
            py::arg("adev_tol"),
            py::arg("ddev_tol"),
            py::arg("newton_tol"),
            py::arg("newton_max_iters"),
            py::arg("early_exit"),
            py::arg("setup_lmda_max"),
            py::arg("setup_lmda_path"),
            py::arg("intercept"),
            py::arg("n_threads"),
            py::arg("screen_set").noconvert(),
            py::arg("screen_beta").noconvert(),
            py::arg("screen_is_active").noconvert(),
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
        .def_readonly("weights", &state_t::weights, R"delimiter(
        Observation weights.
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
        .def_readonly("max_screen_size", &state_t::max_screen_size, R"delimiter(
        Maximum number of screen groups allowed.
        The function will return a valid state and guarantees to have screen set size
        less than or equal to ``max_screen_size``.
        )delimiter")
        .def_readonly("max_active_size", &state_t::max_active_size, R"delimiter(
        Maximum number of active groups allowed.
        The function will return a valid state and guarantees to have active set size
        less than or equal to ``max_active_size``.
        )delimiter")
        .def_readonly("pivot_subset_ratio", &state_t::pivot_subset_ratio, R"delimiter(
        If screening takes place, then the ``(1 + pivot_subset_ratio) * s``
        largest gradient norms are used to determine the pivot point
        where ``s`` is the current screen set size.
        It is only used if ``screen_rule == "pivot"``.
        )delimiter")
        .def_readonly("pivot_subset_min", &state_t::pivot_subset_min, R"delimiter(
        If screening takes place, then at least ``pivot_subset_min``
        number of gradient norms are used to determine the pivot point.
        It is only used if ``screen_rule == "pivot"``.
        )delimiter")
        .def_readonly("pivot_slack_ratio", &state_t::pivot_slack_ratio, R"delimiter(
        If screening takes place, then ``pivot_slack_ratio``
        number of groups with next smallest (new) active scores 
        below the pivot point are also added to the screen set as slack.
        It is only used if ``screen_rule == "pivot"``.
        )delimiter")
        .def_property_readonly("screen_rule", [](const state_t& s) -> std::string {
            switch (s.screen_rule) {
                case ad::util::screen_rule_type::_strong:
                    return "strong";
                case ad::util::screen_rule_type::_pivot:
                    return "pivot";
            }
            throw std::runtime_error("Invalid screen rule type!");
        }, R"delimiter(
        Strong rule type.
        )delimiter")
        .def_readonly("max_iters", &state_t::max_iters, R"delimiter(
        Maximum number of coordinate descents.
        )delimiter")
        .def_readonly("tol", &state_t::tol, R"delimiter(
        Convergence tolerance.
        )delimiter")
        .def_readonly("adev_tol", &state_t::adev_tol, R"delimiter(
        Percent deviance explained tolerance.
        )delimiter")
        .def_readonly("ddev_tol", &state_t::ddev_tol, R"delimiter(
        Difference in percent deviance explained tolerance.
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
        .def_readonly("screen_hashset", &state_t::screen_hashset, R"delimiter(
        Hashmap containing the same values as ``screen_set``.
        It is used to check if a given group is screen or not.
        )delimiter")
        .def_property_readonly("screen_set", [](const state_t& s) {
            return Eigen::Map<const vec_index_t>(s.screen_set.data(), s.screen_set.size());
        }, R"delimiter(
        List of indices into ``groups`` that correspond to the screen groups.
        ``screen_set[i]`` is ``i`` th screen group.
        ``screen_set`` must contain at least the true (optimal) active groups
        when the regularization is given by ``lmda``.
        )delimiter")
        .def_property_readonly("screen_g1", [](const state_t& s) {
            return Eigen::Map<const vec_index_t>(s.screen_g1.data(), s.screen_g1.size());
        }, R"delimiter(
        List of indices into ``screen_set`` that correspond to groups of size ``1``.
        ``screen_set[screen_g1[i]]`` is the ``i`` th screen group of size ``1``
        such that ``group_sizes[screen_set[screen_g1[i]]]`` is ``1``.
        )delimiter")
        .def_property_readonly("screen_g2", [](const state_t& s) {
            return Eigen::Map<const vec_index_t>(s.screen_g2.data(), s.screen_g2.size());
        }, R"delimiter(
        List of indices into ``screen_set`` that correspond to groups more than size ``1``.
        ``screen_set[screen_g2[i]]`` is the ``i`` th screen group of size more than ``1``
        such that ``group_sizes[screen_set[screen_g2[i]]]`` is more than ``1``.
        )delimiter")
        .def_property_readonly("screen_begins", [](const state_t& s) {
            return Eigen::Map<const vec_index_t>(s.screen_begins.data(), s.screen_begins.size());
        }, R"delimiter(
        List of indices that index a corresponding list of values for each screen group.
        ``screen_begins[i]`` is the starting index corresponding to the ``i`` th screen group.
        From this index, reading ``group_sizes[screen_set[i]]`` number of elements
        will grab values corresponding to the full ``i`` th screen group block.
        )delimiter")
        .def_property_readonly("screen_beta", [](const state_t& s) {
            return Eigen::Map<const vec_value_t>(s.screen_beta.data(), s.screen_beta.size());
        }, R"delimiter(
        Coefficient vector on the screen set.
        ``screen_beta[b:b+p]`` is the coefficient for the ``i`` th screen group 
        where
        ``k = screen_set[i]``,
        ``b = screen_begins[i]``,
        and ``p = group_sizes[k]``.
        The values can be arbitrary but it is recommended to be close to the solution at ``lmda``.
        )delimiter")
        .def_property_readonly("screen_is_active", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<safe_bool_t>>(
                s.screen_is_active.data(), 
                s.screen_is_active.size()
            );
        }, R"delimiter(
        Boolean vector that indicates whether each screen group in ``groups`` is active or not.
        ``screen_is_active[i]`` is ``True`` if and only if ``screen_set[i]`` is active.
        )delimiter")
        .def_readonly("rsq", &state_t::rsq, R"delimiter(
        The unnormalized :math:`R^2` given by :math:`\|y_c\|_{W}^2 - \|y_c-X_c\beta\|_{W}^2`
        where :math:`\beta` is given by ``screen_beta``.
        )delimiter")
        .def_readonly("lmda", &state_t::lmda, R"delimiter(
        The last regularization parameter that was attempted to be solved.
        )delimiter")
        .def_readonly("grad", &state_t::grad, R"delimiter(
        The full gradient :math:`X_c^\top W (y_c - X_c\beta)` where
        :math:`\beta` is given by ``screen_beta``.
        )delimiter")
        .def_readonly("abs_grad", &state_t::abs_grad, R"delimiter(
        The :math:`\ell_2` norms of ``grad`` across each group.
        ``abs_grad[i]`` is given by ``np.linalg.norm(grad[g:g+gs] - lmda * penalty[i] * (1-alpha) * beta[g:g+gs])``
        where ``g = groups[i]``,
        ``gs = group_sizes[i]``,
        and ``beta`` is the full solution vector represented by ``screen_beta``.
        )delimiter")
        .def_property_readonly("betas", [](const state_t& s) {
            return convert_betas(
                s.group_sizes.sum(),
                s.betas
            );
        }, R"delimiter(
        ``betas[i]`` is the (untransformed) solution corresponding to ``lmdas[i]``.
        )delimiter")
        .def_property_readonly("devs", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<value_t>>(
                s.devs.data(),
                s.devs.size()
            );
        }, R"delimiter(
        ``devs[i]`` is the (normalized) :math:`R^2` at ``betas[i]``.
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
        .def_property_readonly("benchmark_fit_screen", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<double>>(
                s.benchmark_fit_screen.data(),
                s.benchmark_fit_screen.size()
            );
        }, R"delimiter(
        Fit time on the screen set for a given BASIL iteration.
        )delimiter")
        .def_property_readonly("benchmark_fit_active", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<double>>(
                s.benchmark_fit_active.data(),
                s.benchmark_fit_active.size()
            );
        }, R"delimiter(
        Fit time on the active set for a given BASIL iteration.
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
        .def_property_readonly("n_valid_solutions", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<int>>(
                s.n_valid_solutions.data(),
                s.n_valid_solutions.size()
            );
        }, R"delimiter(
        Number of valid solutions for a given BASIL iteration.
        )delimiter")
        .def_property_readonly("active_sizes", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<int>>(
                s.active_sizes.data(),
                s.active_sizes.size()
            );
        }, R"delimiter(
        Active set size for every saved solution.
        )delimiter")
        .def_property_readonly("screen_sizes", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<int>>(
                s.screen_sizes.data(),
                s.screen_sizes.size()
            );
        }, R"delimiter(
        Strong set size for every saved solution.
        )delimiter")
        ;
}

template <class MatrixType>
class PyStateGaussianNaive : public ad::state::StateGaussianNaive<MatrixType>
{
    using base_t = ad::state::StateGaussianNaive<MatrixType>;
public:
    using base_t::base_t;
    PyStateGaussianNaive(base_t&& base) : base_t(std::move(base)) {}
};

template <class MatrixType>
void state_gaussian_naive(py::module_& m, const char* name)
{
    using matrix_t = MatrixType;
    using state_t = ad::state::StateGaussianNaive<matrix_t>;
    using base_t = typename state_t::base_t;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_bool_t = typename state_t::vec_bool_t;
    py::class_<state_t, base_t, PyStateGaussianNaive<matrix_t>>(m, name)
        .def(py::init<
            matrix_t&,
            const Eigen::Ref<const vec_value_t>&,
            value_t,
            value_t,
            const Eigen::Ref<const vec_value_t>&,
            value_t,
            const Eigen::Ref<const vec_index_t>&,
            const Eigen::Ref<const vec_index_t>&,
            value_t, 
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            value_t,
            value_t,
            size_t,
            size_t,
            size_t,
            value_t,
            size_t,
            value_t,
            const std::string&,
            size_t,
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
            py::arg("y_mean"),
            py::arg("y_var"),
            py::arg("resid").noconvert(),
            py::arg("resid_sum"),
            py::arg("groups").noconvert(),
            py::arg("group_sizes").noconvert(),
            py::arg("alpha"),
            py::arg("penalty").noconvert(),
            py::arg("weights").noconvert(),
            py::arg("lmda_path").noconvert(),
            py::arg("lmda_max"),
            py::arg("min_ratio"),
            py::arg("lmda_path_size"),
            py::arg("max_screen_size"),
            py::arg("max_active_size"),
            py::arg("pivot_subset_ratio"),
            py::arg("pivot_subset_min"),
            py::arg("pivot_slack_ratio"),
            py::arg("screen_rule"),
            py::arg("max_iters"),
            py::arg("tol"),
            py::arg("adev_tol"),
            py::arg("ddev_tol"),
            py::arg("newton_tol"),
            py::arg("newton_max_iters"),
            py::arg("early_exit"),
            py::arg("setup_lmda_max"),
            py::arg("setup_lmda_path"),
            py::arg("intercept"),
            py::arg("n_threads"),
            py::arg("screen_set").noconvert(),
            py::arg("screen_beta").noconvert(),
            py::arg("screen_is_active").noconvert(),
            py::arg("rsq"),
            py::arg("lmda"),
            py::arg("grad").noconvert()
        )
        .def(py::init([](const state_t& s) { return new state_t(s); }))
        .def_readonly("X_means", &state_t::X_means, R"delimiter(
        Column means (weighted by :math:`W`) of ``X``.
        )delimiter")
        .def_readonly("y_mean", &state_t::y_mean, R"delimiter(
        The mean (weighted by :math:`W`) of the offsetted response vector :math:`y-\eta^0`.
        )delimiter")
        .def_readonly("y_var", &state_t::y_var, R"delimiter(
        The variance (weighted by :math:`W`) of the offsetted response vector :math:`y-\eta^0`, i.e. 
        :math:`\|y_c\|_{W}^2`.
        )delimiter")
        .def_readonly("X", &state_t::X, R"delimiter(
        Feature matrix.
        )delimiter")
        .def_readonly("resid", &state_t::resid, R"delimiter(
        Residual :math:`W(y_c - X \beta)` where :math:`\beta` is given by ``screen_beta``.
        )delimiter")
        .def_readonly("resid_sum", &state_t::resid_sum, R"delimiter(
        Sum of ``resid``.
        )delimiter")
        .def_readonly("screen_X_means", &state_t::screen_X_means, R"delimiter(
        Column means (weighted by :math:`W`) of ``X`` on the screen set.
        )delimiter")
        .def_readonly("screen_transforms", &state_t::screen_transforms, R"delimiter(
        The :math:`V` from the SVD of :math:`\sqrt{W} X_{c,k}` where :math:`X_c` 
        is the possibly centered (weighted by :math:`W`) feature matrix and :math:`k` is an index to ``screen_set``.
        )delimiter")
        .def_property_readonly("screen_vars", [](const state_t& s) {
            return Eigen::Map<const vec_value_t>(s.screen_vars.data(), s.screen_vars.size());
        }, R"delimiter(
        The :math:`D^2` from the SVD of :math:`\sqrt{W} X_{c,k}` where :math:`X_c` 
        is the possibly centered (weighted by :math:`W`) feature matrix and :math:`k` is an index to ``screen_set``.
        )delimiter")
        ;
}

template <class MatrixType>
class PyStateMultiGaussianNaive : public ad::state::StateMultiGaussianNaive<MatrixType>
{
    using base_t = ad::state::StateMultiGaussianNaive<MatrixType>;
public:
    using base_t::base_t;
    PyStateMultiGaussianNaive(base_t&& base) : base_t(std::move(base)) {}
};

template <class MatrixType>
void state_multi_gaussian_naive(py::module_& m, const char* name)
{
    using matrix_t = MatrixType;
    using state_t = ad::state::StateMultiGaussianNaive<matrix_t>;
    using base_t = typename state_t::base_t;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_bool_t = typename state_t::vec_bool_t;
    py::class_<state_t, base_t, PyStateMultiGaussianNaive<matrix_t>>(m, name)
        .def(py::init<
            size_t,
            bool,
            matrix_t&,
            const Eigen::Ref<const vec_value_t>&,
            value_t,
            value_t,
            const Eigen::Ref<const vec_value_t>&,
            value_t,
            const Eigen::Ref<const vec_index_t>&,
            const Eigen::Ref<const vec_index_t>&,
            value_t, 
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            value_t,
            value_t,
            size_t,
            size_t,
            size_t,
            value_t,
            size_t,
            value_t,
            const std::string&,
            size_t,
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
            py::arg("n_classes"),
            py::arg("multi_intercept"),
            py::arg("X"),
            py::arg("X_means").noconvert(),
            py::arg("y_mean"),
            py::arg("y_var"),
            py::arg("resid").noconvert(),
            py::arg("resid_sum"),
            py::arg("groups").noconvert(),
            py::arg("group_sizes").noconvert(),
            py::arg("alpha"),
            py::arg("penalty").noconvert(),
            py::arg("weights").noconvert(),
            py::arg("lmda_path").noconvert(),
            py::arg("lmda_max"),
            py::arg("min_ratio"),
            py::arg("lmda_path_size"),
            py::arg("max_screen_size"),
            py::arg("max_active_size"),
            py::arg("pivot_subset_ratio"),
            py::arg("pivot_subset_min"),
            py::arg("pivot_slack_ratio"),
            py::arg("screen_rule"),
            py::arg("max_iters"),
            py::arg("tol"),
            py::arg("adev_tol"),
            py::arg("ddev_tol"),
            py::arg("newton_tol"),
            py::arg("newton_max_iters"),
            py::arg("early_exit"),
            py::arg("setup_lmda_max"),
            py::arg("setup_lmda_path"),
            py::arg("intercept"),
            py::arg("n_threads"),
            py::arg("screen_set").noconvert(),
            py::arg("screen_beta").noconvert(),
            py::arg("screen_is_active").noconvert(),
            py::arg("rsq"),
            py::arg("lmda"),
            py::arg("grad").noconvert()
        )
        .def(py::init([](const state_t& s) { return new state_t(s); }))
        .def_readonly("n_classes", &state_t::n_classes, R"delimiter(
        Number of classes.
        )delimiter")
        .def_readonly("multi_intercept", &state_t::multi_intercept, R"delimiter(
        ``True`` if an intercept is added for each response.
        )delimiter")
        .def_property_readonly("betas", [](const state_t& s) {
            return convert_betas(
                s.group_sizes.sum() -  s.multi_intercept * s.n_classes,
                s.betas
            );
        }, R"delimiter(
        ``betas[i]`` is the (untransformed) solution corresponding to ``lmdas[i]``.
        )delimiter")
        .def_readonly("intercepts", &state_t::intercepts, R"delimiter(
        ``intercepts[i]`` is the intercept solution corresponding to ``lmdas[i]`` for each class.
        )delimiter")
        ;
}

// ========================================================================
// GLM State 
// ========================================================================

template <class GlmType>
void state_glm_base(py::module_& m, const char* name)
{
    using glm_t = GlmType;
    using state_t = ad::state::StateGlmBase<glm_t>;
    using value_t = typename state_t::value_t;
    using safe_bool_t = typename state_t::safe_bool_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_bool_t = typename state_t::vec_bool_t;
    py::class_<state_t>(m, name, R"delimiter(
        Base core state class for all GLM methods.
        )delimiter") 
        .def(py::init<
            glm_t&,
            const Eigen::Ref<const vec_index_t>&, 
            const Eigen::Ref<const vec_index_t>&,
            value_t, 
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            value_t,
            value_t,
            value_t,
            value_t,
            size_t,
            size_t,
            size_t,
            value_t,
            size_t,
            value_t,
            const std::string&,
            size_t,
            value_t,
            size_t,
            value_t,
            value_t,
            value_t,
            value_t,
            size_t,
            bool,
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
            py::arg("glm"),
            py::arg("groups").noconvert(),
            py::arg("group_sizes").noconvert(),
            py::arg("alpha"),
            py::arg("penalty").noconvert(),
            py::arg("weights").noconvert(),
            py::arg("offsets").noconvert(),
            py::arg("lmda_path").noconvert(),
            py::arg("dev_null"),
            py::arg("dev_full"),
            py::arg("lmda_max"),
            py::arg("min_ratio"),
            py::arg("lmda_path_size"),
            py::arg("max_screen_size"),
            py::arg("max_active_size"),
            py::arg("pivot_subset_ratio"),
            py::arg("pivot_subset_min"),
            py::arg("pivot_slack_ratio"),
            py::arg("screen_rule"),
            py::arg("irls_max_iters"),
            py::arg("irls_tol"),
            py::arg("max_iters"),
            py::arg("tol"),
            py::arg("adev_tol"),
            py::arg("ddev_tol"),
            py::arg("newton_tol"),
            py::arg("newton_max_iters"),
            py::arg("early_exit"),
            py::arg("setup_dev_null"),
            py::arg("setup_lmda_max"),
            py::arg("setup_lmda_path"),
            py::arg("intercept"),
            py::arg("n_threads"),
            py::arg("screen_set").noconvert(),
            py::arg("screen_beta").noconvert(),
            py::arg("screen_is_active").noconvert(),
            py::arg("beta0"),
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
        .def_readonly("weights", &state_t::weights, R"delimiter(
        Observation weights.
        )delimiter")
        .def_readonly("offsets", &state_t::offsets, R"delimiter(
        Observation offsets :math:`\eta^0`.
        )delimiter")
        .def_readonly("min_ratio", &state_t::min_ratio, R"delimiter(
        The ratio between the largest and smallest :math:`\lambda` in the regularization sequence
        if it is to be generated.
        )delimiter")
        .def_readonly("lmda_path_size", &state_t::lmda_path_size, R"delimiter(
        Number of regularizations in the path if it is to be generated.
        )delimiter")
        .def_readonly("max_screen_size", &state_t::max_screen_size, R"delimiter(
        Maximum number of screen groups allowed.
        The function will return a valid state and guarantees to have screen size
        less than or equal to ``max_screen_size``.
        )delimiter")
        .def_readonly("max_active_size", &state_t::max_active_size, R"delimiter(
        Maximum number of active groups allowed.
        The function will return a valid state and guarantees to have active set size
        less than or equal to ``max_active_size``.
        )delimiter")
        .def_readonly("pivot_subset_ratio", &state_t::pivot_subset_ratio, R"delimiter(
        If screening takes place, then the ``(1 + pivot_subset_ratio) * s``
        largest gradient norms are used to determine the pivot point
        where ``s`` is the current screen set size.
        It is only used if ``screen_rule == "pivot"``.
        )delimiter")
        .def_readonly("pivot_subset_min", &state_t::pivot_subset_min, R"delimiter(
        If screening takes place, then at least ``pivot_subset_min``
        number of gradient norms are used to determine the pivot point.
        It is only used if ``screen_rule == "pivot"``.
        )delimiter")
        .def_readonly("pivot_slack_ratio", &state_t::pivot_slack_ratio, R"delimiter(
        If screening takes place, then ``pivot_slack_ratio``
        number of groups with next smallest (new) active scores 
        below the pivot point are also added to the screen set as slack.
        It is only used if ``screen_rule == "pivot"``.
        )delimiter")
        .def_property_readonly("screen_rule", [](const state_t& s) -> std::string {
            switch (s.screen_rule) {
                case ad::util::screen_rule_type::_strong:
                    return "strong";
                case ad::util::screen_rule_type::_pivot:
                    return "pivot";
            }
            throw std::runtime_error("Invalid screen rule type!");
        }, R"delimiter(
        Strong rule type.
        )delimiter")
        .def_readonly("irls_max_iters", &state_t::irls_max_iters, R"delimiter(
        Maximum number of IRLS iterations.
        )delimiter")
        .def_readonly("irls_tol", &state_t::irls_tol, R"delimiter(
        IRLS convergence tolerance.
        )delimiter")
        .def_readonly("max_iters", &state_t::max_iters, R"delimiter(
        Maximum number of coordinate descents.
        )delimiter")
        .def_readonly("tol", &state_t::tol, R"delimiter(
        Convergence tolerance.
        )delimiter")
        .def_readonly("adev_tol", &state_t::adev_tol, R"delimiter(
        Percent deviance explained tolerance.
        )delimiter")
        .def_readonly("ddev_tol", &state_t::ddev_tol, R"delimiter(
        Difference in percent deviance explained tolerance.
        )delimiter")
        .def_readonly("newton_tol", &state_t::newton_tol, R"delimiter(
        Convergence tolerance for the BCD update.
        )delimiter")
        .def_readonly("newton_max_iters", &state_t::newton_max_iters, R"delimiter(
        Maximum number of iterations for the BCD update.
        )delimiter")
        .def_readonly("early_exit", &state_t::early_exit, R"delimiter(
        ``True`` if the function should early exit based on training percent deviance explained.
        )delimiter")
        .def_readonly("setup_dev_null", &state_t::setup_dev_null, R"delimiter(
        ``True`` if the function should setup ``dev_null``.
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
        .def_readonly("glm", &state_t::glm, R"delimiter(
        GLM object.
        )delimiter")
        .def_readonly("dev_null", &state_t::dev_null, R"delimiter(
        Null deviance :math:`D(\beta_0^\star \mathbf{1} + \eta^0)`
        from fitting an intercept-only model (if ``intercept`` is ``True``).
        )delimiter")
        .def_readonly("dev_full", &state_t::dev_full, R"delimiter(
        Full deviance :math:`D(\eta^\star)` where :math:`\eta^\star` is the minimizer.
        )delimiter")
        .def_readonly("lmda_max", &state_t::lmda_max, R"delimiter(
        The smallest :math:`\lambda` such that the true solution is zero
        for all coefficients that have a non-vanishing group lasso penalty (:math:`\ell_2`-norm).
        )delimiter")
        .def_readonly("lmda_path", &state_t::lmda_path, R"delimiter(
        The regularization path to solve for.
        The full path is not considered if ``early_exit`` is ``True``.
        It is recommended that the path is sorted in decreasing order.
        )delimiter")
        .def_readonly("screen_hashset", &state_t::screen_hashset, R"delimiter(
        Hashmap containing the same values as ``screen_set``.
        It is used to check if a given group is screen or not.
        )delimiter")
        .def_property_readonly("screen_set", [](const state_t& s) {
            return Eigen::Map<const vec_index_t>(s.screen_set.data(), s.screen_set.size());
        }, R"delimiter(
        List of indices into ``groups`` that correspond to the screen groups.
        ``screen_set[i]`` is ``i`` th screen group.
        ``screen_set`` must contain at least the true (optimal) active groups
        when the regularization is given by ``lmda``.
        )delimiter")
        .def_property_readonly("screen_g1", [](const state_t& s) {
            return Eigen::Map<const vec_index_t>(s.screen_g1.data(), s.screen_g1.size());
        }, R"delimiter(
        List of indices into ``screen_set`` that correspond to groups of size ``1``.
        ``screen_set[screen_g1[i]]`` is the ``i`` th screen group of size ``1``
        such that ``group_sizes[screen_set[screen_g1[i]]]`` is ``1``.
        )delimiter")
        .def_property_readonly("screen_g2", [](const state_t& s) {
            return Eigen::Map<const vec_index_t>(s.screen_g2.data(), s.screen_g2.size());
        }, R"delimiter(
        List of indices into ``screen_set`` that correspond to groups more than size ``1``.
        ``screen_set[screen_g2[i]]`` is the ``i`` th screen group of size more than ``1``
        such that ``group_sizes[screen_set[screen_g2[i]]]`` is more than ``1``.
        )delimiter")
        .def_property_readonly("screen_begins", [](const state_t& s) {
            return Eigen::Map<const vec_index_t>(s.screen_begins.data(), s.screen_begins.size());
        }, R"delimiter(
        List of indices that index a corresponding list of values for each screen group.
        ``screen_begins[i]`` is the starting index corresponding to the ``i`` th screen group.
        From this index, reading ``group_sizes[screen_set[i]]`` number of elements
        will grab values corresponding to the full ``i`` th screen group block.
        )delimiter")
        .def_property_readonly("screen_beta", [](const state_t& s) {
            return Eigen::Map<const vec_value_t>(s.screen_beta.data(), s.screen_beta.size());
        }, R"delimiter(
        Coefficient vector on the screen set.
        ``screen_beta[b:b+p]`` is the coefficient for the ``i`` th screen group 
        where
        ``k = screen_set[i]``,
        ``b = screen_begins[i]``,
        and ``p = group_sizes[k]``.
        The values can be arbitrary but it is recommended to be close to the solution at ``lmda``.
        )delimiter")
        .def_property_readonly("screen_is_active", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<safe_bool_t>>(
                s.screen_is_active.data(), 
                s.screen_is_active.size()
            );
        }, R"delimiter(
        Boolean vector that indicates whether each screen group in ``groups`` is active or not.
        ``screen_is_active[i]`` is ``True`` if and only if ``screen_set[i]`` is active.
        )delimiter")
        .def_readonly("beta0", &state_t::beta0, R"delimiter(
        The current intercept value.
        The value can be arbitrary but it is recommended to be close to the solution at ``lmda``.
        )delimiter")
        .def_readonly("lmda", &state_t::lmda, R"delimiter(
        The last regularization parameter that was attempted to be solved.
        )delimiter")
        .def_readonly("grad", &state_t::grad, R"delimiter(
        The full gradient :math:`X^\top (W y - \nabla A(\eta))` where
        :math:`\eta` is given by ``eta``.
        )delimiter")
        .def_readonly("abs_grad", &state_t::abs_grad, R"delimiter(
        The :math:`\ell_2` norms of ``grad`` across each group.
        ``abs_grad[i]`` is given by ``np.linalg.norm(grad[g:g+gs] - lmda * penalty[i] * (1-alpha) * beta[g:g+gs])``
        where ``g = groups[i]``,
        ``gs = group_sizes[i]``,
        and ``beta`` is the full solution vector represented by ``screen_beta``.
        )delimiter")
        .def_property_readonly("betas", [](const state_t& s) {
            return convert_betas(
                s.group_sizes.sum(),
                s.betas
            );
        }, R"delimiter(
        ``betas[i]`` is the solution corresponding to ``lmdas[i]``.
        )delimiter")
        .def_property_readonly("devs", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<value_t>>(
                s.devs.data(),
                s.devs.size()
            );
        }, R"delimiter(
        ``devs[i]`` is the (normalized) deviance at ``betas[i]``.
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
        .def_property_readonly("benchmark_fit_screen", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<double>>(
                s.benchmark_fit_screen.data(),
                s.benchmark_fit_screen.size()
            );
        }, R"delimiter(
        Fit time on the screen set for a given BASIL iteration.
        )delimiter")
        .def_property_readonly("benchmark_fit_active", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<double>>(
                s.benchmark_fit_active.data(),
                s.benchmark_fit_active.size()
            );
        }, R"delimiter(
        Fit time on the active set for a given BASIL iteration.
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
        .def_property_readonly("n_valid_solutions", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<int>>(
                s.n_valid_solutions.data(),
                s.n_valid_solutions.size()
            );
        }, R"delimiter(
        Number of valid solutions for a given BASIL iteration.
        )delimiter")
        .def_property_readonly("active_sizes", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<int>>(
                s.active_sizes.data(),
                s.active_sizes.size()
            );
        }, R"delimiter(
        Active set size for every saved solution.
        )delimiter")
        .def_property_readonly("screen_sizes", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<int>>(
                s.screen_sizes.data(),
                s.screen_sizes.size()
            );
        }, R"delimiter(
        Strong set size for every saved solution.
        )delimiter")
        ;
}

template <class GlmType, class MatrixType>
class PyStateGlmNaive : public ad::state::StateGlmNaive<GlmType, MatrixType>
{
    using base_t = ad::state::StateGlmNaive<GlmType, MatrixType>;
public:
    using base_t::base_t;
    PyStateGlmNaive(base_t&& base) : base_t(std::move(base)) {}
};

template <class GlmType, class MatrixType>
void state_glm_naive(py::module_& m, const char* name)
{
    using glm_t = GlmType;
    using matrix_t = MatrixType;
    using state_t = ad::state::StateGlmNaive<glm_t, matrix_t>;
    using base_t = typename state_t::base_t;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_bool_t = typename state_t::vec_bool_t;
    py::class_<state_t, base_t, PyStateGlmNaive<glm_t, matrix_t>>(m, name)
        .def(py::init<
            glm_t&,
            matrix_t&,
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_index_t>&,
            const Eigen::Ref<const vec_index_t>&,
            value_t, 
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            value_t,
            value_t,
            value_t,
            value_t,
            size_t,
            size_t,
            size_t,
            value_t,
            size_t,
            value_t,
            const std::string&,
            size_t,
            value_t,
            size_t,
            value_t,
            value_t,
            value_t,
            value_t,
            size_t,
            bool,
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
            py::arg("glm"),
            py::arg("X"),
            py::arg("y").noconvert(),
            py::arg("eta").noconvert(),
            py::arg("mu").noconvert(),
            py::arg("groups").noconvert(),
            py::arg("group_sizes").noconvert(),
            py::arg("alpha"),
            py::arg("penalty").noconvert(),
            py::arg("weights").noconvert(),
            py::arg("offsets").noconvert(),
            py::arg("lmda_path").noconvert(),
            py::arg("dev_null"),
            py::arg("dev_full"),
            py::arg("lmda_max"),
            py::arg("min_ratio"),
            py::arg("lmda_path_size"),
            py::arg("max_screen_size"),
            py::arg("max_active_size"),
            py::arg("pivot_subset_ratio"),
            py::arg("pivot_subset_min"),
            py::arg("pivot_slack_ratio"),
            py::arg("screen_rule"),
            py::arg("irls_max_iters"),
            py::arg("irls_tol"),
            py::arg("max_iters"),
            py::arg("tol"),
            py::arg("adev_tol"),
            py::arg("ddev_tol"),
            py::arg("newton_tol"),
            py::arg("newton_max_iters"),
            py::arg("early_exit"),
            py::arg("setup_dev_null"),
            py::arg("setup_lmda_max"),
            py::arg("setup_lmda_path"),
            py::arg("intercept"),
            py::arg("n_threads"),
            py::arg("screen_set").noconvert(),
            py::arg("screen_beta").noconvert(),
            py::arg("screen_is_active").noconvert(),
            py::arg("beta0"),
            py::arg("lmda"),
            py::arg("grad").noconvert()
        )
        .def(py::init([](const state_t& s) { return new state_t(s); }))
        .def_readonly("y", &state_t::y, R"delimiter(
        Response vector.
        )delimiter")
        .def_readonly("X", &state_t::X, R"delimiter(
        Feature matrix.
        )delimiter")
        .def_readonly("eta", &state_t::eta, R"delimiter(
        The natural parameter :math:`\eta = X\beta + \beta_0 \mathbf{1} + \eta^0`
        where :math:`\beta`, :math:`\beta_0`, and :math:`\eta^0` are given by
        ``screen_beta``, ``beta0``, and ``offsets``.
        )delimiter")
        .def_readonly("mu", &state_t::mu, R"delimiter(
        The mean parameter :math:`\mu = \nabla A(\eta)`
        where :math:`\eta` is given by ``eta``.
        )delimiter")
        ;
}

void register_state(py::module_& m)
{
    state_gaussian_pin_base<double>(m, "StateGaussianPinBase64");
    state_gaussian_pin_base<float>(m, "StateGaussianPinBase32");
    state_gaussian_pin_naive<ad::matrix::MatrixNaiveBase<double>>(m, "StateGaussianPinNaive64");
    state_gaussian_pin_naive<ad::matrix::MatrixNaiveBase<float>>(m, "StateGaussianPinNaive32");
    state_gaussian_pin_cov<ad::matrix::MatrixCovBase<double>>(m, "StateGaussianPinCov64");
    state_gaussian_pin_cov<ad::matrix::MatrixCovBase<float>>(m, "StateGaussianPinCov32");

    state_gaussian_base<double>(m, "StateGaussianBase64");
    state_gaussian_base<float>(m, "StateGaussianBase32");
    state_gaussian_naive<ad::matrix::MatrixNaiveBase<double>>(m, "StateGaussianNaive64");
    state_gaussian_naive<ad::matrix::MatrixNaiveBase<float>>(m, "StateGaussianNaive32");
    state_multi_gaussian_naive<ad::matrix::MatrixNaiveBase<double>>(m, "StateMultiGaussianNaive64");
    state_multi_gaussian_naive<ad::matrix::MatrixNaiveBase<float>>(m, "StateMultiGaussianNaive32");
    state_glm_base<ad::glm::GlmBase<double>>(m, "StateGlmBase64");
    state_glm_base<ad::glm::GlmBase<float>>(m, "StateGlmBase32");
    state_glm_naive<
        ad::glm::GlmBase<double>,
        ad::matrix::MatrixNaiveBase<double>
    >(m, "StateGlmNaive64");
    state_glm_naive<
        ad::glm::GlmBase<float>,
        ad::matrix::MatrixNaiveBase<float>
    >(m, "StateGlmNaive32");
}