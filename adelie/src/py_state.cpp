#include "py_decl.hpp"
#include <state/state.hpp>
#include <adelie_core/util/stopwatch.hpp>

namespace py = pybind11;
namespace ad = adelie_core;
using namespace pybind11::literals; // to bring in the `_a` literal

template <class BetasType>
static auto convert_sparse_to_dense(
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

template <class StateType, class SolveType>
py::dict _solve(
    StateType& state,
    SolveType solve_f
)
{
    using sw_t = ad::util::Stopwatch;

    const auto check_user_interrupt = [&]() {
        if (PyErr_CheckSignals() != 0) {
            throw py::error_already_set();
        }
    };

    std::string error;

    // this is to redirect std::cerr to sys.stderr in Python.
    // https://pybind11.readthedocs.io/en/stable/advanced/pycpp/utilities.html?highlight=cout#capturing-standard-output-from-ostream
    py::scoped_estream_redirect _estream;
    sw_t sw;
    sw.start();
    try {
        solve_f(state, check_user_interrupt);
    } catch(const std::exception& e) {
        error = e.what(); 
    }
    double total_time = sw.elapsed();

    return py::dict("state"_a=state, "error"_a=error, "total_time"_a=total_time);
}

template <class StateType>
py::dict _solve(
    StateType& state
)
{
    return _solve(
        state,
        [&](auto& state, auto c) { state.solve(c); }
    );
}

template <class StateType>
py::dict _solve(
    StateType& state,
    bool display_progress_bar,
    std::function<bool(const StateType&)> exit_cond
)
{
    return _solve(
        state,
        [&](auto& state, auto c) {
            const auto exit_cond_f = [&]() {
                return exit_cond && exit_cond(state);
            };
            auto pb = ad::util::tq::trange(0);
            pb.set_display(display_progress_bar);
            pb.set_ostream(std::cerr);
            state.solve(pb, exit_cond_f, c);
        }
    );
}

template <class StateType, class GlmType>
py::dict _solve(
    StateType& state,
    GlmType& glm,
    bool display_progress_bar,
    std::function<bool(const StateType&)> exit_cond
)
{
    return _solve(
        state,
        [&](auto& state, auto c) {
            const auto exit_cond_f = [&]() {
                return exit_cond && exit_cond(state);
            };
            auto pb = ad::util::tq::trange(0);
            pb.set_display(display_progress_bar);
            pb.set_ostream(std::cerr);
            state.solve(glm, pb, exit_cond_f, c);
        }
    );
}

// ========================================================================
// Pin State 
// ========================================================================

template <class ConstraintType>
void state_gaussian_pin_base(py::module_& m, const char* name)
{
    using state_t = ad::state::StateGaussianPinBase<ConstraintType>;
    using index_t = typename state_t::index_t;
    using value_t = typename state_t::value_t;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_bool_t = typename state_t::vec_bool_t;
    using dyn_vec_mat_value_t = typename state_t::dyn_vec_mat_value_t;
    using dyn_vec_constraint_t = typename state_t::dyn_vec_constraint_t;

    py::class_<state_t>(m, name, R"delimiter(
        Base state class for Gaussian, pin classes.
        )delimiter")
        .def(py::init<
            const dyn_vec_constraint_t&,
            const Eigen::Ref<const vec_index_t>&, 
            const Eigen::Ref<const vec_index_t>&,
            value_t, 
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_index_t>&, 
            const Eigen::Ref<const vec_index_t>&, 
            const Eigen::Ref<const vec_value_t>&,
            const dyn_vec_mat_value_t&,
            const Eigen::Ref<const vec_value_t>&, 
            size_t,
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
            Eigen::Ref<vec_bool_t>,
            size_t,
            Eigen::Ref<vec_index_t>
        >(),
            py::arg("constraints").noconvert(),
            py::arg("groups").noconvert(),
            py::arg("group_sizes").noconvert(),
            py::arg("alpha"),
            py::arg("penalty").noconvert(),
            py::arg("screen_set").noconvert(),
            py::arg("screen_begins").noconvert(),
            py::arg("screen_vars").noconvert(),
            py::arg("screen_transforms").noconvert(),
            py::arg("lmda_path").noconvert(),
            py::arg("constraint_buffer_size"),
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
            py::arg("screen_is_active").noconvert(),
            py::arg("active_set_size"),
            py::arg("active_set").noconvert()
        )
        .def_readonly("constraints", &state_t::constraints, R"delimiter(
        List of constraints for each group.
        ``constraints[i]`` is the constraint object corresponding to group ``i``.
        )delimiter")
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
        )delimiter")
        .def_readonly("penalty", &state_t::penalty, R"delimiter(
        Penalty factor for each group in the same order as ``groups``.
        )delimiter")
        .def_readonly("screen_set", &state_t::screen_set, R"delimiter(
        List of indices into ``groups`` that correspond to the screen groups.
        ``screen_set[i]`` is ``i`` th screen group.
        )delimiter")
        .def_readonly("screen_begins", &state_t::screen_begins, R"delimiter(
        List of indices that index a corresponding list of values for each screen group.
        ``screen_begins[i]`` is the starting index corresponding to the ``i`` th screen group.
        From this index, reading ``group_sizes[screen_set[i]]`` number of elements
        will grab values corresponding to the full ``i`` th screen group block.
        )delimiter")
        .def_readonly("lmda_path", &state_t::lmda_path, R"delimiter(
        The regularization path to solve for.
        )delimiter")
        .def_readonly("constraint_buffer_size", &state_t::constraint_buffer_size, R"delimiter(
        Max constraint buffer size.
        Equivalent to ``np.max([0 if c is None else c.buffer_size() for c in constraints])``.
        )delimiter")
        .def_readonly("intercept", &state_t::intercept, R"delimiter(
        ``True`` if the function should fit with intercept.
        )delimiter")
        .def_readonly("max_active_size", &state_t::max_active_size, R"delimiter(
        Maximum number of active groups allowed.
        )delimiter")
        .def_readonly("max_iters", &state_t::max_iters, R"delimiter(
        Maximum number of coordinate descents.
        )delimiter")
        .def_readonly("tol", &state_t::tol, R"delimiter(
        Coordinate descent convergence tolerance.
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
        .def_readonly("active_set_size", &state_t::active_set_size, R"delimiter(
        Number of active groups.
        ``active_set[i]`` is only well-defined
        for ``i`` in the range ``[0, active_set_size)``.
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
        ``groups[screen_set[active_set[active_order[i]]]]`` is the ``i`` th active group in ascending order.
        )delimiter")
        .def_property_readonly("betas", [](const state_t& s) {
            const auto& groups = s.groups;
            const auto& group_sizes = s.group_sizes;
            const auto G = groups.size();
            const auto p = G ? (groups[G-1] + group_sizes[G-1]) : 0;
            return convert_sparse_to_dense(
                p,
                s.betas
            );
        }, R"delimiter(
        ``betas[i]`` is the solution at ``lmdas[i]``.
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
        ``rsqs[i]`` is the unnormalized :math:`R^2` at ``betas[i]``.
        )delimiter")
        .def_property_readonly("lmdas", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<value_t>>(
                s.lmdas.data(),
                s.lmdas.size()
            );
        }, R"delimiter(
        ``lmdas[i]`` is the regularization :math:`\lambda`
        used for the ``i`` th solution.
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

template <class ConstraintType, class MatrixType>
class PyStateGaussianPinNaive: public ad::state::StateGaussianPinNaive<ConstraintType, MatrixType>
{
    using base_t = ad::state::StateGaussianPinNaive<ConstraintType, MatrixType>;
public:
    using base_t::base_t;
    PyStateGaussianPinNaive(base_t&& base) : base_t(std::move(base)) {}
};

template <class ConstraintType, class MatrixType>
void state_gaussian_pin_naive(py::module_& m, const char* name)
{
    using constraint_t = ConstraintType;
    using matrix_t = MatrixType;
    using state_t = ad::state::StateGaussianPinNaive<constraint_t, matrix_t>;
    using base_t = typename state_t::base_t;
    using value_t = typename state_t::value_t;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_bool_t = typename state_t::vec_bool_t;
    using dyn_vec_mat_value_t = typename state_t::dyn_vec_mat_value_t;
    using dyn_vec_constraint_t = typename state_t::dyn_vec_constraint_t;

    py::class_<state_t, base_t, PyStateGaussianPinNaive<constraint_t, matrix_t>>(m, name, R"delimiter(
        Core state class for Gaussian, pin, naive method.
        )delimiter")
        .def(py::init<
            matrix_t&,
            value_t,
            value_t,
            const dyn_vec_constraint_t&,
            const Eigen::Ref<const vec_index_t>&, 
            const Eigen::Ref<const vec_index_t>&,
            value_t, 
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_index_t>&, 
            const Eigen::Ref<const vec_index_t>&, 
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            const dyn_vec_mat_value_t&,
            const Eigen::Ref<const vec_value_t>&, 
            size_t,
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
            Eigen::Ref<vec_bool_t>,
            size_t,
            Eigen::Ref<vec_index_t>
        >(),
            py::arg("X"),
            py::arg("y_mean"),
            py::arg("y_var"),
            py::arg("constraints").noconvert(),
            py::arg("groups").noconvert(),
            py::arg("group_sizes").noconvert(),
            py::arg("alpha"),
            py::arg("penalty").noconvert(),
            py::arg("weights").noconvert(),
            py::arg("screen_set").noconvert(),
            py::arg("screen_begins").noconvert(),
            py::arg("screen_vars").noconvert(),
            py::arg("screen_X_means").noconvert(),
            py::arg("screen_transforms").noconvert(),
            py::arg("lmda_path").noconvert(),
            py::arg("constraint_buffer_size"),
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
            py::arg("screen_is_active").noconvert(),
            py::arg("active_set_size"),
            py::arg("active_set").noconvert()
        )
        .def(py::init([](const state_t& s) { return new state_t(s); }))
        .def_readonly("weights", &state_t::weights, R"delimiter(
        Observation weights :math:`W`.
        )delimiter")
        .def_readonly("y_mean", &state_t::y_mean, R"delimiter(
        Mean of the response vector :math:`y` (weighted by :math:`W`),
        i.e. :math:`\mathbf{1}^\top W y`.
        )delimiter")
        .def_readonly("y_var", &state_t::y_var, R"delimiter(
        Variance of the response vector :math:`y` (weighted by :math:`W`), 
        i.e. :math:`\|y_c\|_{W}^2`.
        )delimiter")
        .def_readonly("screen_X_means", &state_t::screen_X_means, R"delimiter(
        Column means of :math:`X` for screen groups (weighted by :math:`W`).
        )delimiter")
        .def_readonly("screen_vars", &state_t::screen_vars, R"delimiter(
        List of :math:`D_k^2` where :math:`D_k` is from the SVD of :math:`\sqrt{W} X_{c,k}` 
        along the screen groups :math:`k` and for possibly column-centered (weighted by :math:`W`) :math:`X_k`.
        ``screen_vars[b:b+p]`` is :math:`D_k^2` for the ``i`` th screen group where
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
        .def_readonly("rsq", &state_t::rsq, R"delimiter(
        The change in unnormalized :math:`R^2` given by 
        :math:`\|y_c-X_c\beta_{\mathrm{old}}\|_{W}^2 - \|y_c-X_c\beta_{\mathrm{curr}}\|_{W}^2`.
        )delimiter")
        .def_readonly("X", &state_t::X, R"delimiter(
        Feature matrix.
        )delimiter")
        .def_readonly("resid", &state_t::resid, R"delimiter(
        Residual :math:`y_c - X \beta` where :math:`\beta` is given by ``screen_beta``.

        .. note:: 
            This definition is unconventional.
            This was done deliberately as the algorithm is most efficient 
            when it updates this quantity compared to 
            the conventional quantity :math:`y_c-X_c \beta`.

        )delimiter")
        .def_readonly("resid_sum", &state_t::resid_sum, R"delimiter(
        Weighted (by :math:`W`) sum of ``resid``.
        )delimiter")
        .def("solve", [](state_t state) { return _solve(state); }, R"delimiter(
        Solves the state-specific problem.
        )delimiter")
        ;
}

template <class ConstraintType, class MatrixType>
class PyStateGaussianPinCov : public ad::state::StateGaussianPinCov<ConstraintType, MatrixType>
{
    using base_t = ad::state::StateGaussianPinCov<ConstraintType, MatrixType>;
public:
    using base_t::base_t;
    PyStateGaussianPinCov(base_t&& base) : base_t(std::move(base)) {}
};

template <class ConstraintType, class MatrixType>
void state_gaussian_pin_cov(py::module_& m, const char* name)
{
    using constraint_t = ConstraintType;
    using matrix_t = MatrixType;
    using state_t = ad::state::StateGaussianPinCov<constraint_t, matrix_t>;
    using base_t = typename state_t::base_t;
    using value_t = typename state_t::value_t;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_bool_t = typename state_t::vec_bool_t;
    using dyn_vec_mat_value_t = typename state_t::dyn_vec_mat_value_t;
    using dyn_vec_constraint_t = typename state_t::dyn_vec_constraint_t;

    py::class_<state_t, base_t, PyStateGaussianPinCov<constraint_t, matrix_t>>(m, name, R"delimiter(
        Core state class for Gaussian, pin, covariance method.
        )delimiter")
        .def(py::init<
            matrix_t&,
            const dyn_vec_constraint_t&,
            const Eigen::Ref<const vec_index_t>&, 
            const Eigen::Ref<const vec_index_t>&,
            value_t, 
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_index_t>&, 
            const Eigen::Ref<const vec_index_t>&, 
            const Eigen::Ref<const vec_value_t>&,
            const dyn_vec_mat_value_t&,
            const Eigen::Ref<const vec_index_t>&, 
            const Eigen::Ref<const vec_index_t>&, 
            const Eigen::Ref<const vec_value_t>&, 
            size_t,
            size_t,
            size_t,
            value_t,
            value_t,
            value_t,
            size_t,
            size_t,
            value_t,
            Eigen::Ref<vec_value_t>, 
            Eigen::Ref<vec_value_t>,
            Eigen::Ref<vec_bool_t>,
            size_t,
            Eigen::Ref<vec_index_t>
        >(),
            py::arg("A"),
            py::arg("constraints").noconvert(),
            py::arg("groups").noconvert(),
            py::arg("group_sizes").noconvert(),
            py::arg("alpha"),
            py::arg("penalty").noconvert(),
            py::arg("screen_set").noconvert(),
            py::arg("screen_begins").noconvert(),
            py::arg("screen_vars").noconvert(),
            py::arg("screen_transforms").noconvert(),
            py::arg("screen_subset_order").noconvert(),
            py::arg("screen_subset_ordered").noconvert(),
            py::arg("lmda_path").noconvert(),
            py::arg("constraint_buffer_size"),
            py::arg("max_active_size"),
            py::arg("max_iters"),
            py::arg("tol"),
            py::arg("rdev_tol"),
            py::arg("newton_tol"),
            py::arg("newton_max_iters"),
            py::arg("n_threads"),
            py::arg("rsq"),
            py::arg("screen_beta").noconvert(),
            py::arg("screen_grad").noconvert(),
            py::arg("screen_is_active").noconvert(),
            py::arg("active_set_size"),
            py::arg("active_set").noconvert()
        )
        .def(py::init([](const state_t& s) { return new state_t(s); }))
        .def_readonly("screen_subset_order", &state_t::screen_subset_order, R"delimiter(
        Ordering such that ``screen_subset`` is sorted in ascending order where
        ``screen_subset`` is a list of column indices into :math:`A`
        in the implicit ordering of ``screen_grad``.
        ``screen_subset[screen_subset_order[i]]`` is the ``i`` th
        column index into :math:`A` corresponding to 
        ``screen_grad[screen_subset_order[i]]``.
        )delimiter")
        .def_readonly("screen_subset_ordered", &state_t::screen_subset_ordered, R"delimiter(
        Ordered ``screen_subset`` (see ``screen_subset_order`` for its definition).
        We have the equivalence
        ``screen_subset_ordered[i] == screen_subset[screen_subset_order[i]]``.
        )delimiter")
        .def_readonly("screen_vars", &state_t::screen_vars, R"delimiter(
        List of :math:`D_k^2` where :math:`D_k^2` are the eigenvalues of :math:`A_{kk}` 
        along the screen groups :math:`k`.
        ``screen_vars[b:b+p]`` is :math:`D_k^2` for the ``i`` th screen group where
        ``k = screen_set[i]``,
        ``b = screen_begins[i]``,
        and ``p = group_sizes[k]``.
        )delimiter")
        .def_readonly("screen_transforms", &state_t::screen_transforms, R"delimiter(
        List of :math:`V_k` where :math:`V_k` are the eigenvectors of :math:`A_{kk}`
        along the screen groups :math:`k`.
        It *only* needs to be properly initialized for groups with size > 1.
        ``screen_transforms[i]`` is :math:`V_k` for the ``i`` th screen group where
        ``k = screen_set[i]``.
        )delimiter")
        .def_readonly("rsq", &state_t::rsq, R"delimiter(
        The change in unnormalized :math:`R^2` given by 
        :math:`2(\ell(\beta_{\mathrm{old}}) - \ell(\beta_{\mathrm{curr}}))`.
        )delimiter")
        .def_readonly("rdev_tol", &state_t::rdev_tol, R"delimiter(
        Relative percent deviance explained tolerance.
        )delimiter")
        .def_readonly("A", &state_t::A, R"delimiter(
        Positive semi-definite matrix :math:`A`.
        )delimiter")
        .def_readonly("screen_grad", &state_t::screen_grad, R"delimiter(
        Gradient :math:`v_k - A_{k,\cdot} \beta` on the screen groups :math:`k` 
        where :math:`\beta` is given by ``screen_beta``.
        ``screen_grad[b:b+p]`` is the gradient for the ``i`` th screen group
        where 
        ``k = screen_set[i]``,
        ``b = screen_begins[i]``,
        and ``p = group_sizes[k]``.
        )delimiter")
        .def("solve", [](state_t state) { return _solve(state); }, R"delimiter(
        Solves the state-specific problem.
        )delimiter")
        ;
}

// ========================================================================
// State Base
// ========================================================================

template <class ConstraintType>
void state_base(py::module_& m, const char* name)
{
    using state_t = ad::state::StateBase<ConstraintType>;
    using value_t = typename state_t::value_t;
    using index_t = typename state_t::index_t;
    using safe_bool_t = typename state_t::safe_bool_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_bool_t = typename state_t::vec_bool_t;
    using dyn_vec_constraint_t = typename state_t::dyn_vec_constraint_t;
    py::class_<state_t>(m, name, R"delimiter(
        Base state class for non-pin classes.
        )delimiter") 
        .def(py::init<
            const dyn_vec_constraint_t&,
            const Eigen::Ref<const vec_index_t>&, 
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
            size_t,
            const Eigen::Ref<const vec_index_t>&,
            value_t,
            const Eigen::Ref<const vec_value_t>& 
        >(),
            py::arg("constraints").noconvert(),
            py::arg("groups").noconvert(),
            py::arg("group_sizes").noconvert(),
            py::arg("dual_groups").noconvert(),
            py::arg("alpha"),
            py::arg("penalty").noconvert(),
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
            py::arg("active_set_size"),
            py::arg("active_set").noconvert(),
            py::arg("lmda"),
            py::arg("grad").noconvert()
        )
        .def_readonly("constraints", &state_t::constraints, R"delimiter(
        List of constraints for each group.
        ``constraints[i]`` is the constraint object corresponding to group ``i``.
        )delimiter")
        .def_readonly("groups", &state_t::groups, R"delimiter(
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
        )delimiter")
        .def_readonly("group_sizes", &state_t::group_sizes, R"delimiter(
        List of group sizes corresponding to each element in ``groups``.
        ``group_sizes[i]`` is the group size of the ``i`` th group. 
        )delimiter")
        .def_readonly("dual_groups", &state_t::dual_groups, R"delimiter(
        List of starting indices to each dual group where `G` is the number of groups.
        ``dual_groups[i]`` is the starting index of the ``i`` th dual group. 
        )delimiter")
        .def_readonly("alpha", &state_t::alpha, R"delimiter(
        Elastic net parameter.
        )delimiter")
        .def_readonly("penalty", &state_t::penalty, R"delimiter(
        Penalty factor for each group in the same order as ``groups``.
        )delimiter")
        .def_readonly("constraint_buffer_size", &state_t::constraint_buffer_size, R"delimiter(
        Max constraint buffer size.
        Equivalent to ``np.max([0 if c is None else c.buffer_size() for c in constraints])``.
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
        )delimiter")
        .def_readonly("max_active_size", &state_t::max_active_size, R"delimiter(
        Maximum number of active groups allowed.
        )delimiter")
        .def_readonly("pivot_subset_ratio", &state_t::pivot_subset_ratio, R"delimiter(
        If screening takes place, then the ``(1 + pivot_subset_ratio) * s``
        largest active scores are used to determine the pivot point
        where ``s`` is the current screen set size.
        )delimiter")
        .def_readonly("pivot_subset_min", &state_t::pivot_subset_min, R"delimiter(
        If screening takes place, then at least ``pivot_subset_min``
        number of active scores are used to determine the pivot point.
        )delimiter")
        .def_readonly("pivot_slack_ratio", &state_t::pivot_slack_ratio, R"delimiter(
        If screening takes place, then ``pivot_slack_ratio``
        number of groups with next smallest (new) active scores 
        below the pivot point are also added to the screen set as slack.
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
        Coordinate descent convergence tolerance.
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
        )delimiter")
        .def_readonly("screen_hashset", &state_t::screen_hashset, R"delimiter(
        Hashmap containing the same values as ``screen_set``.
        )delimiter")
        .def_property_readonly("screen_set", [](const state_t& s) {
            return Eigen::Map<const vec_index_t>(s.screen_set.data(), s.screen_set.size());
        }, R"delimiter(
        List of indices into ``groups`` that correspond to the screen groups.
        ``screen_set[i]`` is ``i`` th screen group.
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
        .def_readonly("active_set_size", &state_t::active_set_size, R"delimiter(
        Number of active groups.
        ``active_set[i]`` is only well-defined
        for ``i`` in the range ``[0, active_set_size)``.
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
        .def_readonly("lmda", &state_t::lmda, R"delimiter(
        The last regularization parameter that was attempted to be solved.
        )delimiter")
        .def_readonly("grad", &state_t::grad, R"delimiter(
        The full gradient :math:`-X^\top \nabla \ell(\eta)`.
        )delimiter")
        .def_readonly("abs_grad", &state_t::abs_grad, R"delimiter(
        The :math:`\ell_2` norms of (corrected) ``grad`` across each group.
        ``abs_grad[i]`` is given by ``np.linalg.norm(grad[g:g+gs] - lmda * penalty[i] * (1-alpha) * beta[g:g+gs] - correction)``
        where ``g = groups[i]``,
        ``gs = group_sizes[i]``,
        ``beta`` is the full solution vector represented by ``screen_beta``,
        and ``correction`` is the output from calling ``constraints[i].gradient()``.
        )delimiter")
        .def_property_readonly("duals", [](const state_t& s) {
            const auto& constraints = s.constraints;
            const auto& dual_groups = s.dual_groups;
            const auto G = dual_groups.size();
            const auto constraint = constraints[G-1];
            const auto group_size = constraint ? constraint->duals() : 0;
            const auto n_duals = G ? (dual_groups[G-1] + group_size) : 0;
            return convert_sparse_to_dense(
                n_duals,
                s.duals
            );
        }, R"delimiter(
        ``duals[i]`` is the dual at ``lmdas[i]``.
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
        ``lmdas[i]`` is the regularization :math:`\lambda`
        used for the ``i`` th solution.
        )delimiter")
        .def_property_readonly("intercepts", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<value_t>>(
                s.intercepts.data(),
                s.intercepts.size()
            );
        }, R"delimiter(
        ``intercepts[i]`` is the intercept at ``lmdas[i]``.
        )delimiter")
        .def_property_readonly("benchmark_screen", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<double>>(
                s.benchmark_screen.data(),
                s.benchmark_screen.size()
            );
        }, R"delimiter(
        Screen time for each iteration.
        )delimiter")
        .def_property_readonly("benchmark_fit_screen", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<double>>(
                s.benchmark_fit_screen.data(),
                s.benchmark_fit_screen.size()
            );
        }, R"delimiter(
        Fit time on the screen set for each iteration.
        )delimiter")
        .def_property_readonly("benchmark_fit_active", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<double>>(
                s.benchmark_fit_active.data(),
                s.benchmark_fit_active.size()
            );
        }, R"delimiter(
        Fit time on the active set for each iteration.
        )delimiter")
        .def_property_readonly("benchmark_kkt", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<double>>(
                s.benchmark_kkt.data(),
                s.benchmark_kkt.size()
            );
        }, R"delimiter(
        KKT time for each iteration.
        )delimiter")
        .def_property_readonly("benchmark_invariance", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<double>>(
                s.benchmark_invariance.data(),
                s.benchmark_invariance.size()
            );
        }, R"delimiter(
        Invariance time for each iteration.
        )delimiter")
        .def_property_readonly("n_valid_solutions", [](const state_t& s) {
            return Eigen::Map<const ad::util::rowvec_type<int>>(
                s.n_valid_solutions.data(),
                s.n_valid_solutions.size()
            );
        }, R"delimiter(
        Number of valid solutions for each iteration.
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

template <class ConstraintType, class MatrixType>
class PyStateGaussianNaive : public ad::state::StateGaussianNaive<ConstraintType, MatrixType>
{
    using base_t = ad::state::StateGaussianNaive<ConstraintType, MatrixType>;
public:
    using base_t::base_t;
    PyStateGaussianNaive(base_t&& base) : base_t(std::move(base)) {}
};

template <class ConstraintType, class MatrixType>
void state_gaussian_naive(py::module_& m, const char* name)
{
    using constraint_t = ConstraintType;
    using matrix_t = MatrixType;
    using state_t = ad::state::StateGaussianNaive<constraint_t, matrix_t>;
    using base_t = typename state_t::base_t;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_bool_t = typename state_t::vec_bool_t;
    using dyn_vec_constraint_t = typename state_t::dyn_vec_constraint_t;
    py::class_<state_t, base_t, PyStateGaussianNaive<constraint_t, matrix_t>>(m, name, R"delimiter(
        Core state class for Gaussian, naive method.
        )delimiter")
        .def(py::init<
            matrix_t&,
            const Eigen::Ref<const vec_value_t>&,
            value_t,
            value_t,
            const Eigen::Ref<const vec_value_t>&,
            value_t,
            const dyn_vec_constraint_t&,
            const Eigen::Ref<const vec_index_t>&,
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
            size_t,
            const Eigen::Ref<const vec_index_t>&,
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
            py::arg("constraints").noconvert(),
            py::arg("groups").noconvert(),
            py::arg("group_sizes").noconvert(),
            py::arg("dual_groups").noconvert(),
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
            py::arg("active_set_size"),
            py::arg("active_set").noconvert(),
            py::arg("rsq"),
            py::arg("lmda"),
            py::arg("grad").noconvert()
        )
        .def(py::init([](const state_t& s) { return new state_t(s); }))
        .def_readonly("weights", &state_t::weights, R"delimiter(
        Observation weights :math:`W`.
        )delimiter")
        .def_readonly("X_means", &state_t::X_means, R"delimiter(
        Column means of ``X`` (weighted by :math:`W`).
        )delimiter")
        .def_readonly("y_mean", &state_t::y_mean, R"delimiter(
        Mean of the response vector :math:`y` (weighted by :math:`W`),
        i.e. :math:`\mathbf{1}^\top W y`.
        )delimiter")
        .def_readonly("y_var", &state_t::y_var, R"delimiter(
        Variance of the response vector :math:`y` (weighted by :math:`W`), 
        i.e. :math:`\|y_c\|_{W}^2`.
        )delimiter")
        .def_readonly("loss_null", &state_t::loss_null, R"delimiter(
        Null loss :math:`-\frac{1}{2} \overline{y}^2` where :math:`\overline{y}`
        is given by ``y_mean``.
        )delimiter")
        .def_readonly("loss_full", &state_t::loss_full, R"delimiter(
        Full loss :math:`-\frac{1}{2} \|y\|_W^2`. 
        )delimiter")
        .def_readonly("X", &state_t::X, R"delimiter(
        Feature matrix.
        )delimiter")
        .def_readonly("resid", &state_t::resid, R"delimiter(
        Residual :math:`y_c - X \beta` where :math:`\beta` is given by ``screen_beta``.
        )delimiter")
        .def_readonly("resid_sum", &state_t::resid_sum, R"delimiter(
        Weighted (by :math:`W`) sum of ``resid``.
        )delimiter")
        .def_readonly("rsq", &state_t::rsq, R"delimiter(
        The change in unnormalized :math:`R^2` given by 
        :math:`\|y_c-X_c\beta_{\mathrm{old}}\|_{W}^2 - \|y_c-X_c\beta_{\mathrm{curr}}\|_{W}^2`.
        )delimiter")
        .def_readonly("screen_X_means", &state_t::screen_X_means, R"delimiter(
        Column means of :math:`X` for screen groups (weighted by :math:`W`).
        )delimiter")
        .def_readonly("screen_transforms", &state_t::screen_transforms, R"delimiter(
        List of :math:`V_k` where :math:`V_k` is from the SVD of :math:`\sqrt{W} X_{c,k}`
        along the screen groups :math:`k` and for possibly column-centered (weighted by :math:`W`) :math:`X_k`.
        It *only* needs to be properly initialized for groups with size > 1.
        ``screen_transforms[i]`` is :math:`V_k` for the ``i`` th screen group where
        ``k = screen_set[i]``.
        )delimiter")
        .def_property_readonly("screen_vars", [](const state_t& s) {
            return Eigen::Map<const vec_value_t>(s.screen_vars.data(), s.screen_vars.size());
        }, R"delimiter(
        List of :math:`D_k^2` where :math:`D_k` is from the SVD of :math:`\sqrt{W} X_{c,k}` 
        along the screen groups :math:`k` and for possibly column-centered (weighted by :math:`W`) :math:`X_k`.
        ``screen_vars[b:b+p]`` is :math:`D_k^2` for the ``i`` th screen group where
        ``k = screen_set[i]``,
        ``b = screen_begins[i]``,
        and ``p = group_sizes[k]``.
        )delimiter")
        .def_property_readonly("betas", [](const state_t& s) {
            return convert_sparse_to_dense(
                s.X->cols(),
                s.betas
            );
        }, R"delimiter(
        ``betas[i]`` is the solution at ``lmdas[i]``.
        )delimiter")
        .def("solve", [](
            state_t state, 
            bool pb, 
            std::function<bool(const state_t&)> exit_cond
        ) { 
            return _solve(state, pb, exit_cond); 
        }, R"delimiter(
        Solves the state-specific problem.
        )delimiter")
        ;
}

template <class ConstraintType, class MatrixType>
class PyStateMultiGaussianNaive : public ad::state::StateMultiGaussianNaive<ConstraintType, MatrixType>
{
    using base_t = ad::state::StateMultiGaussianNaive<ConstraintType, MatrixType>;
public:
    using base_t::base_t;
    PyStateMultiGaussianNaive(base_t&& base) : base_t(std::move(base)) {}
};

template <class ConstraintType, class MatrixType>
void state_multigaussian_naive(py::module_& m, const char* name)
{
    using constraint_t = ConstraintType;
    using matrix_t = MatrixType;
    using state_t = ad::state::StateMultiGaussianNaive<constraint_t, matrix_t>;
    using base_t = typename state_t::base_t;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_bool_t = typename state_t::vec_bool_t;
    using dyn_vec_constraint_t = typename state_t::dyn_vec_constraint_t;
    py::class_<state_t, base_t, PyStateMultiGaussianNaive<constraint_t, matrix_t>>(m, name, R"delimiter(
        Core state class for MultiGaussian, naive method.
        )delimiter")
        .def(py::init<
            size_t,
            bool,
            matrix_t&,
            const Eigen::Ref<const vec_value_t>&,
            value_t,
            value_t,
            const Eigen::Ref<const vec_value_t>&,
            value_t,
            const dyn_vec_constraint_t&,
            const Eigen::Ref<const vec_index_t>&,
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
            size_t,
            const Eigen::Ref<const vec_index_t>&,
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
            py::arg("constraints").noconvert(),
            py::arg("groups").noconvert(),
            py::arg("group_sizes").noconvert(),
            py::arg("dual_groups").noconvert(),
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
            py::arg("active_set_size"),
            py::arg("active_set").noconvert(),
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
            return convert_sparse_to_dense(
                s.X->cols() -  s.multi_intercept * s.n_classes,
                s.betas
            );
        }, R"delimiter(
        ``betas[i]`` is the solution at ``lmdas[i]``.
        )delimiter")
        .def_property_readonly("intercepts", [](const state_t& s) {
            ad::util::rowarr_type<value_t> intercepts(
                s.intercepts.size(), s.n_classes
            );
            for (size_t i = 0; i < s.intercepts.size(); ++i) {
                intercepts.row(i) = s.intercepts[i];
            }
            return intercepts;
        }, R"delimiter(
        ``intercepts[i]`` is the intercept at ``lmdas[i]`` for each class.
        )delimiter")
        .def("solve", [](
            state_t state, 
            bool pb, 
            std::function<bool(const state_t&)> exit_cond
        ) { 
            return _solve(state, pb, exit_cond); 
        }, R"delimiter(
        Solves the state-specific problem.
        )delimiter")
        ;
}

template <class ConstraintType, class MatrixType>
class PyStateGaussianCov : public ad::state::StateGaussianCov<ConstraintType, MatrixType>
{
    using base_t = ad::state::StateGaussianCov<ConstraintType, MatrixType>;
public:
    using base_t::base_t;
    PyStateGaussianCov(base_t&& base) : base_t(std::move(base)) {}
};

template <class ConstraintType, class MatrixType>
void state_gaussian_cov(py::module_& m, const char* name)
{
    using constraint_t = ConstraintType;
    using matrix_t = MatrixType;
    using state_t = ad::state::StateGaussianCov<constraint_t, matrix_t>;
    using base_t = typename state_t::base_t;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_bool_t = typename state_t::vec_bool_t;
    using dyn_vec_constraint_t = typename state_t::dyn_vec_constraint_t;
    py::class_<state_t, base_t, PyStateGaussianCov<constraint_t, matrix_t>>(m, name, R"delimiter(
        Core state class for Gaussian, covariance method.
        )delimiter")
        .def(py::init<
            matrix_t&,
            const Eigen::Ref<const vec_value_t>&,
            const dyn_vec_constraint_t&,
            const Eigen::Ref<const vec_index_t>&,
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
            value_t,
            size_t,
            value_t,
            const std::string&,
            size_t,
            value_t,
            value_t,
            value_t,
            size_t,
            bool,
            bool,
            bool,
            size_t,
            const Eigen::Ref<const vec_index_t>&,
            const Eigen::Ref<const vec_value_t>&, 
            const Eigen::Ref<const vec_bool_t>&,
            size_t,
            const Eigen::Ref<const vec_index_t>&,
            value_t,
            value_t,
            const Eigen::Ref<const vec_value_t>& 
        >(),
            py::arg("A"),
            py::arg("v").noconvert(),
            py::arg("constraints").noconvert(),
            py::arg("groups").noconvert(),
            py::arg("group_sizes").noconvert(),
            py::arg("dual_groups").noconvert(),
            py::arg("alpha"),
            py::arg("penalty").noconvert(),
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
            py::arg("rdev_tol"),
            py::arg("newton_tol"),
            py::arg("newton_max_iters"),
            py::arg("early_exit"),
            py::arg("setup_lmda_max"),
            py::arg("setup_lmda_path"),
            py::arg("n_threads"),
            py::arg("screen_set").noconvert(),
            py::arg("screen_beta").noconvert(),
            py::arg("screen_is_active").noconvert(),
            py::arg("active_set_size"),
            py::arg("active_set").noconvert(),
            py::arg("rsq"),
            py::arg("lmda"),
            py::arg("grad").noconvert()
        )
        .def(py::init([](const state_t& s) { return new state_t(s); }))
        .def_readonly("v", &state_t::v, R"delimiter(
        Linear term.
        )delimiter")
        .def_readonly("A", &state_t::A, R"delimiter(
        Positive semi-definite matrix :math:`A`.
        )delimiter")
        .def_readonly("rdev_tol", &state_t::rdev_tol, R"delimiter(
        Relative percent deviance explained tolerance.
        )delimiter")
        .def_readonly("rsq", &state_t::rsq, R"delimiter(
        The change in unnormalized :math:`R^2` given by 
        :math:`2(\ell(\beta_{\mathrm{old}}) - \ell(\beta_{\mathrm{curr}}))`.
        )delimiter")
        .def_readonly("screen_transforms", &state_t::screen_transforms, R"delimiter(
        List of :math:`V_k` where :math:`V_k` are the eigenvectors of :math:`A_{kk}`
        along the screen groups :math:`k`.
        It *only* needs to be properly initialized for groups with size > 1.
        ``screen_transforms[i]`` is :math:`V_k` for the ``i`` th screen group where
        ``k = screen_set[i]``.
        )delimiter")
        .def_property_readonly("screen_vars", [](const state_t& s) {
            return Eigen::Map<const vec_value_t>(s.screen_vars.data(), s.screen_vars.size());
        }, R"delimiter(
        List of :math:`D_k^2` where :math:`D_k^2` are the eigenvalues of :math:`A_{kk}` 
        along the screen groups :math:`k`.
        ``screen_vars[b:b+p]`` is :math:`D_k^2` for the ``i`` th screen group where
        ``k = screen_set[i]``,
        ``b = screen_begins[i]``,
        and ``p = group_sizes[k]``.
        )delimiter")
        .def_readonly("screen_grad", &state_t::screen_grad, R"delimiter(
        Gradient :math:`v_k - A_{k,\cdot} \beta` on the screen groups :math:`k` 
        where :math:`\beta` is given by ``screen_beta``.
        ``screen_grad[b:b+p]`` is the gradient for the ``i`` th screen group
        where 
        ``k = screen_set[i]``,
        ``b = screen_begins[i]``,
        and ``p = group_sizes[k]``.
        )delimiter")
        .def_readonly("grad", &state_t::grad, R"delimiter(
        The full gradient :math:`v - A\beta`.
        )delimiter")
        .def_property_readonly("betas", [](const state_t& s) {
            return convert_sparse_to_dense(
                s.A->cols(),
                s.betas
            );
        }, R"delimiter(
        ``betas[i]`` is the solution at ``lmdas[i]``.
        )delimiter")
        .def("solve", [](
            state_t state, 
            bool pb, 
            std::function<bool(const state_t&)> exit_cond
        ) { 
            return _solve(state, pb, exit_cond); 
        }, R"delimiter(
        Solves the state-specific problem.
        )delimiter")
        ;
}

// ========================================================================
// GLM State 
// ========================================================================

template <class ConstraintType, class MatrixType>
class PyStateGlmNaive : public ad::state::StateGlmNaive<ConstraintType, MatrixType>
{
    using base_t = ad::state::StateGlmNaive<ConstraintType, MatrixType>;
public:
    using base_t::base_t;
    PyStateGlmNaive(base_t&& base) : base_t(std::move(base)) {}
};

template <class ConstraintType, class MatrixType>
void state_glm_naive(py::module_& m, const char* name)
{
    using constraint_t = ConstraintType;
    using matrix_t = MatrixType;
    using state_t = ad::state::StateGlmNaive<constraint_t, matrix_t>;
    using base_t = typename state_t::base_t;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_bool_t = typename state_t::vec_bool_t;
    using dyn_vec_constraint_t = typename state_t::dyn_vec_constraint_t;
    using glm_t = typename state_t::glm_t;
    py::class_<state_t, base_t, PyStateGlmNaive<constraint_t, matrix_t>>(m, name, R"delimiter(
        Core state class for GLM, naive method.
        )delimiter")
        .def(py::init<
            matrix_t&,
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            const dyn_vec_constraint_t&,
            const Eigen::Ref<const vec_index_t>&,
            const Eigen::Ref<const vec_index_t>&,
            const Eigen::Ref<const vec_index_t>&, 
            value_t, 
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
            size_t,
            const Eigen::Ref<const vec_index_t>&,
            value_t,
            value_t,
            const Eigen::Ref<const vec_value_t>& 
        >(),
            py::arg("X"),
            py::arg("eta").noconvert(),
            py::arg("resid").noconvert(),
            py::arg("constraints").noconvert(),
            py::arg("groups").noconvert(),
            py::arg("group_sizes").noconvert(),
            py::arg("dual_groups").noconvert(),
            py::arg("alpha"),
            py::arg("penalty").noconvert(),
            py::arg("offsets").noconvert(),
            py::arg("lmda_path").noconvert(),
            py::arg("loss_null"),
            py::arg("loss_full"),
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
            py::arg("setup_loss_null"),
            py::arg("setup_lmda_max"),
            py::arg("setup_lmda_path"),
            py::arg("intercept"),
            py::arg("n_threads"),
            py::arg("screen_set").noconvert(),
            py::arg("screen_beta").noconvert(),
            py::arg("screen_is_active").noconvert(),
            py::arg("active_set_size"),
            py::arg("active_set").noconvert(),
            py::arg("beta0"),
            py::arg("lmda"),
            py::arg("grad").noconvert()
        )
        .def(py::init([](const state_t& s) { return new state_t(s); }))
        .def_readonly("offsets", &state_t::offsets, R"delimiter(
        Observation offsets :math:`\eta^0`.
        )delimiter")
        .def_readonly("irls_max_iters", &state_t::irls_max_iters, R"delimiter(
        Maximum number of IRLS iterations.
        )delimiter")
        .def_readonly("irls_tol", &state_t::irls_tol, R"delimiter(
        IRLS convergence tolerance.
        )delimiter")
        .def_readonly("setup_loss_null", &state_t::setup_loss_null, R"delimiter(
        ``True`` if the function should setup ``loss_null``.
        )delimiter")
        .def_readonly("loss_null", &state_t::loss_null, R"delimiter(
        Null loss :math:`\ell(\beta_0^\star \mathbf{1} + \eta^0)`
        from fitting an intercept-only model (if ``intercept`` is ``True``)
        and otherwise :math:`\ell(\eta^0)`.
        )delimiter")
        .def_readonly("loss_full", &state_t::loss_full, R"delimiter(
        Full loss :math:`\ell(\eta^\star)` where :math:`\eta^\star` is the minimizer.
        )delimiter")
        .def_readonly("X", &state_t::X, R"delimiter(
        Feature matrix.
        )delimiter")
        .def_readonly("beta0", &state_t::beta0, R"delimiter(
        The current intercept value.
        )delimiter")
        .def_readonly("eta", &state_t::eta, R"delimiter(
        The natural parameter :math:`\eta = X\beta + \beta_0 \mathbf{1} + \eta^0`
        where 
        :math:`\beta`
        and :math:`\beta_0` are given by
        ``screen_beta`` and ``beta0``.
        )delimiter")
        .def_readonly("resid", &state_t::resid, R"delimiter(
        Residual :math:`-\nabla \ell(\eta)`
        where :math:`\eta` is given by ``eta``.
        )delimiter")
        .def_property_readonly("betas", [](const state_t& s) {
            return convert_sparse_to_dense(
                s.X->cols(),
                s.betas
            );
        }, R"delimiter(
        ``betas[i]`` is the solution at ``lmdas[i]``.
        )delimiter")
        .def("solve", [](
            state_t state, 
            glm_t& glm,
            bool pb, 
            std::function<bool(const state_t&)> exit_cond
        ) { 
            return _solve(state, glm, pb, exit_cond); 
        }, R"delimiter(
        Solves the state-specific problem.
        )delimiter")
        ;
}

template <class ConstraintType, class MatrixType>
class PyStateMultiGlmNaive : public ad::state::StateMultiGlmNaive<ConstraintType, MatrixType>
{
    using base_t = ad::state::StateMultiGlmNaive<ConstraintType, MatrixType>;
public:
    using base_t::base_t;
    PyStateMultiGlmNaive(base_t&& base) : base_t(std::move(base)) {}
};

template <class ConstraintType, class MatrixType>
void state_multiglm_naive(py::module_& m, const char* name)
{
    using constraint_t = ConstraintType;
    using matrix_t = MatrixType;
    using state_t = ad::state::StateMultiGlmNaive<constraint_t, matrix_t>;
    using base_t = typename state_t::base_t;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_bool_t = typename state_t::vec_bool_t;
    using dyn_vec_constraint_t = typename state_t::dyn_vec_constraint_t;
    using glm_t = typename state_t::glm_t;
    py::class_<state_t, base_t, PyStateMultiGlmNaive<constraint_t, matrix_t>>(m, name, R"delimiter(
        Core state class for multi-response GLM, naive method.
        )delimiter")
        .def(py::init<
            size_t,
            bool,
            matrix_t&,
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            const dyn_vec_constraint_t&,
            const Eigen::Ref<const vec_index_t>&,
            const Eigen::Ref<const vec_index_t>&,
            const Eigen::Ref<const vec_index_t>&, 
            value_t, 
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
            size_t,
            const Eigen::Ref<const vec_index_t>&,
            value_t,
            value_t,
            const Eigen::Ref<const vec_value_t>& 
        >(),
            py::arg("n_classes"),
            py::arg("multi_intercept"),
            py::arg("X"),
            py::arg("eta").noconvert(),
            py::arg("resid").noconvert(),
            py::arg("constraints").noconvert(),
            py::arg("groups").noconvert(),
            py::arg("group_sizes").noconvert(),
            py::arg("dual_groups").noconvert(),
            py::arg("alpha"),
            py::arg("penalty").noconvert(),
            py::arg("offsets").noconvert(),
            py::arg("lmda_path").noconvert(),
            py::arg("loss_null"),
            py::arg("loss_full"),
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
            py::arg("setup_loss_null"),
            py::arg("setup_lmda_max"),
            py::arg("setup_lmda_path"),
            py::arg("intercept"),
            py::arg("n_threads"),
            py::arg("screen_set").noconvert(),
            py::arg("screen_beta").noconvert(),
            py::arg("screen_is_active").noconvert(),
            py::arg("active_set_size"),
            py::arg("active_set").noconvert(),
            py::arg("beta0"),
            py::arg("lmda"),
            py::arg("grad").noconvert()
        )
        .def(py::init([](const state_t& s) { return new state_t(s); }))
        .def_readonly("multi_intercept", &state_t::multi_intercept, R"delimiter(
        ``True`` if an intercept is added for each response.
        )delimiter")
        .def_property_readonly("betas", [](const state_t& s) {
            return convert_sparse_to_dense(
                s.X->cols() -  s.multi_intercept * s.n_classes,
                s.betas
            );
        }, R"delimiter(
        ``betas[i]`` is the solution at ``lmdas[i]``.
        )delimiter")
        .def_property_readonly("intercepts", [](const state_t& s) {
            ad::util::rowarr_type<value_t> intercepts(
                s.intercepts.size(), s.n_classes
            );
            for (size_t i = 0; i < s.intercepts.size(); ++i) {
                intercepts.row(i) = s.intercepts[i];
            }
            return intercepts;
        }, R"delimiter(
        ``intercepts[i]`` is the intercept at ``lmdas[i]`` for each class.
        )delimiter")
        .def("solve", [](
            state_t state, 
            glm_t& glm,
            bool pb, 
            std::function<bool(const state_t&)> exit_cond
        ) { 
            return _solve(state, glm, pb, exit_cond); 
        }, R"delimiter(
        Solves the state-specific problem.
        )delimiter")
        ;
}

template <class MatrixType>
class PyStateBVLS : public ad::state::StateBVLS<MatrixType>
{
    using base_t = ad::state::StateBVLS<MatrixType>;
public:
    using base_t::base_t;
    PyStateBVLS(base_t&& base) : base_t(std::move(base)) {}
};

template <class MatrixType>
void state_bvls(py::module_& m, const char* name)
{
    using matrix_t = MatrixType;
    using state_t = ad::state::StateBVLS<MatrixType>;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_bool_t = typename state_t::vec_bool_t;
    py::class_<state_t, PyStateBVLS<matrix_t>>(m, name, R"delimiter(
        Core state class for BVLS.
        )delimiter")
        .def(py::init<
            matrix_t&,
            value_t,
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            size_t,
            size_t,
            value_t,
            size_t,
            Eigen::Ref<vec_index_t>,
            Eigen::Ref<vec_bool_t>,
            size_t,
            Eigen::Ref<vec_index_t>,
            Eigen::Ref<vec_bool_t>,
            Eigen::Ref<vec_value_t>,
            Eigen::Ref<vec_value_t>,
            Eigen::Ref<vec_value_t>,
            value_t 
        >(),
            py::arg("X"),
            py::arg("y_var"),
            py::arg("X_vars").noconvert(),
            py::arg("lower").noconvert(),
            py::arg("upper").noconvert(),
            py::arg("weights").noconvert(),
            py::arg("kappa"),
            py::arg("max_iters"),
            py::arg("tol"),
            py::arg("screen_set_size"),
            py::arg("screen_set").noconvert(),
            py::arg("is_screen").noconvert(),
            py::arg("active_set_size"),
            py::arg("active_set").noconvert(),
            py::arg("is_active").noconvert(),
            py::arg("beta").noconvert(),
            py::arg("resid").noconvert(),
            py::arg("grad").noconvert(),
            py::arg("loss")
        )
        .def(py::init([](const state_t& s) { return new state_t(s); }))
        .def_readonly("screen_set_size", &state_t::screen_set_size, R"delimiter(
        Screen set size.
        )delimiter")
        .def_readonly("screen_set", &state_t::screen_set, R"delimiter(
        Screen set buffer.
        )delimiter")
        .def_readonly("is_screen", &state_t::is_screen, R"delimiter(
        Boolean buffer to indicate the screen variables.
        )delimiter")
        .def_readonly("active_set_size", &state_t::active_set_size, R"delimiter(
        Active set size.
        )delimiter")
        .def_readonly("active_set", &state_t::active_set, R"delimiter(
        Active set buffer.
        )delimiter")
        .def_readonly("is_active", &state_t::is_active, R"delimiter(
        Boolean buffer to indicate the active variables.
        )delimiter")
        .def_readonly("beta", &state_t::beta, R"delimiter(
        Coefficient vector.
        )delimiter")
        .def_readonly("resid", &state_t::resid, R"delimiter(
        Residual :math:`y - X \beta`.
        )delimiter")
        .def_readonly("grad", &state_t::grad, R"delimiter(
        Internal buffer that is implementation-defined.
        )delimiter")
        .def_readonly("loss", &state_t::loss, R"delimiter(
        Loss :math:`\frac{1}{2} \|y-X\beta\|_W^2`.
        )delimiter")
        .def_readonly("iters", &state_t::iters, R"delimiter(
        Number of coordinate descent iterations.
        )delimiter")
        .def_readonly("benchmark_fit_active", &state_t::benchmark_fit_active, R"delimiter(
        Benchmark time for fitting on the active set.
        )delimiter")
        .def_readonly("benchmark_fit_screen", &state_t::benchmark_fit_screen, R"delimiter(
        Benchmark time for fitting on the screen set.
        )delimiter")
        .def_readonly("benchmark_gradient", &state_t::benchmark_gradient, R"delimiter(
        Benchmark time for computing the gradient.
        )delimiter")
        .def_readonly("benchmark_viols_sort", &state_t::benchmark_viols_sort, R"delimiter(
        Benchmark time for sorting the violations.
        )delimiter")
        .def_readonly("dbg_beta", &state_t::dbg_beta, R"delimiter(
        List of the coefficient vectors at each outer loop.
        )delimiter")
        .def_readonly("dbg_active_set", &state_t::dbg_active_set, R"delimiter(
        List of the active sets at each outer loop.
        )delimiter")
        .def_readonly("dbg_iter", &state_t::dbg_iter, R"delimiter(
        List of the number of iterations at each outer loop.
        )delimiter")
        .def_readonly("dbg_loss", &state_t::dbg_loss, R"delimiter(
        List of the losses at each outer loop.
        )delimiter")
        .def("solve", [](state_t state) { return _solve(state); }, R"delimiter(
        Solves the state-specific problem.
        )delimiter")
        ;
}

template <class MatrixType>
class PyStateCSSCov : public ad::state::StateCSSCov<MatrixType>
{
    using base_t = ad::state::StateCSSCov<MatrixType>;
public:
    using base_t::base_t;
    PyStateCSSCov(base_t&& base) : base_t(std::move(base)) {}
};

template <class MatrixType>
void state_css_cov(py::module_& m, const char* name)
{
    using matrix_t = MatrixType;
    using state_t = ad::state::StateCSSCov<MatrixType>;
    using vec_index_t = typename state_t::vec_index_t;
    py::class_<state_t, PyStateCSSCov<matrix_t>>(m, name, R"delimiter(
        Core state class for CSS via covariance method.
        )delimiter")
        .def(py::init<
            const Eigen::Ref<const matrix_t>&,
            size_t,
            const Eigen::Ref<const vec_index_t>&,
            const std::string&,
            const std::string&,
            size_t,
            size_t
        >(),
            py::arg("S").noconvert(),
            py::arg("subset_size"),
            py::arg("subset").noconvert(),
            py::arg("method"),
            py::arg("loss"),
            py::arg("max_iters"),
            py::arg("n_threads")
        )
        .def(py::init([](const state_t& s) { return new state_t(s); }))
        .def_property_readonly("subset", [](const state_t& s) {
            return Eigen::Map<const vec_index_t>(
                s.subset.data(),
                s.subset.size()
            );
        }, R"delimiter(
        Selected subset.
        )delimiter")
        .def_readonly("S_resid", &state_t::S_resid, R"delimiter(
        Residual covariance matrix :math:`\Sigma - \Sigma_{\cdot T} \Sigma_{TT}^{\dagger} \Sigma_{T \cdot}`
        where :math:`T` is the current subset.
        )delimiter")
        .def_readonly("L_T", &state_t::L_T, R"delimiter(
        Lower Cholesky factor of :math:`\Sigma_{TT}` 
        where :math:`T` is the current subset.
        )delimiter")
        .def_readonly("benchmark_init", &state_t::benchmark_init, R"delimiter(
        Benchmark time for the initialization.
        )delimiter")
        .def_readonly("benchmark_L_U", &state_t::benchmark_L_U, R"delimiter(
        Benchmark time for updating the Cholesky factor after removing a feature.
        )delimiter")
        .def_readonly("benchmark_S_resid", &state_t::benchmark_S_resid, R"delimiter(
        Benchmark time for updating the residual covariance matrix after removing a feature.
        )delimiter")
        .def_readonly("benchmark_scores", &state_t::benchmark_scores, R"delimiter(
        Benchmark time for computing the scores.
        )delimiter")
        .def_readonly("benchmark_L_T", &state_t::benchmark_L_T, R"delimiter(
        Benchmark time for updating the Cholesky factor after adding a feature.
        )delimiter")
        .def_readonly("benchmark_resid_fwd", &state_t::benchmark_resid_fwd, R"delimiter(
        Benchmark time for updating the residual covariance matrix after adding a feature.
        )delimiter")
        .def("solve", [](state_t state) { return _solve(state); }, R"delimiter(
        Solves the state-specific problem.
        )delimiter")
        ;
}

template <class MatrixType>
class PyStatePinball : public ad::state::StatePinball<MatrixType>
{
    using base_t = ad::state::StatePinball<MatrixType>;
public:
    using base_t::base_t;
    PyStatePinball(base_t&& base) : base_t(std::move(base)) {}
};

template <class MatrixType>
void state_pinball(py::module_& m, const char* name)
{
    using matrix_t = MatrixType;
    using state_t = ad::state::StatePinball<MatrixType>;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_bool_t = typename state_t::vec_bool_t;
    using colmat_value_t = typename state_t::colmat_value_t;
    using rowmat_value_t = typename state_t::rowmat_value_t;
    py::class_<state_t, PyStatePinball<matrix_t>>(m, name, R"delimiter(
        Core state class for pinball least squares.
        )delimiter")
        .def(py::init<
            matrix_t&,
            value_t,
            const Eigen::Ref<const colmat_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            size_t,
            size_t,
            value_t,
            size_t,
            Eigen::Ref<vec_index_t>,
            Eigen::Ref<vec_bool_t>,
            Eigen::Ref<vec_value_t>,
            Eigen::Ref<rowmat_value_t>,
            size_t,
            Eigen::Ref<vec_index_t>,
            Eigen::Ref<vec_bool_t>,
            Eigen::Ref<vec_value_t>,
            Eigen::Ref<vec_value_t>,
            Eigen::Ref<vec_value_t>,
            value_t 
        >(),
            py::arg("A"),
            py::arg("y_var"),
            py::arg("S").noconvert(),
            py::arg("penalty_neg").noconvert(),
            py::arg("penalty_pos").noconvert(),
            py::arg("kappa"),
            py::arg("max_iters"),
            py::arg("tol"),
            py::arg("screen_set_size"),
            py::arg("screen_set").noconvert(),
            py::arg("is_screen").noconvert(),
            py::arg("screen_ASAT_diag").noconvert(),
            py::arg("screen_AS").noconvert(),
            py::arg("active_set_size"),
            py::arg("active_set").noconvert(),
            py::arg("is_active").noconvert(),
            py::arg("beta").noconvert(),
            py::arg("resid").noconvert(),
            py::arg("grad").noconvert(),
            py::arg("loss")
        )
        .def(py::init([](const state_t& s) { return new state_t(s); }))
        .def_readonly("screen_set_size", &state_t::screen_set_size, R"delimiter(
        Screen set size.
        )delimiter")
        .def_readonly("screen_set", &state_t::screen_set, R"delimiter(
        Screen set buffer.
        )delimiter")
        .def_readonly("is_screen", &state_t::is_screen, R"delimiter(
        Boolean buffer to indicate the screen variables.
        )delimiter")
        .def_readonly("screen_ASAT_diag", &state_t::screen_ASAT_diag, R"delimiter(
        Diagonal of :math:`A S A^\top`.
        )delimiter")
        .def_readonly("screen_AS", &state_t::screen_AS, R"delimiter(
        :math:`A S`.
        )delimiter")
        .def_readonly("active_set_size", &state_t::active_set_size, R"delimiter(
        Active set size.
        )delimiter")
        .def_readonly("active_set", &state_t::active_set, R"delimiter(
        Active set buffer.
        )delimiter")
        .def_readonly("is_active", &state_t::is_active, R"delimiter(
        Boolean buffer to indicate the active variables.
        )delimiter")
        .def_readonly("beta", &state_t::beta, R"delimiter(
        Coefficient vector.
        )delimiter")
        .def_readonly("resid", &state_t::resid, R"delimiter(
        Residual :math:`v - S A^\top \beta`.
        )delimiter")
        .def_readonly("grad", &state_t::grad, R"delimiter(
        Internal buffer that is implementation-defined.
        )delimiter")
        .def_readonly("loss", &state_t::loss, R"delimiter(
        Loss :math:`\frac{1}{2} \|S^{-\frac{1}{2}} v - S^{\frac{1}{2}} A^\top \beta\|_2^2`.
        )delimiter")
        .def_readonly("iters", &state_t::iters, R"delimiter(
        Number of coordinate descent iterations.
        )delimiter")
        .def_readonly("benchmark_fit_active", &state_t::benchmark_fit_active, R"delimiter(
        Benchmark time for fitting on the active set.
        )delimiter")
        .def_readonly("benchmark_fit_screen", &state_t::benchmark_fit_screen, R"delimiter(
        Benchmark time for fitting on the screen set.
        )delimiter")
        .def_readonly("benchmark_gradient", &state_t::benchmark_gradient, R"delimiter(
        Benchmark time for computing the gradient.
        )delimiter")
        .def_readonly("benchmark_viols_sort", &state_t::benchmark_viols_sort, R"delimiter(
        Benchmark time for sorting the violations.
        )delimiter")
        .def_readonly("dbg_beta", &state_t::dbg_beta, R"delimiter(
        List of the coefficient vectors at each outer loop.
        )delimiter")
        .def_readonly("dbg_active_set", &state_t::dbg_active_set, R"delimiter(
        List of the active sets at each outer loop.
        )delimiter")
        .def_readonly("dbg_iter", &state_t::dbg_iter, R"delimiter(
        List of the number of iterations at each outer loop.
        )delimiter")
        .def_readonly("dbg_loss", &state_t::dbg_loss, R"delimiter(
        List of the losses at each outer loop.
        )delimiter")
        .def("solve", [](state_t state) { return _solve(state); }, R"delimiter(
        Solves the state-specific problem.
        )delimiter")
        ;
}

void register_state(py::module_& m)
{
    state_gaussian_pin_base<
        constraint_type<double>
    >(m, "StateGaussianPinBase64");
    state_gaussian_pin_base<
        constraint_type<float>
    >(m, "StateGaussianPinBase32");
    state_gaussian_pin_cov<
        constraint_type<double>,
        matrix_cov_type<double>
    >(m, "StateGaussianPinCov64");
    state_gaussian_pin_cov<
        constraint_type<float>,
        matrix_cov_type<float>
    >(m, "StateGaussianPinCov32");
    state_gaussian_pin_naive<
        constraint_type<double>,
        matrix_naive_type<double>
    >(m, "StateGaussianPinNaive64");
    state_gaussian_pin_naive<
        constraint_type<float>,
        matrix_naive_type<float>
    >(m, "StateGaussianPinNaive32");

    state_base<
        constraint_type<double>
    >(m, "StateBase64");
    state_base<
        constraint_type<float>
    >(m, "StateBase32");
    state_gaussian_cov<
        constraint_type<double>,
        matrix_cov_type<double>
    >(m, "StateGaussianCov64");
    state_gaussian_cov<
        constraint_type<float>,
        matrix_cov_type<float>
    >(m, "StateGaussianCov32");
    state_gaussian_naive<
        constraint_type<double>,
        matrix_naive_type<double>
    >(m, "StateGaussianNaive64");
    state_gaussian_naive<
        constraint_type<float>,
        matrix_naive_type<float>
    >(m, "StateGaussianNaive32");
    state_multigaussian_naive<
        constraint_type<double>,
        matrix_naive_type<double>
    >(m, "StateMultiGaussianNaive64");
    state_multigaussian_naive<
        constraint_type<float>,
        matrix_naive_type<float>
    >(m, "StateMultiGaussianNaive32");
    state_glm_naive<
        constraint_type<double>,
        matrix_naive_type<double>
    >(m, "StateGlmNaive64");
    state_glm_naive<
        constraint_type<float>,
        matrix_naive_type<float>
    >(m, "StateGlmNaive32");
    state_multiglm_naive<
        constraint_type<double>,
        matrix_naive_type<double>
    >(m, "StateMultiGlmNaive64");
    state_multiglm_naive<
        constraint_type<float>,
        matrix_naive_type<float>
    >(m, "StateMultiGlmNaive32");

    state_bvls<
        matrix_naive_type<double>
    >(m, "StateBVLS64");
    state_bvls<
        matrix_naive_type<float>
    >(m, "StateBVLS32");

    state_css_cov<
        dense_type<double, Eigen::ColMajor>
    >(m, "StateCSSCov64");
    state_css_cov<
        dense_type<float, Eigen::ColMajor>
    >(m, "StateCSSCov32");

    state_pinball<
        matrix_constraint_type<double>
    >(m, "StatePinball64");
    state_pinball<
        matrix_constraint_type<float>
    >(m, "StatePinball32");
}