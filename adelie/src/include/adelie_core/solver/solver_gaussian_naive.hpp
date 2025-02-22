#pragma once
#include <Eigen/Eigenvalues>
#include <adelie_core/solver/solver_base.hpp>
#include <adelie_core/solver/solver_gaussian_pin_naive.hpp>
#include <adelie_core/state/state_gaussian_pin_naive.hpp>
#include <adelie_core/matrix/utils.hpp>
#include <adelie_core/util/macros.hpp>
#include <adelie_core/util/omp.hpp>

namespace adelie_core {
namespace solver {
namespace gaussian {
namespace naive {

template <class ValueType, class SafeBoolType=int8_t>
struct GaussianNaiveBufferPack
{
    using value_t = ValueType;
    using safe_bool_t = SafeBoolType;
    using vec_value_t = util::rowvec_type<value_t>;
    using dyn_vec_value_t = std::vector<value_t>;
    using dyn_vec_bool_t = std::vector<safe_bool_t>;

    GaussianNaiveBufferPack(
        size_t n
    ):
        resid_prev(n)
    {}

    vec_value_t resid_prev;
    dyn_vec_value_t screen_beta_prev; 
    dyn_vec_bool_t screen_is_active_prev;
};

/**
 * Updates in-place the screen quantities 
 * in the range [begin, end) of the groups in screen_set. 
 * NOTE: X_means only needs to be well-defined on the groups in that range,
 * that is, weighted mean according to weights_sqrt ** 2.
 */
template <
    class XType, 
    class XMType, 
    class WType,
    class GroupsType, 
    class GroupSizesType, 
    class SSType, 
    class SBType,
    class SXMType, 
    class STType, 
    class SVType
>
inline void update_screen_derived(
    XType& X,
    const XMType& X_means,
    const WType& weights_sqrt,
    const GroupsType& groups,
    const GroupSizesType& group_sizes,
    const SSType& screen_set,
    const SBType& screen_begins,
    size_t begin,
    size_t end,
    bool intercept,
    size_t n_threads,
    SXMType& screen_X_means,
    STType& screen_transforms,
    SVType& screen_vars
)
{
    using value_t = typename std::decay_t<XType>::value_t;
    using vec_value_t = util::rowvec_type<value_t>;
    using colmat_value_t = util::colmat_type<value_t>;

    // buffers
    const auto max_gs = group_sizes.maxCoeff();
    const auto n_threads_cap_1 = std::max<size_t>(n_threads, 1);
    util::rowvec_type<value_t> buffer(n_threads_cap_1 * max_gs * max_gs);

    const auto routine = [&](auto i) {
        const auto g = groups[screen_set[i]];
        const auto gs = group_sizes[screen_set[i]];
        const auto sb = screen_begins[i];
        const auto thr_id = util::omp_get_thread_num();

        // compute column-means
        Eigen::Map<vec_value_t> Xi_means(
            screen_X_means.data() + sb, gs
        );
        Xi_means = X_means.segment(g, gs);

        // resize output and buffer 
        Eigen::Map<colmat_value_t> XiTXi(
            buffer.data() + thr_id * max_gs * max_gs, gs, gs
        );

        // compute weighted covariance matrix
        X.cov(g, gs, weights_sqrt, XiTXi);

        if (intercept) {
            auto XiTXi_lower = XiTXi.template selfadjointView<Eigen::Lower>();
            XiTXi_lower.rankUpdate(Xi_means.matrix().transpose(), -1);
            XiTXi.template triangularView<Eigen::Upper>() = XiTXi.transpose();
        }

        if (gs == 1) {
            util::colmat_type<value_t, 1, 1> Q;
            Q(0, 0) = 1;
            screen_transforms[i] = Q;
            screen_vars[sb] = std::max<value_t>(XiTXi(0, 0), 0);
            return;
        }

        Eigen::SelfAdjointEigenSolver<colmat_value_t> solver(XiTXi);

        /* update screen_transforms */
        screen_transforms[i] = std::move(solver.eigenvectors());

        /* update screen_vars */
        const auto& D = solver.eigenvalues();
        Eigen::Map<vec_value_t> svars(screen_vars.data() + sb, gs);
        // numerical stability to remove small negative eigenvalues
        svars.head(D.size()) = D.array() * (D.array() >= 0).template cast<value_t>(); 
    };
    util::omp_parallel_for(routine, begin, end, n_threads * ((begin+n_threads) <= end));
}

/**
 * Updates all derived screen quantities for naive state.
 * See the incoming state requirements in update_screen_derived_base.
 * After the function finishes, all screen quantities in the base + naive class
 * will be consistent with screen_set, and the state is otherwise effectively
 * unchanged in the sense that other quantities dependent on screen states are unchanged.
 */
template <class StateType>
inline void update_screen_derived(
    StateType& state
)
{
    update_screen_derived_base(state);

    const auto& group_sizes = state.group_sizes;
    const auto& screen_set = state.screen_set;
    auto& screen_transforms = state.screen_transforms;
    const auto& screen_begins = state.screen_begins;
    auto& screen_X_means = state.screen_X_means;
    auto& screen_vars = state.screen_vars;

    const auto old_screen_size = screen_transforms.size();
    const auto new_screen_size = screen_set.size();
    const int new_screen_value_size = (
        (screen_begins.size() == 0) ? 0 : (
            screen_begins.back() + group_sizes[screen_set.back()]
        )
    );

    screen_X_means.resize(new_screen_value_size);    
    screen_transforms.resize(new_screen_size);
    screen_vars.resize(new_screen_value_size, 0);

    update_screen_derived(
        *state.X, 
        state.X_means, 
        state.weights_sqrt,
        state.groups, 
        state.group_sizes, 
        state.screen_set, 
        state.screen_begins, 
        old_screen_size, 
        new_screen_size, 
        state.intercept, 
        state.n_threads,
        state.screen_X_means, 
        state.screen_transforms, 
        state.screen_vars
    );
}

template <class StateType, class StateGaussianPinType, class ValueType>
ADELIE_CORE_STRONG_INLINE
void update_solutions(
    StateType& state,
    StateGaussianPinType& state_gaussian_pin_naive,
    ValueType lmda
)
{
    using state_t = std::decay_t<StateType>;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_value_t = typename state_t::vec_value_t;

    const auto y_var = state.y_var;
    auto& betas = state.betas;
    auto& duals = state.duals;
    auto& devs = state.devs;
    auto& lmdas = state.lmdas;
    auto& intercepts = state.intercepts;

    vec_index_t dual_indices; 
    vec_value_t dual_values;

    betas.emplace_back(std::move(state_gaussian_pin_naive.betas.back()));
    duals.emplace_back(sparsify_dual(state, dual_indices, dual_values));
    intercepts.emplace_back(state_gaussian_pin_naive.intercepts.back());
    lmdas.emplace_back(lmda);

    const auto dev = state_gaussian_pin_naive.rsqs.back();
    devs.emplace_back(dev / y_var);
}

template <
    class StateType,
    class BufferPackType,
    class ValueType,
    class CUIType=util::no_op
>
inline auto fit(
    StateType& state,
    BufferPackType& buffer_pack,
    ValueType lmda,
    CUIType check_user_interrupt = CUIType()
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;
    using index_t = typename state_t::index_t;
    using safe_bool_t = typename state_t::safe_bool_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_safe_bool_t = util::rowvec_type<safe_bool_t>;
    using constraint_t = typename state_t::constraint_t;
    using matrix_naive_t = typename state_t::matrix_t;
    using state_gaussian_pin_naive_t = state::StateGaussianPinNaive<
        constraint_t,
        matrix_naive_t,
        typename std::decay_t<matrix_naive_t>::value_t,
        index_t,
        safe_bool_t
    >;

    auto& X = *state.X;
    const auto y_mean = state.y_mean;
    const auto y_var = state.y_var;
    const auto& constraints = state.constraints;
    const auto& groups = state.groups;
    const auto& group_sizes = state.group_sizes;
    const auto alpha = state.alpha;
    const auto& penalty = state.penalty;
    const auto& weights = state.weights;
    const auto& screen_set = state.screen_set;
    const auto& screen_begins = state.screen_begins;
    const auto& screen_vars = state.screen_vars;
    const auto& screen_X_means = state.screen_X_means;
    const auto& screen_transforms = state.screen_transforms;
    const auto constraint_buffer_size = state.constraint_buffer_size;
    const auto intercept = state.intercept;
    const auto max_active_size = state.max_active_size;
    const auto max_iters = state.max_iters;
    const auto tol = state.tol;
    const auto adev_tol = state.adev_tol;
    const auto ddev_tol = state.ddev_tol;
    const auto newton_tol = state.newton_tol;
    const auto newton_max_iters = state.newton_max_iters;
    const auto n_threads = state.n_threads;
    auto& rsq = state.rsq;
    auto& resid_sum = state.resid_sum;
    auto& resid = state.resid;
    auto& screen_beta = state.screen_beta;
    auto& screen_is_active = state.screen_is_active;
    auto& active_set_size = state.active_set_size;
    auto& active_set = state.active_set;

    auto& resid_prev = buffer_pack.resid_prev;
    auto& screen_beta_prev = buffer_pack.screen_beta_prev;
    auto& screen_is_active_prev = buffer_pack.screen_is_active_prev;

    util::rowvec_type<value_t, 1> lmda_path;
    lmda_path = lmda;

    // Save all current valid quantities that will be modified in-place by fit.
    // This is needed in case we exit with exception and need to restore invariance.
    // Saving SHOULD NOT swap since we still need the values of the current containers.
    const auto save_prev_valid = [&]() {
        resid_prev = resid;
        screen_beta_prev = screen_beta;
        screen_is_active_prev = screen_is_active;
    };
    const auto load_prev_valid = [&]() {
        resid.swap(resid_prev);
        screen_beta.swap(screen_beta_prev);
        screen_is_active.swap(screen_is_active_prev);
    };

    save_prev_valid();

    state_gaussian_pin_naive_t state_gaussian_pin_naive(
        X,
        y_mean,
        y_var,
        constraints,
        groups, 
        group_sizes,
        alpha, 
        penalty,
        weights,
        Eigen::Map<const vec_index_t>(screen_set.data(), screen_set.size()), 
        Eigen::Map<const vec_index_t>(screen_begins.data(), screen_begins.size()), 
        Eigen::Map<const vec_value_t>(screen_vars.data(), screen_vars.size()), 
        Eigen::Map<const vec_value_t>(screen_X_means.data(), screen_X_means.size()), 
        screen_transforms,
        lmda_path,
        constraint_buffer_size,
        intercept, max_active_size, max_iters, 
        tol * y_var, 
        adev_tol, ddev_tol, 
        newton_tol, newton_max_iters, n_threads,
        rsq,
        Eigen::Map<vec_value_t>(resid.data(), resid.size()),
        resid_sum,
        Eigen::Map<vec_value_t>(screen_beta.data(), screen_beta.size()), 
        Eigen::Map<vec_safe_bool_t>(screen_is_active.data(), screen_is_active.size()),
        active_set_size,
        active_set
    );

    try {
        state_gaussian_pin_naive.solve(check_user_interrupt);
    } catch(...) {
        load_prev_valid();
        throw;
    }

    resid_sum = state_gaussian_pin_naive.resid_sum;
    rsq = state_gaussian_pin_naive.rsq;
    active_set_size = state_gaussian_pin_naive.active_set_size;

    const auto screen_time = Eigen::Map<const util::rowvec_type<double>>(
        state_gaussian_pin_naive.benchmark_screen.data(),
        state_gaussian_pin_naive.benchmark_screen.size()
    ).sum();
    const auto active_time = Eigen::Map<const util::rowvec_type<double>>(
        state_gaussian_pin_naive.benchmark_active.data(),
        state_gaussian_pin_naive.benchmark_active.size()
    ).sum();

    return std::make_tuple(
        std::move(state_gaussian_pin_naive), 
        screen_time, 
        active_time
    );
}

template <
    class StateType,
    class PBType,
    class ExitCondType,
    class TidyType,
    class CUIType
>
inline void solve(
    StateType&& state,
    PBType&& pb,
    ExitCondType exit_cond_f,
    TidyType tidy_f,
    CUIType check_user_interrupt
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;
    using safe_bool_t = typename state_t::safe_bool_t;

    const auto n = state.X->rows();
    GaussianNaiveBufferPack<value_t, safe_bool_t> buffer_pack(n);

    const auto pb_add_suffix_f = [&](const auto& state, auto& pb) {
        solver::pb_add_suffix(state, pb);
    };
    const auto update_loss_null_f = [](const auto&) {};
    const auto update_invariance_f = [&](auto& state, const auto&, auto lmda) {
        const auto& X_means = state.X_means;
        const auto& weights = state.weights;
        const auto intercept = state.intercept;
        const auto n_threads = state.n_threads;
        const auto& resid = state.resid;
        const auto& resid_sum = state.resid_sum; // MUST be a reference since the most updated value is needed
        auto& X = *state.X;
        auto& grad = state.grad;

        state.lmda = lmda;
        X.mul(resid, weights, grad);
        if (intercept) {
            matrix::dvsubi(grad, resid_sum * X_means, n_threads);
        }
        update_abs_grad(state, lmda);
    };
    const auto update_solutions_f = [&](auto& state, auto& state_gaussian_pin_naive, auto lmda) {
        update_solutions(
            state, 
            state_gaussian_pin_naive,
            lmda
        );
        tidy_f();
    };
    const auto early_exit_f = [&](const auto& state) {
        return solver::early_exit(state) || exit_cond_f();
    };
    const auto screen_f = [](auto& state, auto lmda, auto kkt_passed, auto n_new_active) {
        solver::screen(
            state,
            lmda,
            kkt_passed,
            n_new_active
        );
        update_screen_derived(state);
    };
    const auto fit_f = [&](auto& state, auto lmda) {
        return fit(
            state, 
            buffer_pack, 
            lmda, 
            check_user_interrupt
        );
    };

    solver::solve_core(
        state,
        pb,
        pb_add_suffix_f,
        update_loss_null_f,
        update_invariance_f,
        update_solutions_f,
        early_exit_f,
        screen_f,
        fit_f
    );
}

template <
    class StateType,
    class PBType,
    class ExitCondType,
    class CUIType=util::no_op
>
inline void solve(
    StateType&& state,
    PBType&& pb,
    ExitCondType exit_cond_f,
    CUIType check_user_interrupt = CUIType()
)
{
    solve(
        state,
        pb,
        exit_cond_f,
        [](){},
        check_user_interrupt
    );
}

} // namespace naive
} // namespace gaussian
} // namespace solver
} // namespace adelie_core