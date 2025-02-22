#pragma once
#include <Eigen/Eigenvalues>
#include <adelie_core/solver/solver_base.hpp>
#include <adelie_core/solver/solver_gaussian_pin_cov.hpp>
#include <adelie_core/state/state_gaussian_pin_cov.hpp>
#include <adelie_core/matrix/utils.hpp>
#include <adelie_core/util/macros.hpp>

namespace adelie_core {
namespace solver {
namespace gaussian {
namespace cov {

/**
 * Updates all derived screen quantities for cov state.
 * After the function finishes, all screen quantities in the base + cov class
 * will be consistent with screen_set, and the state is otherwise effectively
 * unchanged in the sense that other quantities dependent on screen states are unchanged.
 */
template <class StateType>
inline void update_screen_derived(
    StateType& state
)
{
    using state_t = typename std::decay_t<StateType>;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_index_t = typename state_t::vec_index_t;

    update_screen_derived_base(state);

    const auto& groups = state.groups;
    const auto& group_sizes = state.group_sizes;
    const auto& screen_set = state.screen_set;
    const auto& screen_begins = state.screen_begins;
    const auto& grad = state.grad;
    const auto n_threads = state.n_threads;
    auto& A = *state.A;
    auto& screen_transforms = state.screen_transforms;
    auto& screen_vars = state.screen_vars;
    auto& screen_grad = state.screen_grad;
    auto& screen_subset = state.screen_subset;
    auto& screen_subset_order = state.screen_subset_order;
    auto& screen_subset_ordered = state.screen_subset_ordered;

    const auto old_screen_size = screen_transforms.size();
    const auto new_screen_size = screen_set.size();
    const int old_screen_value_size = screen_subset.size();
    const int new_screen_value_size = (
        (screen_begins.size() == 0) ? 0 : (
            screen_begins.back() + group_sizes[screen_set.back()]
        )
    );

    screen_transforms.resize(new_screen_size);
    screen_vars.resize(new_screen_value_size, 0);
    screen_grad.resize(new_screen_value_size, 0);

    const auto max_gs = group_sizes.maxCoeff();
    const auto n_threads_cap_1 = std::max<size_t>(n_threads, 1);
    util::rowvec_type<value_t> buffer(n_threads_cap_1 * max_gs * max_gs);

    const auto routine = [&](auto i) {
        const auto g = groups[screen_set[i]];
        const auto gs = group_sizes[screen_set[i]];
        const auto sb = screen_begins[i];
        const auto thr_id = util::omp_get_thread_num();

        Eigen::Map<util::colmat_type<value_t>> A_gg(
            buffer.data() + thr_id * max_gs * max_gs, gs, gs
        );

        // compute covariance matrix
        A.to_dense(g, gs, A_gg);

        if (gs == 1) {
            util::colmat_type<value_t, 1, 1> Q;
            Q(0, 0) = 1;
            screen_transforms[i] = Q;
            screen_vars[sb] = std::max<value_t>(A_gg(0, 0), 0);
            return;
        }

        Eigen::SelfAdjointEigenSolver<util::colmat_type<value_t>> solver(A_gg);

        /* update screen_transforms */
        screen_transforms[i] = std::move(solver.eigenvectors());

        /* update screen_vars */
        const auto& D = solver.eigenvalues();
        Eigen::Map<vec_value_t> svars(screen_vars.data() + sb, gs);
        // numerical stability to remove small negative eigenvalues
        svars.head(D.size()) = D.array() * (D.array() >= 0).template cast<value_t>(); 
    };
    util::omp_parallel_for(routine, old_screen_size, new_screen_size, n_threads * (old_screen_size+n_threads <= new_screen_size));

    /* update screen_grad */
    for (size_t i = 0; i < new_screen_size; ++i) {
        const auto g = groups[screen_set[i]];
        const auto gs = group_sizes[screen_set[i]];
        const auto sb = screen_begins[i];

        Eigen::Map<vec_value_t>(
            screen_grad.data() + sb,
            gs
        ) = grad.segment(g, gs);
    }

    /* update screen_subset */
    screen_subset.resize(new_screen_value_size);
    int n_processed = 0;
    for (size_t ss_idx = old_screen_size; ss_idx < new_screen_size; ++ss_idx) {
        const auto ss = screen_set[ss_idx];
        const auto g = groups[ss];
        const auto gs = group_sizes[ss];
        Eigen::Map<vec_index_t>(
            screen_subset.data() + old_screen_value_size + n_processed, gs
        ) = vec_index_t::LinSpaced(gs, g, g + gs - 1);
        n_processed += gs;
    }

    /* update screen_subset_order */
    screen_subset_order.resize(new_screen_value_size);
    std::iota(
        std::next(screen_subset_order.begin(), old_screen_value_size),
        screen_subset_order.end(),
        old_screen_value_size
    );
    std::sort(
        screen_subset_order.data(),
        screen_subset_order.data() + screen_subset_order.size(),
        [&](auto i, auto j) { return screen_subset[i] < screen_subset[j]; }
    );

    /* update screen_subset_ordered */
    screen_subset_ordered.resize(new_screen_value_size);
    for (size_t i = 0; i < screen_subset_order.size(); ++i) {
        screen_subset_ordered[i] = screen_subset[screen_subset_order[i]];
    }
}

template <class ValueType, class SafeBoolType=int8_t>
struct GaussianCovBufferPack
{
    using value_t = ValueType;
    using safe_bool_t = SafeBoolType;
    using vec_value_t = util::rowvec_type<value_t>;
    using dyn_vec_value_t = std::vector<value_t>;
    using dyn_vec_bool_t = std::vector<safe_bool_t>;

    GaussianCovBufferPack(
        size_t p
    ):
        buffer_p(p)
    {}

    dyn_vec_value_t screen_grad_prev;
    dyn_vec_value_t screen_beta_prev; 
    dyn_vec_bool_t screen_is_active_prev;

    vec_value_t buffer_p;
};

template <class StateType, class PBType>
ADELIE_CORE_STRONG_INLINE
void pb_add_suffix(
    const StateType& state,
    PBType& pb
)
{
    const auto& devs = state.devs;
    // current relative training % dev explained
    const auto rdev = (
        (devs.size() < 2) ? 0.0 :
        (devs[devs.size()-1] - devs[devs.size()-2]) / devs[devs.size()-1]
    );
    pb << " [rdev:" 
        << std::fixed << std::setprecision(1) 
        << rdev * 100
        << "%]"
        ; 
}

template <class StateType>
ADELIE_CORE_STRONG_INLINE
bool early_exit(
    const StateType& state
)
{
    const auto early_exit = state.early_exit;
    const auto& devs = state.devs;

    if (!early_exit || devs.size() < 2) return false;

    const auto rdev_tol = state.rdev_tol;

    const auto dev_u = devs[devs.size()-1];
    const auto dev_m = devs[devs.size()-2];
    if (dev_u-dev_m <= rdev_tol * dev_u) return true;

    return false;
}

template <class StateType, class StateGaussianPinType, class ValueType>
inline void update_solutions(
    StateType& state,
    StateGaussianPinType& state_gaussian_pin_cov,
    ValueType lmda
)
{
    using state_t = std::decay_t<StateType>;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_value_t = typename state_t::vec_value_t;

    auto& betas = state.betas;
    auto& duals = state.duals;
    auto& intercepts = state.intercepts;
    auto& devs = state.devs;
    auto& lmdas = state.lmdas;

    vec_index_t dual_indices; 
    vec_value_t dual_values;

    betas.emplace_back(std::move(state_gaussian_pin_cov.betas.back()));
    duals.emplace_back(sparsify_dual(state, dual_indices, dual_values));
    intercepts.emplace_back(0);
    lmdas.emplace_back(lmda);

    const auto dev = state_gaussian_pin_cov.rsqs.back();
    devs.emplace_back(dev);
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
    using matrix_cov_t = typename state_t::matrix_t;
    using state_gaussian_pin_cov_t = state::StateGaussianPinCov<
        constraint_t,
        matrix_cov_t,
        typename std::decay_t<matrix_cov_t>::value_t,
        index_t,
        safe_bool_t
    >;

    auto& A = *state.A;
    const auto& constraints = state.constraints;
    const auto& groups = state.groups;
    const auto& group_sizes = state.group_sizes;
    const auto alpha = state.alpha;
    const auto& penalty = state.penalty;
    const auto& screen_set = state.screen_set;
    const auto& screen_begins = state.screen_begins;
    const auto& screen_subset_order = state.screen_subset_order;
    const auto& screen_subset_ordered = state.screen_subset_ordered;
    const auto& screen_vars = state.screen_vars;
    const auto& screen_transforms = state.screen_transforms;
    const auto constraint_buffer_size = state.constraint_buffer_size;
    const auto max_active_size = state.max_active_size;
    const auto max_iters = state.max_iters;
    const auto tol = state.tol;
    const auto rdev_tol = state.rdev_tol;
    const auto newton_tol = state.newton_tol;
    const auto newton_max_iters = state.newton_max_iters;
    const auto n_threads = state.n_threads;
    auto& rsq = state.rsq;
    auto& screen_beta = state.screen_beta;
    auto& screen_grad = state.screen_grad;
    auto& screen_is_active = state.screen_is_active;
    auto& active_set_size = state.active_set_size;
    auto& active_set = state.active_set;

    auto& screen_grad_prev = buffer_pack.screen_grad_prev;
    auto& screen_beta_prev = buffer_pack.screen_beta_prev;
    auto& screen_is_active_prev = buffer_pack.screen_is_active_prev;

    util::rowvec_type<value_t, 1> lmda_path;
    lmda_path = lmda;

    // Save all current valid quantities that will be modified in-place by fit.
    // This is needed in case we exit with exception and need to restore invariance.
    // Saving SHOULD NOT swap since we still need the values of the current containers.
    const auto save_prev_valid = [&]() {
        screen_grad_prev = screen_grad;
        screen_beta_prev = screen_beta;
        screen_is_active_prev = screen_is_active;
    };
    const auto load_prev_valid = [&]() {
        screen_grad.swap(screen_grad_prev);
        screen_beta.swap(screen_beta_prev);
        screen_is_active.swap(screen_is_active_prev);
    };

    save_prev_valid();

    state_gaussian_pin_cov_t state_gaussian_pin_cov(
        A,
        constraints,
        groups, 
        group_sizes,
        alpha, 
        penalty,
        Eigen::Map<const vec_index_t>(screen_set.data(), screen_set.size()), 
        Eigen::Map<const vec_index_t>(screen_begins.data(), screen_begins.size()), 
        Eigen::Map<const vec_value_t>(screen_vars.data(), screen_vars.size()), 
        screen_transforms,
        Eigen::Map<const vec_index_t>(screen_subset_order.data(), screen_subset_order.size()),
        Eigen::Map<const vec_index_t>(screen_subset_ordered.data(), screen_subset_ordered.size()),
        lmda_path,
        constraint_buffer_size,
        max_active_size, max_iters, 
        tol, 
        rdev_tol,
        newton_tol, newton_max_iters, n_threads,
        rsq,
        Eigen::Map<vec_value_t>(screen_beta.data(), screen_beta.size()), 
        Eigen::Map<vec_value_t>(screen_grad.data(), screen_grad.size()), 
        Eigen::Map<vec_safe_bool_t>(screen_is_active.data(), screen_is_active.size()),
        active_set_size,
        active_set
    );

    try {
        state_gaussian_pin_cov.solve(check_user_interrupt);
    } catch(...) {
        load_prev_valid();
        throw;
    }

    rsq = state_gaussian_pin_cov.rsq;
    active_set_size = state_gaussian_pin_cov.active_set_size;

    const auto screen_time = Eigen::Map<const util::rowvec_type<double>>(
        state_gaussian_pin_cov.benchmark_screen.data(),
        state_gaussian_pin_cov.benchmark_screen.size()
    ).sum();
    const auto active_time = Eigen::Map<const util::rowvec_type<double>>(
        state_gaussian_pin_cov.benchmark_active.data(),
        state_gaussian_pin_cov.benchmark_active.size()
    ).sum();

    return std::make_tuple(
        std::move(state_gaussian_pin_cov), 
        screen_time, 
        active_time
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
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;
    using safe_bool_t = typename state_t::safe_bool_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_index_t = typename state_t::vec_index_t;

    const auto p = state.A->cols();
    GaussianCovBufferPack<value_t, safe_bool_t> buffer_pack(p);

    const auto pb_add_suffix_f = [&](const auto& state, auto& pb) {
        cov::pb_add_suffix(state, pb);
    };
    const auto update_loss_null_f = [](const auto&) {};
    const auto update_invariance_f = [&](
        auto& state, 
        const auto& state_gaussian_pin_cov,
        auto lmda
    ) {
        const auto& v = state.v;
        const auto n_threads = state.n_threads;
        auto& A = *state.A;
        auto& grad = state.grad;

        state.lmda = lmda;

        const auto& beta = state_gaussian_pin_cov.betas.back();
        const Eigen::Map<const vec_index_t> beta_indices(
            beta.innerIndexPtr(),
            beta.nonZeros()
        );
        const Eigen::Map<const vec_value_t> beta_values(
            beta.valuePtr(),
            beta.nonZeros()
        );
        A.mul(beta_indices, beta_values, grad);
        matrix::dvveq(grad, v - grad, n_threads);

        update_abs_grad(state, lmda);
    };
    const auto update_solutions_f = [](auto& state, auto& state_gaussian_pin_cov, auto lmda) {
        update_solutions(
            state, 
            state_gaussian_pin_cov,
            lmda
        );
    };
    const auto early_exit_f = [&](const auto& state) {
        return cov::early_exit(state) || exit_cond_f();
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

} // namespace cov
} // namespace gaussian
} // namespace solver
} // namespace adelie_core