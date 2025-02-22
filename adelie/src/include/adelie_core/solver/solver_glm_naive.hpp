#pragma once
#include <adelie_core/configs.hpp>
#include <adelie_core/solver/solver_gaussian_naive.hpp>
#include <adelie_core/solver/solver_gaussian_pin_naive.hpp>
#include <adelie_core/state/state_gaussian_pin_naive.hpp>
#include <adelie_core/util/omp.hpp>

namespace adelie_core {
namespace solver {
namespace glm {
namespace naive {

template <class ValueType, class SafeBoolType=int8_t>
struct GlmNaiveBufferPack
{
    using value_t = ValueType;
    using safe_bool_t = SafeBoolType;
    using vec_value_t = util::rowvec_type<value_t>;
    using dyn_vec_value_t = std::vector<value_t>;
    using dyn_vec_bool_t = std::vector<safe_bool_t>;
    using dyn_vec_mat_value_t = std::vector<util::rowmat_type<value_t>>;

    GlmNaiveBufferPack(
        size_t n,
        size_t p
    ):
        X_means(p),
        irls_weights(n),
        irls_weights_sqrt(n),
        irls_y(n),
        irls_resid(n),
        resid_prev(n),
        eta_prev(n),
        hess(n),
        ones(vec_value_t::Ones(n)),
        buffer_n(n)
    {}

    vec_value_t X_means;    // (p,) buffer for X column means (only screen groups need to be well-defined)
    dyn_vec_value_t screen_X_means; // (ss,) buffer for X column means on screen groups
    dyn_vec_mat_value_t screen_transforms; // (s,) buffer for eigenvectors on screen groups
    dyn_vec_value_t screen_vars;    // (s,) buffer for eigenvalues on screen groups
    vec_value_t irls_weights;            // (n,) IRLS weights
    vec_value_t irls_weights_sqrt;       // (n,) IRLS weights sqrt
    vec_value_t irls_y;                  // (n,) IRLS response
    vec_value_t irls_resid;              // (n,) IRLS residual
    vec_value_t resid_prev;            // (n,) previous residual
    vec_value_t eta_prev;            // (n,) previous eta
    vec_value_t hess;                // (n,) hessian 

    dyn_vec_value_t screen_beta_prev;
    dyn_vec_bool_t screen_is_active_prev;

    vec_value_t ones;       // (n,) vector of ones
    vec_value_t buffer_n;   // (n,) extra buffer
};

/**
 * Unlike the similar function in gaussian::naive,
 * this does not call the base version to update the base classes's screen derived quantities.
 * This is because in GLM fitting, the three screen_* inputs are modified at every IRLS loop,
 * while the base quantities remain the same. 
 * It is only when IRLS finishes and we must screen for variables where we have to update the base quantities.
 * In gaussian naive setting, the IRLS has loop size of 1 essentially, so the two versions are synonymous.
 */
template <
    class StateType, 
    class XMType, 
    class WType,
    class SXMType, 
    class STType, 
    class SVType
>
inline void update_screen_derived(
    StateType& state,
    const XMType& X_means,
    const WType& weights_sqrt,
    size_t begin,
    size_t end,
    SXMType& screen_X_means,
    STType& screen_transforms,
    SVType& screen_vars
)
{
    const auto& group_sizes = state.group_sizes;
    const auto& screen_set = state.screen_set;
    const auto& screen_begins = state.screen_begins;

    const auto new_screen_size = screen_set.size();
    const int new_screen_value_size = (
        (screen_begins.size() == 0) ? 0 : (
            screen_begins.back() + group_sizes[screen_set.back()]
        )
    );

    screen_X_means.resize(new_screen_value_size);    
    screen_transforms.resize(new_screen_size);
    screen_vars.resize(new_screen_value_size, 0);

    gaussian::naive::update_screen_derived(
        *state.X,
        X_means,
        weights_sqrt,
        state.groups,
        state.group_sizes,
        state.screen_set,
        state.screen_begins,
        begin,
        end,
        state.intercept,
        state.n_threads,
        screen_X_means,
        screen_transforms,
        screen_vars
    );
}

template <
    class StateType, 
    class GlmType,
    class StateGaussianPinType,
    class ValueType
>
ADELIE_CORE_STRONG_INLINE
void update_solutions(
    StateType& state,
    GlmType& glm,
    StateGaussianPinType& state_gaussian_pin_naive,
    ValueType lmda
)
{
    using state_t = std::decay_t<StateType>;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_value_t = typename state_t::vec_value_t;

    const auto loss_null = state.loss_null;
    const auto loss_full = state.loss_full;
    const auto& eta = state.eta;
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

    const auto loss = glm.loss(eta);
    devs.emplace_back(
        (loss_null - loss) /
        (loss_null - loss_full)
    );
}

template <
    class StateType,
    class GlmType,
    class BufferPackType
>
ADELIE_CORE_STRONG_INLINE
void update_loss_null(
    StateType& state,
    GlmType& glm,
    BufferPackType& buffer_pack
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;

    const auto& offsets = state.offsets;
    const auto intercept = state.intercept;
    auto& loss_null = state.loss_null;

    if (!intercept) {
        loss_null = glm.loss(offsets);
        return;
    }

    const auto irls_max_iters = state.irls_max_iters;
    const auto irls_tol = state.irls_tol;

    // make copies since we do not want to mess with the warm-start.
    // this function is only needed to fit intercept-only model and get loss_null.
    value_t beta0 = state.beta0;
    vec_value_t eta = state.eta;
    vec_value_t resid = state.resid;

    auto& irls_y = buffer_pack.irls_y;
    auto& resid_prev = buffer_pack.resid_prev;
    auto& eta_prev = buffer_pack.eta_prev;
    auto& hess = buffer_pack.hess;

    size_t irls_it = 0;

    while (1) {
        if (irls_it >= irls_max_iters) {
            throw util::adelie_core_solver_error("Maximum IRLS iterations reached.");
        }

        /* compute rest of quadratic approximation quantities */
        glm.hessian(eta, resid, hess);
        glm.inv_hessian_gradient(eta, resid, hess, irls_y);
        // hessian is raised whenever <= 0 for well-defined proximal Newton iterations
        hess = hess.max(0) + value_t(Configs::hessian_min) * (hess <= 0).template cast<value_t>();
        const auto hess_sum = hess.sum();

        /* fit beta0 */
        beta0 = (hess * (irls_y + eta - offsets)).sum() / hess_sum;

        // update eta
        eta.swap(eta_prev);
        eta = beta0 + offsets;

        // update resid
        resid_prev.swap(resid);
        glm.gradient(eta, resid); 

        /* check convergence */
        if (std::abs(((resid - resid_prev) * (eta - eta_prev)).sum()) <= irls_tol) {
            loss_null = glm.loss(eta);
            return;
        }

        ++irls_it;
    }
}

template <
    class StateType,
    class GlmType,
    class BufferPackType,
    class ValueType,
    class CUIType=util::no_op
>
inline auto fit(
    StateType& state,
    GlmType& glm,
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
    const auto& constraints = state.constraints;
    const auto& groups = state.groups;
    const auto& group_sizes = state.group_sizes;
    const auto alpha = state.alpha;
    const auto& penalty = state.penalty;
    const auto& offsets = state.offsets;
    const auto& screen_set = state.screen_set;
    const auto& screen_begins = state.screen_begins;
    const auto constraint_buffer_size = state.constraint_buffer_size;
    const auto intercept = state.intercept;
    const auto max_active_size = state.max_active_size;
    const auto irls_max_iters = state.irls_max_iters;
    const auto irls_tol = state.irls_tol;
    const auto max_iters = state.max_iters;
    const auto tol = state.tol;
    const auto newton_tol = state.newton_tol;
    const auto newton_max_iters = state.newton_max_iters;
    const auto n_threads = state.n_threads;
    const auto loss_null = state.loss_null;
    const auto loss_full = state.loss_full;
    auto& screen_beta = state.screen_beta;
    auto& screen_is_active = state.screen_is_active;
    auto& active_set_size = state.active_set_size;
    auto& active_set = state.active_set;
    auto& beta0 = state.beta0;
    auto& eta = state.eta;
    auto& resid = state.resid;

    const auto& ones = buffer_pack.ones;
    auto& X_means = buffer_pack.X_means;
    auto& screen_X_means = buffer_pack.screen_X_means;
    auto& screen_transforms = buffer_pack.screen_transforms;
    auto& screen_vars = buffer_pack.screen_vars;
    auto& irls_weights = buffer_pack.irls_weights;
    auto& irls_weights_sqrt = buffer_pack.irls_weights_sqrt;
    auto& irls_y = buffer_pack.irls_y;
    auto& irls_resid = buffer_pack.irls_resid;
    auto& eta_prev = buffer_pack.eta_prev;
    auto& resid_prev = buffer_pack.resid_prev;
    auto& hess = buffer_pack.hess;
    auto& screen_beta_prev = buffer_pack.screen_beta_prev;
    auto& screen_is_active_prev = buffer_pack.screen_is_active_prev;

    util::rowvec_type<value_t, 1> lmda_path_adjusted;

    // Save all current valid quantities that will be modified in-place by fit.
    // This is needed in case we exit with exception and need to restore invariance.
    // Saving SHOULD NOT swap since we still need the values of the current containers.
    const auto save_prev_valid = [&]() {
        screen_beta_prev = screen_beta;
        screen_is_active_prev = screen_is_active;
    };
    const auto load_prev_valid = [&]() {
        screen_beta.swap(screen_beta_prev);
        screen_is_active.swap(screen_is_active_prev);
    };

    value_t screen_time = 0;
    value_t active_time = 0;
    size_t irls_it = 0;

    while (1) {
        if (irls_it >= irls_max_iters) {
            throw util::adelie_core_solver_error("Maximum IRLS iterations reached.");
        }

        save_prev_valid();

        /* compute rest of quadratic approximation quantities */
        glm.hessian(eta, resid, hess);
        glm.inv_hessian_gradient(eta, resid, hess, irls_resid);
        // hessian is raised whenever <= 0 for well-defined proximal Newton iterations
        hess = hess.max(0) + value_t(Configs::hessian_min) * (hess <= 0).template cast<value_t>();
        const auto hess_sum = hess.sum();
        irls_weights = hess / hess_sum;
        irls_weights_sqrt = irls_weights.sqrt();
        irls_y = irls_resid + eta - offsets;
        const auto y_mean = (irls_weights * irls_y).sum();
        const auto y_var = (irls_weights * irls_y.square()).sum() - intercept * y_mean * y_mean;
        if (intercept) irls_resid += (beta0 - y_mean);
        const auto resid_sum = (irls_weights * irls_resid).sum();
        lmda_path_adjusted = lmda / hess_sum;
        if (std::isinf(lmda_path_adjusted[0])) {
            if (lmda == std::numeric_limits<value_t>::max()) {
                lmda_path_adjusted = lmda;
            }
            else {
                throw util::adelie_core_solver_error(
                    "IRLS lambda is unexpectedly inf. "
                    "This likely indicates a bug in the code. Please report this!"
                );
            }
        }

        const auto update_X_means = [&](auto ss_idx) {
            const auto i = screen_set[ss_idx];
            const auto g = groups[i];
            const size_t gs = group_sizes[i];
            if (gs == 1) {
                X_means[g] = X.cmul_safe(g, ones, irls_weights);
            } else {
                Eigen::Map<vec_value_t> Xi_means(X_means.data() + g, gs);
                X.bmul_safe(g, gs, ones, irls_weights, Xi_means);
            }
        };
        util::omp_parallel_for(update_X_means, 0, screen_set.size(), n_threads * (n_threads <= screen_set.size()));

        // this call should only adjust the size of screen_* quantities
        // and repopulate every entry using the new weights.
        update_screen_derived(
            state,
            X_means,
            irls_weights_sqrt,
            0,
            screen_set.size(),
            screen_X_means,
            screen_transforms,
            screen_vars
        );

        /* fit gaussian pin */
        // update screen_beta, screen_is_active
        state_gaussian_pin_naive_t state_gaussian_pin_naive(
            X,
            y_mean,
            y_var,
            constraints,
            groups, 
            group_sizes,
            alpha, 
            penalty,
            irls_weights,
            Eigen::Map<const vec_index_t>(screen_set.data(), screen_set.size()), 
            Eigen::Map<const vec_index_t>(screen_begins.data(), screen_begins.size()), 
            Eigen::Map<const vec_value_t>(screen_vars.data(), screen_vars.size()), 
            Eigen::Map<const vec_value_t>(screen_X_means.data(), screen_X_means.size()), 
            screen_transforms,
            lmda_path_adjusted,
            constraint_buffer_size,
            intercept, max_active_size, max_iters, 
            tol * (loss_null - loss_full) / hess_sum, 
            0 /* adev_tol */, 0 /* ddev_tol */,
            newton_tol, newton_max_iters, n_threads,
            0 /* rsq (no need to track) */,
            Eigen::Map<vec_value_t>(irls_resid.data(), irls_resid.size()),
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

        // update benchmark times
        screen_time += Eigen::Map<const util::rowvec_type<double>>(
            state_gaussian_pin_naive.benchmark_screen.data(),
            state_gaussian_pin_naive.benchmark_screen.size()
        ).sum();
        active_time += Eigen::Map<const util::rowvec_type<double>>(
            state_gaussian_pin_naive.benchmark_active.data(),
            state_gaussian_pin_naive.benchmark_active.size()
        ).sum();

        // update invariants
        active_set_size = state_gaussian_pin_naive.active_set_size;
        beta0 = state_gaussian_pin_naive.intercepts[0];

        // update eta
        eta.swap(eta_prev);
        eta = irls_y + offsets - irls_resid;
        if (intercept) eta += beta0 - y_mean;

        // update resid
        resid_prev.swap(resid);
        glm.gradient(eta, resid); 

        /* check convergence */
        if (std::abs(((resid - resid_prev) * (eta - eta_prev)).sum()) <= irls_tol) {
            return std::make_tuple(
                std::move(state_gaussian_pin_naive),
                screen_time,
                active_time
            );
        }

        ++irls_it;
    }
}

template <
    class StateType,
    class GlmType,
    class PBType,
    class ExitCondType,
    class UpdateLossNullType,
    class TidyType,
    class CUIType
>
inline void solve(
    StateType&& state,
    GlmType&& glm,
    PBType&& pb,
    ExitCondType exit_cond_f,
    UpdateLossNullType update_loss_null_f,
    TidyType tidy_f,
    CUIType check_user_interrupt
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;
    using safe_bool_t = typename state_t::safe_bool_t;

    const auto n = state.X->rows();
    const auto p = state.X->cols();
    GlmNaiveBufferPack<value_t, safe_bool_t> buffer_pack(n, p);

    const auto pb_add_suffix_f = [&](const auto& state, auto& pb) {
        solver::pb_add_suffix(state, pb);
    };
    const auto update_loss_null_wrap_f = [&](auto& state) {
        const auto setup_loss_null = state.setup_loss_null;
        if (setup_loss_null) update_loss_null_f(state, glm, buffer_pack);
    };
    const auto update_invariance_f = [&](auto& state, const auto&, auto lmda) {
        const auto& resid = state.resid;
        auto& X = *state.X;
        auto& grad = state.grad;
        const auto& ones = buffer_pack.ones;
        state.lmda = lmda;
        X.mul(resid, ones, grad);
        update_abs_grad(state, lmda);
    };
    const auto update_solutions_f = [&](auto& state, auto& state_gaussian_pin_naive, auto lmda) {
        update_solutions(
            state, 
            glm,
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
        update_screen_derived_base(state);
    };
    const auto fit_f = [&](auto& state, auto lmda) {
        return fit(
            state,
            glm,
            buffer_pack,
            lmda,
            check_user_interrupt
        );
    };

    solver::solve_core(
        state,
        pb,
        pb_add_suffix_f,
        update_loss_null_wrap_f,
        update_invariance_f,
        update_solutions_f,
        early_exit_f,
        screen_f,
        fit_f
    );
}

template <
    class StateType,
    class GlmType,
    class PBType,
    class ExitCondType,
    class CUIType=util::no_op
>
inline void solve(
    StateType&& state,
    GlmType&& glm,
    PBType&& pb,
    ExitCondType exit_cond_f,
    CUIType check_user_interrupt = CUIType()
)
{
    solve(
        std::forward<StateType>(state), 
        std::forward<GlmType>(glm), 
        std::forward<PBType>(pb),
        exit_cond_f,
        [](auto& state, auto& glm, auto& buffer_pack) {
            update_loss_null(state, glm, buffer_pack);
        },
        [](){},
        check_user_interrupt
    );
}

} // namespace naive 
} // namespace glm
} // namespace solver
} // namespace adelie_core