#pragma once
#include <adelie_core/configs.hpp>
#include <adelie_core/solver/solver_base.hpp>
#include <adelie_core/solver/solver_gaussian_pin_naive.hpp>
#include <adelie_core/state/state_gaussian_pin_naive.hpp>
#include <adelie_core/state/state_glm_naive.hpp>

namespace adelie_core {
namespace solver {
namespace glm {
namespace naive {

template <class ValueType>
struct GlmNaiveBufferPack
{
    using value_t = ValueType;
    using safe_bool_t = int8_t;
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

template <class StateType, 
          class GlmType,
          class StateGaussianPinType,
          class ValueType>
ADELIE_CORE_STRONG_INLINE
void update_solutions(
    StateType& state,
    GlmType& glm,
    StateGaussianPinType& state_gaussian_pin_naive,
    ValueType lmda
)
{
    const auto loss_null = state.loss_null;
    const auto loss_full = state.loss_full;
    const auto& eta = state.eta;
    auto& betas = state.betas;
    auto& devs = state.devs;
    auto& lmdas = state.lmdas;
    auto& intercepts = state.intercepts;

    betas.emplace_back(std::move(state_gaussian_pin_naive.betas.back()));
    intercepts.emplace_back(state_gaussian_pin_naive.intercepts.back());
    lmdas.emplace_back(lmda);

    const auto loss = glm.loss(eta);
    devs.emplace_back(
        (loss_null - loss) /
        (loss_null - loss_full)
    );
}

template <class StateType,
          class GlmType,
          class BufferPackType>
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

    auto& irls_weights = buffer_pack.irls_weights;
    auto& irls_y = buffer_pack.irls_y;
    auto& resid_prev = buffer_pack.resid_prev;
    auto& eta_prev = buffer_pack.eta_prev;
    auto& hess = buffer_pack.hess;

    size_t irls_it = 0;

    while (1) {
        if (irls_it >= irls_max_iters) {
            throw std::runtime_error("Maximum IRLS iterations reached.");
        }

        /* compute rest of quadratic approximation quantities */
        glm.hessian(eta, resid, hess);
        glm.inv_hessian_gradient(eta, resid, hess, irls_y);
        // hessian is raised whenever <= 0 for well-defined proximal Newton iterations
        hess = hess.max(0) + value_t(Configs::hessian_min) * (hess <= 0).template cast<value_t>();
        const auto hess_sum = hess.sum();
        irls_weights = hess / hess_sum;
        irls_y += eta - offsets;

        /* fit beta0 */
        beta0 = (irls_weights * irls_y).sum();

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

template <class StateType,
          class GlmType,
          class BufferPackType,
          class ValueType,
          class UpdateCoefficientsType,
          class CUIType=util::no_op>
ADELIE_CORE_STRONG_INLINE
auto fit(
    StateType& state,
    GlmType& glm,
    BufferPackType& buffer_pack,
    ValueType lmda,
    UpdateCoefficientsType update_coefficients_f,
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
    using matrix_naive_t = typename state_t::matrix_t;
    using state_gaussian_pin_naive_t = state::StateGaussianPinNaive<
        matrix_naive_t,
        typename std::decay_t<matrix_naive_t>::value_t,
        index_t,
        safe_bool_t
    >;

    auto& X = *state.X;
    const auto& groups = state.groups;
    const auto& group_sizes = state.group_sizes;
    const auto alpha = state.alpha;
    const auto& penalty = state.penalty;
    const auto& offsets = state.offsets;
    const auto& screen_set = state.screen_set;
    const auto& screen_g1 = state.screen_g1;
    const auto& screen_g2 = state.screen_g2;
    const auto& screen_begins = state.screen_begins;
    const auto intercept = state.intercept;
    const auto max_active_size = state.max_active_size;
    const auto irls_max_iters = state.irls_max_iters;
    const auto irls_tol = state.irls_tol;
    const auto max_iters = state.max_iters;
    const auto tol = state.tol;
    const auto newton_tol = state.newton_tol;
    const auto newton_max_iters = state.newton_max_iters;
    const auto n_threads = state.n_threads;
    auto& screen_beta = state.screen_beta;
    auto& screen_is_active = state.screen_is_active;
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
            throw std::runtime_error("Maximum IRLS iterations reached.");
        }

        save_prev_valid();

        /* compute rest of quadratic approximation quantities */
        glm.hessian(eta, resid, hess);
        glm.inv_hessian_gradient(eta, resid, hess, irls_y);
        // hessian is raised whenever <= 0 for well-defined proximal Newton iterations
        hess = hess.max(0) + value_t(Configs::hessian_min) * (hess <= 0).template cast<value_t>();
        const auto hess_sum = hess.sum();
        irls_weights = hess / hess_sum;
        irls_weights_sqrt = irls_weights.sqrt();
        irls_y += eta - offsets;
        const auto y_mean = (irls_weights * irls_y).sum();
        const auto y_var = (irls_weights * irls_y.square()).sum() - intercept * y_mean * y_mean;
        irls_resid = irls_y + offsets - eta + intercept * (beta0 - y_mean);
        const auto resid_sum = (irls_weights * irls_resid).sum();
        lmda_path_adjusted = lmda / hess_sum;
        if (std::isinf(lmda_path_adjusted[0])) {
            if (lmda == std::numeric_limits<value_t>::max()) {
                lmda_path_adjusted = lmda;
            }
            else {
                throw std::runtime_error(
                    "IRLS lambda is unexpectedly inf. "
                    "This likely indicates a bug in the code. Please report this!"
                );
            }
        }
        for (size_t ss_idx = 0; ss_idx < screen_set.size(); ++ss_idx) {
            const auto i = screen_set[ss_idx];
            const auto g = groups[i];
            const size_t gs = group_sizes[i];
            if (gs == 1) {
                X_means[g] = X.cmul(g, ones, irls_weights);
            } else {
                Eigen::Map<vec_value_t> Xi_means(X_means.data() + g, gs);
                X.bmul(g, gs, ones, irls_weights, Xi_means);
            }
        }
        // this call should only adjust the size of screen_* quantities
        // and repopulate every entry using the new weights.
        state::glm::naive::update_screen_derived(
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
            groups, 
            group_sizes,
            alpha, 
            penalty,
            irls_weights,
            Eigen::Map<const vec_index_t>(screen_set.data(), screen_set.size()), 
            Eigen::Map<const vec_index_t>(screen_g1.data(), screen_g1.size()), 
            Eigen::Map<const vec_index_t>(screen_g2.data(), screen_g2.size()), 
            Eigen::Map<const vec_index_t>(screen_begins.data(), screen_begins.size()), 
            Eigen::Map<const vec_value_t>(screen_vars.data(), screen_vars.size()), 
            Eigen::Map<const vec_value_t>(screen_X_means.data(), screen_X_means.size()), 
            screen_transforms,
            lmda_path_adjusted,
            intercept, max_active_size, max_iters, tol, 0 /* adev_tol */, 0 /* ddev_tol */,
            newton_tol, newton_max_iters, n_threads,
            0 /* rsq (no need to track) */,
            Eigen::Map<vec_value_t>(irls_resid.data(), irls_resid.size()),
            resid_sum,
            Eigen::Map<vec_value_t>(screen_beta.data(), screen_beta.size()), 
            Eigen::Map<vec_safe_bool_t>(screen_is_active.data(), screen_is_active.size())
        );
        try { 
            gaussian::pin::naive::solve(
                state_gaussian_pin_naive, 
                update_coefficients_f, 
                check_user_interrupt
            );
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

        // update beta0
        beta0 = state_gaussian_pin_naive.intercepts[0];

        // update eta
        eta.swap(eta_prev);
        eta = irls_y + offsets - irls_resid + intercept * (beta0 - y_mean);

        // update resid
        resid_prev.swap(resid);
        glm.gradient(eta, resid); 

        /* check convergence */
        // check directional derivative of gradient (resid) as an approximation
        // to the quadratic loss. 
        const auto& active_set = state_gaussian_pin_naive.active_set;
        const auto& active_begins = state_gaussian_pin_naive.active_begins;
        const auto n_active = (
            (active_begins.size() == 0) ? 1 : (
                active_begins.back() + group_sizes[screen_set[active_set.back()]]
            )
        );
        if (std::abs(((resid - resid_prev) * (eta - eta_prev)).sum()) <= irls_tol * n_active) {
            return std::make_tuple(
                std::move(state_gaussian_pin_naive),
                screen_time,
                active_time
            );
        }

        ++irls_it;
    }
}

template <class StateType,
          class GlmType,
          class UpdateLossNullType,
          class UpdateCoefficientsType,
          class CUIType>
inline void solve(
    StateType&& state,
    GlmType&& glm,
    bool display,
    UpdateLossNullType update_loss_null_f,
    UpdateCoefficientsType update_coefficients_f,
    CUIType check_user_interrupt
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;

    const auto n = state.X->rows();
    const auto p = state.X->cols();
    GlmNaiveBufferPack<value_t> buffer_pack(n, p);

    const auto pb_add_suffix_f = [&](const auto& state, auto& pb) {
        if (display) solver::pb_add_suffix(state, pb);
    };
    const auto update_loss_null_wrap_f = [&](auto& state) {
        const auto setup_loss_null = state.setup_loss_null;
        if (setup_loss_null) update_loss_null_f(state, glm, buffer_pack);
    };
    const auto update_invariance_f = [&](auto& state, auto lmda) {
        const auto& resid = state.resid;
        auto& X = *state.X;
        auto& grad = state.grad;
        const auto& ones = buffer_pack.ones;
        state.lmda = lmda;
        X.mul(resid, ones, grad);
        state::update_abs_grad(state, lmda);
    };
    const auto update_solutions_f = [&](auto& state, auto& state_gaussian_pin_naive, auto lmda) {
        update_solutions(
            state, 
            glm,
            state_gaussian_pin_naive,
            lmda
        );
    };
    const auto early_exit_f = [](const auto& state) {
        return solver::early_exit(state);
    };
    const auto screen_f = [](auto& state, auto lmda, auto kkt_passed, auto n_new_active) {
        solver::screen(
            state,
            lmda,
            kkt_passed,
            n_new_active
        );
        state::update_screen_derived_base(state);
    };
    const auto fit_f = [&](auto& state, auto lmda) {
        return fit(
            state,
            glm,
            buffer_pack,
            lmda,
            update_coefficients_f,
            check_user_interrupt
        );
    };

    solver::solve_core(
        state,
        display,
        pb_add_suffix_f,
        update_loss_null_wrap_f,
        update_invariance_f,
        update_solutions_f,
        early_exit_f,
        screen_f,
        fit_f
    );
}

template <class StateType,
          class GlmType,
          class UpdateCoefficientsType,
          class CUIType=util::no_op>
inline void solve(
    StateType&& state,
    GlmType&& glm,
    bool display,
    UpdateCoefficientsType update_coefficients_f,
    CUIType check_user_interrupt = CUIType()
)
{
    solve(
        std::forward<StateType>(state), 
        std::forward<GlmType>(glm), 
        display, 
        [](auto& state, auto& glm, auto& buffer_pack) {
            update_loss_null(state, glm, buffer_pack);
        },
        update_coefficients_f, 
        check_user_interrupt
    );
}

} // namespace naive 
} // namespace glm
} // namespace solver
} // namespace adelie_core