#pragma once
#include <adelie_core/optimization/search_pivot.hpp>
#include <adelie_core/solver/solver_gaussian_pin_naive.hpp>
#include <adelie_core/state/state_gaussian_pin_naive.hpp>
#include <adelie_core/util/algorithm.hpp>
#include <adelie_core/util/stopwatch.hpp>

namespace adelie_core {
namespace solver {
namespace glm {
namespace naive {

template <class ValueType>
struct GlmNaiveBufferPack
{
    using value_t = ValueType;
    using safe_bool_t = int;
    using vec_value_t = util::rowvec_type<value_t>;
    using dyn_vec_value_t = std::vector<value_t>;
    using dyn_vec_bool_t = std::vector<safe_bool_t>;
    using dyn_vec_mat_value_t = std::vector<util::rowmat_type<value_t>>;

    GlmNaiveBufferPack(
        size_t n,
        size_t p
    ):
        ones(vec_value_t::Ones(n)),
        X_means(p),
        weights(n),
        weights_sqrt(n),
        y(n),
        resid(n),
        mu_prev(n),
        var(n),
        buffer_n(n)
    {}

    const vec_value_t ones;       // (n,) vector of ones

    vec_value_t X_means;    // (p,) buffer for X column means (only screen groups need to be well-defined)
    dyn_vec_value_t screen_X_means; // (ss,) buffer for X column means on screen groups
    dyn_vec_mat_value_t screen_transforms; // (s,) buffer for eigenvectors on screen groups
    dyn_vec_value_t screen_vars;    // (s,) buffer for eigenvalues on screen groups
    vec_value_t weights;            // (n,) IRLS weights
    vec_value_t weights_sqrt;       // (n,) IRLS weights sqrt
    vec_value_t y;                  // (n,) IRLS response
    vec_value_t resid;              // (n,) IRLS residual
    vec_value_t mu_prev;            // (n,) previous mean
    vec_value_t var;                // (n,) variance 

    dyn_vec_value_t screen_beta_prev;
    dyn_vec_bool_t screen_is_active_prev;

    vec_value_t buffer_n;   // (n,) extra buffer
};

template <class StateType,
          class BufferPackType,
          class ValueType,
          class UpdateCoefficientsType,
          class CUIType=util::no_op>
ADELIE_CORE_STRONG_INLINE
auto fit(
    StateType& state,
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

    auto& glm = *state.glm;
    auto& X = *state.X;
    const auto& y0 = state.y;
    const auto& groups = state.groups;
    const auto& group_sizes = state.group_sizes;
    const auto alpha = state.alpha;
    const auto& penalty = state.penalty;
    const auto& weights0 = state.weights;
    const auto& screen_set = state.screen_set;
    const auto& screen_g1 = state.screen_g1;
    const auto& screen_g2 = state.screen_g2;
    const auto& screen_begins = state.screen_begins;
    const auto intercept = state.intercept;
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
    auto& mu = state.mu;

    const auto& ones = buffer_pack.ones;
    auto& X_means = buffer_pack.X_means;
    auto& screen_X_means = buffer_pack.screen_X_means;
    auto& screen_transforms = buffer_pack.screen_transforms;
    auto& screen_vars = buffer_pack.screen_vars;
    auto& weights = buffer_pack.weights;
    auto& weights_sqrt = buffer_pack.weights_sqrt;
    auto& y = buffer_pack.y;
    auto& resid = buffer_pack.resid;
    auto& mu_prev = buffer_pack.mu_prev;
    auto& var = buffer_pack.var;
    auto& screen_beta_prev = buffer_pack.screen_beta_prev;
    auto& screen_is_active_prev = buffer_pack.screen_is_active_prev;
    auto& buffer_n = buffer_pack.buffer_n;

    util::rowvec_type<value_t, 1> lmda_path_adjusted;

    const auto save_prev_valid = [&]() {
        screen_beta_prev = screen_beta;
        screen_is_active_prev = screen_is_active;
    };
    const auto load_prev_valid = [&]() {
        screen_beta.swap(screen_beta_prev);
        screen_is_active.swap(screen_is_active_prev);
    };

    size_t irls_it = 0;
    while (1) {
        if (irls_it >= irls_max_iters) {
            throw std::runtime_error("Maximum IRLS iterations reached.");
        }

        save_prev_valid();

        /* compute rest of quadratic approximation quantities */
        // TODO: parallelize?
        glm.hessian(eta, var);
        weights = weights0 * var;
        const auto weights_sum = weights.sum();
        weights /= weights_sum;
        weights_sqrt = weights.sqrt();
        y = (y0 - mu) / var + eta;
        const auto y_mean = (weights * y).sum();
        const auto y_var = (weights * y.square()).sum() - y_mean * y_mean;
        resid = weights * (y - eta + intercept * (beta0 - y_mean));
        const auto resid_sum = resid.sum();
        lmda_path_adjusted = lmda / weights_sum;
        for (size_t ss_idx = 0; ss_idx < screen_set.size(); ++ss_idx) {
            const auto i = screen_set[ss_idx];
            const auto g = groups[i];
            const size_t gs = group_sizes[i];
            for (size_t j = 0; j < gs; ++j) {
                X_means[g+j] = X.cmul(g+j, weights);
            }
        }
        // this call should effectively only adjust the size of screen_* quantities
        // and repopulate every entry using the new weights.
        state::glm::naive::update_screen_derived(
            state,
            X_means,
            weights_sqrt,
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
            weights,
            Eigen::Map<const vec_index_t>(screen_set.data(), screen_set.size()), 
            Eigen::Map<const vec_index_t>(screen_g1.data(), screen_g1.size()), 
            Eigen::Map<const vec_index_t>(screen_g2.data(), screen_g2.size()), 
            Eigen::Map<const vec_index_t>(screen_begins.data(), screen_begins.size()), 
            Eigen::Map<const vec_value_t>(screen_vars.data(), screen_vars.size()), 
            Eigen::Map<const vec_value_t>(screen_X_means.data(), screen_X_means.size()), 
            screen_transforms,
            lmda_path_adjusted,
            intercept, max_iters, tol, 0 /* rsq_tol */, 0 /* rsq_slope_tol */, 0 /* rsq_curv_tol */, 
            newton_tol, newton_max_iters, n_threads,
            0 /* rsq (no need to track) */,
            Eigen::Map<vec_value_t>(resid.data(), resid.size()),
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

        // update beta0
        beta0 = state_gaussian_pin_naive.intercepts[0];

        // update eta
        matrix::dvzero(eta, n_threads);
        for (size_t ss_idx = 0; ss_idx < screen_set.size(); ++ss_idx) {
            const auto i = screen_set[ss_idx];
            const auto g = groups[i];
            const auto gs = group_sizes[i];
            const auto sb = screen_begins[ss_idx];
            const auto beta_g = Eigen::Map<const vec_value_t>(
                screen_beta.data() + sb,
                gs
            );
            if (gs == 1) X.ctmul(g, beta_g[0], ones, buffer_n);
            else X.btmul(g, gs, beta_g, ones, buffer_n);
            matrix::dvaddi(eta, buffer_n, n_threads);
        }
        if (intercept) eta += beta0;

        // update mu
        mu_prev.swap(mu);
        glm.gradient(eta, mu); 

        /* check convergence */
        if ((weights * (mu - mu_prev).square()).sum() <= irls_tol) {
            return state_gaussian_pin_naive;
        }

        ++irls_it;
    }
}

template <class StateType,
          class UpdateCoefficientsType,
          class CUIType=util::no_op>
inline void solve(
    StateType&& state,
    bool /*display*/,
    UpdateCoefficientsType update_coefficients_f,
    CUIType check_user_interrupt = CUIType()
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;
    //using vec_safe_bool_t = typename state_t::vec_safe_bool_t;
    //using sw_t = util::Stopwatch;

    const auto& y0 = state.y;
    const auto alpha = state.alpha;
    const auto& penalty = state.penalty;
    const auto& weights0 = state.weights;
    const auto& screen_set = state.screen_set;
    //const auto early_exit = state.early_exit;
    const auto max_screen_size = state.max_screen_size;
    const auto setup_lmda_max = state.setup_lmda_max;
    //const auto setup_lmda_path = state.setup_lmda_path;
    //const auto lmda_path_size = state.lmda_path_size;
    //const auto min_ratio = state.min_ratio;
    //const auto intercept = state.intercept;
    //const auto n_threads = state.n_threads;
    const auto& abs_grad = state.abs_grad;
    const auto& mu = state.mu;
    auto& X = *state.X;
    auto& lmda_max = state.lmda_max;
    //auto& lmda_path = state.lmda_path;
    //auto& screen_is_active = state.screen_is_active;
    //auto& screen_beta = state.screen_beta;
    auto& grad = state.grad;
    auto& lmda = state.lmda;
    //auto& resid_prev_valid = state.resid_prev_valid;
    //auto& screen_beta_prev_valid = state.screen_beta_prev_valid; 
    //auto& screen_is_active_prev_valid = state.screen_is_active_prev_valid;
    //auto& benchmark_screen = state.benchmark_screen;
    //auto& benchmark_fit_screen = state.benchmark_fit_screen;
    //auto& benchmark_fit_active = state.benchmark_fit_active;
    //auto& benchmark_kkt = state.benchmark_kkt;
    //auto& benchmark_invariance = state.benchmark_invariance;
    //auto& n_valid_solutions = state.n_valid_solutions;
    //auto& active_sizes = state.active_sizes;
    //auto& screen_sizes = state.screen_sizes;

    if (screen_set.size() > max_screen_size) throw util::max_basil_screen_set();

    const auto n = X.rows();
    const auto p = X.cols();
    GlmNaiveBufferPack<value_t> buffer_pack(n, p);

    auto& buffer_n = buffer_pack.buffer_n;

    // ==================================================================================== 
    // Initial fit for lambda ~ infinity to setup lmda_max
    // ==================================================================================== 
    // Only unpenalized (l1) groups are active in this case.
    // State must include all unpenalized groups.
    // We solve for large lambda, then back-track the KKT condition to find the lambda
    // that leads to that solution where all penalized variables have 0 coefficient.
    if (setup_lmda_max) {
        vec_value_t large_lmda_path(1);
        // NOTE: std::numeric_limits<value_t>::max() does not work here
        // because when we call fit, we have to scale lmda, 
        // which may make it exceed the initial value.
        // Instead, we just set it to some very large number O_O (similar to glmnet).
        large_lmda_path[0] = 9.9e35; 

        fit(
            state,
            buffer_pack,
            large_lmda_path,
            update_coefficients_f,
            check_user_interrupt
        );

        /* Invariance */
        lmda = large_lmda_path[0];
        buffer_n = weights0 * (y0 - mu);
        X.mul(buffer_n, grad);
        state::gaussian::update_abs_grad(state, lmda);

        /* Compute lmda_max */
        const auto factor = (alpha <= 0) ? 1e-3 : alpha;
        lmda_max = vec_value_t::NullaryExpr(
            abs_grad.size(),
            [&](auto i) { 
                return (penalty[i] <= 0.0) ? 0.0 : abs_grad[i] / penalty[i];
            }
        ).maxCoeff() / factor;
    }
}

} // namespace naive 
} // namespace glm
} // namespace solver
} // namespace adelie_core