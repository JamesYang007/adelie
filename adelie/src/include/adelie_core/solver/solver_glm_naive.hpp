#pragma once
#include <adelie_core/optimization/search_pivot.hpp>
#include <adelie_core/solver/solver_gaussian_naive.hpp>
#include <adelie_core/solver/solver_gaussian_pin_naive.hpp>
#include <adelie_core/solver/utils.hpp>
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
        hess(n),
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
    vec_value_t hess;                // (n,) hessian 

    dyn_vec_value_t screen_beta_prev;
    dyn_vec_bool_t screen_is_active_prev;

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
    auto& hess = buffer_pack.hess;

    size_t irls_it = 0;

    while (1) {
        if (irls_it >= irls_max_iters) {
            throw std::runtime_error("Maximum IRLS iterations reached.");
        }

        /* compute rest of quadratic approximation quantities */
        glm.hessian(eta, resid, hess);
        const auto hess_sum = hess.sum();
        irls_weights = hess / hess_sum;
        irls_y = resid.NullaryExpr(resid.size(), [&](auto i) {
            const auto ratio = resid[i] / hess[i]; 
            return std::isnan(ratio) ? resid[i] : ratio;
        }) + eta - offsets;

        /* fit beta0 */
        beta0 = (irls_weights * irls_y).sum();

        // update eta
        eta = beta0 + offsets;

        // update resid
        resid_prev.swap(resid);
        glm.gradient(eta, resid); 

        /* check convergence */
        if ((resid - resid_prev).square().sum() <= irls_tol) {
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

    auto& X_means = buffer_pack.X_means;
    auto& screen_X_means = buffer_pack.screen_X_means;
    auto& screen_transforms = buffer_pack.screen_transforms;
    auto& screen_vars = buffer_pack.screen_vars;
    auto& irls_weights = buffer_pack.irls_weights;
    auto& irls_weights_sqrt = buffer_pack.irls_weights_sqrt;
    auto& irls_y = buffer_pack.irls_y;
    auto& irls_resid = buffer_pack.irls_resid;
    auto& resid_prev = buffer_pack.resid_prev;
    auto& hess = buffer_pack.hess;
    auto& screen_beta_prev = buffer_pack.screen_beta_prev;
    auto& screen_is_active_prev = buffer_pack.screen_is_active_prev;

    util::rowvec_type<value_t, 1> lmda_path_adjusted;

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
        const auto hess_sum = hess.sum();
        irls_weights = hess / hess_sum;
        irls_weights_sqrt = irls_weights.sqrt();
        irls_y = resid.NullaryExpr(resid.size(), [&](auto i) {
            const auto ratio = resid[i] / hess[i]; 
            return std::isnan(ratio) ? resid[i] : ratio;
        }) + eta - offsets;
        const auto y_mean = (irls_weights * irls_y).sum();
        const auto y_var = (irls_weights * irls_y.square()).sum() - intercept * y_mean * y_mean;
        irls_resid = irls_weights * (irls_y + offsets - eta + intercept * (beta0 - y_mean));
        const auto resid_sum = irls_resid.sum();
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
                X_means[g] = X.cmul(g, irls_weights);
            } else {
                Eigen::Map<vec_value_t> Xi_means(X_means.data() + g, gs);
                X.bmul(g, gs, irls_weights, Xi_means);
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
        eta = (
            irls_y + offsets - 
            irls_resid / (irls_weights + (irls_weights <= 0).template cast<value_t>()) + 
            intercept * (beta0 - y_mean)
        );

        // update resid
        resid_prev.swap(resid);
        glm.gradient(eta, resid); 

        /* check convergence */
        if ((resid - resid_prev).square().sum() <= irls_tol) {
            return std::make_tuple(
                std::move(state_gaussian_pin_naive),
                screen_time,
                active_time
            );
        }

        ++irls_it;
    }
}

/**
 * Checks the KKT condition on the proposed solutions in state_gaussian_pin_naive.
 */
template <class StateType, 
          class ValueType>
ADELIE_CORE_STRONG_INLINE
size_t kkt(
    StateType& state,
    ValueType lmda
)
{
    const auto& groups = state.groups;
    const auto alpha = state.alpha;
    const auto& penalty = state.penalty;
    const auto& screen_hashset = state.screen_hashset;
    const auto& abs_grad = state.abs_grad;
    const auto& resid = state.resid;
    auto& X = *state.X;
    auto& grad = state.grad;

    const auto is_screen = [&](auto i) {
        return screen_hashset.find(i) != screen_hashset.end();
    };

    // NOTE: no matter what, every gradient must be computed for pivot method.
    X.mul(resid, grad);
    state::gaussian::update_abs_grad(state, lmda);

    for (int k = 0; k < groups.size(); ++k) {
        if (is_screen(k)) continue;
        const auto pk = penalty[k];
        const auto abs_grad_k = abs_grad[k];
        if (abs_grad_k > lmda * alpha * pk) return false;
    }

    return true;
}

template <class StateType,
          class GlmType,
          class UpdateDevNullType,
          class UpdateCoefficientsType,
          class CUIType>
inline void solve(
    StateType&& state,
    GlmType&& glm,
    bool display,
    UpdateDevNullType update_loss_null_f,
    UpdateCoefficientsType update_coefficients_f,
    CUIType check_user_interrupt
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_safe_bool_t = typename state_t::vec_safe_bool_t;
    using sw_t = util::Stopwatch;

    const auto alpha = state.alpha;
    const auto& penalty = state.penalty;
    const auto& screen_set = state.screen_set;
    const auto early_exit = state.early_exit;
    const auto max_screen_size = state.max_screen_size;
    const auto setup_loss_null = state.setup_loss_null;
    const auto setup_lmda_max = state.setup_lmda_max;
    const auto setup_lmda_path = state.setup_lmda_path;
    const auto lmda_path_size = state.lmda_path_size;
    const auto min_ratio = state.min_ratio;
    const auto adev_tol = state.adev_tol;
    const auto ddev_tol = state.ddev_tol;
    const auto& screen_is_active = state.screen_is_active;
    const auto& abs_grad = state.abs_grad;
    const auto& resid = state.resid;
    const auto& devs = state.devs;
    auto& X = *state.X;
    auto& lmda_max = state.lmda_max;
    auto& lmda_path = state.lmda_path;
    auto& grad = state.grad;
    auto& lmda = state.lmda;
    auto& benchmark_screen = state.benchmark_screen;
    auto& benchmark_fit_screen = state.benchmark_fit_screen;
    auto& benchmark_fit_active = state.benchmark_fit_active;
    auto& benchmark_kkt = state.benchmark_kkt;
    auto& benchmark_invariance = state.benchmark_invariance;
    auto& n_valid_solutions = state.n_valid_solutions;
    auto& active_sizes = state.active_sizes;
    auto& screen_sizes = state.screen_sizes;

    if (screen_set.size() > max_screen_size) throw util::max_basil_screen_set();

    const auto n = X.rows();
    const auto p = X.cols();
    GlmNaiveBufferPack<value_t> buffer_pack(n, p);

    // ==================================================================================== 
    // Initial fit with beta = 0 to get loss_null.
    // ==================================================================================== 
    if (setup_loss_null) {
        update_loss_null_f(state, glm, buffer_pack);
    }

    // ==================================================================================== 
    // Initial fit for lambda ~ infinity to setup lmda_max
    // ==================================================================================== 
    // Only unpenalized (l1) groups are active in this case.
    // State must include all unpenalized groups.
    // We solve for large lambda, then back-track the KKT condition to find the lambda
    // that leads to that solution where all penalized variables have 0 coefficient.
    if (setup_lmda_max) {
        const auto large_lmda = std::numeric_limits<value_t>::max(); 

        fit(
            state,
            glm,
            buffer_pack,
            large_lmda,
            update_coefficients_f,
            check_user_interrupt
        );

        /* Invariance */
        lmda = large_lmda;
        X.mul(resid, grad);
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

    // ==================================================================================== 
    // Generate lambda path if needed
    // ==================================================================================== 
    if (setup_lmda_path) {
        if (lmda_path_size <= 0) throw std::runtime_error("lmda_path_size must be > 0.");

        lmda_path.resize(lmda_path_size);

        generate_lmda_path(lmda_path, min_ratio, lmda_max);
    }

    // ==================================================================================== 
    // Initial fit for lambda > lambda_max
    // ==================================================================================== 
    // Only unpenalized (l1) groups are active in this case by definition of lmda_max.
    // Since state is in its invariance (solution at state.lmda) and unpenalized groups
    // are ALWAYS active, state includes all unpenalized groups.
    // If no lambda in lmda_path is > lmda_max and setup at lmda_max is not required, 
    // state is left unchanged.
    // Otherwise, it is in its invariance at lmda = lmda_max.
    // All solutions to lambda > lambda_max are saved.

    // slice lambda_path up to lmda_max
    const auto large_lmda_path_size = std::find_if(
        lmda_path.data(), 
        lmda_path.data() + lmda_path.size(),
        [&](auto x) { return x <= lmda_max; }
    ) - lmda_path.data();

    if (large_lmda_path_size || setup_lmda_max) {
        // create a lambda path containing only lmdas > lambda_max
        // and additionally lambda_max at the end.
        // If large_lmda_path_size > 0, mind as well fit for lambda_max as well to go down the path.
        vec_value_t large_lmda_path(large_lmda_path_size + 1);
        large_lmda_path.head(large_lmda_path_size) = lmda_path.head(large_lmda_path_size);
        large_lmda_path[large_lmda_path_size] = lmda_max;

        for (int i = 0; i < large_lmda_path.size(); ++i) {
            auto tup = fit(
                state, 
                glm,
                buffer_pack,
                large_lmda_path[i], 
                update_coefficients_f, 
                check_user_interrupt
            );
            auto&& state_gaussian_pin_naive = std::get<0>(tup);

            /* Invariance */
            // save only the solutions that the user asked for (up to and not including lmda_max)
            if (i < large_lmda_path.size()-1) {
                update_solutions(
                    state, 
                    glm,
                    state_gaussian_pin_naive,
                    large_lmda_path[i]
                );
            // otherwise, put the state at the last fitted lambda (lmda_max)
            } else {
                lmda = large_lmda_path[i];
                X.mul(resid, grad);
                state::gaussian::update_abs_grad(state, lmda);
            }
        }
    }

    size_t lmda_path_idx = devs.size(); // next index into lmda_path to fit

    // ==================================================================================== 
    // BASIL iterations for lambda <= lambda_max
    // ==================================================================================== 
    // In this case, screen_set may not contain the true active set.
    // We must go through BASIL iterations to solve each lambda.
    sw_t sw;
    int current_active_size = Eigen::Map<const vec_safe_bool_t>(
        screen_is_active.data(),
        screen_is_active.size()
    ).sum();
    bool kkt_passed = true;
    int n_new_active = 0;

    auto pb = util::tq::trange(lmda_path.size() - lmda_path_idx);
    pb.set_display(display);
    const auto pb_add_suffix = [&]() {
        // print extra information with the progress bar
        if (display) {
            // current training R^2
            pb << " [dev:" 
                << std::fixed << std::setprecision(1) 
                << ((devs.size() == 0) ? 0.0 : devs.back()) * 100
                << "%]"
                ; 
        }
    };

    for (int _ : pb)
    {
        static_cast<void>(_);

        // check early exit
        if (early_exit && (devs.size() >= 2)) {
            const auto dev_u = devs[devs.size()-1];
            const auto dev_m = devs[devs.size()-2];
            if ((dev_u >= adev_tol) || (dev_u-dev_m <= ddev_tol)) 
            {
                pb_add_suffix();
                break;
            }
        }

        // batch the next set of lambdas
        const auto lmda_curr = lmda_path[lmda_path_idx];

        // keep doing screen-fit-kkt until KKT passes
        while (1) {
            try {
                // ==================================================================================== 
                // Screening step
                // ==================================================================================== 
                sw.start();
                gaussian::naive::screen(
                    state,
                    lmda_curr,
                    kkt_passed,
                    n_new_active
                );
                state::gaussian::update_screen_derived_base(state);
                benchmark_screen.push_back(sw.elapsed());

                // ==================================================================================== 
                // Fit step
                // ==================================================================================== 
                // Save all current valid quantities that will be modified in-place by fit.
                // This is needed in case we exit with exception and need to restore invariance.
                auto tup = fit(
                    state,
                    glm,
                    buffer_pack,
                    lmda_curr,
                    update_coefficients_f,
                    check_user_interrupt
                );
                auto&& state_gaussian_pin_naive = std::get<0>(tup);
                benchmark_fit_screen.push_back(std::get<1>(tup));
                benchmark_fit_active.push_back(std::get<2>(tup));

                // ==================================================================================== 
                // KKT step
                // ==================================================================================== 
                sw.start();
                kkt_passed = kkt(
                    state,
                    lmda_curr
                );
                benchmark_kkt.push_back(sw.elapsed());
                n_valid_solutions.push_back(kkt_passed);

                // ==================================================================================== 
                // Invariance step
                // ==================================================================================== 
                sw.start();
                lmda_path_idx += kkt_passed;
                lmda = lmda_curr;
                if (kkt_passed) {
                    update_solutions(
                        state, 
                        glm,
                        state_gaussian_pin_naive,
                        lmda_curr
                    );
                }
                benchmark_invariance.push_back(sw.elapsed());

                // ==================================================================================== 
                // Diagnostic step
                // ==================================================================================== 
                if (kkt_passed) {
                    active_sizes.push_back(state_gaussian_pin_naive.active_set.size());
                    screen_sizes.push_back(state.screen_set.size());
                }
                // compute the number of new active groups 
                n_new_active = (
                    kkt_passed ?
                    active_sizes.back() - current_active_size : n_new_active
                );
                current_active_size = (
                    kkt_passed ?
                    active_sizes.back() : current_active_size
                );
            } catch (...) {
                pb_add_suffix();
                throw;
            }

            if (kkt_passed) break;
        } // end while(1)

        pb_add_suffix();
    }
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