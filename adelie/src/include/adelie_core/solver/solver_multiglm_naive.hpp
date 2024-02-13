#pragma once
#include <adelie_core/matrix/utils.hpp>
#include <adelie_core/solver/solver_glm_naive.hpp>

namespace adelie_core {
namespace solver {
namespace multiglm {
namespace naive {

/**
 * Wrapper of Multi-GLM objects.
 * This wrapper is treated like a single-response GLM object,
 * but the inputs are essentially reshaped to be able to pass to Multi-GLM API.
 * We take advantage of the fact that the solver passes the original weights.
 * Since the Multi-GLM expects (n,) weights not scaled by 1/K,
 * (but the solver will pass (nK,) expanded weights scaled by 1/K),
 * we ignore this input and replace it.
 */
template <class GlmType>
struct GlmWrap
{
    using glm_t = GlmType;
    using value_t = typename glm_t::value_t;
    using vec_value_t = typename glm_t::vec_value_t;
    using rowarr_value_t = typename glm_t::rowarr_value_t;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;

    const bool is_symmetric;
    const size_t n_classes;
    const map_cvec_value_t weights_orig;
    glm_t& glm;

    explicit GlmWrap(
        glm_t& glm,
        size_t n_classes,
        const Eigen::Ref<const vec_value_t>& weights_orig
    ):
        is_symmetric(glm.is_symmetric),
        n_classes(n_classes),
        weights_orig(weights_orig.data(), weights_orig.size()),
        glm(glm)
    {}

    void gradient(
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>&,
        Eigen::Ref<vec_value_t> mu
    )
    {
        const auto n = weights_orig.size();
        const auto K = n_classes;
        Eigen::Map<const rowarr_value_t> E(eta.data(), n, K);
        Eigen::Map<rowarr_value_t> M(mu.data(), n, K);
        glm.gradient(E, weights_orig, M);
    }

    void hessian(
        const Eigen::Ref<const vec_value_t>& mu,
        const Eigen::Ref<const vec_value_t>&,
        Eigen::Ref<vec_value_t> var
    )
    {
        const auto n = weights_orig.size();
        const auto K = n_classes;
        Eigen::Map<const rowarr_value_t> M(mu.data(), n, K);
        Eigen::Map<rowarr_value_t> V(var.data(), n, K);
        glm.hessian(M, weights_orig, V);
    }

    value_t loss(
        const Eigen::Ref<const vec_value_t>& y,
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& 
    )
    {
        const auto n = weights_orig.size();
        const auto K = n_classes;
        Eigen::Map<const rowarr_value_t> Y(y.data(), n, K);
        Eigen::Map<const rowarr_value_t> E(eta.data(), n, K);
        return glm.loss(Y, E, weights_orig);
    }

    value_t loss_full(
        const Eigen::Ref<const vec_value_t>& y,
        const Eigen::Ref<const vec_value_t>& 
    )
    {
        const auto n = weights_orig.size();
        const auto K = n_classes;
        Eigen::Map<const rowarr_value_t> Y(y.data(), n, K);
        return glm.loss_full(Y, weights_orig);
    }
};

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
    using vec_value_t = typename state_t::vec_value_t;
    using rowarr_value_t = typename state_t::rowarr_value_t;

    const auto& y0 = state.y;
    const auto& weights0 = state.weights;
    const auto& offsets = state.offsets;
    const auto n_classes = state.n_classes;
    const auto multi_intercept = state.multi_intercept;
    auto& loss_null = state.loss_null;

    if (!multi_intercept) {
        loss_null = glm.loss(y0, offsets, weights0);
        return;
    }

    const auto irls_max_iters = state.irls_max_iters;
    const auto irls_tol = state.irls_tol;

    // make copies since we do not want to mess with the warm-start.
    // this function is only needed to fit intercept-only model and get loss_null.
    vec_value_t beta0 = Eigen::Map<const vec_value_t>(
        state.screen_beta.data(),
        n_classes
    );
    vec_value_t eta = state.eta;
    vec_value_t mu = state.mu;

    auto& weights = buffer_pack.weights;
    auto& y = buffer_pack.y;
    auto& mu_prev = buffer_pack.mu_prev;
    auto& var = buffer_pack.var;

    size_t irls_it = 0;

    while (1) {
        if (irls_it >= irls_max_iters) {
            throw std::runtime_error("Maximum IRLS iterations reached.");
        }

        /* compute rest of quadratic approximation quantities */
        glm.hessian(mu, weights0, var);
        const auto var_sum = var.sum();
        weights = var / var_sum;
        y = weights0 * y0 - mu;
        y = y.NullaryExpr(y.size(), [&](auto i) {
            const auto ratio = y[i] / var[i]; 
            return std::isnan(ratio) ? y[i] : ratio;
        }) + eta - offsets;

        /* fit beta0 */
        const auto n = weights.size() / n_classes;
        Eigen::Map<const rowarr_value_t> weights_arr(
            weights.data(), n, n_classes
        );
        Eigen::Map<const rowarr_value_t> y_arr(
            y.data(), n, n_classes
        );
        beta0 = (weights_arr * y_arr).colwise().sum() / weights_arr.colwise().sum();

        // TODO: this doesn't seem necessary either..
        //if (glm.is_symmetric) beta0 -= beta0.mean();

        // update eta
        Eigen::Map<rowarr_value_t> eta_arr(
            eta.data(), n, n_classes
        );
        Eigen::Map<const rowarr_value_t> offsets_arr(
            offsets.data(), n, n_classes
        );
        eta_arr = offsets_arr.rowwise() + beta0;

        // update mu
        mu_prev.swap(mu);
        glm.gradient(eta, weights0, mu); 

        /* check convergence */
        if ((mu - mu_prev).square().sum() <= irls_tol) {
            loss_null = glm.loss(y0, eta, weights0);
            return;
        }

        ++irls_it;
    }
}

/**
 * TODO: glmnet performs a modification to the coefficients to accelerate convergence
 * for the case when the log-likelihood is symmetric in the groups (e.g. multinomial).
 * Doesn't seem to work well...
 */
template <class StateType,
          class GlmType,
          class BufferPackType,
          class OnesType>
inline void update_symmetric(
    StateType& state,
    GlmType& glm,
    BufferPackType& buffer_pack,
    const OnesType& ones
)
{
    using state_t = std::decay_t<StateType>;
    using vec_value_t = typename state_t::vec_value_t;

    if (!glm.is_symmetric) return;

    const auto group_type = state.group_type;
    const auto& groups = state.groups;
    const auto& group_sizes = state.group_sizes;
    const auto& screen_set = state.screen_set;
    const auto& screen_begins = state.screen_begins;
    const auto& screen_is_active = state.screen_is_active;
    const auto n_threads = state.n_threads;
    auto& X = *state.X;
    auto& screen_beta = state.screen_beta;
    auto& eta = state.eta;
    auto& buffer_n = buffer_pack.buffer_n;

    const auto n = X.rows();

    if (group_type == util::multi_group_type::_grouped) {
        for (int ss_idx = 0; ss_idx < screen_set.size(); ++ss_idx) {
            const auto ss = screen_set[ss_idx];
            const auto sb = screen_begins[ss_idx];
            const auto g = groups[ss];
            const auto gs = group_sizes[ss];

            if (!screen_is_active[ss_idx]) continue;

            Eigen::Map<vec_value_t> beta_g(screen_beta.data() + sb, gs);
            const auto beta_g_mean = beta_g.mean();
            beta_g -= beta_g_mean;

            X.btmul(g, gs, ones.head(gs), ones.head(n), buffer_n);
            matrix::dvsubi(eta, beta_g_mean * buffer_n, n_threads);
        }
    } else if (group_type == util::multi_group_type::_ungrouped) {
        // TODO
    } else {
        throw std::runtime_error("Group type must be _grouped or _ungrouped.");
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
    using state_t = std::decay_t<StateType>;
    using glm_t = std::decay_t<GlmType>;
    using vec_value_t = typename state_t::vec_value_t;
    using state_glm_naive_t = typename state_t::base_t;

    const auto n_classes = state.n_classes;
    const auto multi_intercept = state.multi_intercept;
    const auto& weights_orig = state.weights_orig;
    auto& betas = state.betas;
    auto& intercepts = state.intercepts;

    GlmWrap<glm_t> glm_wrap(glm, n_classes, weights_orig);

    const auto tidy = [&]() {
        intercepts.resize(betas.size(), n_classes);
        if (multi_intercept) {
            for (int i = 0; i < betas.size(); ++i) {
                intercepts.row(i) = Eigen::Map<const vec_value_t>(betas[i].valuePtr(), n_classes);
                betas[i] = betas[i].tail(betas[i].size() - n_classes);
            }
        } else {
            intercepts.setZero();
        }
    };

    // TODO: only needed for the update_symmetric.
    //vec_value_t ones = vec_value_t::Ones(
    //    std::max<size_t>(n_classes, state.X->rows())
    //);

    try {
        glm::naive::solve(
            static_cast<state_glm_naive_t&>(state),
            glm_wrap,
            display,
            [&](auto&, auto& glm, auto& buffer_pack) {
                // ignore casted down state and use derived state
                multiglm::naive::update_loss_null(state, glm, buffer_pack);
            },
            [&](auto&, auto& glm, auto& buffer_pack) {
                // TODO: keep? This update doesn't seem to make things converge.
                //multiglm::naive::update_symmetric(state, glm, buffer_pack, ones);
            },
            update_coefficients_f,
            check_user_interrupt
        );
        tidy();
    } catch(...) {
        tidy();
        throw;
    }
}

} // namespace naive 
} // namespace multiglm
} // namespace solver
} // namespace adelie_core