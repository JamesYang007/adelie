#pragma once
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

    value_t deviance(
        const Eigen::Ref<const vec_value_t>& y,
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& 
    )
    {
        const auto n = weights_orig.size();
        const auto K = n_classes;
        Eigen::Map<const rowarr_value_t> Y(y.data(), n, K);
        Eigen::Map<const rowarr_value_t> E(eta.data(), n, K);
        return glm.deviance(Y, E, weights_orig);
    }

    value_t deviance_full(
        const Eigen::Ref<const vec_value_t>& y,
        const Eigen::Ref<const vec_value_t>& 
    )
    {
        const auto n = weights_orig.size();
        const auto K = n_classes;
        Eigen::Map<const rowarr_value_t> Y(y.data(), n, K);
        return glm.deviance_full(Y, weights_orig);
    }
};

template <class StateType,
          class GlmType,
          class BufferPackType>
ADELIE_CORE_STRONG_INLINE
void update_dev_null(
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
    auto& dev_null = state.dev_null;

    if (!multi_intercept) {
        dev_null = glm.deviance(y0, offsets, weights0);
        return;
    }

    const auto irls_max_iters = state.irls_max_iters;
    const auto irls_tol = state.irls_tol;

    // make copies since we do not want to mess with the warm-start.
    // this function is only needed to fit intercept-only model and get dev_null.
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

        if (glm.is_symmetric) {
            if (state.group_type == util::multi_group_type::_grouped) {
                //beta0 -= beta0.mean();
            } else if (state.group_type == util::multi_group_type::_ungrouped) {
                // TODO
            } else {
                throw std::runtime_error("Unexpected multi-response group type!");
            }
        }

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
            dev_null = glm.deviance(y0, eta, weights0);
            return;
        }

        ++irls_it;
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

    glm::naive::solve(
        static_cast<state_glm_naive_t&>(state),
        glm_wrap,
        display,
        [&](auto&, auto& glm, auto& buffer_pack) {
            // ignore casted down state and use derived state
            multiglm::naive::update_dev_null(state, glm, buffer_pack);
        },
        update_coefficients_f,
        check_user_interrupt
    );

    intercepts.resize(betas.size(), n_classes);
    if (multi_intercept) {
        for (int i = 0; i < betas.size(); ++i) {
            intercepts.row(i) = Eigen::Map<const vec_value_t>(betas[i].valuePtr(), n_classes);
            betas[i] = betas[i].tail(betas[i].size() - n_classes);
        }
    } else {
        intercepts.setZero();
    }
}

} // namespace naive 
} // namespace multiglm
} // namespace solver
} // namespace adelie_core