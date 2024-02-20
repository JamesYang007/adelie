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
    using map_carr_value_t = Eigen::Map<const rowarr_value_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;

    glm_t& glm;
    const map_carr_value_t y;
    const map_cvec_value_t weights;

    explicit GlmWrap(
        glm_t& glm
    ):
        glm(glm),
        y(glm.y.data(), glm.y.rows(), glm.y.cols()),
        weights(glm.weights.data(), glm.weights.size())
    {}

    void gradient(
        const Eigen::Ref<const vec_value_t>& eta,
        Eigen::Ref<vec_value_t> resid
    )
    {
        const auto& y = glm.y;
        const auto n = y.rows();
        const auto K = y.cols();
        Eigen::Map<const rowarr_value_t> E(eta.data(), n, K);
        Eigen::Map<rowarr_value_t> R(resid.data(), n, K);
        glm.gradient(E, R);
    }

    void hessian(
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& resid,
        Eigen::Ref<vec_value_t> hess
    )
    {
        const auto& y = glm.y;
        const auto n = y.rows();
        const auto K = y.cols();
        Eigen::Map<const rowarr_value_t> E(eta.data(), n, K);
        Eigen::Map<const rowarr_value_t> R(resid.data(), n, K);
        Eigen::Map<rowarr_value_t> H(hess.data(), n, K);
        glm.hessian(E, R, H);
    }

    value_t loss(
        const Eigen::Ref<const vec_value_t>& eta
    )
    {
        const auto& y = glm.y;
        const auto n = y.rows();
        const auto K = y.cols();
        Eigen::Map<const rowarr_value_t> E(eta.data(), n, K);
        return glm.loss(E);
    }

    value_t loss_full()
    {
        return glm.loss_full();
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

    const auto n_classes = glm.y.cols();
    const auto& offsets = state.offsets;
    const auto multi_intercept = state.multi_intercept;
    auto& loss_null = state.loss_null;

    if (!multi_intercept) {
        loss_null = glm.loss(offsets);
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
        const auto n = irls_weights.size() / n_classes;
        Eigen::Map<const rowarr_value_t> irls_weights_arr(
            irls_weights.data(), n, n_classes
        );
        Eigen::Map<const rowarr_value_t> irls_y_arr(
            irls_y.data(), n, n_classes
        );
        beta0 = (irls_weights_arr * irls_y_arr).colwise().sum() / irls_weights_arr.colwise().sum();

        // update eta
        Eigen::Map<rowarr_value_t> eta_arr(
            eta.data(), n, n_classes
        );
        Eigen::Map<const rowarr_value_t> offsets_arr(
            offsets.data(), n, n_classes
        );
        eta_arr = offsets_arr.rowwise() + beta0;

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

    const auto n_classes = glm.y.cols();
    const auto multi_intercept = state.multi_intercept;
    auto& betas = state.betas;
    auto& intercepts = state.intercepts;

    GlmWrap<glm_t> glm_wrap(glm);

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

    try {
        glm::naive::solve(
            static_cast<state_glm_naive_t&>(state),
            glm_wrap,
            display,
            [&](auto&, auto& glm, auto& buffer_pack) {
                // ignore casted down state and use derived state
                multiglm::naive::update_loss_null(state, glm, buffer_pack);
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