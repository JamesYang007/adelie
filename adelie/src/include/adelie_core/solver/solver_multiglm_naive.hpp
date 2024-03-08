#pragma once
#include <adelie_core/configs.hpp>
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
    const bool is_symmetric;

    explicit GlmWrap(
        glm_t& glm
    ):
        glm(glm),
        y(glm.y.data(), glm.y.rows(), glm.y.cols()),
        weights(glm.weights.data(), glm.weights.size()),
        is_symmetric(glm.is_symmetric)
    {}

    void gradient(
        const Eigen::Ref<const vec_value_t>& eta,
        Eigen::Ref<vec_value_t> resid
    )
    {
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
        const auto n = y.rows();
        const auto K = y.cols();
        Eigen::Map<const rowarr_value_t> E(eta.data(), n, K);
        Eigen::Map<const rowarr_value_t> R(resid.data(), n, K);
        Eigen::Map<rowarr_value_t> H(hess.data(), n, K);
        glm.hessian(E, R, H);
    }

    void inv_hessian_gradient(
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& resid,
        const Eigen::Ref<const vec_value_t>& hess,
        Eigen::Ref<vec_value_t> inv_hess_grad
    )
    {
        const auto n = y.rows();
        const auto K = y.cols();
        Eigen::Map<const rowarr_value_t> E(eta.data(), n, K);
        Eigen::Map<const rowarr_value_t> R(resid.data(), n, K);
        Eigen::Map<const rowarr_value_t> H(hess.data(), n, K);
        Eigen::Map<rowarr_value_t> IHG(inv_hess_grad.data(), n, K);
        glm.inv_hessian_gradient(E, R, H, IHG);
    }

    value_t loss(
        const Eigen::Ref<const vec_value_t>& eta
    )
    {
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
    using value_t = typename state_t::value_t;
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
    vec_value_t beta0(n_classes);
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
        const auto n = irls_weights.size() / n_classes;
        Eigen::Map<const rowarr_value_t> irls_weights_arr(
            irls_weights.data(), n, n_classes
        );
        Eigen::Map<const rowarr_value_t> irls_y_arr(
            irls_y.data(), n, n_classes
        );
        beta0 = (
            (irls_weights_arr * irls_y_arr).colwise().sum() / 
            irls_weights_arr.colwise().sum()
        );

        // update eta
        eta.swap(eta_prev);
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
        if (std::abs(((resid - resid_prev) * (eta - eta_prev)).sum()) <= irls_tol) {
            loss_null = glm.loss(eta);
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