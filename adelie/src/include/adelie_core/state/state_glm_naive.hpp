#pragma once
#include <adelie_core/state/state_gaussian_naive.hpp>
#include <adelie_core/state/state_glm_base.hpp>

namespace adelie_core {
namespace state {
namespace glm {
namespace naive {

/**
 * Unlike the similar function in gaussian::naive,
 * this does not call the base version to update the base classes's screen derived quantities.
 * This is because in GLM fitting, the three screen_* inputs are modified at every IRLS loop,
 * while the base quantities remain the same. 
 * It is only when IRLS finishes and we must screen for variables where we have to update the base quantities.
 * In gaussian naive setting, the IRLS has loop size of 1 essentially, so the two versions are synonymous.
 */
template <class StateType, class XMType, class WType,
          class SXMType, class STType, class SVType>
void update_screen_derived(
    StateType& state,
    const XMType& X_means,
    const WType& weights_sqrt,
    size_t begin,
    size_t size,
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
        size,
        state.intercept,
        screen_X_means,
        screen_transforms,
        screen_vars
    );
}

} // namespace naive
} // namespace glm

template <class MatrixType, 
          class ValueType=typename std::decay_t<MatrixType>::value_t,
          class IndexType=Eigen::Index,
          class BoolType=bool
        >
struct StateGlmNaive: StateGlmBase<
        ValueType,
        IndexType,
        BoolType
    >
{
    using base_t = StateGlmBase<
        ValueType,
        IndexType,
        BoolType
    >;
    using typename base_t::value_t;
    using typename base_t::index_t;
    using typename base_t::uset_index_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::vec_bool_t;
    using typename base_t::map_cvec_value_t;
    using typename base_t::dyn_vec_value_t;
    using typename base_t::dyn_vec_index_t;
    using typename base_t::dyn_vec_bool_t;
    using matrix_t = MatrixType;

    /* static states */
    const map_cvec_value_t y;

    /* configurations */

    /* dynamic states */
    matrix_t* X;
    vec_value_t eta;
    vec_value_t mu;

    explicit StateGlmNaive(
        matrix_t& X,
        const Eigen::Ref<const vec_value_t>& y,
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& mu,
        const Eigen::Ref<const vec_index_t>& groups, 
        const Eigen::Ref<const vec_index_t>& group_sizes,
        value_t alpha, 
        const Eigen::Ref<const vec_value_t>& penalty,
        const Eigen::Ref<const vec_value_t>& weights,
        const Eigen::Ref<const vec_value_t>& offsets,
        const Eigen::Ref<const vec_value_t>& lmda_path,
        value_t loss_null,
        value_t loss_full,
        value_t lmda_max,
        value_t min_ratio,
        size_t lmda_path_size,
        size_t max_screen_size,
        size_t max_active_size,
        value_t pivot_subset_ratio,
        size_t pivot_subset_min,
        value_t pivot_slack_ratio,
        const std::string& screen_rule,
        size_t irls_max_iters,
        value_t irls_tol,
        size_t max_iters,
        value_t tol,
        value_t adev_tol,
        value_t ddev_tol,
        value_t newton_tol,
        size_t newton_max_iters,
        bool early_exit,
        bool setup_loss_null, 
        bool setup_lmda_max,
        bool setup_lmda_path,
        bool intercept,
        size_t n_threads,
        const Eigen::Ref<const vec_index_t>& screen_set,
        const Eigen::Ref<const vec_value_t>& screen_beta,
        const Eigen::Ref<const vec_bool_t>& screen_is_active,
        value_t beta0,
        value_t lmda,
        const Eigen::Ref<const vec_value_t>& grad
    ):
        base_t(
            groups, group_sizes, alpha, penalty, weights, offsets, lmda_path, 
            loss_null, loss_full, lmda_max, min_ratio, lmda_path_size, max_screen_size, max_active_size,
            pivot_subset_ratio, pivot_subset_min, pivot_slack_ratio, screen_rule, 
            irls_max_iters, irls_tol, max_iters, tol, adev_tol, ddev_tol,
            newton_tol, newton_max_iters, early_exit, setup_loss_null, setup_lmda_max, setup_lmda_path, intercept, n_threads,
            screen_set, screen_beta, screen_is_active, beta0, lmda, grad
        ),
        y(y.data(), y.size()),
        X(&X),
        eta(eta),
        mu(mu)
    {}
};

} // namespace state
} // namespace adelie_core