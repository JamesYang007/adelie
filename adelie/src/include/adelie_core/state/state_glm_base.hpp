#pragma once
#include <adelie_core/state/state_gaussian_base.hpp>
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace state {

template <class ValueType,
          class IndexType=Eigen::Index,
          class BoolType=bool
        >
struct StateGlmBase : StateGaussianBase<
        ValueType,
        IndexType,
        BoolType
    >
{
    using base_t = StateGaussianBase<
        ValueType,
        IndexType,
        BoolType
    >;
    using typename base_t::value_t;
    using typename base_t::index_t;
    using typename base_t::bool_t;
    using typename base_t::safe_bool_t;
    using typename base_t::uset_index_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::vec_bool_t;
    using typename base_t::vec_safe_bool_t;
    using typename base_t::sp_vec_value_t;
    using typename base_t::map_cvec_value_t;
    using typename base_t::map_cvec_index_t;
    using typename base_t::dyn_vec_value_t;
    using typename base_t::dyn_vec_index_t;
    using typename base_t::dyn_vec_bool_t;
    using typename base_t::dyn_vec_sp_vec_t;

    // convergence configs
    const size_t max_irls_iters;

    explicit StateGlmBase(
        /* TODO: input GLM object */
        const Eigen::Ref<const vec_index_t>& groups, 
        const Eigen::Ref<const vec_index_t>& group_sizes,
        value_t alpha, 
        const Eigen::Ref<const vec_value_t>& penalty,
        const Eigen::Ref<const vec_value_t>& weights,
        const Eigen::Ref<const vec_value_t>& lmda_path,
        value_t lmda_max,
        value_t min_ratio,
        size_t lmda_path_size,
        size_t max_screen_size,
        value_t pivot_subset_ratio,
        size_t pivot_subset_min,
        value_t pivot_slack_ratio,
        const std::string& screen_rule,
        size_t max_irls_iters,
        size_t max_iters,
        value_t tol,
        value_t rsq_tol,
        value_t rsq_slope_tol,
        value_t rsq_curv_tol,
        value_t newton_tol,
        size_t newton_max_iters,
        bool early_exit,
        bool setup_lmda_max,
        bool setup_lmda_path,
        bool intercept,
        size_t n_threads,
        const Eigen::Ref<const vec_index_t>& screen_set,
        const Eigen::Ref<const vec_value_t>& screen_beta,
        const Eigen::Ref<const vec_bool_t>& screen_is_active,
        value_t rsq,
        value_t lmda,
        const Eigen::Ref<const vec_value_t>& grad
    ): 
        base_t(
            groups, group_sizes, alpha, penalty, weights,
            lmda_path, lmda_max, min_ratio, lmda_path_size, max_screen_size,
            pivot_subset_ratio, pivot_subset_min, pivot_slack_ratio,
            screen_rule, max_iters, tol, rsq_tol, rsq_slope_tol, rsq_curv_tol,
            newton_tol, newton_max_iters, early_exit,
            setup_lmda_max, setup_lmda_path, intercept, n_threads,
            screen_set, screen_beta, screen_is_active,
            rsq, lmda, grad 
        ),
        max_irls_iters(max_irls_iters)
    {}
};

} // namespace state
} // namespace adelie_core