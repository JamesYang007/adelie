#pragma once
#include <adelie_core/state/state_glm_naive.hpp>
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace state {

template <class MatrixType, 
          class ValueType=typename std::decay_t<MatrixType>::value_t,
          class IndexType=Eigen::Index,
          class BoolType=bool
        >
struct StateMultiGlmNaive: StateGlmNaive<
        MatrixType,
        ValueType,
        IndexType,
        BoolType
    >
{
    using base_t = StateGlmNaive<
        MatrixType,
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
    using typename base_t::matrix_t;
    using rowarr_value_t = util::rowarr_type<value_t>;

    /* static states */
    const util::multi_group_type group_type;
    const size_t n_classes;
    const bool multi_intercept;

    /* dynamic states */
    rowarr_value_t intercepts;

    explicit StateMultiGlmNaive(
        const std::string& group_type,
        size_t n_classes,
        bool multi_intercept,
        matrix_t& X,
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& resid,
        const Eigen::Ref<const vec_index_t>& groups, 
        const Eigen::Ref<const vec_index_t>& group_sizes,
        value_t alpha, 
        const Eigen::Ref<const vec_value_t>& penalty,
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
            X, eta, resid, groups, group_sizes, alpha, penalty, offsets, lmda_path, 
            loss_null, loss_full, lmda_max, min_ratio, lmda_path_size, max_screen_size, max_active_size,
            pivot_subset_ratio, pivot_subset_min, pivot_slack_ratio, screen_rule, 
            irls_max_iters, irls_tol, max_iters, tol, adev_tol, ddev_tol,
            newton_tol, newton_max_iters, early_exit, setup_loss_null, setup_lmda_max, setup_lmda_path, intercept, n_threads,
            screen_set, screen_beta, screen_is_active, beta0, lmda, grad
        ),
        group_type(util::convert_multi_group(group_type)),
        n_classes(n_classes),
        multi_intercept(multi_intercept)
    {}
};

} // namespace state
} // namespace adelie_core