#pragma once
#include <numeric>
#include <adelie_core/matrix/utils.hpp>
#include <adelie_core/state/state_gaussian_naive.hpp>

namespace adelie_core {
namespace state {

template <class MatrixType, 
          class ValueType=typename std::decay_t<MatrixType>::value_t,
          class IndexType=Eigen::Index,
          class BoolType=bool
        >
struct StateMultiGaussianNaive : StateGaussianNaive<
        MatrixType,
        ValueType,
        IndexType,
        BoolType
    >
{
    using base_t = StateGaussianNaive<
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
    using matrix_t = MatrixType;
    using rowarr_value_t = util::rowarr_type<value_t>;

    /* static states */
    const util::multi_group_type group_type;
    const size_t n_classes;
    const bool multi_intercept;

    /* dynamic states */
    rowarr_value_t intercepts;

    explicit StateMultiGaussianNaive(
        const std::string& group_type,
        size_t n_classes,
        bool multi_intercept,
        matrix_t& X,
        const Eigen::Ref<const vec_value_t>& X_means,
        value_t y_mean,
        value_t y_var,
        const Eigen::Ref<const vec_value_t>& resid,
        value_t resid_sum,
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
        size_t max_active_size,
        value_t pivot_subset_ratio,
        size_t pivot_subset_min,
        value_t pivot_slack_ratio,
        const std::string& screen_rule,
        size_t max_iters,
        value_t tol,
        value_t adev_tol,
        value_t ddev_tol,
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
            X, X_means, y_mean, y_var, resid, resid_sum,
            groups, group_sizes, alpha, penalty, weights, lmda_path, lmda_max, min_ratio, lmda_path_size,
            max_screen_size, max_active_size,
            pivot_subset_ratio, pivot_subset_min, pivot_slack_ratio, screen_rule, 
            max_iters, tol, adev_tol, ddev_tol, 
            newton_tol, newton_max_iters, early_exit, setup_lmda_max, setup_lmda_path, intercept, n_threads,
            screen_set, screen_beta, screen_is_active, rsq, lmda, grad
        ),
        group_type(util::convert_multi_group(group_type)),
        n_classes(n_classes),
        multi_intercept(multi_intercept)
    {}
};

} // namespace state
} // namespace adelie_core