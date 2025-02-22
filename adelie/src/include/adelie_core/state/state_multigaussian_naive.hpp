#pragma once
#include <adelie_core/state/state_gaussian_naive.hpp>
#include <adelie_core/util/functional.hpp>
#include <adelie_core/util/tqdm.hpp>

#ifndef ADELIE_CORE_STATE_MULTI_GAUSSIAN_NAIVE_TP
#define ADELIE_CORE_STATE_MULTI_GAUSSIAN_NAIVE_TP \
    template <\
        class ConstraintType,\
        class MatrixType,\
        class ValueType,\
        class IndexType,\
        class BoolType,\
        class SafeBoolType\
    >
#endif
#ifndef ADELIE_CORE_STATE_MULTI_GAUSSIAN_NAIVE
#define ADELIE_CORE_STATE_MULTI_GAUSSIAN_NAIVE \
    StateMultiGaussianNaive<\
        ConstraintType,\
        MatrixType,\
        ValueType,\
        IndexType,\
        BoolType,\
        SafeBoolType\
    >
#endif

namespace adelie_core {
namespace state {

template <
    class ConstraintType,
    class MatrixType, 
    class ValueType=typename std::decay_t<MatrixType>::value_t,
    class IndexType=Eigen::Index,
    class BoolType=bool,
    class SafeBoolType=int8_t
>
class StateMultiGaussianNaive: public StateGaussianNaive<
    ConstraintType,
    MatrixType,
    ValueType,
    IndexType,
    BoolType,
    SafeBoolType
>
{
public:
    using base_t = StateGaussianNaive<
        ConstraintType,
        MatrixType,
        ValueType,
        IndexType,
        BoolType,
        SafeBoolType
    >;
    using typename base_t::value_t;
    using typename base_t::index_t;
    using typename base_t::uset_index_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::vec_bool_t;
    using typename base_t::map_cvec_value_t;
    using typename base_t::dyn_vec_constraint_t;
    using typename base_t::dyn_vec_value_t;
    using typename base_t::dyn_vec_index_t;
    using typename base_t::dyn_vec_bool_t;
    using matrix_t = MatrixType;

    /* static states */
    const size_t n_classes;
    const bool multi_intercept;

    /* dynamic states */
    std::vector<vec_value_t> intercepts;

    explicit StateMultiGaussianNaive(
        size_t n_classes,
        bool multi_intercept,
        matrix_t& X,
        const Eigen::Ref<const vec_value_t>& X_means,
        value_t y_mean,
        value_t y_var,
        const Eigen::Ref<const vec_value_t>& resid,
        value_t resid_sum,
        const dyn_vec_constraint_t& constraints,
        const Eigen::Ref<const vec_index_t>& groups, 
        const Eigen::Ref<const vec_index_t>& group_sizes,
        const Eigen::Ref<const vec_index_t>& dual_groups, 
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
        size_t active_set_size,
        const Eigen::Ref<const vec_index_t>& active_set,
        value_t rsq,
        value_t lmda,
        const Eigen::Ref<const vec_value_t>& grad
    ):
        base_t(
            X, X_means, y_mean, y_var, resid, resid_sum,
            constraints, groups, group_sizes, dual_groups, alpha, penalty, weights, lmda_path, lmda_max, min_ratio, lmda_path_size,
            max_screen_size, max_active_size,
            pivot_subset_ratio, pivot_subset_min, pivot_slack_ratio, screen_rule, 
            max_iters, tol, adev_tol, ddev_tol, 
            newton_tol, newton_max_iters, early_exit, setup_lmda_max, setup_lmda_path, intercept, n_threads,
            screen_set, screen_beta, screen_is_active, active_set_size, active_set, rsq, lmda, grad
        ),
        n_classes(n_classes),
        multi_intercept(multi_intercept)
    {}

    void solve(
        util::tq::progress_bar_t& pb,
        std::function<bool()> exit_cond,
        std::function<void()> check_user_interrupt =util::no_op()
    );
};

} // namespace state
} // namespace adelie_core