#pragma once
#include <adelie_core/state/state_base.hpp>
#include <adelie_core/util/functional.hpp>
#include <adelie_core/util/tqdm.hpp>

#ifndef ADELIE_CORE_STATE_GAUSSIAN_NAIVE_TP
#define ADELIE_CORE_STATE_GAUSSIAN_NAIVE_TP \
    template <\
        class ConstraintType,\
        class MatrixType,\
        class ValueType,\
        class IndexType,\
        class BoolType,\
        class SafeBoolType\
    >
#endif
#ifndef ADELIE_CORE_STATE_GAUSSIAN_NAIVE
#define ADELIE_CORE_STATE_GAUSSIAN_NAIVE \
    StateGaussianNaive<\
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
class StateGaussianNaive: public StateBase<
    ConstraintType,
    ValueType,
    IndexType,
    BoolType,
    SafeBoolType
>
{
public:
    using base_t = StateBase<
        ConstraintType,
        ValueType,
        IndexType,
        BoolType,
        SafeBoolType
    >;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::vec_bool_t;
    using typename base_t::map_cvec_value_t;
    using typename base_t::dyn_vec_constraint_t;
    using typename base_t::dyn_vec_value_t;
    using matrix_t = MatrixType;
    using dyn_vec_mat_value_t = std::vector<util::rowmat_type<value_t>>;

    /* static states */
    const map_cvec_value_t weights;
    const vec_value_t weights_sqrt;
    const map_cvec_value_t X_means;
    const value_t y_mean;
    const value_t y_var;
    const value_t loss_null;
    const value_t loss_full;

    /* dynamic states */
    matrix_t* X;
    vec_value_t resid;
    value_t resid_sum;
    value_t rsq;
    dyn_vec_value_t screen_X_means;
    dyn_vec_mat_value_t screen_transforms;
    dyn_vec_value_t screen_vars;

private:
    void initialize();

public:
    explicit StateGaussianNaive(
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
            constraints, groups, group_sizes, dual_groups, alpha, penalty, lmda_path, lmda_max, min_ratio, lmda_path_size,
            max_screen_size, max_active_size,
            pivot_subset_ratio, pivot_subset_min, pivot_slack_ratio, screen_rule, 
            max_iters, tol, adev_tol, ddev_tol, newton_tol, newton_max_iters, early_exit, 
            setup_lmda_max, setup_lmda_path, intercept, n_threads,
            screen_set, screen_beta, screen_is_active, active_set_size, active_set, lmda, grad
        ),
        weights(weights.data(), weights.size()),
        weights_sqrt(weights.sqrt()),
        X_means(X_means.data(), X_means.size()),
        y_mean(y_mean),
        y_var(y_var),
        loss_null(-0.5 * y_mean * y_mean),
        loss_full(-0.5 * y_var + loss_null),
        X(&X),
        resid(resid),
        resid_sum(resid_sum),
        rsq(rsq)
    { 
        initialize();
    }

    void solve(
        util::tq::progress_bar_t& pb,
        std::function<bool()> exit_cond,
        std::function<void()> check_user_interrupt =util::no_op()
    );
};

} // namespace state
} // namespace adelie_core