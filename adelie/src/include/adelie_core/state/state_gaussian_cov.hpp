#pragma once
#include <adelie_core/state/state_base.hpp>
#include <adelie_core/util/functional.hpp>
#include <adelie_core/util/tqdm.hpp>

#ifndef ADELIE_CORE_STATE_GAUSSIAN_COV_TP
#define ADELIE_CORE_STATE_GAUSSIAN_COV_TP \
    template <\
        class ConstraintType,\
        class MatrixType,\
        class ValueType,\
        class IndexType,\
        class BoolType,\
        class SafeBoolType\
    >
#endif
#ifndef ADELIE_CORE_STATE_GAUSSIAN_COV
#define ADELIE_CORE_STATE_GAUSSIAN_COV \
    StateGaussianCov<\
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
class StateGaussianCov: public StateBase<
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
    using typename base_t::dyn_vec_index_t;
    using matrix_t = MatrixType;
    using dyn_vec_mat_value_t = std::vector<util::rowmat_type<value_t>>;

    /* configurations */
    const map_cvec_value_t v;
    const value_t rdev_tol;    

    /* dynamic states */
    matrix_t* A;
    value_t rsq;
    dyn_vec_mat_value_t screen_transforms;
    dyn_vec_value_t screen_vars;
    dyn_vec_value_t screen_grad;
    dyn_vec_index_t screen_subset;
    dyn_vec_index_t screen_subset_order;
    dyn_vec_index_t screen_subset_ordered;

private:
    void initialize();

public:
    explicit StateGaussianCov(
        matrix_t& A,
        const Eigen::Ref<const vec_value_t>& v,
        const dyn_vec_constraint_t& constraints,
        const Eigen::Ref<const vec_index_t>& groups, 
        const Eigen::Ref<const vec_index_t>& group_sizes,
        const Eigen::Ref<const vec_index_t>& dual_groups, 
        value_t alpha, 
        const Eigen::Ref<const vec_value_t>& penalty,
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
        value_t rdev_tol,
        value_t newton_tol,
        size_t newton_max_iters,
        bool early_exit,
        bool setup_lmda_max,
        bool setup_lmda_path,
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
            max_iters, tol, 0, 0, newton_tol, newton_max_iters, early_exit, 
            setup_lmda_max, setup_lmda_path, false, n_threads,
            screen_set, screen_beta, screen_is_active, active_set_size, active_set, lmda, grad
        ),
        v(v.data(), v.size()),
        rdev_tol(rdev_tol),
        A(&A),
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