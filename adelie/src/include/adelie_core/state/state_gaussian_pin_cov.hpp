#pragma once
#include <adelie_core/state/state_gaussian_pin_base.hpp>
#include <adelie_core/util/functional.hpp>

#ifndef ADELIE_CORE_STATE_GAUSSIAN_PIN_COV_TP
#define ADELIE_CORE_STATE_GAUSSIAN_PIN_COV_TP \
    template <\
        class ConstraintType,\
        class MatrixType,\
        class ValueType,\
        class IndexType,\
        class BoolType\
    >
#endif
#ifndef ADELIE_CORE_STATE_GAUSSIAN_PIN_COV
#define ADELIE_CORE_STATE_GAUSSIAN_PIN_COV \
    StateGaussianPinCov<\
        ConstraintType,\
        MatrixType,\
        ValueType,\
        IndexType,\
        BoolType\
    >
#endif

namespace adelie_core {
namespace state {

template <
    class ConstraintType,
    class MatrixType, 
    class ValueType=typename std::decay_t<MatrixType>::value_t,
    class IndexType=Eigen::Index,
    class BoolType=bool
>
class StateGaussianPinCov: public StateGaussianPinBase<
    ConstraintType,
    ValueType,
    IndexType,
    BoolType
>
{
public:
    using base_t = StateGaussianPinBase<
        ConstraintType,
        ValueType,
        IndexType,
        BoolType
    >;
    using typename base_t::constraint_t;
    using typename base_t::value_t;
    using typename base_t::index_t;
    using typename base_t::bool_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::vec_bool_t;
    using typename base_t::sp_vec_value_t;
    using typename base_t::map_vec_value_t;
    using typename base_t::map_vec_bool_t;
    using typename base_t::map_cvec_value_t;
    using typename base_t::map_cvec_index_t;
    using typename base_t::dyn_vec_constraint_t;
    using typename base_t::dyn_vec_index_t;
    using typename base_t::dyn_vec_value_t;
    using typename base_t::dyn_vec_sp_vec_t;
    using typename base_t::dyn_vec_mat_value_t;
    using typename base_t::dyn_vec_vec_value_t;
    using matrix_t = MatrixType;

    /* static states */
    const map_cvec_index_t screen_subset_order;
    const map_cvec_index_t screen_subset_ordered;

    /* configurations */
    const value_t rdev_tol;

    /* dynamic states */
    matrix_t* A;    // covariance matrix-like
    map_vec_value_t screen_grad;
    vec_bool_t screen_is_active_subset;
    dyn_vec_index_t active_subset_order;
    dyn_vec_index_t active_subset_ordered;
    dyn_vec_index_t inactive_subset_order; 
    dyn_vec_index_t inactive_subset_ordered;

private:
    void initialize();

public:
    explicit StateGaussianPinCov(
        matrix_t& A,
        const dyn_vec_constraint_t& constraints,
        const Eigen::Ref<const vec_index_t>& groups, 
        const Eigen::Ref<const vec_index_t>& group_sizes,
        value_t alpha, 
        const Eigen::Ref<const vec_value_t>& penalty,
        const Eigen::Ref<const vec_index_t>& screen_set, 
        const Eigen::Ref<const vec_index_t>& screen_begins, 
        const Eigen::Ref<const vec_value_t>& screen_vars,
        const dyn_vec_mat_value_t& screen_transforms,
        const Eigen::Ref<const vec_index_t>& screen_subset_order,
        const Eigen::Ref<const vec_index_t>& screen_subset_ordered,
        const Eigen::Ref<const vec_value_t>& lmda_path, 
        size_t constraint_buffer_size,
        size_t max_active_size,
        size_t max_iters,
        value_t tol,
        value_t rdev_tol,
        value_t newton_tol,
        size_t newton_max_iters,
        size_t n_threads,
        value_t rsq,
        Eigen::Ref<vec_value_t> screen_beta, 
        Eigen::Ref<vec_value_t> screen_grad,
        Eigen::Ref<vec_bool_t> screen_is_active,
        size_t active_set_size,
        Eigen::Ref<vec_index_t> active_set
    ): 
        base_t(
            constraints, groups, group_sizes, alpha, penalty, 
            screen_set, screen_begins, screen_vars, screen_transforms, lmda_path, 
            constraint_buffer_size, false, max_active_size, max_iters, tol, 0, 0, newton_tol, newton_max_iters, n_threads,
            rsq, screen_beta, screen_is_active, active_set_size, active_set
        ),
        screen_subset_order(screen_subset_order.data(), screen_subset_order.size()),
        screen_subset_ordered(screen_subset_ordered.data(), screen_subset_ordered.size()),
        rdev_tol(rdev_tol),
        A(&A),
        screen_grad(screen_grad.data(), screen_grad.size()),
        screen_is_active_subset(screen_grad.size())
    {
        initialize();
    }

    void solve(
        std::function<void()> check_user_interrupt =util::no_op()
    );
};

} // namespace state
} // namespace adelie_core