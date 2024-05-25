#pragma once
#include <adelie_core/state/state_gaussian_pin_base.hpp>

namespace adelie_core {
namespace state {
namespace gaussian {
namespace pin {
namespace cov {

template <class StateType>
ADELIE_CORE_STRONG_INLINE
void update_active_inactive_subset(StateType& state)
{
    using state_t = std::decay_t<StateType>;
    using vec_bool_t = typename state_t::vec_bool_t;

    const auto& groups = state.groups;
    const auto& group_sizes = state.group_sizes;
    const auto& screen_set = state.screen_set;
    const auto& screen_subset_order = state.screen_subset_order;
    const auto& screen_subset_ordered = state.screen_subset_ordered;
    const auto& screen_is_active = state.screen_is_active;
    auto& screen_is_active_subset = state.screen_is_active_subset;
    auto& active_subset_order = state.active_subset_order;
    auto& active_subset_ordered = state.active_subset_ordered;
    auto& inactive_subset_order = state.inactive_subset_order;
    auto& inactive_subset_ordered = state.inactive_subset_ordered;

    // update screen_is_active_subset
    int n_processed = 0;
    for (int ss_idx = 0; ss_idx < screen_set.size(); ++ss_idx) {
        const auto ss = screen_set[ss_idx];
        const auto gs = group_sizes[ss];
        Eigen::Map<vec_bool_t>(
            screen_is_active_subset.data() + n_processed, gs
        ) = screen_is_active[ss_idx];
        n_processed += gs;
    }

    // update active/inactive subset order/ordered
    active_subset_order.clear();
    active_subset_ordered.clear();
    inactive_subset_order.clear();
    inactive_subset_ordered.clear();
    for (int i = 0; i < screen_subset_order.size(); ++i) {
        const auto ssoi = screen_subset_order[i];
        const auto sso = screen_subset_ordered[i];
        if (screen_is_active_subset[ssoi]) {
            active_subset_order.push_back(i);
            active_subset_ordered.push_back(sso);
        } else {
            inactive_subset_order.push_back(i);
            inactive_subset_ordered.push_back(sso);
        }
    }
}

} // namespace cov
} // namespace pin
} // namespace gaussian

template <class ConstraintType,
          class MatrixType, 
          class ValueType=typename std::decay_t<MatrixType>::value_t,
          class IndexType=Eigen::Index,
          class BoolType=bool
        >
struct StateGaussianPinCov: StateGaussianPinBase<
        ConstraintType,
        ValueType,
        IndexType,
        BoolType
    >
{
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
        const Eigen::Ref<const vec_index_t>& screen_dual_begins, 
        const Eigen::Ref<const vec_index_t>& screen_subset_order,
        const Eigen::Ref<const vec_index_t>& screen_subset_ordered,
        const Eigen::Ref<const vec_value_t>& lmda_path, 
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
        Eigen::Ref<vec_value_t> screen_dual,
        size_t active_set_size,
        Eigen::Ref<vec_index_t> active_set
    ): 
        base_t(
            constraints, groups, group_sizes, alpha, penalty, 
            screen_set, screen_begins, screen_vars, screen_transforms, screen_dual_begins, lmda_path, 
            false, max_active_size, max_iters, tol, 0, 0, newton_tol, newton_max_iters, n_threads,
            rsq, screen_beta, screen_is_active, screen_dual, active_set_size, active_set
        ),
        screen_subset_order(screen_subset_order.data(), screen_subset_order.size()),
        screen_subset_ordered(screen_subset_ordered.data(), screen_subset_ordered.size()),
        rdev_tol(rdev_tol),
        A(&A),
        screen_grad(screen_grad.data(), screen_grad.size()),
        screen_is_active_subset(screen_grad.size())
    {
        // optimization
        active_subset_order.reserve(screen_subset_order.size());
        active_subset_ordered.reserve(screen_subset_order.size());
        inactive_subset_order.reserve(screen_subset_order.size());
        inactive_subset_ordered.reserve(screen_subset_order.size());

        gaussian::pin::cov::update_active_inactive_subset(*this);
    }
};

} // namespace state
} // namespace adelie_core