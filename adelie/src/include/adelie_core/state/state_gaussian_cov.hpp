#pragma once
#include <Eigen/Eigenvalues>
#include <adelie_core/state/state_base.hpp>

namespace adelie_core {
namespace state {
namespace gaussian {
namespace cov {

/**
 * Updates all derived screen quantities for cov state.
 * After the function finishes, all screen quantities in the base + cov class
 * will be consistent with screen_set, and the state is otherwise effectively
 * unchanged in the sense that other quantities dependent on screen states are unchanged.
 */
template <class StateType>
void update_screen_derived(
    StateType& state
)
{
    using state_t = typename std::decay_t<StateType>;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_index_t = typename state_t::vec_index_t;

    update_screen_derived_base(state);

    const auto& groups = state.groups;
    const auto& group_sizes = state.group_sizes;
    const auto& screen_set = state.screen_set;
    const auto& screen_begins = state.screen_begins;
    const auto& grad = state.grad;
    auto& A = *state.A;
    auto& screen_transforms = state.screen_transforms;
    auto& screen_vars = state.screen_vars;
    auto& screen_grad = state.screen_grad;
    auto& screen_subset = state.screen_subset;
    auto& screen_subset_order = state.screen_subset_order;
    auto& screen_subset_ordered = state.screen_subset_ordered;

    const auto old_screen_size = screen_transforms.size();
    const auto new_screen_size = screen_set.size();
    const int old_screen_value_size = screen_subset.size();
    const int new_screen_value_size = (
        (screen_begins.size() == 0) ? 0 : (
            screen_begins.back() + group_sizes[screen_set.back()]
        )
    );

    screen_transforms.resize(new_screen_size);
    screen_vars.resize(new_screen_value_size, 0);
    screen_grad.resize(new_screen_value_size, 0);

    const auto max_gs = group_sizes.maxCoeff();
    util::rowvec_type<value_t> buffer(max_gs * max_gs);

    for (size_t i = old_screen_size; i < new_screen_size; ++i) {
        const auto g = groups[screen_set[i]];
        const auto gs = group_sizes[screen_set[i]];
        const auto sb = screen_begins[i];

        Eigen::Map<util::colmat_type<value_t>> A_gg(
            buffer.data(), gs, gs
        );

        // compute covariance matrix
        A.to_dense(g, gs, A_gg);

        if (gs == 1) {
            util::colmat_type<value_t, 1, 1> Q;
            Q(0, 0) = 1;
            screen_transforms[i] = Q;
            screen_vars[sb] = A_gg(0, 0);
            continue;
        }

        Eigen::SelfAdjointEigenSolver<util::colmat_type<value_t>> solver(A_gg);

        /* update screen_transforms */
        screen_transforms[i] = std::move(solver.eigenvectors());

        /* update screen_vars */
        const auto& D = solver.eigenvalues();
        Eigen::Map<vec_value_t> svars(screen_vars.data() + sb, gs);
        // numerical stability to remove small negative eigenvalues
        svars.head(D.size()) = D.array() * (D.array() >= 0).template cast<value_t>(); 
    }

    /* update screen_grad */
    for (size_t i = 0; i < new_screen_size; ++i) {
        const auto g = groups[screen_set[i]];
        const auto gs = group_sizes[screen_set[i]];
        const auto sb = screen_begins[i];

        Eigen::Map<vec_value_t>(
            screen_grad.data() + sb,
            gs
        ) = grad.segment(g, gs);
    }

    /* update screen_subset */
    screen_subset.resize(new_screen_value_size);
    int n_processed = 0;
    for (size_t ss_idx = old_screen_size; ss_idx < new_screen_size; ++ss_idx) {
        const auto ss = screen_set[ss_idx];
        const auto g = groups[ss];
        const auto gs = group_sizes[ss];
        Eigen::Map<vec_index_t>(
            screen_subset.data() + old_screen_value_size + n_processed, gs
        ) = vec_index_t::LinSpaced(gs, g, g + gs - 1);
        n_processed += gs;
    }

    /* update screen_subset_order */
    screen_subset_order.resize(new_screen_value_size);
    std::iota(
        std::next(screen_subset_order.begin(), old_screen_value_size),
        screen_subset_order.end(),
        old_screen_value_size
    );
    std::sort(
        screen_subset_order.data(),
        screen_subset_order.data() + screen_subset_order.size(),
        [&](auto i, auto j) { return screen_subset[i] < screen_subset[j]; }
    );

    /* update screen_subset_ordered */
    screen_subset_ordered.resize(new_screen_value_size);
    for (size_t i = 0; i < screen_subset_order.size(); ++i) {
        screen_subset_ordered[i] = screen_subset[screen_subset_order[i]];
    }
}

} // namespace cov
} // namespace gaussian

template <class ConstraintType,
          class MatrixType,
          class ValueType=typename std::decay_t<MatrixType>::value_t,
          class IndexType=Eigen::Index,
          class BoolType=bool,
          class SafeBoolType=int8_t
        >
struct StateGaussianCov: public StateBase<
        ConstraintType,
        ValueType,
        IndexType,
        BoolType,
        SafeBoolType
    >
{
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
        if (v.size() != A.cols()) {
            throw util::adelie_core_error("v must be (p,) where A is (p, p).");
        }
        /* initialize the rest of the screen quantities */
        gaussian::cov::update_screen_derived(*this);
    }
};

} // namespace state
} // namespace adelie_core