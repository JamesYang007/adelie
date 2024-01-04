#pragma once
#include <numeric>
#include <unordered_map>
#include <Eigen/Eigenvalues>
#include <adelie_core/matrix/utils.hpp>
#include <adelie_core/state/state_gaussian_base.hpp>

namespace adelie_core {
namespace state {
namespace gaussian {

/**
 * Updates all derived strong quantities for naive state.
 * See the incoming state requirements in update_screen_derived_base.
 * After the function finishes, all strong quantities in the base + naive class
 * will be consistent with screen_set, and the state is otherwise effectively
 * unchanged in the sense that other quantities dependent on strong states are unchanged.
 */
template <class StateType>
void update_screen_derived_naive(
    StateType& state
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;

    update_screen_derived_base(state);

    const auto& weights_sqrt = state.weights_sqrt;
    const auto& X_means = state.X_means;
    const auto& groups = state.groups;
    const auto& group_sizes = state.group_sizes;
    const auto& screen_set = state.screen_set;
    const auto& screen_begins = state.screen_begins;
    const auto intercept = state.intercept;
    auto& X = *state.X;
    auto& screen_X_means = state.screen_X_means;
    auto& screen_transforms = state.screen_transforms;
    auto& screen_vars = state.screen_vars;

    const auto old_screen_size = screen_transforms.size();
    const auto new_screen_size = screen_set.size();
    const int new_screen_value_size = (
        (screen_begins.size() == 0) ? 0 : (
            screen_begins.back() + group_sizes[screen_set.back()]
        )
    );

    screen_X_means.resize(new_screen_value_size);    
    screen_transforms.resize(new_screen_size);
    screen_vars.resize(new_screen_value_size, 0);

    // buffers
    const auto n = X.rows();
    const auto max_gs = group_sizes.maxCoeff();
    util::colmat_type<value_t> buffer(n, max_gs);
    util::colmat_type<value_t> XiTXi;

    for (size_t i = old_screen_size; i < new_screen_size; ++i) {
        const auto g = groups[screen_set[i]];
        const auto gs = group_sizes[screen_set[i]];
        const auto sb = screen_begins[i];

        // resize output and buffer 
        auto Xi = buffer.leftCols(gs);
        XiTXi.resize(gs, gs);

        // compute column-means
        Eigen::Map<vec_value_t> Xi_means(
            screen_X_means.data() + sb, gs
        );
        Xi_means = X_means.segment(g, gs);

        // compute weighted covariance matrix
        X.cov(g, gs, weights_sqrt, XiTXi, Xi);

        if (intercept) {
            XiTXi.noalias() -= Xi_means.matrix().transpose() * Xi_means.matrix();
        }

        Eigen::SelfAdjointEigenSolver<util::colmat_type<value_t>> solver(XiTXi);

        /* update screen_transforms */
        screen_transforms[i] = std::move(solver.eigenvectors());

        /* update screen_vars */
        const auto& D = solver.eigenvalues();
        Eigen::Map<vec_value_t> svars(screen_vars.data() + sb, gs);
        svars.head(D.size()) = D.array();
    }
}


template <class MatrixType, 
          class ValueType=typename std::decay_t<MatrixType>::value_t,
          class IndexType=Eigen::Index,
          class BoolType=bool
        >
struct StateGaussianNaive : StateGaussianBase<
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
    using typename base_t::uset_index_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::vec_bool_t;
    using typename base_t::map_cvec_value_t;
    using typename base_t::dyn_vec_value_t;
    using typename base_t::dyn_vec_index_t;
    using typename base_t::dyn_vec_bool_t;
    using umap_index_t = std::unordered_map<index_t, index_t>;
    using matrix_t = MatrixType;
    using dyn_vec_mat_value_t = std::vector<util::rowmat_type<value_t>>;

    /* static states */
    const vec_value_t weights_sqrt;
    const map_cvec_value_t X_means;
    const value_t y_mean;
    const value_t y_var;

    /* configurations */

    /* dynamic states */
    matrix_t* X;
    vec_value_t resid;
    value_t resid_sum;
    dyn_vec_value_t screen_X_means;
    dyn_vec_mat_value_t screen_transforms;
    dyn_vec_value_t screen_vars;

    /* buffers */
    vec_value_t resid_prev_valid;
    dyn_vec_value_t screen_beta_prev_valid; 
    dyn_vec_bool_t screen_is_active_prev_valid;

    explicit StateGaussianNaive(
        matrix_t& X,
        const Eigen::Ref<const vec_value_t>& X_means,
        value_t y_mean,
        value_t y_var,
        const Eigen::Ref<const vec_value_t>& resid,
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
            groups, group_sizes, alpha, penalty, weights, lmda_path, lmda_max, min_ratio, lmda_path_size,
            max_screen_size, 
            pivot_subset_ratio, pivot_subset_min, pivot_slack_ratio, screen_rule, 
            max_iters, tol, rsq_tol, rsq_slope_tol, rsq_curv_tol, 
            newton_tol, newton_max_iters, early_exit, setup_lmda_max, setup_lmda_path, intercept, n_threads,
            screen_set, screen_beta, screen_is_active, rsq, lmda, grad
        ),
        weights_sqrt(weights.sqrt()),
        X_means(X_means.data(), X_means.size()),
        y_mean(y_mean),
        y_var(y_var),
        X(&X),
        resid(resid),
        resid_sum(resid.sum())
    { 
        initialize();
    }

    /**
     * The state invariance just needs to hold when lmda > lmda_max.
     */
    void initialize() 
    {
        /* initialize the rest of the strong quantities */
        update_screen_derived_naive(*this); 
    }
};

} // namespace gaussian
} // namespace state
} // namespace adelie_core