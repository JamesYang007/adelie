#pragma once
#include <numeric>
#include <unordered_map>
#include <Eigen/Eigenvalues>
#include <adelie_core/matrix/utils.hpp>
#include <adelie_core/state/state_basil_base.hpp>

namespace adelie_core {
namespace state {

/**
 * Updates all derived strong quantities for naive state.
 * See the incoming state requirements in update_strong_derived_base.
 * After the function finishes, all strong quantities in the base + naive class
 * will be consistent with strong_set, and the state is otherwise effectively
 * unchanged in the sense that other quantities dependent on strong states are unchanged.
 */
template <class StateType>
void update_strong_derived_naive(
    StateType& state
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;

    update_strong_derived_base(state);

    const auto& X_means = state.X_means;
    const auto& groups = state.groups;
    const auto& group_sizes = state.group_sizes;
    const auto& strong_set = state.strong_set;
    const auto& strong_begins = state.strong_begins;
    const auto intercept = state.intercept;
    const auto n_threads = state.n_threads;
    auto& X = *state.X;
    auto& strong_X_means = state.strong_X_means;
    auto& strong_transforms = state.strong_transforms;
    auto& strong_vars = state.strong_vars;

    const auto old_strong_size = strong_transforms.size();
    const auto new_strong_size = strong_set.size();
    const int new_strong_value_size = (
        (strong_begins.size() == 0) ? 0 : (
            strong_begins.back() + group_sizes[strong_set.back()]
        )
    );

    strong_X_means.resize(new_strong_value_size);    
    strong_transforms.resize(new_strong_size);
    strong_vars.resize(new_strong_value_size, 0);

    // buffers
    util::colmat_type<value_t> Xi;
    util::colmat_type<value_t> XiTXi;

    for (size_t i = old_strong_size; i < new_strong_size; ++i) {
        const auto g = groups[strong_set[i]];
        const auto gs = group_sizes[strong_set[i]];
        const auto sb = strong_begins[i];
        const auto n = X.rows();

        // get dense version of the group matrix block
        Xi.resize(n, gs);
        X.to_dense(g, gs, Xi);

        // compute column-means
        Eigen::Map<vec_value_t> Xi_means(
            strong_X_means.data() + sb, gs
        );
        Xi_means = X_means.segment(g, gs);

        // if intercept, must center first
        if (intercept) {
            auto Xia = Xi.array();
            matrix::dmvsubi(Xia, Xi_means, n_threads);
        }

        // transform data
        Eigen::setNbThreads(n_threads);
        XiTXi.noalias() = Xi.transpose() * Xi;
        Eigen::setNbThreads(0);

        Eigen::SelfAdjointEigenSolver<util::colmat_type<value_t>> solver(XiTXi);

        /* update strong_transforms */
        strong_transforms[i] = std::move(solver.eigenvectors());

        /* update strong_vars */
        const auto& D = solver.eigenvalues();
        Eigen::Map<vec_value_t> svars(strong_vars.data() + sb, gs);
        svars.head(D.size()) = D.array();
    }
}

/**
 * Updates the EDPP states.
 * The state must be in its invariance with lmda == lmda_max.
 * After the function finishes, the state is still in its invariance
 * with EDPP states initialized.
 */
template <class StateType>
void update_edpp_states(
    StateType& state
)
{
    using state_t = std::decay_t<StateType>;
    using vec_value_t = typename state_t::vec_value_t;

    if (!state.setup_edpp || !state.use_edpp) return;

    const auto& X_means = state.X_means;
    const auto& groups = state.groups;
    const auto& group_sizes = state.group_sizes;
    const auto& penalty = state.penalty;
    const auto& resid = state.resid;
    const auto& abs_grad = state.abs_grad;
    const auto& strong_X_means = state.strong_X_means;
    const auto& strong_beta = state.strong_beta;
    const auto intercept = state.intercept;
    auto& X = *state.X;
    auto& edpp_resid_0 = state.edpp_resid_0;
    auto& edpp_v1_0 = state.edpp_v1_0;

    edpp_resid_0 = resid;
    if (intercept) {
        const auto xbar_beta = (
            Eigen::Map<const vec_value_t>(strong_X_means.data(), strong_X_means.size()) * 
            Eigen::Map<const vec_value_t>(strong_beta.data(), strong_beta.size())
        ).sum();
        // TODO: parallelize
        edpp_resid_0 += xbar_beta;
    }

    Eigen::Index g_star;
    vec_value_t::NullaryExpr(
        abs_grad.size(),
        [&](auto i) {
            return (penalty[i] <= 0.0) ? 0.0 : abs_grad[i] / penalty[i];
        }
    ).maxCoeff(&g_star);

    vec_value_t out(group_sizes[g_star]);
    X.bmul(groups[g_star], group_sizes[g_star], edpp_resid_0, out);
    edpp_v1_0.resize(resid.size());
    X.btmul(groups[g_star], group_sizes[g_star], out, edpp_v1_0);
    if (intercept) {
        const auto xbar_star_out = (
            out * X_means.segment(groups[g_star], group_sizes[g_star])
        ).sum();
        // TODO: parallelize
        edpp_v1_0 -= xbar_star_out;
    }
}


template <class MatrixType, 
          class ValueType=typename std::decay_t<MatrixType>::value_t,
          class IndexType=Eigen::Index,
          class BoolType=bool
        >
struct StateBasilNaive : StateBasilBase<
        ValueType,
        IndexType,
        BoolType
    >
{
    using base_t = StateBasilBase<
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
    const map_cvec_value_t X_means;
    const map_cvec_value_t X_group_norms;
    const value_t y_mean;
    const value_t y_var;

    /* configurations */
    const bool use_edpp;
    const bool setup_edpp;

    /* dynamic states */
    matrix_t* X;
    vec_value_t resid;
    value_t resid_sum;
    dyn_vec_value_t strong_X_means;
    dyn_vec_mat_value_t strong_transforms;
    dyn_vec_value_t strong_vars;

    dyn_vec_index_t edpp_safe_set;
    uset_index_t edpp_safe_hashset;
    vec_value_t edpp_v1_0;
    vec_value_t edpp_resid_0;

    /* buffers */
    vec_value_t resid_prev_valid;
    dyn_vec_value_t strong_beta_prev_valid; 
    dyn_vec_bool_t strong_is_active_prev_valid;

    explicit StateBasilNaive(
        matrix_t& X,
        const Eigen::Ref<const vec_value_t>& X_means,
        const Eigen::Ref<const vec_value_t>& X_group_norms,
        value_t y_mean,
        value_t y_var,
        bool setup_edpp,
        const Eigen::Ref<const vec_value_t>& resid,
        const Eigen::Ref<const vec_index_t>& edpp_safe_set,
        const Eigen::Ref<const vec_value_t>& edpp_v1_0,
        const Eigen::Ref<const vec_value_t>& edpp_resid_0,
        const Eigen::Ref<const vec_index_t>& groups, 
        const Eigen::Ref<const vec_index_t>& group_sizes,
        value_t alpha, 
        const Eigen::Ref<const vec_value_t>& penalty,
        const Eigen::Ref<const vec_value_t>& lmda_path,
        value_t lmda_max,
        value_t min_ratio,
        size_t lmda_path_size,
        size_t delta_strong_size,
        size_t max_strong_size,
        value_t pivot_subset_ratio,
        size_t pivot_subset_min,
        value_t pivot_slack_ratio,
        const std::string& screen_rule,
        bool lazify_screen,
        value_t lazy_ratio,
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
        const Eigen::Ref<const vec_index_t>& strong_set,
        const Eigen::Ref<const vec_value_t>& strong_beta,
        const Eigen::Ref<const vec_bool_t>& strong_is_active,
        value_t rsq,
        value_t lmda,
        const Eigen::Ref<const vec_value_t>& grad
    ):
        base_t(
            groups, group_sizes, alpha, penalty, lmda_path, lmda_max, min_ratio, lmda_path_size,
            delta_strong_size, max_strong_size, 
            pivot_subset_ratio, pivot_subset_min, pivot_slack_ratio, screen_rule, lazify_screen, lazy_ratio,
            max_iters, tol, rsq_tol, rsq_slope_tol, rsq_curv_tol, 
            newton_tol, newton_max_iters, early_exit, setup_lmda_max, setup_lmda_path, intercept, n_threads,
            strong_set, strong_beta, strong_is_active, rsq, lmda, grad
        ),
        X_means(X_means.data(), X_means.size()),
        X_group_norms(X_group_norms.data(), X_group_norms.size()),
        y_mean(y_mean),
        y_var(y_var),
        use_edpp(alpha == 1),
        setup_edpp(setup_edpp),
        X(&X),
        resid(resid),
        resid_sum(resid.sum()),
        edpp_safe_set(edpp_safe_set.data(), edpp_safe_set.data() + edpp_safe_set.size()),
        edpp_safe_hashset(edpp_safe_set.data(), edpp_safe_set.data() + edpp_safe_set.size()),
        edpp_v1_0(edpp_v1_0),
        edpp_resid_0(edpp_resid_0)
    { 
        initialize();
    }

    /**
     * The state invariance just needs to hold when lmda > lmda_max.
     * Guarantees that EDPP superset of strong set and outside EDPP is truly 0 coefficient.
     */
    void initialize() 
    {
        /* initialize the rest of the strong quantities */
        update_strong_derived_naive(*this); 

        /* initialize edpp_safe_set */
        if (setup_edpp) {
            edpp_safe_hashset.clear();
            edpp_safe_set.clear();

            if (use_edpp) {
                // EDPP safe set must be a super-set of strong set.
                // Since strong_set is guaranteed to contain the true active set,
                // everything outside strong_set has 0 coefficient.
                edpp_safe_set.insert(
                    edpp_safe_set.end(),
                    base_t::strong_set.begin(), 
                    base_t::strong_set.end()
                );
            } else {
                // If EDPP is not used, every variable is safe.
                edpp_safe_set.resize(base_t::groups.size());
                std::iota(edpp_safe_set.begin(), edpp_safe_set.end(), 0);
            }
            
            /* initialize edpp_safe_hashset */
            edpp_safe_hashset.insert(edpp_safe_set.begin(), edpp_safe_set.end());

            // other EDPP quantities get initialized during the fit.
        }
        // if no setup EDPP, we assume user passed valid EDPP states.
    }
};

} // namespace state
} // namespace adelie_core