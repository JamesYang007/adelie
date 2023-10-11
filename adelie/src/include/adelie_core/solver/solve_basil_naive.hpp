#pragma once
#include <adelie_core/matrix/matrix_pin_naive_subset.hpp>
#include <adelie_core/solver/solve_basil_base.hpp>
#include <adelie_core/solver/solve_pin_naive.hpp>
#include <adelie_core/state/state_pin_naive.hpp>
#include <adelie_core/state/state_basil_naive.hpp>
#include <adelie_core/util/algorithm.hpp>
#include <adelie_core/util/stopwatch.hpp>

namespace adelie_core {
namespace solver {
namespace basil_naive {

template <class StateType, class StatePinType, class ValueType>
ADELIE_CORE_STRONG_INLINE
void update_solutions(
    StateType& state,
    StatePinType& state_pin_naive,
    const Eigen::Ref<const util::rowvec_type<ValueType>>& lmda_path,
    size_t n_sols
)
{
    const auto X = *state.X;
    const auto y_mean = state.y_mean;
    const auto y_var = state.y_var;
    const auto& X_mean = state.X_mean;
    const auto intercept = state.intercept;
    auto& betas = state.betas;
    auto& rsqs = state.rsqs;
    auto& lmdas = state.lmdas;
    auto& intercepts = state.intercepts;

    // first, rotate all betas to the original space
    untransform_beta(
        state_pin_naive.betas,
        state.strong_X_block_vs,
        state.group_sizes,
        state.strong_set,
        state_pin_naive.active_set,
        state_pin_naive.active_order
    );

    for (size_t i = 0; i < n_sols; ++i) {
        betas.emplace_back(std::move(state_pin_naive.betas[i]));
        lmdas.emplace_back(lmda_path[i]);
        rsqs.emplace_back(state_pin_naive.rsqs[i]);
        const auto b0 = y_mean - X_mean.matrix().dot(betas.back());
        if (intercept) {
            intercepts.emplace_back(b0);
        } else {
            intercepts.emplace_back(0);
            // correction for R^2 so that it always computes for X and y centered
            const auto n = X.rows();
            rsqs.back() += n * (b0 * b0 - y_mean * y_mean);
        }
        rsqs.back() /= y_var;
    }
}

/**
 * Screens for new variables to include in the safe set
 * for fitting with lmda = lmda_next.
 * 
 * State MUST be a valid state satisfying its invariance.
 * The state after the function is finished is also a valid state.
 */
template <class StateType, class ValueType>
ADELIE_CORE_STRONG_INLINE
void screen_edpp(
    StateType& state,
    ValueType lmda_next
)
{
    using state_t = std::decay_t<StateType>;
    using vec_value_t = typename state_t::vec_value_t;

    const auto use_edpp = state.use_edpp;

    if (!use_edpp) return;

    const auto& X_group_norms = state.X_group_norms;
    const auto& groups = state.groups;
    const auto& group_sizes = state.group_sizes;
    const auto& penalty = state.penalty;
    const auto lmda = state.lmda;
    const auto lmda_max = state.lmda_max;
    const auto& edpp_v1_0 = state.edpp_v1_0;
    const auto& edpp_resid_0 = state.edpp_resid_0;
    const auto& resid = state.resid;
    const auto& grad = state.grad;
    const auto n_threads = state.n_threads;
    auto& X = *state.X;
    auto& edpp_safe_set = state.edpp_safe_set;
    auto& edpp_safe_hashset = state.edpp_safe_hashset;

    vec_value_t v1 = (
        (lmda == lmda_max) ?
        edpp_v1_0 :
        (edpp_resid_0 - resid) / lmda
    );
    vec_value_t v2 = edpp_resid_0 / lmda_next - resid / lmda;
    vec_value_t v2_perp = v2 - (v1.matrix().dot(v2.matrix()) / v1.matrix().squaredNorm()) * v1;
    const auto v2_perp_norm = v2_perp.matrix().norm();

    vec_value_t buffer(grad.size());    

    for (size_t i = 0; i < groups.size(); ++i) {
        if (edpp_safe_hashset.find(i) != edpp_safe_hashset.end()) continue;
        const auto g = groups[i];
        const auto gs = group_sizes[i];
        const auto grad_g = grad.segment(g, gs);
        auto buff_g = buffer.segment(g, gs);

        X.bmul(g, gs, v2_perp, buff_g);
        buff_g = 0.5 * buff_g + grad_g / lmda;
        const auto buff_g_norm = buff_g.matrix().norm();

        if (buff_g_norm >= penalty[i] - 0.5 * v2_perp_norm * X_group_norms[i]) {
            edpp_safe_set.push_back(i);
            edpp_safe_hashset.push_back(i);
        }
    }    
}

/**
 * Screens for new variables to include in the strong set
 * for fitting with lmda = lmda_next.
 * 
 * State MUST be a valid state satisfying its invariance.
 * The state after the function is finished is also a valid state.
 */
template <class StateType, class ValueType>
ADELIE_CORE_STRONG_INLINE 
void screen_strong(
    StateType& state,
    ValueType lmda_next
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;

    const auto& abs_grad = state.abs_grad;
    const auto lmda = state.lmda;
    const auto alpha = state.alpha;
    const auto& penalty = state.penalty;
    const auto& strong_hashset = state.strong_hashset;
    const auto& edpp_safe_set = state.edpp_safe_set;
    const auto delta_strong_size = state.delta_strong_size;
    const auto max_strong_size = state.max_strong_size;
    const auto strong_rule = state.strong_rule;
    auto& strong_set = state.strong_set;

    const auto old_strong_set_size = strong_set.size();
    const auto new_safe_size = edpp_safe_set.size() - old_strong_set_size;

    assert(strong_set.size() <= abs_grad.size());

    const auto is_strong = [&](auto i) { 
        return strong_hashset.find(i) != strong_hashset.end(); 
    };

    /* update strong_set */
    // Use either the fixed-increment rule or strong rule to increase strong set.
    if (!strong_rule) {
        size_t size_capped = std::min(delta_strong_size, new_safe_size);

        strong_set.insert(strong_set.end(), size_capped, -1);
        const auto factor = (alpha <= 0) ? 1e-3 : alpha;
        const auto abs_grad_p = vec_value_t::NullaryExpr(
            abs_grad.size(), [&](auto i) {
                return (penalty[i] <= 0) ? 0.0 : abs_grad[i] / penalty[i];
            }
        ) / factor;
        util::k_imax(
            abs_grad_p, 
            is_strong,
            size_capped, 
            std::next(strong_set.begin(), old_strong_set_size)
        );
    } else {
        const auto strong_rule_lmda = (2 * lmda_next - lmda) * alpha;
        // TODO: PARALLELIZE!
        for (int i = 0; i < abs_grad.size(); ++i) {
            if (is_strong(i)) continue;
            if (abs_grad[i] > strong_rule_lmda * penalty[i]) {
                strong_set.push_back(i);
            }
        }
    }

    // If adding new amount went over max strong size, 
    // undo the change to keep invariance from before, then throw exception.
    if (strong_set.size() > max_strong_size) {
        strong_set.erase(
            std::next(strong_set.begin(), old_strong_set_size),
            strong_set.end()
        );
        throw util::max_basil_strong_set();
    }

    /* update derived strong quantities */
    state::update_strong_derived_base(state, old_strong_set_size);
    state::update_strong_derived_naive(state, old_strong_set_size);

}

template <class StateType, class ValueType>
ADELIE_CORE_STRONG_INLINE
void screen(
    StateType& state,
    ValueType lmda_next
)
{
    screen_edpp(state, lmda_next);
    screen_strong(state, lmda_next);
}

template <class StateType,
          class ValueType,
          class UpdateCoefficientsType,
          class CUIType=util::no_op>
ADELIE_CORE_STRONG_INLINE
auto fit(
    StateType& state,
    const Eigen::Ref<const util::rowvec_type<ValueType>>& lmda_path,
    UpdateCoefficientsType update_coefficients_f,
    CUIType check_user_interrupt = CUIType()
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;
    using index_t = typename state_t::index_t;
    using safe_bool_t = typename state_t::safe_bool_t;
    using vec_value_t = typename state_t::vec_value_t;
    using vec_index_t = typename state_t::vec_index_t;
    using vec_bool_t = typename state_t::vec_bool_t;
    using matrix_pin_naive_t = matrix::MatrixPinNaiveSubset<
        util::colmat_type<value_t>,
        index_t
    >;
    using state_pin_naive_t = state::StatePinNaive<
        matrix_pin_naive_t,
        typename std::decay_t<matrix_pin_naive_t>::value_t,
        index_t,
        safe_bool_t
    >;

    const auto& X = *state.X;
    const auto& groups = state.groups;
    const auto& group_sizes = state.group_sizes;
    const auto alpha = state.alpha;
    const auto& penalty = state.penalty;
    const auto& strong_X_blocks = state.strong_X_blocks;
    const auto& strong_idx_map = state.strong_idx_map;
    const auto& strong_slice_map = state.strong_slice_map;
    const auto& strong_set = state.strong_set;
    const auto& strong_g1 = state.strong_g1;
    const auto& strong_g2 = state.strong_g2;
    const auto& strong_begins = state.strong_begins;
    const auto& strong_vars = state.strong_vars;
    const auto max_iters = state.max_iters;
    const auto tol = state.tol;
    const auto newton_tol = state.newton_tol;
    const auto newton_max_iters = state.newton_max_iters;
    const auto n_threads = state.n_threads;
    const auto rsq = state.rsq;
    auto& resid = state.resid;
    auto& strong_beta = state.strong_beta;
    auto& strong_grad = state.strong_grad;
    auto& strong_is_active = state.strong_is_active;

    matrix_pin_naive_t X_subset(
        X.rows(),
        X.cols(),
        strong_X_blocks,
        strong_idx_map,
        strong_slice_map,
        n_threads
    );

    // create a pin, naive state object
    state_pin_naive_t state_pin_naive(
        X_subset,
        groups, 
        group_sizes,
        alpha, 
        penalty,
        Eigen::Map<const vec_index_t>(strong_set.data(), strong_set.data() + strong_set.size()), 
        Eigen::Map<const vec_index_t>(strong_g1.data(), strong_g1.data() + strong_g1.size()), 
        Eigen::Map<const vec_index_t>(strong_g2.data(), strong_g2.data() + strong_g2.size()), 
        Eigen::Map<const vec_index_t>(strong_begins.data(), strong_begins.data() + strong_begins.size()), 
        Eigen::Map<const vec_value_t>(strong_vars.data(), strong_vars.data() + strong_vars.size()), 
        lmda_path,
        max_iters, tol, 0, 0, newton_tol, newton_max_iters, n_threads,
        rsq,
        Eigen::Map<vec_value_t>(resid.data(), resid.data() + resid.size()),
        Eigen::Map<vec_value_t>(strong_beta.data(), strong_beta.data() + strong_beta.size()), 
        Eigen::Map<vec_value_t>(strong_grad.data(), strong_grad.data() + strong_grad.size()),
        Eigen::Map<vec_bool_t>(strong_is_active.data(), strong_is_active.data() + strong_is_active.size())
    );

    // run pin, naive method to fit
    solve_pin_naive(
        state_pin_naive, 
        update_coefficients_f, 
        check_user_interrupt
    );

    return state_pin_naive;
}

/**
 * Checks the KKT condition on the proposed solutions in state_pin_naive.
 */
template <class StateType, class StatePinNaiveType, class GradsType>
ADELIE_CORE_STRONG_INLINE
size_t kkt(
    StateType& state,
    const StatePinNaiveType& state_pin_naive,
    GradsType& grads
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;
    using dyn_vec_value_t = typename state_t::dyn_vec_value_t;

    const auto& groups = state.groups;
    const auto& group_sizes = state.group_sizes;
    const auto alpha = state.alpha;
    const auto& penalty = state.penalty;
    const auto& strong_begins = state.strong_begins;
    const auto& strong_hashset = state.strong_hashset;
    const auto& strong_hashmap = state.strong_hashmap;
    const auto& strong_X_block_vs = state.strong_X_block_vs;
    const auto& resid = state.resid;
    auto& X = state.X;

    const auto n_lmdas = state_pin_naive.lmdas.size();
    const auto p = X.cols();

    // Keep track of the number of lmdas that are viable.
    // Viable means KKT hasn't failed yet for this lmda.
    int n_valid_solutions = n_lmdas;

    // Important to loop over groups first.
    // X usually loads new matrices when column blocks change.
    // If we loop over lmdas first, we will re-read column blocks multiple time.
    // Better to read once and process on all lmdas.
    for (int k = 0; k < groups.size(); ++k) {
        const auto gk = groups[k];
        const auto gk_size = group_sizes[k];
        const auto pk = penalty[k];

        for (int l = 0; l < n_valid_solutions; ++l) {
            auto grad_lk = grads[l].segment(gk, gk_size);

            if (strong_hashset.find(k) != strong_hashset.end()) {
                const auto strong_idx = strong_hashmap[k];
                const auto& strong_grad_l = state_pin_naive.strong_grads[l];
                const auto strong_grad_lk = strong_grad_l.segment(
                    strong_begins[strong_idx], gk_size
                );
                grad_lk.noalias() = (
                    strong_grad_lk.matrix() * strong_X_block_vs[strong_idx].transpose()
                );
            } else {
                X.bmul(gk, gk_size, state_pin_naive.resids[l], grad_lk);
                const auto abs_grad_lk = grad_lk.matrix().norm();
                const auto lmda_l = state_pin_naive.lmdas[l];
                if (abs_grad_lk > lmda_l * alpha * pk) {
                    n_valid_solutions = l;
                    break;
                }
            }
        }
         
        if (n_valid_solutions <= 0) break;
    }

    return n_valid_solutions;
}

} // namespace basil_naive

template <class StateType,
          class UpdateCoefficientsType,
          class CUIType=util::no_op>
inline void solve_basil_naive(
    StateType&& state,
    UpdateCoefficientsType update_coefficients_f,
    CUIType check_user_interrupt = CUIType()
)
{
    using state_t = std::decay_t<StateType>;
    using vec_value_t = typename state_t::vec_value_t;
    using dyn_vec_bool_t = typename state_t::dyn_vec_bool_t;
    using dyn_vec_value_t = typename state_t::dyn_vec_value_t;

    const auto y_var = state.y_var;
    const auto& lmda_path = state.lmda_path;
    const auto& strong_set = state.strong_set;
    const auto& rsqs = state.rsqs;
    const auto delta_lmda_path_size = state.delta_lmda_path_size;
    const auto early_exit = state.early_exit;
    const auto max_strong_size = state.max_strong_size;
    const auto rsq_slope_tol = state.rsq_slope_tol;
    const auto rsq_curv_tol = state.rsq_curv_tol;
    const auto lmda_max = state.lmda_max;
    const auto setup_edpp = state.setup_edpp;
    auto& X = *state.X;
    auto& strong_is_active = state.strong_is_active;
    auto& strong_beta = state.strong_beta;
    auto& strong_grad = state.strong_grad;
    auto& resid = state.resid;
    auto& grad = state.grad;
    auto& abs_grad = state.abs_grad;
    auto& rsq = state.rsq;
    auto& lmda = state.lmda;

    const auto p = grad.size();

    if (strong_set.size() > max_strong_size) throw util::max_basil_strong_set();

    // ==================================================================================== 
    // Initial fit for lambda > lambda_max
    // ==================================================================================== 
    // Only unpenalized (l1) groups are active in this case by definition of lmda_max.
    // Since state is in its invariance (solution at state.lmda) and unpenalized groups
    // are ALWAYS active, state includes all unpenalized groups.
    // If no lambda in lmda_path is > lmda_max and EDPP setup not needed, state is left unchanged.
    // Otherwise, it is in its invariance at lmda = lmda_max.
    // All solutions to lambda > lambda_max are saved.

    // slice lambda_path up to lmda_max
    const auto large_lmda_path_size = std::find_if(
        lmda_path.data(), 
        lmda_path.data() + lmda_path.size(),
        [&](auto x) { return x <= lmda_max; }
    ) - lmda_path.data();

    // if there is some large lambda or EDPP setup is needed, do initial fit.
    if (large_lmda_path_size || setup_edpp) {
        // create a lambda path containing only lmdas > lambda_max
        // and additionally lambda_max at the end.
        // If large_lmda_path_size > 0, mind as well fit for lambda_max as well to go down the path.
        // Else, setup_edpp is true, in which case, we must fit lambda_max.
        vec_value_t large_lmda_path(large_lmda_path_size + 1);
        large_lmda_path.head(large_lmda_path_size) = lmda_path.head(large_lmda_path_size);
        large_lmda_path[large_lmda_path_size] = lmda_max;

        const auto& state_pin_naive = basil_naive::fit(
            state, 
            large_lmda_path, 
            update_coefficients_f, 
            check_user_interrupt
        );

        /* Invariance */
        rsq = state_pin_naive.rsq;
        lmda = lmda_max;
        X.bmul(0, p, resid, grad);
        state::update_abs_grad(state);
        basil_naive::update_solutions(
            state, 
            state_pin_naive, 
            large_lmda_path,
            state_pin_naive.lmdas.size()-1
        );
    }

    size_t lmda_path_idx = rsqs.size();

    // ==================================================================================== 
    // Update state after initial fit
    // ==================================================================================== 
    // The state is in its invariance and 
    // is right at the solution at lambda_max if it needed to be fit from before.
    // The latter state is needed if EDPP states need to be updated.
    // Otherwise, EDPP states are unmodified.
    state::update_edpp_states(state);

    // From this point on, all EDPP states are valid, whether they were updated or not.

    // ==================================================================================== 
    // BASIL iterations for lambda <= lambda_max
    // ==================================================================================== 
    // In this case, strong_set may not contain the true active set.
    // We must go through BASIL iterations to solve each lambda.
    const auto lmda_path_size = lmda_path.size();
    vec_value_t lmda_batch;
    vec_value_t resid_prev_valid;
    dyn_vec_bool_t strong_is_active_prev_valid;
    dyn_vec_value_t strong_beta_prev_valid; 
    dyn_vec_value_t strong_grad_prev_valid;
    std::vector<vec_value_t> grads;

    while (1) 
    {
        // check early exit
        if (early_exit && (rsqs.size() >= 3)) {
            const auto rsq_u = rsqs[rsqs.size()-1];
            const auto rsq_m = rsqs[rsqs.size()-2];
            const auto rsq_l = rsqs[rsqs.size()-3];
            if (check_early_stop_rsq(rsq_l, rsq_m, rsq_u, rsq_slope_tol, rsq_curv_tol) || 
                (rsqs.back() >= 0.99 * y_var)) break;
        }

        // check if any lambdas left to fit
        if (lmda_path_idx >= lmda_path.size()) break;

        // batch the next set of lambdas
        lmda_batch = lmda_path.segment(
            lmda_path_idx, 
            std::min(delta_lmda_path_size, lmda_path_size - lmda_path_idx)
        );

        // ==================================================================================== 
        // Screening step
        // ==================================================================================== 
        basil_naive::screen(
            state,
            lmda_batch[0]
        );

        // ==================================================================================== 
        // Fit step
        // ==================================================================================== 
        // Save all current valid quantities that will be modified in-place by fit.
        // This is needed for the invariance step in case no valid solutions are found.
        strong_beta_prev_valid = strong_beta;
        strong_is_active_prev_valid = strong_is_active;
        strong_grad_prev_valid = strong_grad;
        resid_prev_valid = resid;
        auto& state_pin_naive = basil_naive::fit(
            state,
            lmda_batch,
            update_coefficients_f,
            check_user_interrupt
        );

        // ==================================================================================== 
        // KKT step
        // ==================================================================================== 
        grads.resize(lmda_batch.size());
        for (auto& g : grads) g.resize(p);
        const auto n_valid_solutions = basil_naive::kkt(
            state,
            state_pin_naive,
            grads
        );

        // ==================================================================================== 
        // Invariance step
        // ==================================================================================== 
        lmda_path_idx += n_valid_solutions;
        // If no valid solutions found, restore to the previous valid state,
        // so that we can start over after screening more variables.
        if (n_valid_solutions <= 0) {
            std::swap(strong_beta, strong_beta_prev_valid);
            std::swap(strong_is_active, strong_is_active_prev_valid);
            std::swap(strong_grad, strong_grad_prev_valid);
            std::swap(resid, resid_prev_valid);
        // Otherwise, use the last valid state found in kkt.
        } else {
            const auto idx = n_valid_solutions - 1;
            std::swap(strong_beta, state_pin_naive.strong_betas[idx]);
            std::swap(strong_is_active, state_pin_naive.strong_is_actives[idx]);
            std::swap(strong_grad, state_pin_naive.strong_grads[idx]);
            std::swap(resid, state_pin_naive.resids[idx]);
            rsq = state_pin_naive.rsqs[idx];
            lmda = lmda_batch[idx];
            grad.swap(grads[idx]);
            state::update_abs_grad(state);
        }
        basil_naive::update_solutions(
            state, 
            state_pin_naive, 
            lmda_batch,
            n_valid_solutions
        );
    }
}

} // namespace solver
} // namespace adelie_core