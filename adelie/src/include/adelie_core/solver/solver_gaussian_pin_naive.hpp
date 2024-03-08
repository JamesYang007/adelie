#pragma once
#include <numeric>
#include <adelie_core/util/exceptions.hpp>
#include <adelie_core/util/functional.hpp>
#include <adelie_core/util/stopwatch.hpp>
#include <adelie_core/util/eigen/map_sparsevector.hpp>
#include <adelie_core/solver/solver_gaussian_pin_base.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace solver {
namespace gaussian {
namespace pin {
namespace naive {

template <class StateType, class G1Iter, class G2Iter,
          class ValueType, class BufferType,
          class UpdateCoefficientsType,
          class AdditionalStepType=util::no_op>
ADELIE_CORE_STRONG_INLINE
void coordinate_descent(
    StateType&& state,
    G1Iter g1_begin,
    G1Iter g1_end,
    G2Iter g2_begin,
    G2Iter g2_end,
    size_t lmda_idx,
    ValueType& convg_measure,
    BufferType& buffer1,
    BufferType& buffer2,
    BufferType& buffer3,
    BufferType& buffer4_n,
    UpdateCoefficientsType update_coefficients_f,
    AdditionalStepType additional_step=AdditionalStepType()
)
{
    auto& X = *state.X;
    const auto& penalty = state.penalty;
    const auto& weights = state.weights;
    const auto& screen_set = state.screen_set;
    const auto& screen_begins = state.screen_begins;
    const auto& screen_X_means = state.screen_X_means;
    const auto& screen_transforms = *state.screen_transforms;
    const auto& screen_vars = state.screen_vars;
    const auto& groups = state.groups;
    const auto& group_sizes = state.group_sizes;
    const auto intercept = state.intercept;
    const auto alpha = state.alpha;
    const auto lmda = state.lmda_path[lmda_idx];
    const auto newton_tol = state.newton_tol;
    const auto newton_max_iters = state.newton_max_iters;
    const auto n_threads = state.n_threads;
    auto& screen_beta = state.screen_beta;
    auto& screen_grad = state.screen_grad;
    auto& resid = state.resid;
    auto& resid_sum = state.resid_sum;
    auto& rsq = state.rsq;

    const auto l1 = lmda * alpha;
    const auto l2 = lmda * (1-alpha);

    convg_measure = 0;
    // iterate over the groups of size 1
    for (auto it = g1_begin; it != g1_end; ++it) {
        const auto ss_idx = *it;              // index to screen set
        const auto k = screen_set[ss_idx];    // actual group index
        const auto ss_value_begin = screen_begins[ss_idx]; // value begin index at ss_idx
        auto& ak = screen_beta[ss_value_begin]; // corresponding beta
        auto& gk = screen_grad[ss_value_begin]; // corresponding gradient
        const auto Xk_mean = screen_X_means[ss_value_begin]; // corresponding X[:,k] mean
        const auto A_kk = screen_vars[ss_value_begin];  // corresponding A diagonal 
        const auto pk = penalty[k]; // corresponding penalty

        const auto ak_old = ak;

        // compute gradient
        gk = X.cmul(groups[k], resid, weights) - Xk_mean * resid_sum * intercept;

        update_coefficient(
            ak, A_kk, l1, l2, pk, gk
        );

        if (ak_old == ak) continue;

        const auto del = ak - ak_old;

        update_convergence_measure(convg_measure, del, A_kk);

        update_rsq(rsq, del, A_kk, gk);

        // update residual 
        auto dresid = buffer4_n.head(resid.size());
        X.ctmul(groups[k], del, dresid);
        matrix::dvsubi(resid, dresid, n_threads);
        resid_sum -= Xk_mean * del;

        additional_step(ss_idx);
    }
    
    // iterate over the groups of dynamic size
    for (auto it = g2_begin; it != g2_end; ++it) {
        const auto ss_idx = *it;              // index to screen set
        const auto k = screen_set[ss_idx];    // actual group index
        const auto ss_value_begin = screen_begins[ss_idx]; // value begin index at ss_idx
        const auto gsize = group_sizes[k]; // group size  
        auto ak = screen_beta.segment(ss_value_begin, gsize); // corresponding beta
        auto gk = screen_grad.segment(ss_value_begin, gsize); // corresponding gradient
        const auto Xk_mean = screen_X_means.segment(ss_value_begin, gsize); // corresponding X[:, g:g+gs] means
        const auto& Vk = screen_transforms[ss_idx]; // corresponding V in SVD of X_c
        const auto A_kk = screen_vars.segment(ss_value_begin, gsize);  // corresponding A diagonal 
        const auto pk = penalty[k]; // corresponding penalty

        // compute current gradient
        X.bmul(groups[k], gsize, resid, weights, gk);
        if (intercept) {
            gk -= resid_sum * Xk_mean;
        }

        auto gk_transformed = buffer3.head(ak.size());
        gk_transformed.matrix().noalias() = (
            gk.matrix() * Vk
        );

        // save old beta in buffer with transformation
        auto ak_old = buffer4_n.head(ak.size());
        ak_old = ak;
        auto ak_old_transformed = buffer4_n.segment(ak.size(), ak.size());
        ak_old_transformed.matrix().noalias() = ak_old.matrix() * Vk; 
        auto ak_transformed = buffer4_n.segment(2 * ak.size(), ak.size());

        // update group coefficients
        size_t iters;
        gk_transformed += A_kk * ak_old_transformed; 
        update_coefficients_f(
            A_kk, gk_transformed, l1 * pk, l2 * pk, 
            newton_tol, newton_max_iters,
            ak_transformed, iters, buffer1, buffer2
        );
        gk_transformed -= A_kk * ak_old_transformed; 
        
        if ((ak_old_transformed - ak_transformed).matrix().norm() <= 1e-12 * std::sqrt(gsize)) continue;

        auto del_transformed = buffer1.head(ak.size());
        del_transformed = ak_transformed - ak_old_transformed;

        update_convergence_measure(convg_measure, del_transformed, A_kk);

        update_rsq(rsq, del_transformed, A_kk, gk_transformed);

        // update new coefficient
        ak.matrix().noalias() = ak_transformed.matrix() * Vk.transpose();

        // update residual
        auto del = buffer1.head(ak.size());
        del = ak - ak_old;
        auto dresid = buffer4_n.head(resid.size());
        X.btmul(groups[k], gsize, del, dresid);
        matrix::dvsubi(resid, dresid, n_threads);
        resid_sum -= (Xk_mean * del).sum();

        additional_step(ss_idx);
    }
}

/**
 * Applies multiple blockwise coordinate descent on the active set.
 */
template <class StateType, 
          class BufferType, 
          class UpdateCoefficientsType,
          class CUIType>
ADELIE_CORE_STRONG_INLINE
void solve_active(
    StateType&& state,
    size_t lmda_idx,
    BufferType& buffer1,
    BufferType& buffer2,
    BufferType& buffer3,
    BufferType& buffer4_n,
    UpdateCoefficientsType update_coefficients_f,
    CUIType check_user_interrupt
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;

    const auto& active_g1 = state.active_g1;
    const auto& active_g2 = state.active_g2;
    const auto tol = state.tol;
    const auto max_iters = state.max_iters;
    auto& iters = state.iters;

    while (1) {
        check_user_interrupt();
        ++iters;
        value_t convg_measure;
        coordinate_descent(
            state, 
            active_g1.data(), active_g1.data() + active_g1.size(),
            active_g2.data(), active_g2.data() + active_g2.size(),
            lmda_idx, convg_measure, buffer1, buffer2, buffer3, buffer4_n,
            update_coefficients_f
        );
        if (convg_measure < tol) break;
        if (iters >= max_iters) throw util::max_cds_error(lmda_idx);
    }
}

template <class StateType,
          class UpdateCoefficientsType,
          class CUIType = util::no_op>
inline void solve(
    StateType&& state,
    UpdateCoefficientsType update_coefficients_f,
    CUIType check_user_interrupt = CUIType()
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;
    using index_t = typename state_t::index_t;
    using sp_vec_value_t = typename state_t::sp_vec_value_t;
    using sw_t = util::Stopwatch;

    auto& X = *state.X;
    const auto y_mean = state.y_mean;
    const auto y_var = state.y_var;
    const auto& groups = state.groups;
    const auto& group_sizes = state.group_sizes;
    const auto& screen_set = state.screen_set;
    const auto& screen_g1 = state.screen_g1;
    const auto& screen_g2 = state.screen_g2;
    const auto& screen_beta = state.screen_beta;
    const auto& screen_X_means = state.screen_X_means;
    const auto& lmda_path = state.lmda_path;
    const auto& rsq = state.rsq;
    const auto intercept = state.intercept;
    const auto tol = state.tol;
    const auto max_active_size = state.max_active_size;
    const auto max_iters = state.max_iters;
    const auto adev_tol = state.adev_tol;
    const auto ddev_tol = state.ddev_tol;
    auto& screen_is_active = state.screen_is_active;
    auto& active_set = state.active_set;
    auto& active_g1 = state.active_g1;
    auto& active_g2 = state.active_g2;
    auto& active_begins = state.active_begins;
    auto& active_order = state.active_order;
    auto& betas = state.betas;
    auto& intercepts = state.intercepts;
    auto& rsqs = state.rsqs;
    auto& lmdas = state.lmdas;
    auto& iters = state.iters;
    auto& benchmark_screen = state.benchmark_screen;
    auto& benchmark_active = state.benchmark_active;
    
    sw_t stopwatch;
    const auto n = X.rows();
    const auto p = X.cols();

    // buffers for the routine
    const auto max_group_size = group_sizes.maxCoeff();
    GaussianPinBufferPack<value_t> buffer_pack(
        max_group_size, 
        std::max<size_t>(3 * max_group_size, n)
    );
    
    // buffer to store final result
    std::vector<index_t> active_beta_indices;
    std::vector<value_t> active_beta_ordered;

    // allocate buffers for optimization
    active_beta_indices.reserve(screen_beta.size());
    active_beta_ordered.reserve(screen_beta.size());

    // compute number of active coefficients
    size_t active_beta_size = 0;
    if (active_set.size()) {
        const auto last_idx = active_set.size()-1;
        const auto last_group = screen_set[active_set[last_idx]];
        const auto group_size = group_sizes[last_group];
        active_beta_size = active_begins[last_idx] + group_size;
    }
    
    bool lasso_active_called = false;

    const auto add_active_set = [&](auto ss_idx) {
        if (!screen_is_active[ss_idx]) {
            if (active_set.size() >= max_active_size) {
                throw std::runtime_error("Maximum number of active groups reached.");
            }
            screen_is_active[ss_idx] = true;

            active_set.push_back(ss_idx);

            const auto group = screen_set[ss_idx];
            const auto group_size = group_sizes[group];
            if (group_size == 1) {
                active_g1.push_back(ss_idx);
            } else {
                active_g2.push_back(ss_idx);
            }
        }
    };

    const auto lasso_active_and_update = [&](size_t l) {
        solve_active(
            state, l, 
            buffer_pack.buffer1,
            buffer_pack.buffer2,
            buffer_pack.buffer3,
            buffer_pack.buffer4,
            update_coefficients_f,
            check_user_interrupt
        );
        lasso_active_called = true;
    };

    for (int l = 0; l < lmda_path.size(); ++l) {
        double screen_time = 0;
        double active_time = 0;

        if (lasso_active_called) {
            stopwatch.start();
            lasso_active_and_update(l);
            active_time += stopwatch.elapsed();
        }

        while (1) {
            check_user_interrupt();
            ++iters;
            value_t convg_measure;
            const auto old_active_size = active_set.size();
            stopwatch.start();
            coordinate_descent(
                state,
                screen_g1.data(), screen_g1.data() + screen_g1.size(),
                screen_g2.data(), screen_g2.data() + screen_g2.size(),
                l, convg_measure,
                buffer_pack.buffer1,
                buffer_pack.buffer2,
                buffer_pack.buffer3,
                buffer_pack.buffer4,
                update_coefficients_f,
                add_active_set
            );
            screen_time += stopwatch.elapsed();
            const bool new_active_added = (old_active_size < active_set.size());

            if (new_active_added) {
                active_begins.resize(active_set.size());
                for (size_t i = old_active_size; i < active_begins.size(); ++i) {
                    active_begins[i] = active_beta_size;
                    const auto curr_group = screen_set[active_set[i]];
                    const auto curr_size = group_sizes[curr_group];
                    active_beta_size += curr_size;
                }
            }

            if (convg_measure < tol) break;
            if (iters >= max_iters) throw util::max_cds_error(l);

            stopwatch.start();
            lasso_active_and_update(l);
            active_time += stopwatch.elapsed();
        }

        // update active_order
        const auto old_active_size = active_order.size();
        active_order.resize(active_set.size());
        std::iota(
            std::next(active_order.begin(), old_active_size), 
            active_order.end(), 
            old_active_size
        );
        std::sort(
            active_order.begin(), active_order.end(),
            [&](auto i, auto j) { 
                return groups[screen_set[active_set[i]]] < groups[screen_set[active_set[j]]];
            }
        );

        // order the active betas
        active_beta_indices.resize(active_beta_size);
        active_beta_ordered.resize(active_beta_size);
        sparsify_active_beta(
            state,
            active_beta_indices,
            active_beta_ordered
        );
        Eigen::Map<const sp_vec_value_t> beta_map(
            p,
            active_beta_indices.size(),
            active_beta_indices.data(),
            active_beta_ordered.data()
        );

        betas.emplace_back(beta_map);
        if (intercept) {
            intercepts.emplace_back(y_mean - (screen_X_means * screen_beta).sum());
        } else {
            intercepts.emplace_back(0);
        }
        rsqs.emplace_back(rsq);
        lmdas.emplace_back(lmda_path[l]);
        benchmark_screen.emplace_back(screen_time);
        benchmark_active.emplace_back(active_time);

        if (rsq >= adev_tol * y_var) break;
        if ((l >= 1) && (rsqs[l]-rsqs[l-1] <= ddev_tol * y_var)) break;
    }
}

} // namespace naive    
} // namespace pin
} // namespace gaussian
} // namespace solver
} // namespace adelie_core