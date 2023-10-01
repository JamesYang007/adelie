#pragma once
#include <array>
#include <vector>
#include <numeric>
#include <adelie_core/util/exceptions.hpp>
#include <adelie_core/util/functional.hpp>
#include <adelie_core/util/stopwatch.hpp>
#include <adelie_core/util/functor_iterator.hpp>
#include <adelie_core/util/counting_iterator.hpp>
#include <adelie_core/util/eigen/map_sparsevector.hpp>
#include <adelie_core/optimization/group_elnet_base.hpp>

namespace adelie_core {
namespace naive {
    
/**
 * One blockwise coordinate descent loop to solve the objective.
 *  
 * @param   pack            see GroupElnetState.
 * @param   g1_begin        begin iterator to indices into strong set of group type 1, i.e.
 *                          strong_set[*begin] is the current group to descend.
 * @param   g1_end          end iterator to indices into strong set of group type 1.
 * @param   g2_begin        begin iterator to indices into strong set of group type 2, i.e.
 *                          strong_set[*begin] is the current group to descend.
 * @param   g2_end          end iterator to indices into strong set of group type 2.
 * @param   lmda_idx        index into lambda sequence.
 * @param   convg_measure   stores the convergence measure of the call.
 * @param   buffer1         see update_coefficient.
 * @param   buffer2         see update_coefficient.
 * @param   buffer3         any vector of size larger than the largest strong set group size.
 * @param   update_coefficients_f  any functor that updates the coefficient for group 2.
 * @param   additional_step     any functor to run at the end of each loop given current looping value. 
 */
template <class PackType, class G1Iter, class G2Iter,
          class ValueType, class BufferType,
          class UpdateCoefficientsType,
          class AdditionalStepType=util::no_op>
ADELIE_CORE_STRONG_INLINE
void coordinate_descent(
    PackType&& pack,
    G1Iter g1_begin,
    G1Iter g1_end,
    G2Iter g2_begin,
    G2Iter g2_end,
    size_t lmda_idx,
    ValueType& convg_measure,
    BufferType& buffer1,
    BufferType& buffer2,
    BufferType& buffer3,
    UpdateCoefficientsType update_coefficients_f,
    AdditionalStepType additional_step=AdditionalStepType()
)
{
    const auto& X = pack.X;
    const auto& penalty = pack.penalty;
    const auto& strong_set = pack.strong_set;
    const auto& strong_begins = pack.strong_begins;
    const auto& strong_A_diag = pack.strong_A_diag;
    const auto& groups = pack.groups;
    const auto& group_sizes = pack.group_sizes;
    const auto alpha = pack.alpha;
    const auto lmda = pack.lmdas[lmda_idx];
    const auto newton_tol = pack.newton_tol;
    const auto newton_max_iters = pack.newton_max_iters;
    auto& strong_beta = pack.strong_beta;
    auto& strong_grad = pack.strong_grad;
    auto& resid = pack.resid;
    auto& rsq = pack.rsq;

    const auto l1 = lmda * alpha;
    const auto l2 = lmda * (1-alpha);

    convg_measure = 0;
    // iterate over the groups of size 1
    for (auto it = g1_begin; it != g1_end; ++it) {
        const auto ss_idx = *it;              // index to strong set
        const auto k = strong_set[ss_idx];    // actual group index
        const auto ss_value_begin = strong_begins[ss_idx]; // value begin index at ss_idx
        auto& ak = strong_beta[ss_value_begin]; // corresponding beta
        auto& gk = strong_grad[ss_value_begin]; // corresponding residuals
        const auto A_kk = strong_A_diag[ss_value_begin];  // corresponding A diagonal 
        const auto pk = penalty[k]; // corresponding penalty

        const auto Xk = X.col(groups[k]);

        gk = Xk.dot(resid.matrix());
        const auto ak_old = ak;

        // update coefficient
        update_coefficient(
            ak, A_kk, l1, l2, pk, gk
        );

        if (ak_old == ak) continue;

        additional_step(ss_idx);

        auto del = ak - ak_old;

        // update measure of convergence
        update_convergence_measure(convg_measure, del, A_kk);

        update_rsq(rsq, ak_old, ak, A_kk, gk);

        // update residual 
        resid.matrix() -= del * Xk;
    }
    
    // iterate over the groups of dynamic size
    for (auto it = g2_begin; it != g2_end; ++it) {
        const auto ss_idx = *it;              // index to strong set
        const auto k = strong_set[ss_idx];    // actual group index
        const auto ss_value_begin = strong_begins[ss_idx]; // value begin index at ss_idx
        const auto gsize = group_sizes[k]; // group size  
        auto ak = strong_beta.segment(ss_value_begin, gsize); // corresponding beta
        auto gk = strong_grad.segment(ss_value_begin, gsize); // corresponding residuals
        const auto A_kk = strong_A_diag.segment(ss_value_begin, gsize);  // corresponding A diagonal 
        const auto pk = penalty[k]; // corresponding penalty

        const auto Xk = X.block(0, groups[k], X.rows(), gsize);

        // TODO: parallelize
        gk.matrix().noalias() = Xk.transpose() * resid.matrix();

        // save old beta in buffer
        auto ak_old = buffer3.head(ak.size());
        ak_old = ak; 

        // update group coefficients
        size_t iters;
        gk += A_kk * ak_old; 
        update_coefficients_f(
            A_kk, gk, l1 * pk, l2 * pk, 
            newton_tol, newton_max_iters,
            ak, iters, buffer1, buffer2
        );

        if ((ak_old - ak).abs().maxCoeff() <= 1e-14) continue;
        
        // NOTE: MUST undo the correction from before
        gk -= A_kk * ak_old; 

        additional_step(ss_idx);

        // use same buffer as ak_old to store difference
        auto& del = ak_old;
        del = ak - ak_old;

        // update measure of convergence
        update_convergence_measure(convg_measure, del, A_kk);

        // update rsq
        update_rsq(rsq, del, A_kk, gk);

        // update residual
        // TODO: dot-product may allocate new array!!
        // TODO: parallelize
        resid.matrix() -= Xk * del.matrix(); 
    }
}

/**
 * Applies multiple blockwise coordinate descent on the active set 
 * to minimize the adelie_core objective with group-lasso penalty.
 * See "objective" function for the objective of interest.
 *
 * @param   pack        see GroupElnetState.
 * @param   lmda_idx    index into the lambda sequence for logging purposes.
 * @param   buffer1     see coordinate_descent.
 * @param   buffer2     see coordinate_descent.
 * @param   buffer3     see coordinate_descent.
 * @param   update_coefficients_f  any functor that updates the coefficient for group 2.
 * @param   check_user_interrupt    functor that checks for user interruption.
 */
template <class PackType, 
          class BufferType, 
          class UpdateCoefficientsType,
          class CUIType = util::no_op>
ADELIE_CORE_STRONG_INLINE
void fit_active(
    PackType&& pack,
    size_t lmda_idx,
    BufferType& buffer1,
    BufferType& buffer2,
    BufferType& buffer3,
    UpdateCoefficientsType update_coefficients_f,
    CUIType check_user_interrupt = CUIType()
)
{
    using pack_t = std::decay_t<PackType>;
    using value_t = typename pack_t::value_t;
    using sw_t = util::Stopwatch;

    const auto& active_set = pack.active_set;
    const auto& active_g1 = pack.active_g1;
    const auto& active_g2 = pack.active_g2;
    const auto thr = pack.thr;
    const auto max_cds = pack.max_cds;
    auto& n_cds = pack.n_cds;
    auto& diagnostic = pack.diagnostic;

    const auto g1_active_f = [&](auto i) {
        return active_set[active_g1[i]];
    };
    const auto g2_active_f = [&](auto i) {
        return active_set[active_g2[i]];
    };
    const auto ag1_begin = util::make_functor_iterator<size_t>(0, g1_active_f);
    const auto ag1_end = util::make_functor_iterator<size_t>(active_g1.size(), g1_active_f);
    const auto ag2_begin = util::make_functor_iterator<size_t>(0, g2_active_f);
    const auto ag2_end = util::make_functor_iterator<size_t>(active_g2.size(), g2_active_f);
    
    diagnostic.time_active_cd.push_back(0);
    {
        sw_t stopwatch(diagnostic.time_active_cd.back());
        while (1) {
            check_user_interrupt(n_cds);
            ++n_cds;
            value_t convg_measure;
            coordinate_descent(
                pack, ag1_begin, ag1_end, ag2_begin, ag2_end,
                lmda_idx, convg_measure, buffer1, buffer2, buffer3, 
                update_coefficients_f
            );
            if (convg_measure < thr) break;
            if (n_cds >= max_cds) throw util::max_cds_error(lmda_idx);
        }
    }
}

/**
 * Minimizes the objective described in the function group_elnet_objective()
 * with the additional constraint:
 * \f[
 *      x_{i} = 0, \, \forall i \notin S
 * \f]
 * i.e. all betas not in the strong set \f$S\f$ are fixed to be 0.
 * 
 * @param   pack                    see GroupElnetState.
 * @param   update_coefficients_f   see fit_active
 * @param   check_user_interrupt    see fit_active.
 */
template <class PackType,
          class UpdateCoefficientsType,
          class CUIType = util::no_op>
inline void fit(
    PackType&& pack,
    UpdateCoefficientsType update_coefficients_f,
    CUIType check_user_interrupt = CUIType()
)
{
    using pack_t = std::decay_t<PackType>;
    using value_t = typename pack_t::value_t;
    using index_t = typename pack_t::index_t;
    using sp_vec_value_t = typename pack_t::sp_vec_value_t;
    using sw_t = util::Stopwatch;

    const auto& X = pack.X;
    const auto& group_sizes = pack.group_sizes;
    const auto& strong_set = pack.strong_set;
    const auto& strong_g1 = pack.strong_g1;
    const auto& strong_g2 = pack.strong_g2;
    const auto& strong_beta = pack.strong_beta;
    const auto& lmdas = pack.lmdas;
    const auto& resid = pack.resid;
    const auto thr = pack.thr;
    const auto max_cds = pack.max_cds;
    const auto cond_0_thresh = pack.cond_0_thresh;
    const auto cond_1_thresh = pack.cond_1_thresh;
    auto& active_set = pack.active_set;
    auto& active_g1 = pack.active_g1;
    auto& active_g2 = pack.active_g2;
    auto& active_begins = pack.active_begins;
    auto& active_order = pack.active_order;
    auto& is_active = pack.is_active;
    auto& betas = pack.betas;
    auto& rsqs = pack.rsqs;
    auto& resids = pack.resids;
    auto& rsq = pack.rsq;
    auto& n_cds = pack.n_cds;
    auto& diagnostic = pack.diagnostic;
    
    const auto p = X.cols();

    assert(betas.size() == lmdas.size());
    assert(rsqs.size() == lmdas.size());

    // buffers for the routine
    const auto max_group_size = group_sizes.maxCoeff();
    GroupElnetBufferPack<value_t> buffer_pack(max_group_size);
    
    // buffer to store final result
    std::vector<index_t> active_beta_indices;
    std::vector<value_t> active_beta_ordered;

    // allocate buffers with optimization
    active_beta_indices.reserve(strong_beta.size());
    active_beta_ordered.reserve(strong_beta.size());

    // compute number of active coefficients
    size_t active_beta_size = 0;
    if (active_set.size()) {
        const auto last_idx = active_set.size()-1;
        const auto last_group = strong_set[active_set[last_idx]];
        const auto group_size = group_sizes[last_group];
        active_beta_size = active_begins[last_idx] + group_size;
    }
    
    bool lasso_active_called = false;

    const auto add_active_set = [&](auto ss_idx) {
        if (!is_active[ss_idx]) {
            is_active[ss_idx] = true;

            const auto next_idx = active_set.size();
            active_set.push_back(ss_idx);

            const auto group = strong_set[ss_idx];
            const auto group_size = group_sizes[group];
            if (group_size == 1) {
                active_g1.push_back(next_idx);
            } else {
                active_g2.push_back(next_idx);
            }
        }
    };

    const auto lasso_active_and_update = [&](size_t l) {
        fit_active(
            pack, l, 
            buffer_pack.buffer1,
            buffer_pack.buffer2,
            buffer_pack.buffer3,
            update_coefficients_f,
            check_user_interrupt
        );
        lasso_active_called = true;
    };

    for (int l = 0; l < lmdas.size(); ++l) {
        if (lasso_active_called) {
            lasso_active_and_update(l);
        }

        while (1) {
            check_user_interrupt(n_cds);
            ++n_cds;
            value_t convg_measure;
            const auto old_active_size = active_set.size();
            diagnostic.time_strong_cd.push_back(0);
            {
                sw_t stopwatch(diagnostic.time_strong_cd.back());
                coordinate_descent(
                    pack,
                    strong_g1.data(), strong_g1.data() + strong_g1.size(),
                    strong_g2.data(), strong_g2.data() + strong_g2.size(),
                    l, convg_measure,
                    buffer_pack.buffer1,
                    buffer_pack.buffer2,
                    buffer_pack.buffer3,
                    update_coefficients_f,
                    add_active_set
                );
            }
            const bool new_active_added = (old_active_size < active_set.size());

            if (new_active_added) {
                active_begins.resize(active_set.size());
                for (size_t i = old_active_size; i < active_begins.size(); ++i) {
                    active_begins[i] = active_beta_size;
                    const auto curr_group = strong_set[active_set[i]];
                    const auto curr_size = group_sizes[curr_group];
                    active_beta_size += curr_size;
                }
            }

            if (convg_measure < thr) break;
            if (n_cds >= max_cds) throw util::max_cds_error(l);

            lasso_active_and_update(l);
        }

        // update active_order
        const auto old_active_size = active_order.size();
        active_order.resize(active_set.size());
        std::iota(std::next(active_order.begin(), old_active_size), 
                  active_order.end(), 
                  old_active_size);
        std::sort(active_order.begin(), active_order.end(),
                  [&](auto i, auto j) { 
                        return strong_set[active_set[i]] < strong_set[active_set[j]];
                    });

        // order the active betas
        sparsify_active_beta(
            pack,
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
        rsqs.emplace_back(rsq);
        resids.emplace_back(resid);

        // make sure to do at least 3 lambdas.
        if (l < 2) continue;

        // early stop if R^2 criterion is fulfilled.
        if (check_early_stop_rsq(rsqs[l-2], rsqs[l-1], rsqs[l], cond_0_thresh, cond_1_thresh)) break;
    }
}

} // namespace naive
} // namespace adelie_core