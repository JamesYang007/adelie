#pragma once
#include <array>
#include <vector>
#include <numeric>
#include <ghostbasil/util/exceptions.hpp>
#include <ghostbasil/util/functional.hpp>
#include <ghostbasil/util/stopwatch.hpp>
#include <ghostbasil/util/functor_iterator.hpp>
#include <ghostbasil/util/counting_iterator.hpp>
#include <ghostbasil/optimization/group_lasso_base.hpp>
#include <ghostbasil/optimization/lasso.hpp>

namespace ghostbasil {
namespace group_lasso {
namespace naive {
    
struct GroupLassoDiagnostic
{
    using value_t = double;
    using dyn_vec_value_t = std::vector<value_t>;
    
    dyn_vec_value_t time_strong_cd;
    dyn_vec_value_t time_active_cd;
};

/**
 * Parameter pack for group lasso procedure.
 * 
 * @param   X           data matrix (n, p) with diagonal covariance blocks X_i^T X_i. 
 * @param   groups      vector of indices into columns of A that define
 *                      the beginning index of the groups.
 * @param   group_sizes vector of sizes of each group.
 *                      group_sizes[i] = size of group i.
 * @param   alpha       elastic net proportion. 
 *                      It is undefined behavior if alpha is not in [0,1].
 * @param   penalty     penalty factor for each group.
 * @param   strong_set  strong set as a dense vector of indices in [0, I),
 *                      where I is the total number of groups.
 *                      strong_set[i] = ith strong group.
 * @param   strong_g1   indices into strong_set that correspond to groups of size 1.
 * @param   strong_g2   indices into strong_set that correspond to groups of size > 1.
 * @param   strong_begins   vector of indices that define the beginning index to values
 *                          corresponding to the strong groups.
 *                          MUST have strong_begins.size() == strong_set.size().
 * @param   strong_A_diag       dense vector representing diagonal of A restricted to strong_set.
 *                              strong_A_diag[b:(b+p)] = diag(A_{kk})
 *                              where k := strong_set[i], b := strong_begins[i], p := group_sizes[k].
 * @param   lmdas       regularization parameter lambda sequence.
 * @param   max_cds     max number of coordinate descents.
 * @param   thr         convergence threshold.
 * @param   newton_tol  see update_coefficients argument tol.
 * @param   newton_max_iters     see update_coefficients argument max_iters.
 * @param   rsq         unnormalized R^2 estimate.
 *                      It is only well-defined if it is (approximately) 
 *                      the R^2 that corresponds to strong_beta.
 * @param   resid       residual vector y - X * beta where beta is all 0 on non-strong groups 
 *                      and the rest correspond to strong_beta.
 * @param   strong_beta dense vector of coefficients of size strong_A_diag.size().
 *                      strong_beta[b:(b+p)] = coefficient for group i,
 *                      where i := strong_set[j], b := strong_begins[j], and p := group_sizes[i].
 *                      The updated coefficients will be stored here.
 * @param   strong_grad dense vector gradients. It is simply a computation buffer.
 *                      Only needs to be at least the size of strong_beta.
 * @param   active_set  active set as a dense vector of indices in [0, strong_set.size()).
 *                      strong_set[active_set[i]] = ith active group.
 *                      This set must at least contain ALL indices into strong_set 
 *                      where the corresponding strong_beta is non-zero, that is,
 *                      if strong_beta[b:(b+p)] != 0, then i is in active_set,
 *                      where i := strong_set[j], b := strong_begins[j], and p := group_sizes[i].
 * @param   active_g1   indices into active_set that correspond to groups of size 1.
 * @param   active_g2   indices into active_set that correspond to groups of size > 1.
 * @param   active_begins   vector of indices that define the beginning index to values
 *                          corresponding to the active groups.
 *                          MUST have active_begins.size() == active_set.size().
 * @param   active_order    order of active_set that results in sorted (ascending) 
 *                          values of strong_set.
 *                          strong_set[active_set[active_order[i]]] < 
 *                              strong_set[active_set[active_order[j]]] if i < j.
 * @param   is_active   dense vector of bool of size strong_set.size(). 
 *                      is_active[i] = true if group strong_set[i] is active.
 *                      active_set should contain i.
 * @param   betas       vector of length lmdas.size() of 
 *                      output coefficient sparse vectors of size (p,).
 *                      betas[j](i) = ith coefficient for jth lambda
 *                      j < n_lmdas.
 * @param   rsqs        vector of length lmdas.size() 
 *                      to store (unnormalized) R^2 values for each lambda in lmdas
 *                      up to (non-including) index n_lmdas.
 * @param   n_cds       number of coordinate descents.
 * @param   n_lmdas     number of values in lmdas processed.
 * @param   cond_0_thresh   0th order threshold for early exit.
 * @param   cond_1_thresh   1st order threshold for early exit.
 * @param   diagnostic      instance of GroupLassoDiagnostic.
 */
template <class XType, 
          class ValueType,
          class IndexType,
          class BoolType,
          class DynamicVectorIndexType=std::vector<IndexType>,
          class DynamicVectorSpVecType=util::vec_type<
                util::sp_vec_type<ValueType, Eigen::ColMajor, IndexType>
            > 
          >
struct GroupLassoParamPack
{
    using value_t = ValueType;
    using index_t = IndexType;
    using bool_t = BoolType;
    using sp_vec_value_t = util::sp_vec_type<value_t, Eigen::ColMajor, index_t>;
    using vec_value_t = util::vec_type<value_t>;
    using vec_index_t = util::vec_type<index_t>;
    using map_vec_value_t = Eigen::Map<vec_value_t>;
    using map_vec_index_t = Eigen::Map<vec_index_t>;
    using map_vec_bool_t = Eigen::Map<util::vec_type<bool_t>>;
    using map_mat_value_t = Eigen::Map<util::mat_type<value_t>>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;
    using map_cvec_index_t = Eigen::Map<const vec_index_t>;
    using dyn_vec_index_t = DynamicVectorIndexType;
    using dyn_vec_sp_vec_t = DynamicVectorSpVecType;
    using diagnostic_t = GroupLassoDiagnostic;

    const XType* X = nullptr;
    map_cvec_index_t groups;
    map_cvec_index_t group_sizes;
    value_t alpha;
    map_cvec_value_t penalty;
    map_cvec_index_t strong_set;
    map_cvec_index_t strong_g1;
    map_cvec_index_t strong_g2;
    map_cvec_index_t strong_begins;
    map_cvec_value_t strong_A_diag;
    map_cvec_value_t lmdas;
    size_t max_cds;
    value_t thr;
    value_t cond_0_thresh;
    value_t cond_1_thresh;
    value_t newton_tol;
    size_t newton_max_iters;
    value_t rsq;
    map_vec_value_t resid;
    map_vec_value_t strong_beta;
    map_vec_value_t strong_grad;
    dyn_vec_index_t* active_set = nullptr;
    dyn_vec_index_t* active_g1 = nullptr;
    dyn_vec_index_t* active_g2 = nullptr;
    dyn_vec_index_t* active_begins = nullptr;
    dyn_vec_index_t* active_order = nullptr;
    map_vec_bool_t is_active;
    dyn_vec_sp_vec_t* betas = nullptr;
    map_vec_value_t rsqs;
    map_mat_value_t resids;
    size_t n_cds;
    size_t n_lmdas;
    diagnostic_t diagnostic;
    
    explicit GroupLassoParamPack()
        : groups(nullptr, 0),
          group_sizes(nullptr, 0),
          penalty(nullptr, 0),
          strong_set(nullptr, 0),
          strong_g1(nullptr, 0),
          strong_g2(nullptr, 0),
          strong_begins(nullptr, 0),
          strong_A_diag(nullptr, 0),
          lmdas(nullptr, 0),
          resid(nullptr, 0),
          strong_beta(nullptr, 0),
          strong_grad(nullptr, 0),
          is_active(nullptr, 0),
          rsqs(nullptr, 0),
          resids(nullptr, 0)
    {}
         
    template <class GroupsType, class GroupSizesType,
              class PenaltyType, class SSType, class SG1Type,
              class SG2Type, class SBeginsType, 
              class SADType, class LmdasType,
              class ResidType, class SBType, class SGType,
              class IAType, class RsqsType, class ResidsType>
    explicit GroupLassoParamPack(
        const XType& X_,
        const GroupsType& groups_, 
        const GroupSizesType& group_sizes_,
        value_t alpha_, 
        const PenaltyType& penalty_,
        const SSType& strong_set_, 
        const SG1Type& strong_g1_,
        const SG2Type& strong_g2_,
        const SBeginsType& strong_begins_, 
        const SADType& strong_A_diag_,
        const LmdasType& lmdas_, 
        size_t max_cds_,
        value_t thr_,
        value_t cond_0_thresh,
        value_t cond_1_thresh,
        value_t newton_tol_,
        size_t newton_max_iters_,
        value_t rsq_,
        ResidType& resid_,
        SBType& strong_beta_, 
        SGType& strong_grad_,
        dyn_vec_index_t& active_set_,
        dyn_vec_index_t& active_g1_,
        dyn_vec_index_t& active_g2_,
        dyn_vec_index_t& active_begins_,
        dyn_vec_index_t& active_order_,
        IAType& is_active_,
        dyn_vec_sp_vec_t& betas_, 
        RsqsType& rsqs_,
        ResidsType& resids_,
        size_t n_cds_,
        size_t n_lmdas_
    )
        : X(&X_),
          groups(groups_.data(), groups_.size()),
          group_sizes(group_sizes_.data(), group_sizes_.size()),
          alpha(alpha_),
          penalty(penalty_.data(), penalty_.size()),
          strong_set(strong_set_.data(), strong_set_.size()),
          strong_g1(strong_g1_.data(), strong_g1_.size()),
          strong_g2(strong_g2_.data(), strong_g2_.size()),
          strong_begins(strong_begins_.data(), strong_begins_.size()),
          strong_A_diag(strong_A_diag_.data(), strong_A_diag_.size()),
          lmdas(lmdas_.data(), lmdas_.size()),
          max_cds(max_cds_),
          thr(thr_),
          cond_0_thresh(cond_0_thresh),
          cond_1_thresh(cond_1_thresh),
          newton_tol(newton_tol_),
          newton_max_iters(newton_max_iters_),
          rsq(rsq_),
          resid(resid_.data(), resid_.size()),
          strong_beta(strong_beta_.data(), strong_beta_.size()),
          strong_grad(strong_grad_.data(), strong_grad_.size()),
          active_set(&active_set_),
          active_g1(&active_g1_),
          active_g2(&active_g2_),
          active_begins(&active_begins_),
          active_order(&active_order_),
          is_active(is_active_.data(), is_active_.size()),
          betas(&betas_),
          rsqs(rsqs_.data(), rsqs_.size()),
          resids(resids_.data(), resids_.rows(), resids_.cols()),
          n_cds(n_cds_),
          n_lmdas(n_lmdas_)
    {}
};

/**
 * One blockwise coordinate descent loop to solve the objective.
 *  
 * @param   pack            see GroupLassoParamPack.
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
 * @param   additional_step     any functor to run at the end of each loop given current looping value. 
 */
template <class PackType, class G1Iter, class G2Iter,
          class ValueType, class BufferType,
          class UpdateCoefficientsType,
          class AdditionalStepType=util::no_op>
GHOSTBASIL_STRONG_INLINE
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
    const auto& X = *pack.X;
    const auto& penalty = pack.penalty;
    const auto& strong_set = pack.strong_set;
    const auto& strong_begins = pack.strong_begins;
    const auto& strong_A_diag = pack.strong_A_diag;
    const auto& groups = pack.groups;
    const auto& group_sizes = pack.group_sizes;
    const auto alpha = pack.alpha;
    const auto lmda = pack.lmdas[lmda_idx];
    const auto l1 = lmda * alpha;
    const auto l2 = lmda * (1-alpha);
    const auto newton_tol = pack.newton_tol;
    const auto newton_max_iters = pack.newton_max_iters;
    auto& strong_beta = pack.strong_beta;
    auto& strong_grad = pack.strong_grad;
    auto& resid = pack.resid;
    auto& rsq = pack.rsq;

    convg_measure = 0;
    // iterate over the groups of size 1
    for (auto it = g1_begin; it != g1_end; ++it) {
        const auto ss_idx = *it;              // index to strong set
        const auto k = strong_set[ss_idx];    // actual group index
        const auto ss_value_begin = strong_begins[ss_idx]; // value begin index at ss_idx
        auto& ak = strong_beta[ss_value_begin]; // corresponding beta
        auto& gk = strong_grad[ss_value_begin]; // corresponding residuals
        const auto A_kk = strong_A_diag[ss_value_begin];  // corresponding A diagonal 
        const auto pk = penalty[k];

        gk = X.col(groups[k]).dot(resid);
        const auto ak_old = ak;

        // update coefficient
        lasso::update_coefficient(
            ak, A_kk, l1, l2, pk, gk
        );

        if (ak_old == ak) continue;

        // Note: order matters! Purposely put here so that A gets updated properly if iterating over strong set.
        // See fit() with active set update.
        additional_step(ss_idx);

        auto del = ak - ak_old;

        // update measure of convergence
        lasso::update_convergence_measure(convg_measure, del, A_kk);

        lasso::update_rsq(rsq, ak_old, ak, A_kk, gk);

        // update residual 
        resid -= del * X.col(groups[k]);
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
        const auto pk = penalty[k];

        gk.noalias() = X.block(0, groups[k], X.rows(), gsize).transpose() * resid;

        // save old beta in buffer
        auto ak_old = buffer3.head(ak.size());
        ak_old = ak; 

        // update group coefficients
        size_t iters;
        gk += A_kk.cwiseProduct(ak_old); 
        update_coefficients_f(
            A_kk, gk, l1 * pk, l2 * pk, 
            newton_tol, newton_max_iters,  // try only 1 newton iteration
            ak, iters, buffer1, buffer2
        );

        if ((ak_old.array() == ak.array()).all()) continue;
        
        gk -= A_kk.cwiseProduct(ak_old); 

        // Note: order matters! Purposely put here so that A gets updated properly if iterating over strong set.
        // See fit() with active set update.
        additional_step(ss_idx);

        // use same buffer as ak_old to store difference
        auto& del = ak_old;
        del = ak - ak_old;

        // update measure of convergence
        update_convergence_measure(convg_measure, del, A_kk);

        // update rsq
        // NOTE: MUST undo the correction from before
        update_rsq(rsq, del, A_kk, gk);

        // update residual
        resid -= X.block(0, groups[k], X.rows(), gsize) * del;        
    }
}

/**
 * Applies multiple blockwise coordinate descent on the active set 
 * to minimize the ghostbasil objective with group-lasso penalty.
 * See "objective" function for the objective of interest.
 *
 * @param   pack        see GroupLassoParamPack.
 * @param   lmda_idx    index into the lambda sequence for logging purposes.
 * @param   buffer1     see coordinate_descent.
 * @param   buffer2     see coordinate_descent.
 * @param   buffer3     see coordinate_descent.
 * @param   check_user_interrupt    functor that checks for user interruption.
 */
template <class PackType, 
          class BufferType, 
          class UpdateCoefficientsType,
          class CUIType = util::no_op>
GHOSTBASIL_STRONG_INLINE
void group_lasso_active(
    PackType&& pack,
    size_t lmda_idx,
    BufferType& buffer1,
    BufferType& buffer2,
    BufferType& buffer3,
    UpdateCoefficientsType update_coefficients_f,
    CUIType check_user_interrupt = CUIType())
{
    using pack_t = std::decay_t<PackType>;
    using value_t = typename pack_t::value_t;
    using sw_t = util::Stopwatch;

    const auto& active_set = *pack.active_set;
    const auto& active_g1 = *pack.active_g1;
    const auto& active_g2 = *pack.active_g2;
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
 * Minimizes the objective described in the function objective()
 * with the additional constraint:
 * \f[
 *      x_{i} = 0, \, \forall i \notin S
 * \f]
 * i.e. all betas not in the strong set \f$S\f$ are fixed to be 0.
 * 
 * @param   pack                    see GroupLassoParamPack.
 * @param   check_user_interrupt    see group_lasso_active.
 */
template <class PackType,
          class UpdateCoefficientsType,
          class CUIType = util::no_op>
inline void fit(
    PackType&& pack,
    UpdateCoefficientsType update_coefficients_f,
    CUIType check_user_interrupt = CUIType())
{
    using pack_t = std::decay_t<PackType>;
    using value_t = typename pack_t::value_t;
    using index_t = typename pack_t::index_t;
    using sp_vec_value_t = typename pack_t::sp_vec_value_t;
    using sw_t = util::Stopwatch;

    const auto& X = *pack.X;
    const auto& groups = pack.groups;
    const auto& group_sizes = pack.group_sizes;
    const auto& strong_set = pack.strong_set;
    const auto& strong_g1 = pack.strong_g1;
    const auto& strong_g2 = pack.strong_g2;
    const auto& strong_beta = pack.strong_beta;
    const auto& lmdas = pack.lmdas;
    const auto thr = pack.thr;
    const auto max_cds = pack.max_cds;
    const auto cond_0_thresh = pack.cond_0_thresh;
    const auto cond_1_thresh = pack.cond_1_thresh;
    auto& active_set = *pack.active_set;
    auto& active_g1 = *pack.active_g1;
    auto& active_g2 = *pack.active_g2;
    auto& active_begins = *pack.active_begins;
    auto& active_order = *pack.active_order;
    auto& is_active = pack.is_active;
    auto& betas = *pack.betas;
    auto& rsqs = pack.rsqs;
    auto& resids = pack.resids;
    auto& rsq = pack.rsq;
    auto& n_cds = pack.n_cds;
    auto& n_lmdas = pack.n_lmdas;
    auto& diagnostic = pack.diagnostic;
    
    const auto n = X.rows();
    const auto p = X.cols();

    assert(betas.size() == lmdas.size());
    assert(rsqs.size() == lmdas.size());
    assert(resids.rows() == n);
    assert(resids.cols() == lmdas.size());

    // buffers for the routine
    const auto max_group_size = group_sizes.maxCoeff();
    GroupLassoBufferPack<value_t> buffer_pack(max_group_size);
    
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
    n_cds = 0;
    n_lmdas = 0;

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
        group_lasso_active(
                pack, l, 
                buffer_pack.buffer1,
                buffer_pack.buffer2,
                buffer_pack.buffer3,
                check_user_interrupt);
        lasso_active_called = true;
    };

    for (size_t l = 0; l < lmdas.size(); ++l) {
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
                        add_active_set);
            }
            const bool new_active_added = (old_active_size < active_set.size());

            if (new_active_added) {
                // update active_begins
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

        // update active_beta_indices
        active_beta_indices.resize(active_beta_size);
        get_active_indices(pack, active_beta_indices);
        
        // update active_beta_ordered
        active_beta_ordered.resize(active_beta_size);
        get_active_values(pack, active_beta_ordered);

        // order the strong betas 
        Eigen::Map<const sp_vec_value_t> beta_map(
                p,
                active_beta_indices.size(),
                active_beta_indices.data(),
                active_beta_ordered.data());

        betas[l] = beta_map;
        rsqs[l] = rsq;
        resids.col(l) = pack.resid;
        ++n_lmdas;

        // make sure to do at least 3 lambdas.
        if (l < 2) continue;

        // early stop if R^2 criterion is fulfilled.
        if (lasso::check_early_stop_rsq(rsqs[l-2], rsqs[l-1], rsqs[l], cond_0_thresh, cond_1_thresh)) break;
    }
}

} // namespace naive
} // namespace group_lasso
} // namespace ghostbasil