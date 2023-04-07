#pragma once
#include <numeric>
#include <ghostbasil/util/exceptions.hpp>
#include <ghostbasil/util/functional.hpp>
#include <ghostbasil/util/macros.hpp>
#include <ghostbasil/util/types.hpp>
#include <ghostbasil/util/functor_iterator.hpp>
#include <ghostbasil/util/eigen/map_sparsevector.hpp>
#include <ghostbasil/matrix/forward_decl.hpp>

namespace ghostbasil {
namespace lasso {

/**
 * Parameter pack for lasso procedure.
 * 
 * @param   A           PSD matrix (p, p). 
 *                      This matrix only needs to satisfy the properties
 *                      when looking at the sub-matrix of all strong_set features.
 * @param   alpha       elastic net proportion. 
 *                      It is undefined behavior if alpha is not in [0,1].
 * @param   penalty     penalty factor for each feature.
 * @param   strong_set  strong set as a dense vector of indices in [0, p).
 *                      strong_set[i] = ith strong feature.
 * @param   strong_order    order of strong_set that results in sorted (ascending) values.
 *                          strong_set[strong_order[i]] < strong_set[strong_order[j]] if i < j.
 * @param   strong_A_diag       dense vector representing the diagonal of A restricted to strong_set.
 *                              strong_A_diag[i] = A(i,i) where k := strong_set[i].
 * @param   lmdas       regularization parameter lambda sequence (recommended to be decreasing).
 * @param   max_cds     max number of coordinate descents.
 * @param   thr         convergence threshold.
 * @param   rsq         unnormalized difference in R^2 estimate from current state to the end of lasso.
 * @param   strong_beta dense vector of coefficients of size strong_A_diag.size().
 *                      strong_beta[j] = coefficient for feature i,
 *                      where i := strong_set[j].
 *                      The updated coefficients will be stored here.
 * @param   strong_grad dense vector of gradient estimates.
 *                      strong_grad[j] = residuals for group i where i := strong_set[j].
 *                      The updated gradients will be stored here.
 * @param   active_set  active set as a dense vector of indices in [0, strong_set.size()).
 *                      strong_set[active_set[i]] = ith active feature.
 *                      This set must at least contain ALL indices into strong_set 
 *                      where the corresponding strong_beta is non-zero, that is,
 *                      if strong_beta[strong_set[i]] != 0, then i is in active_set.
 * @param   active_order    order of active_set that results in sorted (ascending) 
 *                          values of strong_set.
 *                          strong_set[active_set[active_order[i]]] < 
 *                              strong_set[active_set[active_order[j]]] if i < j.
 * @param   active_set_ordered  ordered *features corresponding to* active_set.
 *                              active_set_ordered[i] == strong_set[active_set[active_order[i]]].
 * @param   is_active   dense vector of bool of size strong_set.size(). 
 *                      is_active[i] = true if feature strong_set[i] is active.
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
 */
template <class AType, 
          class ValueType,
          class IndexType,
          class BoolType,
          class DynamicVectorIndexType=std::vector<IndexType>
          >
struct LassoParamPack
{
    using value_t = ValueType;
    using index_t = IndexType;
    using bool_t = BoolType;
    using sp_vec_value_t = util::sp_vec_type<value_t, Eigen::ColMajor, index_t>;
    using vec_value_t = util::vec_type<value_t>;
    using vec_index_t = util::vec_type<index_t>;
    using map_vec_value_t = Eigen::Map<vec_value_t>;
    using map_vec_bool_t = Eigen::Map<util::vec_type<bool_t>>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;
    using map_cvec_index_t = Eigen::Map<const vec_index_t>;
    using map_vec_sp_vec_value_t = Eigen::Map<util::vec_type<sp_vec_value_t>>;
    using dyn_vec_index_t = DynamicVectorIndexType;

    const AType* A = nullptr;
    value_t alpha;
    map_cvec_value_t penalty;
    map_cvec_index_t strong_set;
    map_cvec_index_t strong_order;
    map_cvec_value_t strong_A_diag;
    map_cvec_value_t lmdas;
    size_t max_cds;
    value_t thr;
    value_t rsq;
    map_vec_value_t strong_beta;
    map_vec_value_t strong_grad;
    dyn_vec_index_t* active_set = nullptr;
    dyn_vec_index_t* active_order = nullptr;
    dyn_vec_index_t* active_set_ordered = nullptr;
    map_vec_bool_t is_active;
    map_vec_sp_vec_value_t betas;
    map_vec_value_t rsqs;
    size_t n_cds;
    size_t n_lmdas;
    bool do_early_stop = true;
    
    explicit LassoParamPack()
        : penalty(nullptr, 0),
          strong_set(nullptr, 0),
          strong_order(nullptr, 0),
          strong_A_diag(nullptr, 0),
          lmdas(nullptr, 0),
          strong_beta(nullptr, 0),
          strong_grad(nullptr, 0),
          is_active(nullptr, 0),
          betas(nullptr, 0),
          rsqs(nullptr, 0)
    {}
         
    template <class PenaltyType, class SSType, class SOType, 
              class SADType, class LmdasType,
              class SBType, class SGType,
              class IAType, class BetasType, class RsqsType>
    explicit LassoParamPack(
        const AType& A_,
        value_t alpha_, 
        const PenaltyType& penalty_,
        const SSType& strong_set_, 
        const SOType& strong_order_, 
        const SADType& strong_A_diag_,
        const LmdasType& lmdas_, 
        size_t max_cds_,
        value_t thr_,
        value_t rsq_,
        SBType& strong_beta_, 
        SGType& strong_grad_,
        dyn_vec_index_t& active_set_,
        dyn_vec_index_t& active_order_,
        dyn_vec_index_t& active_set_ordered_,
        IAType& is_active_,
        BetasType& betas_, 
        RsqsType& rsqs_,
        size_t n_cds_,
        size_t n_lmdas_,
        bool do_early_stop_ = true
    )
        : A(&A_),
          alpha(alpha_),
          penalty(penalty_.data(), penalty_.size()),
          strong_set(strong_set_.data(), strong_set_.size()),
          strong_order(strong_order_.data(), strong_order_.size()),
          strong_A_diag(strong_A_diag_.data(), strong_A_diag_.size()),
          lmdas(lmdas_.data(), lmdas_.size()),
          max_cds(max_cds_),
          thr(thr_),
          rsq(rsq_),
          strong_beta(strong_beta_.data(), strong_beta_.size()),
          strong_grad(strong_grad_.data(), strong_grad_.size()),
          active_set(&active_set_),
          active_order(&active_order_),
          active_set_ordered(&active_set_ordered_),
          is_active(is_active_.data(), is_active_.size()),
          betas(betas_.data(), betas_.size()),
          rsqs(rsqs_.data(), rsqs_.size()),
          n_cds(n_cds_),
          n_lmdas(n_lmdas_),
          do_early_stop(do_early_stop_)
    {}
};

namespace internal {

template <class PackType>
GHOSTBASIL_STRONG_INLINE
void lasso_assert_valid_inputs(const PackType& pack)
{
#ifndef NDEBUG
    using pack_t = std::decay_t<PackType>;
    using vec_value_t = typename pack_t::vec_value_t;
    using vec_index_t = typename pack_t::vec_index_t;

    const auto& A = *pack.A;
    const auto alpha = pack.alpha;
    const auto& strong_set = pack.strong_set;
    const auto& strong_order = pack.strong_order;
    const auto& active_set = *pack.active_set;
    const auto& active_order = *pack.active_order;
    const auto& active_set_ordered = *pack.active_set_ordered;
    const auto& is_active = pack.is_active;
    const auto& strong_A_diag = pack.strong_A_diag;
    const auto& strong_beta = pack.strong_beta;
    const auto& strong_grad = pack.strong_grad;

    // check that A is square
    assert((A.rows() == A.cols()) && A.size());

    // check that alpha is in [0,1]
    assert((0 <= alpha) && (alpha <= 1));

    {
        // check that strong set contains values in [0, p)
        Eigen::Map<const vec_index_t> ss_view(
                strong_set.data(),
                strong_set.size());
        assert(!ss_view.size() || (0 <= ss_view.minCoeff() && ss_view.maxCoeff() < A.cols()));

        // check that strong order results in sorted strong set
        vec_index_t ss_copy = ss_view;
        std::sort(ss_copy.data(), ss_copy.data()+ss_copy.size());
        assert(strong_order.size() == strong_set.size());
        assert((ss_copy.array() == 
                    vec_index_t::NullaryExpr(ss_view.size(),
                        [&](auto i) { return ss_view[strong_order[i]]; }
                        ).array()).all());
    }

    {
        // check that strong_A_diag satisfies the conditions.
        assert(strong_A_diag.size() == strong_set.size());
        for (size_t i = 0; i < strong_A_diag.size(); ++i) {
            auto k = strong_set[i];
            auto A_kk = A.coeff(k, k);
            assert(strong_A_diag[i] == A_kk);
        }
    }

    {
        // check that active set is of right size and contains values in [0, strong_set.size()).
        Eigen::Map<const vec_index_t> as_view(
                active_set.data(),
                active_set.size());
        assert(active_set.size() <= strong_set.size());
        assert(!active_set.size() || (0 <= as_view.minCoeff() && as_view.maxCoeff() < strong_set.size()));

        // check that active order is sorted.
        Eigen::Map<const vec_index_t> ao_view(
                active_order.data(),
                active_order.size());
        vec_index_t ao_copy = ao_view;
        std::sort(ao_copy.data(), ao_copy.data()+ao_copy.size(),
                  [&](auto i, auto j) { 
                        return strong_set[active_set[i]] < strong_set[active_set[j]]; 
                    });
        assert((ao_copy.array() == ao_view.array()).all());

        // check that active set contains at least the non-zero betas
        for (size_t i = 0; i < strong_set.size(); ++i) {
            if (strong_beta[i] == 0) continue;
            auto it = std::find(active_set.begin(), active_set.end(), i);
            assert(it != active_set.end());
        }

        // check that active_set_ordered is truly ordered
        vec_index_t aso_copy = vec_index_t::NullaryExpr(
                    active_order.size(),
                    [&](auto i) { return strong_set[active_set[active_order[i]]]; });
        Eigen::Map<const vec_index_t> aso_view(
                active_set_ordered.data(),
                active_set_ordered.size());
        assert((aso_copy.array() == aso_view.array()).all());
    }

    // check that is_active is right size and contains correct active set variables.
    { 
        assert(is_active.size() == strong_set.size());
        size_t n_active = 0;
        for (size_t i = 0; i < active_set.size(); ++i) {
            if (is_active[active_set[i]]) ++n_active;
        }
        assert(n_active == active_set.size());
    }

    // check that strong_beta and strong_grad agree in size with strong_set.
    {
        assert(strong_beta.size() == strong_set.size());
        assert(strong_grad.size() == strong_set.size());
    }
#endif
}

template<class ForwardIt, class F, class T>
ForwardIt lower_bound(ForwardIt first, ForwardIt last, F f, T value)
{
    ForwardIt it = first;
    typename std::iterator_traits<ForwardIt>::difference_type count, step;
    count = std::distance(first, last);

    while (count > 0) {
        it = first;
        step = count / 2;
        std::advance(it, step);
        if (f(*it) < value) {
            first = ++it;
            count -= step + 1;
        }
        else
            count = step;
    }
    return first;
}

} // namespace internal

/**
 * Checks early stopping based on R^2 values.
 * Returns true (early stopping should occur) if both are true:
 *
 *      delta_u := (R^2_u - R^2_m)/R^2_u
 *      delta_m := (R^2_m - R^2_l)/R^2_m 
 *      delta_u < cond_0_thresh 
 *      AND
 *      (delta_u - delta_m) < cond_1_thresh
 *
 * @param   rsq_l   third to last R^2 value.
 * @param   rsq_m   second to last R^2 value.
 * @param   rsq_u   last R^2 value.
 * @param   cond_0_thresh   threshold for derivative condition.
 * @param   cond_1_thresh   threshold for second derivative condition.
 */
template <class ValueType>
GHOSTBASIL_STRONG_INLINE
bool check_early_stop_rsq(
        ValueType rsq_l,
        ValueType rsq_m,
        ValueType rsq_u,
        ValueType cond_0_thresh = 1e-5,
        ValueType cond_1_thresh = 1e-5)
{
    const auto delta_u = (rsq_u-rsq_m);
    const auto delta_m = (rsq_m-rsq_l);
    return ((delta_u <= cond_0_thresh*rsq_u) &&
            ((delta_m*rsq_u-delta_u*rsq_m) <= cond_1_thresh*rsq_m*rsq_u));
}
  
/**
 * Computes the objective that we wish to minimize.
 * The objective is the quadratic loss + regularization:
 * \f[
 *      \frac{1}{2} \beta^\top A \beta - \beta^\top r
 *          + \lambda \sum\limits_{i} p_i \left(
 *              \frac{1-\alpha}{2} \beta_i^2 + \alpha |\beta_i|
 *            \right)
 * \f]
 * 
 * @param   A   any (p,p) matrix.
 * @param   r   any (p,) vector.
 * @param   penalty penalty factor on each coefficient.
 * @param   alpha   elastic net proportion.
 * @param   lmda    lasso regularization.
 * @param   beta    coefficient vector.
 */
template <class AType, class RType, class PenaltyType,
          class ValueType, class BetaType>
GHOSTBASIL_STRONG_INLINE 
auto objective(
        const AType& A,
        const RType& r,
        const PenaltyType& penalty,
        ValueType alpha,
        ValueType lmda,
        const BetaType& beta)
{
    return 0.5 * A.quad_form(beta) - beta.dot(r) 
        + lmda * (
            (1-alpha)/2 * beta.cwiseProduct(penalty).cwiseProduct(beta).sum()
            + alpha * beta.cwiseAbs().cwiseProduct(penalty).sum()
        );
}

/**
 * Updates the coefficient given the current state via coordinate descent rule.
 *
 * @param   coeff   current coefficient to update.
 * @param   x_var   variance of feature. A[k,k] where k is the feature corresponding to coeff.
 * @param   l1      L1 regularization part in elastic net.
 * @param   l2      L2 regularization part in elastic net.
 * @param   penalty penalty value for current coefficient.
 * @param   grad    current (negative) gradient for coeff.
 */
template <class ValueType>
GHOSTBASIL_STRONG_INLINE
void update_coefficient(
        ValueType& coeff,
        ValueType x_var,
        ValueType l1,
        ValueType l2,
        ValueType penalty,
        ValueType grad)
{
    const auto denom = x_var + l2 * penalty;
    const auto u = grad + coeff * x_var;
    const auto v = std::abs(u) - l1 * penalty;
    coeff = (v > 0.0) ? std::copysign(v,u)/denom : 0;
}

/**
 * Updates the convergence measure using variance of feature and coefficient change.
 *
 * @param   convg_measure   current convergence measure to update.
 * @param   coeff_diff      new coefficient minus old coefficient.
 * @param   x_var           variance of feature. A[k,k] where k is the feature corresponding to coeff_diff.
 */
template <class ValueType>
GHOSTBASIL_STRONG_INLINE
void update_convergence_measure(
        ValueType& convg_measure,
        ValueType coeff_diff,
        ValueType x_var)
{
    const auto convg_measure_curr = x_var * coeff_diff * coeff_diff;
    convg_measure = std::max(convg_measure_curr, convg_measure);
}

/**
 * Increments rsq with the difference in R^2.
 *
 * @param   rsq         R^2 to update.
 * @param   old_coeff   old coefficient.
 * @param   new_coeff   new coefficient.
 * @param   x_var       variance of feature (A[k,k]).
 * @param   grad        (negative) gradient corresponding to the coefficient.
 * @param   s           regularization of A towards identity.
 */
template <class ValueType>
GHOSTBASIL_STRONG_INLINE
void update_rsq(
        ValueType& rsq, 
        ValueType old_coeff, 
        ValueType new_coeff, 
        ValueType x_var, 
        ValueType grad)
{
    const auto del = new_coeff - old_coeff;
    rsq += del * (2 * grad - del * x_var);
}

/**
 * Coordinate descent (one loop) over a subset of strong variables.
 *
 * @param   A               see LassoParamPack.
 * @param   pack            see LassoParamPack.
 * @param   begin           begin iterator to indices into strong set, i.e.
 *                          strong_set[*begin] is the current feature to descend.
 *                          The resulting sequence of indices from calling 
 *                          strong_set[*(begin++)] MUST be ordered.
 * @param   end             end iterator to indices into strong set.
 * @param   lmda_idx        index into current L1 regularization value.
 * @param   convg_measure   stores the convergence measure of the call.
 * @param   additional_step any additional step at the end of each iteration.
 */
template <class DerivedType, class PackType, class Iter, class ValueType,
          class AdditionalStepType=util::no_op>
GHOSTBASIL_STRONG_INLINE
void coordinate_descent(
        const Eigen::DenseBase<DerivedType>& A_base,
        PackType& pack,
        Iter begin,
        Iter end,
        size_t lmda_idx,
        ValueType& convg_measure,
        AdditionalStepType additional_step=AdditionalStepType())
{
    const auto& A = A_base.derived();
    const auto alpha = pack.alpha;
    const auto lmda = pack.lmdas[lmda_idx];
    const auto l1 = lmda * alpha;
    const auto l2 = lmda * (1-alpha);
    const auto& penalty = pack.penalty;
    const auto& strong_set = pack.strong_set;
    const auto& strong_A_diag = pack.strong_A_diag;
    auto& strong_beta = pack.strong_beta;
    auto& strong_grad = pack.strong_grad;
    auto& rsq = pack.rsq;

    convg_measure = 0;
    for (auto it = begin; it != end; ++it) {
        const auto ss_idx = *it;              // index to strong set
        const auto k = strong_set[ss_idx];    // actual feature index
        const auto ak = strong_beta[ss_idx];  // corresponding beta
        const auto gk = strong_grad[ss_idx];  // corresponding gradient
        const auto A_kk = strong_A_diag[ss_idx];  // corresponding A diagonal element
        const auto pk = penalty[k]; 
                                    
        auto& ak_ref = strong_beta[ss_idx];
        update_coefficient(ak_ref, A_kk, l1, l2, pk, gk);

        if (ak_ref == ak) continue;

        const auto del = ak_ref - ak;

        // update measure of convergence
        update_convergence_measure(convg_measure, del, A_kk);

        // update rsq
        update_rsq(rsq, ak, ak_ref, A_kk, gk);

        // update gradient
        for (auto jt = begin; jt != end; ++jt) {
            const auto ss_idx_j = *jt;
            const auto j = strong_set[ss_idx_j];
            const auto A_jk = A.coeff(j, k);
            strong_grad[ss_idx_j] -= del * A_jk;
        }

        // additional step
        additional_step(ss_idx);
    }
}

template <class MatType, class VecType,
          class PackType, class Iter, class ValueType,
          class AdditionalStepType=util::no_op>
GHOSTBASIL_STRONG_INLINE
void coordinate_descent(
        const GhostMatrix<MatType, VecType>& A,
        PackType& pack,
        Iter begin,
        Iter end,
        size_t lmda_idx,
        ValueType& convg_measure,
        AdditionalStepType additional_step=AdditionalStepType())
{
    const auto alpha = pack.alpha;
    const auto lmda = pack.lmdas[lmda_idx];
    const auto l1 = lmda * alpha;
    const auto l2 = lmda * (1-alpha);
    const auto& penalty = pack.penalty;
    const auto& strong_set = pack.strong_set;
    const auto& strong_A_diag = pack.strong_A_diag;
    auto& strong_beta = pack.strong_beta;
    auto& strong_grad = pack.strong_grad;
    auto& rsq = pack.rsq;

    convg_measure = 0;
    for (auto it = begin; it != end; ++it) {
        const auto ss_idx = *it;              // index to strong set
        const auto k = strong_set[ss_idx];    // actual feature index
        const auto ak = strong_beta[ss_idx];  // corresponding beta
        const auto gk = strong_grad[ss_idx];  // corresponding gradient
        const auto A_kk = strong_A_diag[ss_idx];  // corresponding A diagonal element
        const auto pk = penalty[k]; 
                                    
        auto& ak_ref = strong_beta[ss_idx];
        update_coefficient(ak_ref, A_kk, l1, l2, pk, gk);

        if (ak_ref == ak) continue;

        const auto del = ak_ref - ak;

        // update measure of convergence
        update_convergence_measure(convg_measure, del, A_kk);

        // update rsq
        update_rsq(rsq, ak, ak_ref, A_kk, gk);

        // update gradient
        const auto& S = A.matrix();
        const auto& D = A.vector();
        const auto k_shift = A.shift(k);
        const auto D_kk = D[k_shift];
        strong_grad[ss_idx] -= del * D_kk;
        for (auto jt = begin; jt != end; ++jt) {
            const auto ss_idx_j = *jt;
            const auto j = strong_set[ss_idx_j];
            const auto j_shift = A.shift(j);
            const auto S_jk = S.coeff(j_shift, k_shift);
            strong_grad[ss_idx_j] -= del * S_jk;
        }

        // additional step
        additional_step(ss_idx);
    }
}

template <class MatType,
          class PackType, class Iter, class ValueType,
          class AdditionalStepType=util::no_op>
GHOSTBASIL_STRONG_INLINE
void coordinate_descent(
        const GroupGhostMatrix<MatType>& A,
        PackType& pack,
        Iter begin,
        Iter end,
        size_t lmda_idx,
        ValueType& convg_measure,
        AdditionalStepType additional_step=AdditionalStepType())
{
    const auto alpha = pack.alpha;
    const auto lmda = pack.lmdas[lmda_idx];
    const auto l1 = lmda * alpha;
    const auto l2 = lmda * (1-alpha);
    const auto& penalty = pack.penalty;
    const auto& strong_set = pack.strong_set;
    const auto& strong_A_diag = pack.strong_A_diag;
    auto& strong_beta = pack.strong_beta;
    auto& strong_grad = pack.strong_grad;
    auto& rsq = pack.rsq;

    convg_measure = 0;
    for (auto it = begin; it != end; ++it) {
        const auto ss_idx = *it;              // index to strong set
        const auto k = strong_set[ss_idx];    // actual feature index
        const auto ak = strong_beta[ss_idx];  // corresponding beta
        const auto gk = strong_grad[ss_idx];  // corresponding gradient
        const auto A_kk = strong_A_diag[ss_idx];  // corresponding A diagonal element
        const auto pk = penalty[k]; 
                                    
        auto& ak_ref = strong_beta[ss_idx];
        update_coefficient(ak_ref, A_kk, l1, l2, pk, gk);

        if (ak_ref == ak) continue;

        const auto del = ak_ref - ak;

        // update measure of convergence
        update_convergence_measure(convg_measure, del, A_kk);

        // update rsq
        update_rsq(rsq, ak, ak_ref, A_kk, gk);

        // update gradient
        const auto& S = A.get_S();
        const auto& D = A.get_D();
        const auto k_shift = A.shift(k);
        const auto k_block = k / S.cols();
        const auto D_k = D.col(k_shift);
        for (auto jt = begin; jt != end; ++jt) {
            const auto ss_idx_j = *jt;
            const auto j = strong_set[ss_idx_j];
            const auto j_shift = A.shift(j);
            const auto j_block = j / S.cols();
            const auto S_jk = S.coeff(j_shift, k_shift);
            strong_grad[ss_idx_j] -= del * (S_jk + ((j_block == k_block) ? D_k[j_shift] : 0));
        }

        // additional step
        additional_step(ss_idx);
    }
}

template <class MatType, class DType,
          class PackType, class Iter, class ValueType,
          class AdditionalStepType=util::no_op>
GHOSTBASIL_STRONG_INLINE
void coordinate_descent(
        const BlockGroupGhostMatrix<MatType, DType>& A,
        PackType& pack,
        Iter begin,
        Iter end,
        size_t lmda_idx,
        ValueType& convg_measure,
        AdditionalStepType additional_step=AdditionalStepType())
{
    const auto alpha = pack.alpha;
    const auto lmda = pack.lmdas[lmda_idx];
    const auto l1 = lmda * alpha;
    const auto l2 = lmda * (1-alpha);
    const auto& penalty = pack.penalty;
    const auto& strong_set = pack.strong_set;
    const auto& strong_A_diag = pack.strong_A_diag;
    auto& strong_beta = pack.strong_beta;
    auto& strong_grad = pack.strong_grad;
    auto& rsq = pack.rsq;

    convg_measure = 0;
    for (auto it = begin; it != end; ++it) {
        const auto ss_idx = *it;              // index to strong set
        const auto k = strong_set[ss_idx];    // actual feature index
        const auto ak = strong_beta[ss_idx];  // corresponding beta
        const auto gk = strong_grad[ss_idx];  // corresponding gradient
        const auto A_kk = strong_A_diag[ss_idx];  // corresponding A diagonal element
        const auto pk = penalty[k]; 
                                    
        auto& ak_ref = strong_beta[ss_idx];
        update_coefficient(ak_ref, A_kk, l1, l2, pk, gk);

        if (ak_ref == ak) continue;

        const auto del = ak_ref - ak;

        // update measure of convergence
        update_convergence_measure(convg_measure, del, A_kk);

        // update rsq
        update_rsq(rsq, ak, ak_ref, A_kk, gk);

        // update gradient
        const auto& S = A.get_S();
        const auto& D = A.get_D();
        const auto k_shift = A.shift(k);
        const auto k_block = k / S.cols();
        for (auto jt = begin; jt != end; ++jt) {
            const auto ss_idx_j = *jt;
            const auto j = strong_set[ss_idx_j];
            const auto j_shift = A.shift(j);
            const auto j_block = j / S.cols();
            const auto S_jk = S.coeff(j_shift, k_shift);
            strong_grad[ss_idx_j] -= del * (S_jk + ((j_block == k_block) ? D.coeff(j_shift, k_shift) : 0));
        }

        // additional step
        additional_step(ss_idx);
    }
}

/*
 * DEPRECATED: keeping just in case as a fall-back option.
 */
template <class MatType, class PackType, class Iter, class ValueType,
          class AdditionalStepType=util::no_op>
GHOSTBASIL_STRONG_INLINE
void coordinate_descent(
        const BlockMatrix<MatType>& A,
        PackType& pack,
        Iter begin,
        Iter end,
        size_t lmda_idx,
        ValueType& convg_measure,
        AdditionalStepType additional_step=AdditionalStepType())
{
    const auto alpha = pack.alpha;
    const auto lmda = pack.lmdas[lmda_idx];
    const auto l1 = lmda * alpha;
    const auto l2 = lmda * (1-alpha);
    const auto& penalty = pack.penalty;
    const auto& strong_set = pack.strong_set;
    const auto& strong_A_diag = pack.strong_A_diag;
    auto& strong_beta = pack.strong_beta;
    auto& strong_grad = pack.strong_grad;
    auto& rsq = pack.rsq;

    const auto get_strong_set = [&](auto i) { return strong_set[i]; };

    // Note: here we really assume sorted-ness!
    // Since begin->end results in increasing sequence of strong set indices,
    // we can update the block iterator as we increment begin.
    auto block_it = A.block_begin();

    // We can also keep track of the range of strong set indices
    // that produce indices within the current block.
    auto range_begin = begin;
    auto range_end = internal::lower_bound(
            range_begin, end, get_strong_set,
            block_it.stride() + block_it.block().cols());

    convg_measure = 0;
    for (auto it = begin; it != end; ++it) {
        const auto ss_idx = *it;              // index to strong set
        const auto k = strong_set[ss_idx];    // actual feature index
        const auto ak = strong_beta[ss_idx];  // corresponding beta
        const auto gk = strong_grad[ss_idx];  // corresponding gradient
        const auto A_kk = strong_A_diag[ss_idx];  // corresponding A diagonal element
        const auto pk = penalty[k]; 
                                    
        auto& ak_ref = strong_beta[ss_idx];
        update_coefficient(ak_ref, A_kk, l1, l2, pk, gk);

        if (ak_ref == ak) continue;

        const auto del = ak_ref - ak;

        // update measure of convergence
        update_convergence_measure(convg_measure, del, A_kk);

        // update rsq
        update_rsq(rsq, ak, ak_ref, A_kk, gk);

        // update outer iterators to preserve invariant
        if (!block_it.is_in_block(k)) {
            block_it.advance_at(k);
            range_begin = internal::lower_bound(
                    range_end, end, get_strong_set,
                    block_it.stride());
            range_end = internal::lower_bound(
                    range_begin, end, get_strong_set,
                    block_it.stride() + block_it.block().cols());
        }
        const auto k_shifted = block_it.shift(k);
        const auto& block = block_it.block();
        
        // update gradient
        for (auto jt = range_begin; jt != range_end; ++jt) {
            const auto ss_idx_j = *jt;
            const auto j = strong_set[ss_idx_j];
            const auto j_shifted = block_it.shift(j);
            const auto A_jk = block.coeff(j_shifted, k_shifted);
            strong_grad[ss_idx_j] -= del * A_jk;
        }

        // additional step
        additional_step(ss_idx);
    }
}

/*
 * DEPRECATED: keeping just in case as a fall-back option.
 */
template <class MatType, class VecType,
          class PackType, class Iter, class ValueType,
          class AdditionalStepType=util::no_op>
GHOSTBASIL_STRONG_INLINE
void coordinate_descent(
        const BlockMatrix<GhostMatrix<MatType, VecType>>& A,
        PackType& pack,
        Iter begin,
        Iter end,
        size_t lmda_idx,
        ValueType& convg_measure,
        AdditionalStepType additional_step=AdditionalStepType())
{
    const auto alpha = pack.alpha;
    const auto lmda = pack.lmdas[lmda_idx];
    const auto l1 = lmda * alpha;
    const auto l2 = lmda * (1-alpha);
    const auto& penalty = pack.penalty;
    const auto& strong_set = pack.strong_set;
    const auto& strong_A_diag = pack.strong_A_diag;
    auto& strong_beta = pack.strong_beta;
    auto& strong_grad = pack.strong_grad;
    auto& rsq = pack.rsq;

    const auto get_strong_set = [&](auto i) { return strong_set[i]; };

    // Note: here we really assume sorted-ness!
    // Since begin->end results in increasing sequence of strong set indices,
    // we can update the block iterator as we increment begin.
    auto block_it = A.block_begin();

    // We can also keep track of the range of strong set indices
    // that produce indices within the current block.
    auto range_begin = begin;
    auto range_end = internal::lower_bound(
            range_begin, end, get_strong_set,
            block_it.stride() + block_it.block().cols());

    convg_measure = 0;
    for (auto it = begin; it != end; ++it) {
        const auto ss_idx = *it;              // index to strong set
        const auto k = strong_set[ss_idx];    // actual feature index
        const auto ak = strong_beta[ss_idx];  // corresponding beta
        const auto gk = strong_grad[ss_idx];  // corresponding gradient
        const auto A_kk = strong_A_diag[ss_idx];  // corresponding A diagonal element
        const auto pk = penalty[k]; 
                                    
        auto& ak_ref = strong_beta[ss_idx];
        update_coefficient(ak_ref, A_kk, l1, l2, pk, gk);

        if (ak_ref == ak) continue;

        const auto del = ak_ref - ak;

        // update measure of convergence
        update_convergence_measure(convg_measure, del, A_kk);

        // update rsq
        update_rsq(rsq, ak, ak_ref, A_kk, gk);

        // update outer iterators to preserve invariant
        if (!block_it.is_in_block(k)) {
            block_it.advance_at(k);
            range_begin = internal::lower_bound(
                    range_end, end, get_strong_set,
                    block_it.stride());
            range_end = internal::lower_bound(
                    range_begin, end, get_strong_set,
                    block_it.stride() + block_it.block().cols());
        }
        const auto k_block_shift = block_it.shift(k);
        const auto& block = block_it.block();
        
        // update gradient
        const auto& S = block.matrix();
        const auto& D = block.vector();
        const auto k_shift = block.shift(k_block_shift);
        const auto D_kk = D[k_shift];
        strong_grad[ss_idx] -= del * D_kk;
        for (auto jt = range_begin; jt != range_end; ++jt) {
            const auto ss_idx_j = *jt;
            const auto j = strong_set[ss_idx_j];
            const auto j_block_shift = block_it.shift(j);
            const auto j_shift = block.shift(j_block_shift);
            const auto S_jk = S.coeff(j_shift, k_shift);
            strong_grad[ss_idx_j] -= del * S_jk;
        }

        // additional step
        additional_step(ss_idx);
    }
}

/**
 * Applies multiple coordinate descent on the active set 
 * to minimize the ghostbasil objective.
 * See "objective" function for the objective of interest.
 *
 * @param   pack        see LassoParamPack.
 * @param   lmda_idx    index into the lambda sequence for logging purposes.
 * @param   active_beta_diff_ordered    dense vector to store coefficient difference corresponding
 *                                      to the active set between
 *                                      the new coefficients and the current coefficients
 *                                      after performing coordinate descent on the active set.
 *                                      It must be initialized to be of size active_order.size(),
 *                                      though the values need not be initialized.
 *                                      active_beta_diff_ordered[i] = 
 *                                          (new minus old of) 
 *                                          strong_beta[active_set[active_order[i]]].
 * @param   sg_update   functor that updates strong_gradient.
 * @param   check_user_interrupt    functor that checks user interruption and 
 *                                  takes any user-specified action.
 */
template <class PackType, class ABDiffOType, class SGUpdateType,
          class CUIType = util::no_op>
GHOSTBASIL_STRONG_INLINE
void lasso_active_impl(
    PackType& pack,
    size_t lmda_idx,
    ABDiffOType& active_beta_diff_ordered,
    SGUpdateType sg_update,
    CUIType check_user_interrupt = CUIType())
{
    using pack_t = std::decay_t<PackType>;
    using value_t = typename pack_t::value_t;
    using index_t = typename pack_t::index_t;
    using sp_vec_value_t = typename pack_t::sp_vec_value_t;

    const auto& A = *pack.A;
    const auto& strong_beta = pack.strong_beta;
    const auto thr = pack.thr;
    const auto max_cds = pack.max_cds;
    const auto& active_set = *pack.active_set;
    const auto& active_order = *pack.active_order;
    const auto& active_set_ordered = *pack.active_set_ordered;
    auto& n_cds = pack.n_cds;

    internal::lasso_assert_valid_inputs(pack);

    assert(active_beta_diff_ordered.size() == active_order.size());
    Eigen::Map<util::vec_type<value_t>> ab_diff_o_view(
            active_beta_diff_ordered.data(), 
            active_beta_diff_ordered.size());
    const auto active_beta_ordered_expr = 
        util::vec_type<value_t>::NullaryExpr(
            active_order.size(),
            [&](auto i) { return strong_beta[active_set[active_order[i]]]; });
    ab_diff_o_view = active_beta_ordered_expr;

    const auto active_set_iter_f = [&](auto i) {
        return active_set[active_order[i]];
    };

    while (1) {
        check_user_interrupt(n_cds);
        ++n_cds;
        value_t convg_measure;
        coordinate_descent(
                A, pack,
                util::make_functor_iterator<index_t>(
                    0, active_set_iter_f),
                util::make_functor_iterator<index_t>(
                    active_order.size(), active_set_iter_f),
                lmda_idx, convg_measure);
        if (convg_measure < thr) break;
        if (n_cds >= max_cds) throw util::max_cds_error(lmda_idx);
    }
    
    ab_diff_o_view = active_beta_ordered_expr - ab_diff_o_view;

    Eigen::Map<const sp_vec_value_t> active_beta_diff_map(
            A.cols(),
            active_set_ordered.size(),
            active_set_ordered.data(),
            active_beta_diff_ordered.data());

    // update strong gradient
    sg_update(active_beta_diff_map);
}

/**
 * Calls lasso_active_impl with a specialized gradient update routine
 * given the matrix type of A. For parameter descriptions,
 * see lasso_active_impl.
 */
template <class DerivedType, class PackType,
          class ABDiffOType, class CUIType = util::no_op>
GHOSTBASIL_STRONG_INLINE 
void lasso_active(
    const Eigen::DenseBase<DerivedType>& A_base,
    PackType& pack,
    size_t lmda_idx,
    ABDiffOType& active_beta_diff_ordered,
    CUIType check_user_interrupt = CUIType())
{
    const auto& A = A_base.derived();
    const auto& strong_set = pack.strong_set;
    const auto& is_active = pack.is_active;
    const auto& active_set = *pack.active_set;
    auto& strong_grad = pack.strong_grad;

    const auto sg_update = [&](const auto& sp_beta_diff) {
        if ((sp_beta_diff.nonZeros() == 0) ||
            (active_set.size() == strong_set.size())) return;

        // update gradient in non-active positions
        for (size_t ss_idx = 0; ss_idx < strong_set.size(); ++ss_idx) {
            if (is_active[ss_idx]) continue;
            const auto k = strong_set[ss_idx];
            strong_grad[ss_idx] -= A.col_dot(k, sp_beta_diff);
        }
    };

    lasso_active_impl(
            pack, lmda_idx, active_beta_diff_ordered, sg_update,
            check_user_interrupt);
}

template <class MatType, class VecType, class PackType,
          class ABDiffOType, class CUIType = util::no_op>
GHOSTBASIL_STRONG_INLINE 
void lasso_active(
    const GhostMatrix<MatType, VecType>& A,
    PackType& pack,
    size_t lmda_idx,
    ABDiffOType& active_beta_diff_ordered,
    CUIType check_user_interrupt = CUIType())
{
    using pack_t = std::decay_t<PackType>;
    using value_t = typename pack_t::value_t;
    
    const auto& strong_set = pack.strong_set;
    const auto& strong_order = pack.strong_order;
    const auto& is_active = pack.is_active;
    const auto& active_set = *pack.active_set;
    auto& strong_grad = pack.strong_grad;

    const auto sg_update = [&](const auto& sp_beta_diff) {
        if ((sp_beta_diff.nonZeros() == 0) || 
            (active_set.size() == strong_set.size())) return;

        const auto& S = A.matrix();
        const auto& D = A.vector();
        const auto v_inner = sp_beta_diff.innerIndexPtr();
        const auto v_value = sp_beta_diff.valuePtr();
        const auto v_nnz = sp_beta_diff.nonZeros();
        auto v_begin = 0;

        util::vec_type<value_t> sum_vs(S.cols());
        util::vec_type<value_t> S_sum_vs(S.cols());
        sum_vs.setZero();

        static constexpr value_t inf = 
            std::numeric_limits<value_t>::infinity();
        S_sum_vs.fill(inf);

        // compute sum of v blocks
        for (size_t j = 0; j < v_nnz; ++j) {
            const auto j_inner = v_inner[j];
            const auto j_shift = A.shift(j_inner);
            sum_vs[j_shift] += v_value[j];
        }

        // update gradient in non-active positions
        for (size_t ii = 0; ii < strong_order.size(); ++ii) {
            const auto ss_idx = strong_order[ii];
            if (is_active[ss_idx]) continue;
            const auto k = strong_set[ss_idx];
            const auto k_shifted = A.shift(k);
            if (S_sum_vs[k_shifted] == inf) {
                S_sum_vs[k_shifted] = sum_vs.dot(S.col(k_shifted));
            }
            const auto D_kk = D[k_shifted];
            v_begin = std::lower_bound(
                    v_inner+v_begin, v_inner+v_nnz, k)-v_inner;
            const auto D_kk_v_k = ((v_begin != v_nnz) && (v_inner[v_begin] == k)) ?
                D_kk*v_value[v_begin] : 0;
            strong_grad[ss_idx] -= (S_sum_vs[k_shifted] + D_kk_v_k);
        }
    };

    lasso_active_impl(
            pack, lmda_idx, active_beta_diff_ordered, sg_update,
            check_user_interrupt);
}

template <class MatType, class PackType,
          class ABDiffOType, class CUIType = util::no_op>
GHOSTBASIL_STRONG_INLINE 
void lasso_active(
    const GroupGhostMatrix<MatType>& A,
    PackType& pack,
    size_t lmda_idx,
    ABDiffOType& active_beta_diff_ordered,
    CUIType check_user_interrupt = CUIType())
{
    using pack_t = std::decay_t<PackType>;
    using value_t = typename pack_t::value_t;
    
    const auto& strong_set = pack.strong_set;
    const auto& strong_order = pack.strong_order;
    const auto& is_active = pack.is_active;
    const auto& active_set = *pack.active_set;
    auto& strong_grad = pack.strong_grad;

    const auto sg_update = [&](const auto& sp_beta_diff) {
        if ((sp_beta_diff.nonZeros() == 0) || 
            (active_set.size() == strong_set.size())) return;

        const auto& S = A.get_S();
        const auto& D = A.get_D();
        const auto v_inner = sp_beta_diff.innerIndexPtr();
        const auto v_value = sp_beta_diff.valuePtr();
        const auto v_nnz = sp_beta_diff.nonZeros();
        auto v_begin = 0;

        util::vec_type<value_t> sum_vs(S.cols());
        util::vec_type<value_t> S_sum_vs(S.cols());
        sum_vs.setZero();

        static constexpr value_t inf = 
            std::numeric_limits<value_t>::infinity();
        S_sum_vs.fill(inf);

        // compute sum of v blocks
        for (size_t j = 0; j < v_nnz; ++j) {
            const auto j_inner = v_inner[j];
            const auto j_shift = A.shift(j_inner);
            sum_vs[j_shift] += v_value[j];
        }

        // update gradient in non-active positions
        for (size_t ii = 0; ii < strong_order.size(); ++ii) {
            const auto ss_idx = strong_order[ii];
            if (is_active[ss_idx]) continue;
            const auto k = strong_set[ss_idx];
            const auto k_shifted = A.shift(k);
            if (S_sum_vs[k_shifted] == inf) {
                S_sum_vs[k_shifted] = sum_vs.dot(S.col(k_shifted));
            }
            const auto D_k = D.col(k_shifted);
            const auto k_block_begin = (k / S.cols()) * S.cols();
            v_begin = std::lower_bound(v_inner+v_begin, v_inner+v_nnz, k_block_begin)-v_inner;
            const auto v_end = std::lower_bound(v_inner+v_begin, v_inner+v_nnz, k_block_begin+S.cols())-v_inner;
            value_t D_kk_v_k = 0;
            for (size_t l = v_begin; l < v_end; ++l) {
                D_kk_v_k += v_value[l] * D_k[v_inner[l]-k_block_begin];
            }
            strong_grad[ss_idx] -= (S_sum_vs[k_shifted] + D_kk_v_k);
        }
    };

    lasso_active_impl(
            pack, lmda_idx, active_beta_diff_ordered, sg_update,
            check_user_interrupt);
}

template <class MatType, class DType, class PackType,
          class ABDiffOType, class CUIType = util::no_op>
GHOSTBASIL_STRONG_INLINE 
void lasso_active(
    const BlockGroupGhostMatrix<MatType, DType>& A,
    PackType& pack,
    size_t lmda_idx,
    ABDiffOType& active_beta_diff_ordered,
    CUIType check_user_interrupt = CUIType())
{
    using pack_t = std::decay_t<PackType>;
    using value_t = typename pack_t::value_t;
    
    const auto& strong_set = pack.strong_set;
    const auto& strong_order = pack.strong_order;
    const auto& is_active = pack.is_active;
    const auto& active_set = *pack.active_set;
    auto& strong_grad = pack.strong_grad;

    const auto sg_update = [&](const auto& sp_beta_diff) {
        if ((sp_beta_diff.nonZeros() == 0) || 
            (active_set.size() == strong_set.size())) return;

        const auto& S = A.get_S();
        const auto& D = A.get_D();
        const auto v_inner = sp_beta_diff.innerIndexPtr();
        const auto v_value = sp_beta_diff.valuePtr();
        const auto v_nnz = sp_beta_diff.nonZeros();
        auto v_begin = 0;

        util::vec_type<value_t> sum_vs(S.cols());
        util::vec_type<value_t> S_sum_vs(S.cols());
        sum_vs.setZero();

        static constexpr value_t inf = 
            std::numeric_limits<value_t>::infinity();
        S_sum_vs.fill(inf);

        // compute sum of v blocks
        for (size_t j = 0; j < v_nnz; ++j) {
            const auto j_inner = v_inner[j];
            const auto j_shift = A.shift(j_inner);
            sum_vs[j_shift] += v_value[j];
        }

        // update gradient in non-active positions
        for (size_t ii = 0; ii < strong_order.size(); ++ii) {
            const auto ss_idx = strong_order[ii];
            if (is_active[ss_idx]) continue;
            const auto k = strong_set[ss_idx];
            const auto k_shifted = A.shift(k);
            if (S_sum_vs[k_shifted] == inf) {
                S_sum_vs[k_shifted] = sum_vs.dot(S.col(k_shifted));
            }

            // prepare for D slicing
            {
                const auto& n_cum_sum = D.strides();
                const auto ik_end = std::upper_bound(
                        n_cum_sum.begin(),
                        n_cum_sum.end(),
                        k_shifted);
                const auto ik_begin = std::next(ik_end, -1);
                const auto ik = std::distance(n_cum_sum.begin(), ik_begin);  

                // Find i(k)th block matrix.
                const auto& B = D.blocks()[ik];
                const auto stride = n_cum_sum[ik];
                const size_t k_shifted_block = k_shifted - stride;

                // Find v_{i(k)}, i(k)th block of vector. 
                const auto k_block_begin = (k / S.cols()) * S.cols();
                const auto B_begin = k_block_begin + stride;
                const auto B_end = k_block_begin+n_cum_sum[ik+1];
                v_begin = std::lower_bound(v_inner+v_begin, v_inner+v_nnz, B_begin)-v_inner;
                const auto v_end = std::lower_bound(v_inner+v_begin, v_inner+v_nnz, B_end)-v_inner;

                value_t D_kk_v_k = 0;
                for (size_t l = v_begin; l < v_end; ++l) {
                    D_kk_v_k += v_value[l] * B.coeff(v_inner[l]-B_begin, k_shifted_block);
                }
                strong_grad[ss_idx] -= (S_sum_vs[k_shifted] + D_kk_v_k);
            }
        }
    };

    lasso_active_impl(
            pack, lmda_idx, active_beta_diff_ordered, sg_update,
            check_user_interrupt);
}

/*
 * DEPRECATED: keeping just in case as a fall-back option.
 */
template <class MatType, class PackType, 
          class ABDiffOType, class CUIType=util::no_op>
GHOSTBASIL_STRONG_INLINE 
void lasso_active(
    const BlockMatrix<MatType>& A,
    PackType& pack,
    size_t lmda_idx,
    ABDiffOType& active_beta_diff_ordered,
    CUIType check_user_interrupt = CUIType())
{
    using pack_t = std::decay_t<PackType>;
    using value_t = typename pack_t::value_t;
    
    const auto& strong_set = pack.strong_set;
    const auto& strong_order = pack.strong_order;
    const auto& is_active = pack.is_active;
    const auto& active_set = *pack.active_set;
    auto& strong_grad = pack.strong_grad;

    auto sg_update = [&](const auto& sp_beta_diff) {
        if ((sp_beta_diff.nonZeros() == 0) || 
            (active_set.size() == strong_set.size())) return;

        auto block_it = A.block_begin();
        const auto bd_inner = sp_beta_diff.innerIndexPtr();
        const auto bd_value = sp_beta_diff.valuePtr();
        const auto bd_nnz = sp_beta_diff.nonZeros();

        // initialized below
        size_t bd_seg_begin;
        size_t bd_seg_end; 

        {
            const auto curr_stride = block_it.stride(); // should be 0
            assert(curr_stride == 0);

            // find first time a non-zero index of beta is inside the current block
            const auto it = std::lower_bound(bd_inner, bd_inner+bd_nnz, curr_stride);
            bd_seg_begin = std::distance(bd_inner, it);

            // find first time a non-zero index of beta is outside the current block
            const auto next_stride = curr_stride+block_it.block().cols();
            const auto end = std::lower_bound(bd_inner+bd_seg_begin, bd_inner+bd_nnz, next_stride);
            bd_seg_end = std::distance(bd_inner, end);

            // these two define the index of bd_inner and size to read
            // to perform any dot product with a column of current block.
        }

        // update gradient in non-active positions
        for (size_t ii = 0; ii < strong_order.size(); ++ii) {
            const auto ss_idx = strong_order[ii];
            if (is_active[ss_idx]) continue;
            const auto k = strong_set[ss_idx];
            // update A block stride pointer if current feature is not in the block.
            if (!block_it.is_in_block(k)) {
                block_it.advance_at(k);

                const auto curr_stride = block_it.stride();
                const auto it = std::lower_bound(
                        bd_inner+bd_seg_end, 
                        bd_inner+bd_nnz, 
                        curr_stride);
                bd_seg_begin = std::distance(bd_inner, it);

                const auto next_stride = curr_stride+block_it.block().cols();
                const auto end = std::lower_bound(
                        bd_inner+bd_seg_begin, 
                        bd_inner+bd_nnz, 
                        next_stride);
                bd_seg_end = std::distance(bd_inner, end);
            }
            const auto k_shifted = block_it.shift(k);
            const auto& A_block = block_it.block();
            
            value_t dp = 0;
            for (auto i = bd_seg_begin; i < bd_seg_end; ++i) {
                dp += A_block.coeff(block_it.shift(bd_inner[i]), k_shifted) * bd_value[i];
            }
            strong_grad[ss_idx] -= dp;
        }
    };

    lasso_active_impl(
            pack, lmda_idx, active_beta_diff_ordered, sg_update,
            check_user_interrupt);
}

/**
 * Minimizes the objective described in the function "objective"
 * with the additional constraint:
 * \f[
 *      \beta_{i} = 0, \, \forall i \notin \{S\}
 * \f]
 * i.e. all betas not in the strong set, \f$S\f$, are fixed to be 0.
 *
 * @param   pack    see LassoParamPack.
 * @param   check_user_interrupt    see lasso_active_impl().
 */
template <class PackType,
          class CUIType = util::no_op>
inline void fit(
    PackType&& pack,
    CUIType check_user_interrupt = CUIType())
{
    using pack_t = typename std::decay_t<PackType>;
    using value_t = typename pack_t::value_t;
    using vec_index_t = typename pack_t::vec_index_t;
    using vec_value_t = typename pack_t::vec_value_t;
    using sp_vec_value_t = typename pack_t::sp_vec_value_t;

    const auto& A = *pack.A;
    const auto& strong_set = pack.strong_set;
    const auto& strong_order = pack.strong_order;
    const auto& strong_beta = pack.strong_beta;
    const auto& lmdas = pack.lmdas;
    const auto thr = pack.thr;
    const auto max_cds = pack.max_cds;
    const auto do_early_stop = pack.do_early_stop;
    const auto& rsq = pack.rsq;
    auto& active_set = *pack.active_set;
    auto& active_order = *pack.active_order;
    auto& active_set_ordered = *pack.active_set_ordered;
    auto& is_active = pack.is_active;
    auto& betas = pack.betas;
    auto& rsqs = pack.rsqs;
    auto& n_cds = pack.n_cds;
    auto& n_lmdas = pack.n_lmdas;

    internal::lasso_assert_valid_inputs(pack);

    assert(betas.size() == lmdas.size());
    assert(rsqs.size() == lmdas.size());

    // common buffers for the routine
    std::vector<value_t> active_beta_ordered;
    std::vector<value_t> active_beta_diff_ordered;
    active_beta_ordered.reserve(strong_set.size());
    active_beta_diff_ordered.reserve(strong_set.size());
    active_beta_diff_ordered.resize(active_order.size());
    
    bool lasso_active_called = false;
    n_cds = 0;
    n_lmdas = 0;

    const auto add_active_set = [&](auto ss_idx) {
        if (!is_active[ss_idx]) {
            is_active[ss_idx] = true;
            active_set.push_back(ss_idx);
        }
    };

    const auto lasso_active_and_update = [&](size_t l) {
        lasso_active(
                A, pack, l, active_beta_diff_ordered, 
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
            size_t old_active_size = active_set.size();
            coordinate_descent(
                    A, pack, strong_order.data(), strong_order.data() + strong_order.size(),
                    l, convg_measure, add_active_set);
            const bool new_active_added = (old_active_size < active_set.size());

            // since coordinate descent could have added new active variables,
            // we update active_order and active_set_ordered to preserve invariant.
            // NOTE: speed shouldn't be affected since most of the array is already sorted
            // and std::sort is really fast in this setting!
            // See: https://solarianprogrammer.com/2012/10/24/cpp-11-sort-benchmark/
            if (new_active_added) {
                active_order.resize(active_set.size());
                std::iota(std::next(active_order.begin(), old_active_size), 
                          active_order.end(), 
                          old_active_size);
                std::sort(active_order.begin(), active_order.end(),
                          [&](auto i, auto j) { 
                                return strong_set[active_set[i]] < strong_set[active_set[j]];
                            });

                active_set_ordered.resize(active_set.size());
                Eigen::Map<vec_index_t> aso_map(
                        active_set_ordered.data(),
                        active_set_ordered.size());
                aso_map = vec_index_t::NullaryExpr(
                        active_order.size(),
                        [&](auto i) { return strong_set[active_set[active_order[i]]]; });

                active_beta_diff_ordered.resize(active_order.size());
            }

            if (convg_measure < thr) break;
            if (n_cds >= max_cds) throw util::max_cds_error(l);

            lasso_active_and_update(l);
        }

        // order the strong betas 
        active_beta_ordered.resize(active_order.size());
        Eigen::Map<vec_value_t> ab_o_view(
                active_beta_ordered.data(),
                active_beta_ordered.size());
        ab_o_view = vec_value_t::NullaryExpr(
                active_order.size(),
                [&](auto i) { return strong_beta[active_set[active_order[i]]]; });
        assert(active_set_ordered.size() == active_order.size());
        Eigen::Map<const sp_vec_value_t> beta_map(
                A.cols(),
                active_set_ordered.size(),
                active_set_ordered.data(),
                active_beta_ordered.data());

        betas[l] = beta_map;
        rsqs[l] = rsq;
        ++n_lmdas;

        // make sure to do at least 3 lambdas.
        if (l < 2) continue;

        // early stop if R^2 criterion is fulfilled.
        if (do_early_stop && check_early_stop_rsq(rsqs[l-2], rsqs[l-1], rsqs[l])) break;
    }
}

} // namespace lasso
} // namespace ghostbasil
