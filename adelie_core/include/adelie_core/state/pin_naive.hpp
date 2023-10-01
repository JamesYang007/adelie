#pragma once
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace state {

/**
 * Parameter pack for group lasso procedure.
 * 
 * @param   X           data matrix (n, p) with diagonal covariance blocks X_i^T X_i. 
 * @param   groups      vector of indices into columns of X that define
 *                      the beginning index of the groups.
 * @param   group_sizes vector of sizes of each group.
 *                      group_sizes[i] = size of group i.
 * @param   alpha       elastic net proportion. 
 *                      It is undefined behavior if alpha is not in [0,1].
 * @param   penalty     penalty factor for each group.
 * @param   strong_set  strong set as a dense vector of indices in [0, G),
 *                      where G is the total number of groups.
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
 * @param   betas       vector of solution coefficients for each lambda in lmdas.
 *                      betas[j](i) = ith coefficient for jth lambda.
 *                      fit() will emplace_back only the solved coefficients.
 * @param   rsqs        vector of (unnormalized) R^2 values for each lambda in lmdas.
 *                      rsqs[j] = R^2 for jth lambda.
 *                      fit() will emplace_back only the solved rsq.
 * @param   resids      vector of residuals at the solution coefficient for each lambda in lmdas.
 *                      resids[j] = residual using betas[j].
 *                      fit() will emplace_back only the solved resid.
 * @param   n_cds       number of coordinate descents.
 * @param   cond_0_thresh   0th order threshold for early exit.
 * @param   cond_1_thresh   1st order threshold for early exit.
 * @param   diagnostic      instance of GroupElnetDiagnostic.
 */
template <class MatrixType, 
          class ValueType=typename std::decay_t<MatrixType>::value_t,
          class IndexType=Eigen::Index,
          class BoolType=bool,
          class DynamicVectorIndexType=std::vector<IndexType>,
          class DynamicVectorValueType=std::vector<ValueType>,
          class DynamicVectorVecValueType=std::vector<util::rowvec_type<ValueType>>,
          class DynamicVectorSpVecType=std::vector<
                util::sp_vec_type<ValueType, Eigen::ColMajor, IndexType>
            > 
          >
struct PinNaive
{
    using value_t = ValueType;
    using index_t = IndexType;
    using bool_t = BoolType;
    using sp_vec_value_t = util::sp_vec_type<value_t, Eigen::ColMajor, index_t>;
    using vec_value_t = util::rowvec_type<value_t>;
    using vec_index_t = util::rowvec_type<index_t>;
    using vec_bool_t = util::rowvec_type<bool_t>;
    using map_vec_value_t = Eigen::Map<vec_value_t>;
    using map_vec_index_t = Eigen::Map<vec_index_t>;
    using map_vec_bool_t = Eigen::Map<vec_bool_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;
    using map_cvec_index_t = Eigen::Map<const vec_index_t>;
    using dyn_vec_index_t = DynamicVectorIndexType;
    using dyn_vec_value_t = DynamicVectorValueType;
    using dyn_vec_vec_value_t = DynamicVectorVecValueType;
    using dyn_vec_sp_vec_t = DynamicVectorSpVecType;
    using diagnostic_t = GroupElnetDiagnostic;

    // Static states
    const MatrixType X;
    const map_cvec_index_t groups;
    const map_cvec_index_t group_sizes;
    const value_t alpha;
    const map_cvec_value_t penalty;
    const map_cvec_index_t strong_set;
    const map_cvec_index_t strong_g1;
    const map_cvec_index_t strong_g2;
    const map_cvec_index_t strong_begins;
    const map_cvec_value_t strong_A_diag;
    const map_cvec_value_t lmdas;

    // Configurations
    const size_t max_cds;
    const value_t thr;
    const value_t cond_0_thresh;
    const value_t cond_1_thresh;
    const value_t newton_tol;
    const size_t newton_max_iters;

    // Dynamic states
    value_t rsq;
    map_vec_value_t resid;
    map_vec_value_t strong_beta;
    map_vec_value_t strong_grad;
    dyn_vec_index_t active_set;
    dyn_vec_index_t active_g1;
    dyn_vec_index_t active_g2;
    dyn_vec_index_t active_begins;
    dyn_vec_index_t active_order;
    map_vec_bool_t is_active;
    dyn_vec_sp_vec_t betas;
    dyn_vec_value_t rsqs;
    dyn_vec_vec_value_t resids;
    size_t n_cds = 0;
    dyn_vec_value_t time_strong_cd;
    dyn_vec_value_t time_active_cd;
    
    explicit PinNaive(
        const MatrixType& X,
        const Eigen::Ref<const vec_index_t>& groups, 
        const Eigen::Ref<const vec_index_t>& group_sizes,
        value_t alpha, 
        const Eigen::Ref<const vec_value_t>& penalty,
        const Eigen::Ref<const vec_index_t>& strong_set, 
        const Eigen::Ref<const vec_index_t>& strong_g1,
        const Eigen::Ref<const vec_index_t>& strong_g2,
        const Eigen::Ref<const vec_index_t>& strong_begins, 
        const Eigen::Ref<const vec_value_t>& strong_A_diag,
        const Eigen::Ref<const vec_value_t>& lmdas, 
        size_t max_cds,
        value_t thr,
        value_t cond_0_thresh,
        value_t cond_1_thresh,
        value_t newton_tol,
        size_t newton_max_iters,
        value_t rsq,
        Eigen::Ref<vec_value_t> resid,
        Eigen::Ref<vec_value_t> strong_beta, 
        Eigen::Ref<vec_value_t> strong_grad,
        dyn_vec_index_t active_set,
        dyn_vec_index_t active_g1,
        dyn_vec_index_t active_g2,
        dyn_vec_index_t active_begins,
        dyn_vec_index_t active_order,
        Eigen::Ref<vec_bool_t> is_active,
        dyn_vec_sp_vec_t betas, 
        dyn_vec_value_t rsqs,
        dyn_vec_vec_value_t resids
    )
        : X(X),
          groups(groups.data(), groups.size()),
          group_sizes(group_sizes.data(), group_sizes.size()),
          alpha(alpha),
          penalty(penalty.data(), penalty.size()),
          strong_set(strong_set.data(), strong_set.size()),
          strong_g1(strong_g1.data(), strong_g1.size()),
          strong_g2(strong_g2.data(), strong_g2.size()),
          strong_begins(strong_begins.data(), strong_begins.size()),
          strong_A_diag(strong_A_diag.data(), strong_A_diag.size()),
          lmdas(lmdas.data(), lmdas.size()),
          max_cds(max_cds),
          thr(thr),
          cond_0_thresh(cond_0_thresh),
          cond_1_thresh(cond_1_thresh),
          newton_tol(newton_tol),
          newton_max_iters(newton_max_iters),
          rsq(rsq),
          resid(resid.data(), resid.size()),
          strong_beta(strong_beta.data(), strong_beta.size()),
          strong_grad(strong_grad.data(), strong_grad.size()),
          active_set(active_set),
          active_g1(active_g1),
          active_g2(active_g2),
          active_begins(active_begins),
          active_order(active_order),
          is_active(is_active.data(), is_active.size()),
          betas(betas),
          rsqs(rsqs),
          resids(resids)
    {}
};

} // namespace state
} // namespace adelie_core