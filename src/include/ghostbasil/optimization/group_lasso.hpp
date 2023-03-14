#pragma once
#include <array>
#include <vector>
#include <numeric>
#include <ghostbasil/util/exceptions.hpp>
#include <ghostbasil/util/functional.hpp>
#include <ghostbasil/util/macros.hpp>
#include <ghostbasil/util/types.hpp>
#include <ghostbasil/util/functor_iterator.hpp>
#include <ghostbasil/util/counting_iterator.hpp>
#include <ghostbasil/util/eigen/map_sparsevector.hpp>
#include <ghostbasil/matrix/forward_decl.hpp>
#include <ghostbasil/optimization/lasso.hpp>

// TODO: probably needs to change header location
#include <newton.hpp>

namespace ghostbasil {
namespace group_lasso {
    
/**
 * Pack of buffers used for fit().
 * This class is purely for convenience purposes.
 */
template <class ValueType>
struct GroupLassoBufferPack 
{
    using value_t = ValueType;
    
    util::vec_type<value_t> buffer1;
    util::vec_type<value_t> buffer2;
    util::vec_type<value_t> buffer3;

    explicit GroupLassoBufferPack(
        size_t buffer_size
    )
        : GroupLassoBufferPack(
            buffer_size, buffer_size, buffer_size
        ) 
    {}

    explicit GroupLassoBufferPack(
            size_t buffer1_size, 
            size_t buffer2_size,
            size_t buffer3_size
    )
        : buffer1(buffer1_size),
          buffer2(buffer2_size),
          buffer3(buffer3_size)
    {}
};

/**
 * Parameter pack for group lasso procedure.
 * 
 * @param   A           PSD matrix (p, p) with diagonal blocks A_{ii}. 
 *                      This matrix only needs to satisfy the properties
 *                      when looking at the sub-matrix of all strong_set groups.
 *                      The diagonal blocks are never read, 
 *                      so they can be used as storage for something else.
 * @param   groups      vector of indices into columns of A that define
 *                      the beginning index of the groups.
 *                      TODO: I think the following can be removed.
 *                      groups[i] = beginning index of A columns for group i.
 *                      Must be of size I+1 where groups[I] = p.
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
 * @param   strong_beta dense vector of coefficients of size strong_A_diag.size().
 *                      strong_beta[b:(b+p)] = coefficient for group i,
 *                      where i := strong_set[j], b := strong_begins[j], and p := group_sizes[i].
 *                      The updated coefficients will be stored here.
 * @param   strong_grad dense vector of current residuals.
 *                      See update_residual() for formula.
 *                      strong_grad[b:(b+p)] = residuals for group i,
 *                      where i := strong_set[j], b := strong_begins[j], and p := group_sizes[i].
 *                      The updated residuals will be stored here.
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
 */
template <class AType, 
          class ValueType,
          class IndexType,
          class BoolType,
          class DynamicVectorIndexType=std::vector<IndexType>,
          class DynamicVectorSpVecType=std::vector<
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
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;
    using map_cvec_index_t = Eigen::Map<const vec_index_t>;
    using dyn_vec_index_t = DynamicVectorIndexType;
    using dyn_vec_sp_vec_t = DynamicVectorSpVecType;

    AType* A = nullptr;
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
    value_t newton_tol;
    size_t newton_max_iters;
    value_t rsq;
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
    size_t n_cds;
    size_t n_lmdas;
    
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
          strong_beta(nullptr, 0),
          strong_grad(nullptr, 0),
          is_active(nullptr, 0),
          rsqs(nullptr, 0)
    {}
         
    template <class GroupsType, class GroupSizesType,
              class PenaltyType, class SSType, class SG1Type,
              class SG2Type, class SBeginsType, 
              class SADType, class LmdasType,
              class SBType, class SGType,
              class IAType, class RsqsType>
    explicit GroupLassoParamPack(
        AType& A_,
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
        value_t newton_tol_,
        size_t newton_max_iters_,
        value_t rsq_,
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
        size_t n_cds_,
        size_t n_lmdas_
    )
        : A(&A_),
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
          newton_tol(newton_tol_),
          newton_max_iters(newton_max_iters_),
          rsq(rsq_),
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
          n_cds(n_cds_),
          n_lmdas(n_lmdas_)
    {}
};

/**
 * Constructs active (feature) indices in increasing order
 * expanding group ranges as a dense vector.
 * The result is stored in out.
 * 
 * @param   pack    see GroupLassoParamPack.
 * @param   out     output vector.
 */
template <class PackType, class OutType>
GHOSTBASIL_STRONG_INLINE
void get_active_indices(
    const PackType& pack,
    OutType& out
)
{
    using index_t = typename PackType::index_t;
    using vec_t = util::vec_type<index_t>;
    
    const auto& active_set = *pack.active_set;
    const auto& active_order = *pack.active_order;
    const auto& strong_set = pack.strong_set;
    const auto& group_sizes = pack.group_sizes;
    const auto& groups = pack.groups;

    auto out_begin = out.data();
    for (size_t i = 0; i < active_order.size(); ++i) {
        const auto ss_idx = active_set[active_order[i]];
        const auto group = strong_set[ss_idx];
        const auto group_size = group_sizes[group];
        Eigen::Map<vec_t> seg(out_begin, group_size);
        seg = vec_t::LinSpaced(
            group_size, groups[group], groups[group] + group_size - 1
        );
        out_begin += group_size;
    }
    assert(out.size() == std::distance(out.data(), out_begin));
}

/**
 * Constructs active (feature) values in increasing index order.
 * The result is stored in out.
 * 
 * @param   pack    see GroupLassoParamPack.
 * @param   out     output vector.
 */
template <class PackType, class OutType>
GHOSTBASIL_STRONG_INLINE
void get_active_values(
    const PackType& pack,
    OutType& out 
)
{
    using value_t = typename PackType::value_t;
    using vec_t = util::vec_type<value_t>;

    const auto& active_set = *pack.active_set;
    const auto& active_order = *pack.active_order;
    const auto& strong_set = pack.strong_set;
    const auto& group_sizes = pack.group_sizes;
    const auto& strong_beta = pack.strong_beta;
    const auto& strong_begins = pack.strong_begins;

    auto out_begin = out.data();
    for (size_t i = 0; i < active_order.size(); ++i) {
        const auto ss_idx = active_set[active_order[i]];
        const auto group = strong_set[ss_idx];
        const auto group_size = group_sizes[group];
        Eigen::Map<vec_t> seg(out_begin, group_size);
        seg = strong_beta.segment(strong_begins[ss_idx], group_size);
        out_begin += group_size;
    }        
    assert(out.size() == std::distance(out.data(), out_begin));
}

/**
 * Computes the objective that we wish to minimize.
 * The objective is the quadratic loss + group-lasso regularization:
 * \f[
 *      \frac{1}{2} \sum_{ij} x_i^\top A_{ij} x_j - \sum_{i} x_i^\top r 
 *          + \lambda \sum_i p_i \left(
 *              \alpha ||x_i||_2 + \frac{1-\alpha}{2} ||x_i||_2^2
 *              \right)
 * \f]
 *          
 * @param   A       any square (p, p) matrix. 
 * @param   r       any vector (p,).
 * @param   groups  see description in GroupLassoParamPack.
 * @param   group_sizes see description in GroupLassoParamPack.
 * @param   alpha       elastic net proportion.
 * @param   penalty penalty factor for each group.
 * @param   lmda    group-lasso regularization.
 * @param   beta    coefficient vector.
 */
template <class AType, class RType, 
          class GroupsType, class GroupSizesType,
          class ValueType, class PenaltyType, class BetaType>
GHOSTBASIL_STRONG_INLINE 
auto objective(
    const AType& A,
    const RType& r,
    const GroupsType& groups,
    const GroupSizesType& group_sizes,
    ValueType alpha,
    const PenaltyType& penalty,
    ValueType lmda,
    const BetaType& beta)
{
    ValueType p_ = 0.0;
    for (size_t j = 0; j < groups.size()-1; ++j) {
        const auto begin = groups[j];
        const auto size = group_sizes[j];
        const auto b_norm2 = beta.segment(begin, size).norm();
        p_ += penalty[j] * b_norm2 * (
            alpha + (1-alpha) / 2 * b_norm2
        );
    }
    p_ *= lmda;
    return 0.5 * A.quad_form(beta) - beta.dot(r) + p_;
}

/**
 * Updates the convergence measure using variance of each direction.
 * 
 * @param   convg_measure   convergence measure to update.
 * @param   del             vector difference in a group coefficient.
 * @param   var             vector of variance along each direction of coefficient.
 */
template <class ValueType, class DelType, class VarType>
GHOSTBASIL_STRONG_INLINE 
void update_convergence_measure(
    ValueType& convg_measure, 
    const DelType& del, 
    const VarType& var)
{
    const auto convg_measure_curr = del.dot(var.cwiseProduct(del)) / del.size();
    convg_measure = std::max(convg_measure, convg_measure_curr);
}

/**
 * Updates \f$R^2\f$ given the group variance vector, 
 * group coefficient difference (new minus old), 
 * and the current residual vector.
 * 
 * @param   rsq     \f$R^2\f$ to update.
 * @param   del     new coefficient minus old coefficient.
 * @param   coeff_new   new coefficient.
 * @param   var     variance along each coordinate of group.
 * @param   r       current residual correlation vector for group.
 * @param   s       PGR regularization parameter.
 */
template <class ValueType, class DelType, class CoeffNewType,
          class VarType, class RType>
GHOSTBASIL_STRONG_INLINE
void update_rsq(
    ValueType& rsq,
    const DelType& del,
    const CoeffNewType& coeff_new,
    const VarType& var,
    const RType& r
)
{
    const auto sum = 2 * coeff_new.array() - del.array();
    rsq += (
        del.array() * (2 * r.array() - var.array() * sum)
    ).sum();
}

/**
 * Solves the solution for the equation (w.r.t. \f$x\f$):
 * \f[
 *      minimize \frac{1}{2} x^\top L x - x^\top v 
 *          + l_1 ||x||_2 + \frac{l_2}{2} ||x||_2^2
 * \f]
 *      
 * @param   L       vector representing a diagonal PSD matrix.
 *                  Must have max(L + s) > 0. 
 *                  L.size() <= buffer1.size().
 * @param   v       any vector.  
 * @param   l1      L2-norm penalty. Must be >= 0.
 * @param   l2      L2 penalty. Must be >= 0.
 * @param   tol         Newton's method tolerance of closeness to 0.
 * @param   max_iters   maximum number of iterations of Newton's method.
 * @param   x           solution vector.
 * @param   iters       number of Newton's method iterations taken.
 * @param   buffer1     any vector with L.size() <= buffer1.size().
 * @param   buffer2     any vector with L.size() <= buffer2.size().
 */
template <class LType, class VType, class ValueType, 
          class XType, class BufferType>
GHOSTBASIL_STRONG_INLINE
void update_coefficients(
    const LType& L,
    const VType& v,
    ValueType l1,
    ValueType l2,
    ValueType tol,
    size_t max_iters,
    XType& x,
    size_t& iters,
    BufferType& buffer1,
    BufferType& buffer2
)
{
    glstudy::newton_abs_solver(
        L, v, l1, l2, tol, max_iters,
        x, iters, buffer1, buffer2
    );
}

namespace internal {
    
template <int i_pol=-1, int j_pol=-1>
struct UpdateResidual;

template <>
struct UpdateResidual<0, 0>
{
    template <class AijType, class DelType, 
              class GradType, class BufferType>
    GHOSTBASIL_STRONG_INLINE
    static void eval(
        const AijType& A_ij,
        const DelType& del_j,
        GradType& grad_i,
        BufferType& 
    )
    {
        grad_i[0] -= A_ij(0, 0) * del_j[0];
    }
};

template <>
struct UpdateResidual<0, 1>
{
    template <class AijType, class DelType, 
              class GradType, class BufferType>
    GHOSTBASIL_STRONG_INLINE
    static void eval(
        const AijType& A_ij,
        const DelType& del_j,
        GradType& grad_i,
        BufferType& 
    )
    {
        const auto A_ij_ = A_ij.row(0);
        grad_i[0] -= A_ij_.dot(del_j);
    }
};

template <>
struct UpdateResidual<0, -1>
{
    template <class AijType, class DelType, 
              class GradType, class BufferType>
    GHOSTBASIL_STRONG_INLINE
    static void eval(
        const AijType& A_ij,
        const DelType& del_j,
        GradType& grad_i,
        BufferType& buffer
    )
    {
        if (A_ij.cols() == 1) {
            UpdateResidual<0,0>::eval(
                A_ij, del_j, grad_i, buffer
            );
        } else {
            UpdateResidual<0,1>::eval(
                A_ij, del_j, grad_i, buffer
            );
        }
    }
};

template <>
struct UpdateResidual<1, 0>
{
    template <class AijType, class DelType, 
              class GradType, class BufferType>
    GHOSTBASIL_STRONG_INLINE
    static void eval(
        const AijType& A_ij,
        const DelType& del_j,
        GradType& grad_i,
        BufferType& 
    )
    {
        const auto A_ij_ = A_ij.col(0);
        grad_i -= A_ij_ * del_j[0];
    }
};

template <>
struct UpdateResidual<1, 1>
{
    template <class AijType, class DelType, 
              class GradType, class BufferType>
    GHOSTBASIL_STRONG_INLINE
    static void eval(
        const AijType& A_ij,
        const DelType& del_j,
        GradType& grad_i,
        BufferType& buffer
    )
    {
        // TODO: performance may be boosted with grad_i.noalias()
        auto buffer_ = buffer.head(A_ij.rows());
        std::cerr << "del_j size: " << del_j.size() << std::endl;
        std::cerr << "A_ij shape: " << A_ij.rows() << ' ' << A_ij.cols() << std::endl;
        buffer_.noalias() = A_ij * del_j;
        grad_i -= buffer_;
    }
};

template <>
struct UpdateResidual<1, -1>
{
    template <class AijType, class DelType, 
              class GradType, class BufferType>
    GHOSTBASIL_STRONG_INLINE
    static void eval(
        const AijType& A_ij,
        const DelType& del_j,
        GradType& grad_i,
        BufferType& buffer
    )
    {
        if (A_ij.cols() == 1) {
            UpdateResidual<1,0>::eval(
                A_ij, del_j, grad_i, buffer
            );
        } else {
            UpdateResidual<1,1>::eval(
                A_ij, del_j, grad_i, buffer
            );
        }
    }
};

template <int j_pol>
struct UpdateResidual<-1, j_pol>
{
    template <class AijType, class DelType, 
              class GradType, class BufferType>
    GHOSTBASIL_STRONG_INLINE
    static void eval(
        const AijType& A_ij,
        const DelType& del_j,
        GradType& grad_i,
        BufferType& buffer
    )
    {
        if (A_ij.rows() == 1) {
            UpdateResidual<0, j_pol>::eval(
                A_ij, del_j, grad_i, buffer
            );
        } else {
            UpdateResidual<1, j_pol>::eval(
                A_ij, del_j, grad_i, buffer
            );
        }
    }
};


} // namespace internal

/**
 * Updates residual vector
 * for block i with the updated block j coefficient.
 * This function assumes that i != j.
 * The current residual vector for block i is:
 * \f[
 *      r_i - \sum_{k\neq i} A_{ik} \beta_k
 * \f]
 *
 * The template parameters denote policies for the two group sizes.
 * A value of:
 *  - -1 == dynamically check if group is size 1 or not.
 *  - 0 == group is size 1.
 *  - 1 == group is size > 1.
 *
 * @tparam  i_pol       policy for group i based on size.
 * @tparam  j_pol       policy for group j based on size.
 * @param   A_ij        matrix in objective.
 * @param   del_j       \beta_j^{new} - \beta_j^{old}.
 * @param   grad_i      vector of current residual vector for block i.
 */
template <int i_pol=-1, int j_pol=-1,
          class AijType, class DelType, 
          class GradType, class BufferType>
GHOSTBASIL_STRONG_INLINE
void update_residual(
    const AijType& A_ij,
    const DelType& del_j,
    GradType& grad_i,
    BufferType& buffer
)
{
    internal::UpdateResidual<i_pol, j_pol>::eval(
        A_ij, del_j, grad_i, buffer
    );
}

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
    AdditionalStepType additional_step=AdditionalStepType()
)
{
    using pack_t = std::decay_t<PackType>;
    using value_t = typename pack_t::value_t;

    const auto& A = *pack.A;
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
    auto& rsq = pack.rsq;

    convg_measure = 0;
    // iterate over the groups of size 1
    for (auto it = g1_begin; it != g1_end; ++it) {
        const auto ss_idx = *it;              // index to strong set
        const auto k = strong_set[ss_idx];    // actual group index
        const auto ss_value_begin = strong_begins[ss_idx]; // value begin index at ss_idx
        auto& ak = strong_beta[ss_value_begin]; // corresponding beta
        const auto gk = strong_grad[ss_value_begin]; // corresponding residuals
        const auto A_kk = strong_A_diag[ss_value_begin];  // corresponding A diagonal 
        const auto pk = penalty[k];

        const auto ak_old = ak;
        const auto gk_shifted = gk - A_kk * ak_old;

        // update coefficient
        lasso::update_coefficient(
            ak, A_kk, l1, l2, pk, gk_shifted
        );

        if (ak_old == ak) continue;

        // Note: order matters! Purposely put here so that A gets updated properly if iterating over strong set.
        // See fit() with active set update.
        additional_step(ss_idx);

        auto del = ak - ak_old;

        // update measure of convergence
        lasso::update_convergence_measure(convg_measure, del, A_kk);

        lasso::update_rsq(rsq, ak_old, ak, A_kk, gk_shifted);

        // update gradient-like quantity
        
        // iterate over the groups of size 1
            std::cerr << "b" << std::endl;
        for (auto jt = g1_begin; jt != it; ++jt) {
            const auto ss_idx_j = *jt;
            const auto j = strong_set[ss_idx_j];
            const auto A_jk = A.template block<1, 1>(groups[j], groups[k]);
            const util::vec_type<value_t, 1> del_k(del);
            auto sg_j = strong_grad.template segment<1>(strong_begins[ss_idx_j]);
            update_residual<0, 0>(A_jk, del_k, sg_j, buffer1);
        }
            std::cerr << "bb" << std::endl;
        
            std::cerr << "c" << std::endl;
        for (auto jt = std::next(it); jt != g1_end; ++jt) {
            const auto ss_idx_j = *jt;
            const auto j = strong_set[ss_idx_j];
            const auto A_jk = A.template block<1, 1>(groups[j], groups[k]);
            const util::vec_type<value_t, 1> del_k(del);
            auto sg_j = strong_grad.template segment<1>(strong_begins[ss_idx_j]);
            update_residual<0, 0>(A_jk, del_k, sg_j, buffer1);
        }
            std::cerr << "cc" << std::endl;

        // iterate over the groups of dynamic size
            std::cerr << "d" << std::endl;
        for (auto jt = g2_begin; jt != g2_end; ++jt) {
            const auto ss_idx_j = *jt;
            const auto j = strong_set[ss_idx_j];
            const auto groupj_size = group_sizes[j];
            const auto A_jk = A.col(groups[k]).segment(groups[j], groupj_size);
            const util::vec_type<value_t, 1> del_k(del);
            auto sg_j = strong_grad.segment(
                strong_begins[ss_idx_j], groupj_size
            );
            update_residual<1, 0>(A_jk, del_k, sg_j, buffer1);
        }
            std::cerr << "dd" << std::endl;
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

        // save old beta in buffer
        auto ak_old = buffer3.head(ak.size());
        ak_old = ak; 

        // update group coefficients
        size_t iters;
        update_coefficients(
            A_kk, gk, l1 * pk, l2 * pk, 
            newton_tol, newton_max_iters,  // try only 1 newton iteration
            ak, iters, buffer1, buffer2
        );
        //if (iters >= newton_max_iters) {
        //    throw util::group_lasso_max_newton_iters();
        //}

        if ((ak_old.array() == ak.array()).all()) continue;
        
        // Note: order matters! Purposely put here so that A gets updated properly if iterating over strong set.
        // See fit() with active set update.
        additional_step(ss_idx);

        // use same buffer as ak_old to store difference
        auto& del = ak_old;
        del = ak - ak_old;

        // update measure of convergence
        update_convergence_measure(convg_measure, del, A_kk);

        // update rsq
        update_rsq(rsq, del, ak, A_kk, gk);

        // update gradient-like quantity
        
        // iterate over the groups of size 1
            std::cerr << "e" << std::endl;
        for (auto jt = g1_begin; jt != g1_end; ++jt) {
            const auto ss_idx_j = *jt;
            const auto j = strong_set[ss_idx_j];
            const auto A_jk = A.row(groups[j]).segment(groups[k], gsize);
            auto sg_j = strong_grad.template segment<1>(strong_begins[ss_idx_j]);
            update_residual<0, 1>(A_jk, del, sg_j, buffer1);
        }
            std::cerr << "ee" << std::endl;

        // iterate over the groups of dynamic size
            std::cerr << "f" << std::endl;
        for (auto jt = g2_begin; jt != it; ++jt) {
            const auto ss_idx_j = *jt;
            const auto j = strong_set[ss_idx_j];
            const auto groupj_size = group_sizes[j];
            const auto A_jk = A.block(
                groups[j], groups[k], groupj_size, gsize
            );
            auto sg_j = strong_grad.segment(
                strong_begins[ss_idx_j], groupj_size
            );
            update_residual<1, 1>(A_jk, del, sg_j, buffer1);
        }
            std::cerr << "ff" << std::endl;

            std::cerr << "g" << std::endl;
        for (auto jt = std::next(it); jt != g2_end; ++jt) {
            const auto ss_idx_j = *jt;
            const auto j = strong_set[ss_idx_j];
            const auto groupj_size = group_sizes[j];
            const auto A_jk = A.block(
                groups[j], groups[k], groupj_size, gsize
            );
            auto sg_j = strong_grad.segment(
                strong_begins[ss_idx_j], groupj_size
            );
            update_residual<1, 1>(A_jk, del, sg_j, buffer1);
        }
            std::cerr << "gg" << std::endl;
    }
}

/**
 * Applies multiple blockwise coordinate descent on the active set 
 * to minimize the ghostbasil objective with group-lasso penalty.
 * See "objective" function for the objective of interest.
 *
 * @param   pack        see GroupLassoParamPack.
 * @param   lmda_idx    index into the lambda sequence for logging purposes.
 * @param   active_beta_diff    buffer to store difference between active betas
 *                              before and after running coordinate descent.
 * @param   buffer1     see coordinate_descent.
 * @param   buffer2     see coordinate_descent.
 * @param   buffer3     see coordinate_descent.
 * @param   check_user_interrupt    functor that checks for user interruption.
 */
template <class PackType, 
          class ABDiffType,
          class BufferType, 
          class CUIType = util::no_op>
GHOSTBASIL_STRONG_INLINE
void group_lasso_active(
    PackType&& pack,
    size_t lmda_idx,
    ABDiffType& active_beta_diff,
    BufferType& buffer1,
    BufferType& buffer2,
    BufferType& buffer3,
    CUIType check_user_interrupt = CUIType())
{
    using pack_t = std::decay_t<PackType>;
    using value_t = typename pack_t::value_t;
    using vec_value_t = typename pack_t::vec_value_t;

    const auto& A = *pack.A;
    const auto& groups = pack.groups;
    const auto& group_sizes = pack.group_sizes;
    const auto& strong_set = pack.strong_set;
    const auto& strong_g1 = pack.strong_g1;
    const auto& strong_g2 = pack.strong_g2;
    const auto& strong_begins = pack.strong_begins;
    const auto& active_set = *pack.active_set;
    const auto& active_g1 = *pack.active_g1;
    const auto& active_g2 = *pack.active_g2;
    const auto& active_begins = *pack.active_begins;
    const auto& strong_beta = pack.strong_beta;
    const auto& is_active = pack.is_active;
    const auto thr = pack.thr;
    const auto max_cds = pack.max_cds;
    auto& strong_grad = pack.strong_grad;
    auto& n_cds = pack.n_cds;

    // TODO: write some similar assert checks
    //internal::lasso_assert_valid_inputs(
    //        A, s, strong_set, strong_order, 
    //        active_set, active_order, active_set_ordered,
    //        is_active, strong_A_diag,
    //        strong_beta, strong_grad);

    Eigen::Map<vec_value_t> ab_diff_view(
        active_beta_diff.data(), active_beta_diff.size()
    );
    
    // save old active beta
    for (size_t i = 0; i < active_set.size(); ++i) {
        const auto ss_idx_group = active_set[i];
        const auto ss_group = strong_set[ss_idx_group];
        const auto ss_group_size = group_sizes[ss_group];
        const auto sb_begin = strong_begins[ss_idx_group];
        const auto sb = strong_beta.segment(sb_begin, ss_group_size);
        const auto ab_begin = active_begins[i];
        auto ab_diff_view_curr = ab_diff_view.segment(ab_begin, ss_group_size);
        ab_diff_view_curr = sb;
    }
    
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

    while (1) {
        check_user_interrupt(n_cds);
        ++n_cds;
        value_t convg_measure;
        coordinate_descent(
            pack, ag1_begin, ag1_end, ag2_begin, ag2_end,
            lmda_idx, convg_measure, buffer1, buffer2, buffer3
        );
        if (convg_measure < thr) break;
        if (n_cds >= max_cds) throw util::max_cds_error(lmda_idx);
    }
    
    // compute new active beta - old active beta
    for (size_t i = 0; i < active_set.size(); ++i) {
        const auto ss_idx_group = active_set[i];
        const auto ss_group = strong_set[ss_idx_group];
        const auto ss_group_size = group_sizes[ss_group];
        const auto sb_begin = strong_begins[ss_idx_group];
        const auto sb = strong_beta.segment(sb_begin, ss_group_size);
        const auto ab_begin = active_begins[i];
        auto ab_diff_view_curr = ab_diff_view.segment(ab_begin, ss_group_size);
        ab_diff_view_curr = sb - ab_diff_view_curr;
    }

    // update strong gradient for non-active strong variables

    // optimization: if active set is empty or active set is the same as strong set.
    if ((ab_diff_view.size() == 0) ||
        (active_set.size() == strong_set.size())) return;

    const auto sg1_begin = strong_g1.data();
    const auto sg1_end = strong_g1.data() + strong_g1.size();
    const auto sg2_begin = strong_g2.data();
    const auto sg2_end = strong_g2.data() + strong_g2.size();
 
    // iterate over the strong groups of size 1
    for (auto jt = sg1_begin; jt != sg1_end; ++jt) {
        const auto j_idx = *jt;
        if (is_active[j_idx]) continue;
         
        const auto j = strong_set[j_idx];
        //const auto groupj_size = group_sizes[j];
        auto sg_j = strong_grad.template segment<1>(
            strong_begins[j_idx]//, groupj_size
        );

        // iterate over the active groups of size 1
        for (auto it = active_g1.begin(); it != active_g1.end(); ++it) {
            const auto i_idx = *it;
            const auto i = strong_set[active_set[i_idx]];
            //const auto groupi_size = group_sizes[i];
            const auto ab_begin = active_begins[i_idx];
            const auto ab_diff_view_curr = ab_diff_view.template segment<1>(
                ab_begin//, groupi_size
            );
            const auto A_ji = A.template block<1, 1>(
                groups[j], groups[i]//, groupj_size, groupi_size
            );
            update_residual<0, 0>(A_ji, ab_diff_view_curr, sg_j, buffer3);
        }
        
        // iterate over the active groups of dynamic size
        for (auto it = active_g2.begin(); it != active_g2.end(); ++it) {
            const auto i_idx = *it;
            const auto i = strong_set[active_set[i_idx]];
            const auto groupi_size = group_sizes[i];
            const auto ab_begin = active_begins[i_idx];
            const auto ab_diff_view_curr = ab_diff_view.segment(ab_begin, groupi_size);
            const auto A_ji = A.row(groups[j]).segment(
                groups[i], groupi_size
            );
            update_residual<0, 1>(A_ji, ab_diff_view_curr, sg_j, buffer3);
        }
    }

    // iterate over the groups of dynamic size
    for (auto jt = sg2_begin; jt != sg2_end; ++jt) {
        const auto j_idx = *jt;
        if (is_active[j_idx]) continue;

        const auto j = strong_set[j_idx];
        const auto groupj_size = group_sizes[j];
        auto sg_j = strong_grad.segment(
            strong_begins[j_idx], groupj_size
        );

        // iterate over the active groups of size 1
        for (auto it = active_g1.begin(); it != active_g1.end(); ++it) {
            const auto i_idx = *it;
            const auto i = strong_set[active_set[i_idx]];
            //const auto groupi_size = group_sizes[i];
            const auto ab_begin = active_begins[i_idx];
            const auto ab_diff_view_curr = ab_diff_view.template segment<1>(ab_begin);
            const auto A_ji = A.col(groups[i]).segment(
                groups[j], groupj_size
            );
            update_residual<1, 0>(A_ji, ab_diff_view_curr, sg_j, buffer3);
        }
        
        // iterate over the active groups of dynamic size
        for (auto it = active_g2.begin(); it != active_g2.end(); ++it) {
            const auto i_idx = *it;
            const auto i = strong_set[active_set[i_idx]];
            const auto groupi_size = group_sizes[i];
            const auto ab_begin = active_begins[i_idx];
            const auto ab_diff_view_curr = ab_diff_view.segment(ab_begin, groupi_size);
            const auto A_ji = A.block(
                groups[j], groups[i], groupj_size, groupi_size
            );
            update_residual<1, 1>(A_ji, ab_diff_view_curr, sg_j, buffer3);
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
          class CUIType = util::no_op>
inline void fit(
    PackType&& pack,
    CUIType check_user_interrupt = CUIType())
{
    using pack_t = std::decay_t<PackType>;
    using value_t = typename pack_t::value_t;
    using index_t = typename pack_t::index_t;
    using sp_vec_value_t = typename pack_t::sp_vec_value_t;

    auto& A = *pack.A;
    const auto& groups = pack.groups;
    const auto& group_sizes = pack.group_sizes;
    const auto& strong_set = pack.strong_set;
    const auto& strong_g1 = pack.strong_g1;
    const auto& strong_g2 = pack.strong_g2;
    const auto& strong_beta = pack.strong_beta;
    const auto& lmdas = pack.lmdas;
    const auto thr = pack.thr;
    const auto max_cds = pack.max_cds;
    auto& active_set = *pack.active_set;
    auto& active_g1 = *pack.active_g1;
    auto& active_g2 = *pack.active_g2;
    auto& active_begins = *pack.active_begins;
    auto& active_order = *pack.active_order;
    auto& is_active = pack.is_active;
    auto& betas = *pack.betas;
    auto& rsqs = pack.rsqs;
    auto& rsq = pack.rsq;
    auto& n_cds = pack.n_cds;
    auto& n_lmdas = pack.n_lmdas;
    const auto p = A.cols();

    //internal::lasso_assert_valid_inputs(
    //        A, s, strong_set, strong_order, 
    //        active_set, active_order, active_set_ordered, 
    //        is_active, strong_A_diag,
    //        strong_beta, strong_grad);

    assert(betas.size() == lmdas.size());
    assert(rsqs.size() == lmdas.size());

    // buffers for the routine
    const auto max_group_size = group_sizes.maxCoeff();
    GroupLassoBufferPack<value_t> buffer_pack(max_group_size);
    
    // buffer to store final result
    std::vector<index_t> active_beta_indices;
    std::vector<value_t> active_beta_ordered;
    
    // buffer for internal routine group_lasso_active
    std::vector<value_t> active_beta_diff;
    
    // compute size of active_beta_diff (needs to be exact)
    size_t active_beta_size = 0;
    if (active_set.size()) {
        const auto last_idx = active_set.size()-1;
        const auto last_group = strong_set[active_set[last_idx]];
        const auto group_size = group_sizes[last_group];
        active_beta_size = active_begins[last_idx] + group_size;
    }

    // allocate buffers with optimization
    active_beta_indices.reserve(strong_beta.size());
    active_beta_ordered.reserve(strong_beta.size());
    active_beta_diff.reserve(strong_beta.size());
    active_beta_diff.resize(active_beta_size);
    
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
            
            std::cerr << "a" << std::endl;
            A.cache(groups[group], group_size);
            std::cerr << "aa" << std::endl;
        }
    };

    const auto lasso_active_and_update = [&](size_t l) {
        group_lasso_active(
                pack, l, 
                active_beta_diff, 
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
            coordinate_descent(
                    pack,
                    strong_g1.data(), strong_g1.data() + strong_g1.size(),
                    strong_g2.data(), strong_g2.data() + strong_g2.size(),
                    l, convg_measure,
                    buffer_pack.buffer1,
                    buffer_pack.buffer2,
                    buffer_pack.buffer3,
                    add_active_set);
            const bool new_active_added = (old_active_size < active_set.size());

            if (new_active_added) {
                // update active_begins
                auto new_abd_size = active_beta_diff.size();
                active_begins.resize(active_set.size());
                for (size_t i = old_active_size; i < active_begins.size(); ++i) {
                    active_begins[i] = new_abd_size;
                    const auto curr_group = strong_set[active_set[i]];
                    const auto curr_size = group_sizes[curr_group];
                    new_abd_size += curr_size;
                }

                // update active_beta_diff size
                active_beta_diff.resize(new_abd_size);
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
        active_beta_indices.resize(active_beta_diff.size());
        get_active_indices(pack, active_beta_indices);
        
        // update active_beta_ordered
        active_beta_ordered.resize(active_beta_diff.size());
        get_active_values(pack, active_beta_ordered);

        // order the strong betas 
        Eigen::Map<const sp_vec_value_t> beta_map(
                p,
                active_beta_indices.size(),
                active_beta_indices.data(),
                active_beta_ordered.data());

        betas[l] = beta_map;
        rsqs[l] = rsq;
        ++n_lmdas;

        // make sure to do at least 3 lambdas.
        if (l < 2) continue;

        // early stop if R^2 criterion is fulfilled.
        if (lasso::check_early_stop_rsq(rsqs[l-2], rsqs[l-1], rsqs[l])) break;
    }
}

/**
 * TODO:
 *  - (nothing to change in current impl of lasso but basil will need to know this)
 *      modify A to have Q in the diagonal blocks 
 *      and off-diagonals are Q_i^T A_{ij} Q_j among the strong variables.
 */
    
} // namespace group_lasso
} // namespace ghostbasil