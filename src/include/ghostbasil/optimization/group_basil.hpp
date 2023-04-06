#pragma once
#include <vector>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <ghostbasil/optimization/basil.hpp>
#include <ghostbasil/util/types.hpp>

namespace ghostbasil {
namespace group_lasso {

/**
 * @brief Transforms the data matrix X so that each block of data corresponding to a group is 
 * replaced with the principal components. In the process, we also save the l2 norm squared of each column.
 * 
 * @tparam XType    float matrix type.
 * @tparam GroupsType   integer vector type.
 * @tparam GroupSizesType   integer vector type.
 * @tparam DType    float vector type.
 * @param X             data matrix.
 * @param groups        indices to the beginning of each group.
 * @param group_sizes   sizes of each group.
 * @param n_threads     number of threads to parallelize.
 * @param d             l2 norm squared of each column.
 */
template <class XType, class GroupsType, class GroupSizesType, class DType>
void transform_data(
    XType& X,
    const GroupsType& groups,
    const GroupSizesType& group_sizes,
    size_t n_threads,
    DType& d,
    std::vector<Eigen::BDCSVD<Eigen::MatrixXd>>& decomps
) 
{
    decomps.resize(groups.size());

#pragma omp parallel for schedule(auto) num_threads(n_threads)
    for (size_t i = 0; i < groups.size(); ++i) {
        const auto gi = groups[i];
        const auto gi_size = group_sizes[i];
        auto Xi = X.block(0, gi, X.rows(), gi_size);
        const auto n = Xi.rows();
        const auto p = Xi.cols();
        const auto m = std::min(n, p);
        auto& solver = decomps[i];
        solver.compute(Xi, Eigen::ComputeThinU | Eigen::ComputeFullV);
        const auto& U = solver.matrixU();
        const auto& D = solver.singularValues();

        Xi.block(0, m, n, p-m).setZero();
        auto Xi_sub = Xi.block(0, 0, n, m);
        Xi_sub.array() = U.array().rowwise() * D.transpose().array();
        d.segment(gi, m) = D.array().square();
        d.segment(gi + m, p-m).setZero();
    }
}

/**
 * Checks the KKT condition on the sequence of lambdas and the fitted coefficients.
 * 
 * NOTE: This version cannot be parallelized like in lasso.
 * The lasso version had all group sizes of 1, which makes KKT check very easy.
 * Group lasso version needs to read arbitrary groups of gradient to check KKT.
 * Then, synchronization is required, which is no bueno.
 *
 * @param   A       covariance matrix.
 * @param   r       correlation vector.
 * @param   alpha   elastic net penalty.
 * @param   lmdas   downward sequence of L1 regularization parameter.
 * @param   betas   each element i corresponds to a (sparse) vector
 *                  of the solution at lmdas[i].
 * @param   is_strong   a functor that checks if feature i is strong.
 * @param   n_threads   number of threads to use in OpenMP.
 * @param   grad        a dense vector that represents the (negative) gradient
 *                      right before the first value in lmdas (going downwards)
 *                      that fails the KKT check.
 *                      If KKT fails at the first lambda, grad is unchanged.
 * @param   grad_next   a dense vector that represents the (negative) gradient
 *                      right at the first value in lmdas (going downwards)
 *                      that fails the KKT check.
 *                      This is really used for optimizing memory allocation.
 *                      User should not need to access this directly.
 *                      It is undefined-behavior accessing this after the call.
 *                      It just has to be initialized to the same size as grad.
 */
template <class AType, class RType, class GroupsType,
          class GroupSizesType, class ValueType, class PenaltyType,
          class LmdasType, class BetasType, class ISType, class GradType>
GHOSTBASIL_STRONG_INLINE 
auto check_kkt(
    const AType& A, 
    const RType& r,
    const GroupsType& groups,
    const GroupSizesType& group_sizes,
    ValueType alpha,
    const PenaltyType& penalty,
    const LmdasType& lmdas, 
    const BetasType& betas,
    const ISType& is_strong,
    size_t n_threads,
    GradType& grad,
    GradType& grad_next,
    GradType& abs_grad,
    GradType& abs_grad_next)
{
    assert(r.size() == grad.size());
    assert(grad.size() == grad_next.size());
    assert(betas.size() == lmdas.size());
    assert(abs_grad.size() == groups.size());
    assert(abs_grad.size() == abs_grad_next.size());

    size_t i = 0;
    auto alpha_c = 1 - alpha;

    for (; i < lmdas.size(); ++i) {
        const auto& beta_i = betas[i];
        const auto lmda = lmdas[i];
        
        // compute full gradient vector
        const auto beta_i_inner = beta_i.innerIndexPtr();
        const auto beta_i_value = beta_i.valuePtr();
        const auto beta_i_nnz = beta_i.nonZeros();
        grad_next = r;
        for (size_t i = 0; i < beta_i_nnz; ++i) {
            grad_next -= A.col(beta_i_inner[i]) * beta_i_value[i];
        } 

        // check KKT
        std::atomic<bool> kkt_fail(false);

#pragma omp parallel for schedule(static) num_threads(n_threads)
        for (size_t k = 0; k < groups.size(); ++k) {            
            const auto gk = groups[k];
            const auto gk_size = group_sizes[k];
            const auto grad_k = grad_next.segment(gk, gk_size);

            // Just omit the KKT check for strong variables.
            // If KKT failed previously, just do a no-op until loop finishes.
            // This is because OpenMP doesn't allow break statements.
            bool kkt_fail_raw = kkt_fail.load(std::memory_order_relaxed);
            if (kkt_fail_raw) continue;
            
            const auto pk = penalty[k];
            const auto abs_grad_k = (grad_k - (lmda * alpha_c * pk) * beta_i.segment(gk, gk_size)).norm();
            abs_grad_next[k] = abs_grad_k;
            if (is_strong(k) || (abs_grad_k <= lmda * alpha * pk)) continue;
            kkt_fail.store(true, std::memory_order_relaxed);
        }

        if (kkt_fail.load(std::memory_order_relaxed)) break; 
        else {
            grad.swap(grad_next);
            abs_grad.swap(abs_grad_next);
        }
    }

    return i;
}

/**
 * @brief 
 * Checkpoint class for group basil routine.
 * This class contains the minimal state variables that must be provided
 * to generate a full group basil state object.
 * 
 * @tparam ValueType    float type.
 * @tparam IndexType    index type.
 * @tparam BoolType     boolean type.
 */
template <class ValueType,
          class IndexType,
          class BoolType>
struct GroupBasilCheckpoint
{
    using value_t = ValueType;
    using index_t = IndexType;
    using bool_t = BoolType;
    using vec_value_t = util::vec_type<value_t>;

    template <class T>
    using dyn_vec_t = std::vector<T>;
    using dyn_vec_value_t = dyn_vec_t<value_t>;
    using dyn_vec_index_t = dyn_vec_t<index_t>;
    using dyn_vec_bool_t = dyn_vec_t<bool_t>;

    bool is_initialized = false;
    dyn_vec_index_t strong_set; 
    dyn_vec_index_t strong_g1; 
    dyn_vec_index_t strong_g2; 
    dyn_vec_index_t strong_begins; 
    dyn_vec_index_t strong_order;
    dyn_vec_value_t strong_beta;
    dyn_vec_value_t strong_grad;
    dyn_vec_value_t strong_A_diag;
    dyn_vec_index_t active_set;
    dyn_vec_index_t active_g1;
    dyn_vec_index_t active_g2;
    dyn_vec_index_t active_begins;
    dyn_vec_index_t active_order;
    //dyn_vec_index_t active_set_ordered;
    dyn_vec_bool_t is_active;
    vec_value_t grad;
    vec_value_t abs_grad;
    value_t rsq;

    explicit GroupBasilCheckpoint() =default;

    template <class VecIndexType, class VecValueType, 
              class VecBoolType, class GradType, class AbsGradType>
    explicit GroupBasilCheckpoint(
        const VecIndexType& strong_set_,
        const VecIndexType& strong_g1_,
        const VecIndexType& strong_g2_,
        const VecIndexType& strong_begins_,
        const VecIndexType& strong_order_,
        const VecValueType& strong_beta_,
        const VecValueType& strong_grad_,
        const VecValueType& strong_A_diag_,
        const VecIndexType& active_set_,
        const VecIndexType& active_g1_,
        const VecIndexType& active_g2_,
        const VecIndexType& active_begins_,
        const VecIndexType& active_order_,
        //const VecIndexType& active_set_ordered_,
        const VecBoolType& is_active_,
        const GradType& grad_,
        const AbsGradType& abs_grad_,
        value_t rsq_
    )
        : is_initialized(true),
          strong_set(strong_set_),
          strong_g1(strong_g1_),
          strong_g2(strong_g2_),
          strong_begins(strong_begins_),
          strong_order(strong_order_),
          strong_beta(strong_beta_),
          strong_grad(strong_grad_),
          strong_A_diag(strong_A_diag_),
          active_set(active_set_),
          active_g1(active_g1_),
          active_g2(active_g2_),
          active_begins(active_begins_),
          active_order(active_order_),
          //active_set_ordered(active_set_ordered_),
          is_active(is_active_),
          grad(grad_),
          abs_grad(abs_grad_),
          rsq(rsq_)
    {}

    template <class BasilStateType>
    GroupBasilCheckpoint& operator=(BasilStateType&& bs)
    {
        is_initialized = true;
        strong_set = std::move(bs.strong_set);
        strong_g1 = std::move(bs.strong_g1);
        strong_g2 = std::move(bs.strong_g2);
        strong_begins = std::move(bs.strong_begins);
        strong_order = std::move(bs.strong_order);
        strong_beta = std::move(bs.strong_beta);
        strong_grad = std::move(bs.strong_grad);
        strong_A_diag = std::move(bs.strong_A_diag);
        active_set = std::move(bs.active_set);
        active_g1 = std::move(bs.active_g1);
        active_g2 = std::move(bs.active_g2);
        active_begins = std::move(bs.active_begins);
        active_order = std::move(bs.active_order);
        //active_set_ordered = std::move(bs.active_set_ordered);
        is_active = std::move(bs.is_active);
        grad = std::move(bs.grad);
        abs_grad = std::move(bs.abs_grad);
        rsq = std::move(bs.rsq_prev_valid);
        return *this;
    }
};

/**
 * @brief 
 * State class for the group basil routine.
 * This class contains the full state variables that describes the state of the group basil algorithm.
 * 
 * @tparam AType        matrix type for covariance matrix.
 * @tparam ValueType    float type.
 * @tparam IndexType    index type.
 * @tparam BoolType     boolean type.
 */
template <class AType,
          class ValueType,
          class IndexType,
          class BoolType>
struct GroupBasilState
{
    using value_t = ValueType;
    using index_t = IndexType;
    using bool_t = BoolType;
    using sp_vec_value_t = util::sp_vec_type<value_t, Eigen::ColMajor, index_t>;
    using vec_value_t = util::vec_type<value_t>;
    using vec_index_t = util::vec_type<index_t>;
    using vec_bool_t = util::vec_type<bool_t>;
    using map_vec_value_t = Eigen::Map<vec_value_t>;
    using map_vec_index_t = Eigen::Map<vec_index_t>;
    using map_vec_bool_t = Eigen::Map<vec_bool_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;
    using map_cvec_index_t = Eigen::Map<const vec_index_t>;

    template <class T>
    using dyn_vec_t = std::vector<T>;
    using dyn_vec_value_t = dyn_vec_t<value_t>;
    using dyn_vec_index_t = dyn_vec_t<index_t>;
    using dyn_vec_bool_t = dyn_vec_t<bool_t>;
    using dyn_vec_sp_vec_value_t = dyn_vec_t<sp_vec_value_t>;

    const size_t initial_size;
    const size_t initial_size_groups;
    const value_t alpha;
    const map_cvec_value_t penalty;
    const AType& A;
    const map_cvec_value_t r;
    const map_cvec_index_t groups;
    const map_cvec_index_t group_sizes;
    const map_cvec_value_t A_diag;

    std::unordered_set<index_t> strong_hashset;
    dyn_vec_index_t strong_set; 
    dyn_vec_index_t strong_g1;
    dyn_vec_index_t strong_g2;
    dyn_vec_index_t strong_begins;
    dyn_vec_index_t strong_order;
    dyn_vec_value_t strong_beta;
    dyn_vec_value_t strong_beta_prev_valid;
    dyn_vec_value_t strong_grad;
    dyn_vec_value_t strong_A_diag;
    dyn_vec_index_t active_set;
    dyn_vec_index_t active_g1;
    dyn_vec_index_t active_g2;
    dyn_vec_index_t active_begins;
    dyn_vec_index_t active_order;
    //dyn_vec_index_t active_set_ordered;
    dyn_vec_bool_t is_active;
    vec_value_t grad;
    vec_value_t grad_next;
    vec_value_t abs_grad;
    vec_value_t abs_grad_next;
    util::vec_type<sp_vec_value_t> betas_curr;
    vec_value_t rsqs_curr;
    value_t rsq_prev_valid = 0;
    dyn_vec_sp_vec_value_t betas;
    dyn_vec_value_t rsqs;

    template <class RType, class GroupsType, class GroupSizesType, 
              class ADiagType, class PenaltyType>
    explicit GroupBasilState(
        const AType& A_,
        const RType& r_,
        const GroupsType& groups_,
        const GroupSizesType& group_sizes_,
        const ADiagType& A_diag_,
        value_t alpha_,
        const PenaltyType& penalty_,
        const GroupBasilCheckpoint<value_t, index_t, bool_t>& bc
    )
        : initial_size(std::min(static_cast<size_t>(r_.size()), 1uL << 20)),
          initial_size_groups(groups_.size()),
          alpha(alpha_),
          penalty(penalty_.data(), penalty_.size()),
          A(A_),
          r(r_.data(), r_.size()),
          groups(groups_.data(), groups_.size()),
          group_sizes(group_sizes_.data(), group_sizes_.size()),
          A_diag(A_diag_.data(), A_diag_.size()),
          strong_hashset(bc.strong_set.begin(), bc.strong_set.end()),
          strong_set(bc.strong_set),
          strong_g1(bc.strong_g1),
          strong_g2(bc.strong_g2),
          strong_begins(bc.strong_begins),
          strong_order(bc.strong_order),
          strong_beta(bc.strong_beta),
          strong_beta_prev_valid(strong_beta),
          strong_grad(bc.strong_grad),
          strong_A_diag(bc.strong_A_diag),
          active_set(bc.active_set),
          active_g1(bc.active_g1),
          active_g2(bc.active_g2),
          active_begins(bc.active_begins),
          active_order(bc.active_order),
          //active_set_ordered(bc.active_set_ordered),
          is_active(bc.is_active),
          grad(bc.grad),
          grad_next(bc.grad.size()),
          abs_grad(bc.abs_grad),
          abs_grad_next(bc.abs_grad.size()),
          rsq_prev_valid(bc.rsq)
    {}

    template <class RType, class GroupsType, class GroupSizesType, 
              class ADiagType, class PenaltyType>
    explicit GroupBasilState(
        const AType& A_,
        const RType& r_,
        const GroupsType& groups_,
        const GroupSizesType& group_sizes_,
        const ADiagType& A_diag_,
        value_t alpha_,
        const PenaltyType& penalty_
    )
        : initial_size(std::min(static_cast<size_t>(r_.size()), 1uL << 20)),
          initial_size_groups(groups_.size()),
          alpha(alpha_),
          penalty(penalty_.data(), penalty_.size()),
          A(A_),
          r(r_.data(), r_.size()),
          groups(groups_.data(), groups_.size()),
          group_sizes(group_sizes_.data(), group_sizes_.size()),
          A_diag(A_diag_.data(), A_diag_.size()),
          grad(r_),
          grad_next(r_.size()),
          abs_grad(groups.size()),
          abs_grad_next(groups.size()),
          betas_curr(1),
          rsqs_curr(1)
    { 
        /* compute abs_grad */ 
        for (size_t i = 0; i < groups.size(); ++i) {
            const auto k = groups[i];
            const auto size_k = group_sizes[i];
            abs_grad[i] = grad.segment(k, size_k).norm();
        }

        /* initialize strong_set */
        // if no L1 penalty, every variable is active
        if (alpha <= 0) {
            strong_set.resize(groups.size());
            std::iota(strong_set.begin(), strong_set.end(), 0);
        // otherwise, add all non-penalized variables
        } else {
            strong_set.reserve(initial_size_groups); 
            for (size_t i = 0; i < penalty.size(); ++i) {
                if (penalty[i] <= 0.0) {
                    strong_set.push_back(i);        
                }
            }
        }
        
        /* initialize g1/g2 parts */
        update_strong_g1_2(0, initial_size_groups);
        
        /* initialize strong_begins */
        const auto total_strong_size = update_strong_begins(0, 0, initial_size_groups);

        /* initialize strong_hashset */
        strong_hashset.insert(strong_set.begin(), strong_set.end());

        /* initialize strong_order */
        update_strong_order(0, initial_size_groups);

        /* initialize strong_beta */
        strong_beta.reserve(initial_size);
        strong_beta.resize(total_strong_size, 0);
        
        /* OK to leave strong_beta_prev uninitialized (see update_after_initial_fit) */

        /* initialize strong_grad */
        strong_grad.reserve(initial_size);
        strong_grad.resize(total_strong_size);
        for (size_t i = 0; i < strong_set.size(); ++i) {
            const auto begin_i = strong_begins[i];
            const auto size_i = group_sizes[strong_set[i]];
            strong_grad.segment(begin_i, size_i) = grad.segment(groups[strong_set[i]], size_i);
        }

        /* initialize strong_A_diag */
        update_strong_A_diag(0, strong_set.size(), total_strong_size, initial_size);

        /* initialize active_set */
        active_set.reserve(initial_size_groups); 
        
        /* initialize active_g1/g2 */
        active_g1.reserve(initial_size_groups);
        active_g2.reserve(initial_size_groups);
        
        /* initialize active_begins */
        active_begins.reserve(initial_size_groups);

        /* initialize active_order */
        active_order.reserve(initial_size_groups);

        ///* initialize active_set_ordered */
        //active_set_ordered.reserve(initial_size_groups);

        /* initialize is_active */
        is_active.reserve(initial_size_groups);
        is_active.resize(strong_set.size(), false);

        /* initialize betas, rsqs */
        betas.reserve(100);
        rsqs.reserve(100);
    }
    
    GHOSTBASIL_STRONG_INLINE
    void update_after_initial_fit(
        value_t rsq
    ) 
    {
        /* update grad */
        const auto& beta = betas_curr[0];
        const auto beta_inner = beta.innerIndexPtr();
        const auto beta_value = beta.valuePtr();
        const auto beta_nnz = beta.nonZeros();
        for (size_t i = 0; i < beta_nnz; ++i) {
            grad -= A.col(beta_inner[i]) * beta_value[i];
        } 
        
        /* update abs_grad */
        for (size_t i = 0; i < groups.size(); ++i) {
            const auto k = groups[i];
            const auto size_k = group_sizes[i];
            // strong means it has no l1 penalty for initial fit
            // otherwise, coef == 0 because lambda was chosen like that for initial fit,
            // so the KKT check quantity doesn't have any correction and is equivalent to grad.
            abs_grad[i] = is_strong(i) ? 0 : grad.segment(k, size_k).norm();
        }

        /* update rsq_prev_valid */
        rsq_prev_valid = rsq;

        /* update strong_beta_prev_valid */
        strong_beta_prev_valid = strong_beta; 
    }

    GHOSTBASIL_STRONG_INLINE
    void screen(
        size_t beta_last_valid,
        size_t current_size,
        size_t delta_strong_size,
        bool do_strong_rule,
        bool all_lambdas_failed,
        bool some_lambdas_failed,
        value_t lmda_prev_valid,
        value_t lmda_next
    )
    {
        // reset current lasso estimates to next lambda sequence length
        betas_curr.resize(current_size);
        rsqs_curr.resize(current_size);

        // screen to append to strong set and strong hashset.
        // Must use the previous valid gradient vector.
        // Only screen if KKT failure happened somewhere in the current lambda vector.
        // Otherwise, the current set might be enough for the next lambdas, so we try the current list first.
        bool new_strong_added = false;
        const auto old_strong_set_size = strong_set.size();
        const auto old_total_strong_size = strong_beta.size();

        lasso::screen(abs_grad, lmda_prev_valid, lmda_next, 
            alpha, penalty, [&](auto i) { return is_strong(i); }, 
            delta_strong_size, strong_set, do_strong_rule);

        new_strong_added = (old_strong_set_size < strong_set.size());

        const auto strong_set_new_begin = std::next(strong_set.begin(), old_strong_set_size);
        strong_hashset.insert(strong_set_new_begin, strong_set.end());
        
        // update strong_begins
        const auto new_total_strong_size = update_strong_begins(old_total_strong_size);

        // update strong_g1/g2 
        update_strong_g1_2(old_strong_set_size);

        // Note: DO NOT UPDATE strong_order YET!
        // Updating previously valid beta requires the old order.
        
        // only need to update on the new strong variables
        update_strong_A_diag(old_strong_set_size, strong_set.size(), new_total_strong_size);

        // updates on these will be done later!
        strong_beta.resize(new_total_strong_size, 0);
        strong_beta_prev_valid.resize(strong_beta.size(), 0);

        // update is_active to set the new strong variables to false
        is_active.resize(strong_set.size(), false);

        // update strong grad to last valid gradient
        strong_grad.resize(new_total_strong_size);
        for (size_t i = 0; i < strong_set.size(); ++i) {
            const auto k = strong_set[i];
            const auto sbegin_k = strong_begins[i];
            const auto begin_k = groups[k];
            const auto size_k = group_sizes[k];
            strong_grad.segment(sbegin_k, size_k) = grad.segment(begin_k, size_k);
        }

        // At this point, strong_set is ordered for all old variables
        // and unordered for the last few (new variables).
        // But all referencing quantities (strong_beta, strong_grad, is_active, strong_A_diag)
        // match up in size and positions with strong_set.
        // Note: strong_grad has been updated properly to previous valid version in all cases.

        // create dense viewers of old strong betas
        Eigen::Map<util::vec_type<value_t>> old_strong_beta_view(
                strong_beta.data(), old_total_strong_size);
        Eigen::Map<util::vec_type<value_t>> strong_beta_prev_valid_view(
                strong_beta_prev_valid.data(),
                old_total_strong_size);

        // Save last valid strong beta and put current state to that point.
        if (all_lambdas_failed) {
            // warm-start using previously valid solution.
            // Note: this is crucial for correctness in grad update step below.
            old_strong_beta_view = strong_beta_prev_valid_view;
        }
        else if (some_lambdas_failed) {
            // save last valid solution (this logic assumes the ordering is in old one)
            assert(betas.size() > 0);
            const auto& last_valid_sol = betas.back();
            if (last_valid_sol.nonZeros() == 0) {
                old_strong_beta_view.setZero();
            } else {
                auto last_valid_sol_inner = last_valid_sol.innerIndexPtr();
                auto last_valid_sol_value = last_valid_sol.valuePtr();
                size_t osb_pos = 0;
                // zero-out all entries in the range (inner[i-1], inner[i]) 
                // and replace at inner[i] with valid solution.
                for (size_t i = 0; i < last_valid_sol.nonZeros(); ++i) {
                    assert(osb_pos < old_strong_beta_view.size());
                    auto lvs_i = last_valid_sol_inner[i];
                    auto lvs_x = last_valid_sol_value[i];
                    for (; strong_set[strong_order[osb_pos]] < lvs_i; ++osb_pos) {
                        assert(osb_pos < old_strong_beta_view.size());
                        old_strong_beta_view[strong_order[osb_pos]] = 0;
                    }
                    // here, we exploit the fact that the last valid solution
                    // is non-zero on the ever-active set, which is a subset
                    // of the old strong set, so we must have hit a common position.
                    auto ss_idx = strong_order[osb_pos];
                    assert((osb_pos < old_strong_beta_view.size()) &&
                           (strong_set[ss_idx] == lvs_i));
                    old_strong_beta_view[ss_idx] = lvs_x;
                    ++osb_pos;
                }
                for (; osb_pos < old_strong_beta_view.size(); ++osb_pos) {
                    old_strong_beta_view[strong_order[osb_pos]] = 0;
                }
            }

            strong_beta_prev_valid_view = old_strong_beta_view;
        } else {
            strong_beta_prev_valid_view = old_strong_beta_view;
        }

        // save last valid R^2
        rsq_prev_valid = (rsqs.size() == 0) ? rsq_prev_valid : rsqs.back();

        // update strong_order for new order of strong_set
        // only if new variables were added to the strong set.
        if (new_strong_added) {
            update_strong_order(old_strong_set_size);
        }
    }

    GHOSTBASIL_STRONG_INLINE
    bool is_strong(size_t i) const
    {
        return strong_hashset.find(i) != strong_hashset.end();
    }

private:
    
    GHOSTBASIL_STRONG_INLINE
    void update_strong_g1_2(
        size_t old_strong_set_size,
        size_t capacity=0
    )
    {
        strong_g1.reserve(capacity);
        strong_g2.reserve(capacity);
        for (size_t i = old_strong_set_size; i < strong_set.size(); ++i) {
            if (group_sizes[strong_set[i]] == 1) {
                strong_g1.push_back(i);
            } else{
                strong_g2.push_back(i);
            }
        }
    }
    
    GHOSTBASIL_STRONG_INLINE
    size_t update_strong_begins(
        size_t old_total_strong_size,
        size_t capacity=0
    )
    {
        size_t old_strong_begins_size = strong_begins.size();
        size_t total_strong_size = old_total_strong_size;
        strong_begins.reserve(capacity);
        strong_begins.resize(strong_set.size());
        for (size_t i = old_strong_begins_size; i < strong_begins.size(); ++i) {
            strong_begins[i] = total_strong_size;
            total_strong_size += group_sizes[strong_set[i]];
        }
        return total_strong_size;
    }

    GHOSTBASIL_STRONG_INLINE
    void update_strong_order(
        size_t old_strong_set_size,
        size_t capacity = 0
    )
    {
        strong_order.reserve(capacity);
        strong_order.resize(strong_set.size());
        std::iota(std::next(strong_order.begin(), old_strong_set_size), 
                  strong_order.end(), 
                  old_strong_set_size);
        std::sort(strong_order.begin(), strong_order.end(),
                  [&](auto i, auto j) { return strong_set[i] < strong_set[j]; });
    }

    GHOSTBASIL_STRONG_INLINE
    void update_strong_A_diag(
            size_t begin,
            size_t end,
            size_t new_size,
            size_t capacity=0)
    {
        assert((begin <= end) && (end <= strong_set.size()));
        
        auto old_size = strong_A_diag.size();
        
        // subsequent calls does not affect capacity
        strong_A_diag.reserve(capacity);
        strong_A_diag.resize(new_size);

        for (size_t i = begin; i < end; ++i) {
            const auto k = strong_set[i];
            const auto begin_k = groups[k];
            const auto size_k = group_sizes[k];
            strong_A_diag.segment(old_size, size_k) = A_diag.segment(begin_k, size_k);
            old_size += size_k;
        }
        assert(old_size == new_size);
    }
};    


} // namespace group_lasso 
} // namespace ghostbasil