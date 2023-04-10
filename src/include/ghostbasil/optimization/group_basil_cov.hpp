#pragma once
#include <atomic>
#include <unordered_set>
#include <memory>
#include <vector>
#include <ghostbasil/optimization/group_basil_base.hpp>
#include <ghostbasil/optimization/group_lasso_cov.hpp>
#include <ghostbasil/util/stopwatch.hpp>

namespace ghostbasil {
namespace group_lasso {
namespace cov {

/**
 * Checks the KKT condition on the sequence of lambdas and the fitted coefficients.
 * 
 * @param   A       covariance matrix.
 * @param   r       correlation vector.
 * @param   alpha   elastic net penalty.
 * @param   lmdas   downward sequence of L1 regularization parameter.
 * @param   betas   each element i corresponds to a (sparse) vector
 *                  of the solution at lmdas[i].
 * @param   is_strong   a functor that checks if feature i is strong.
 * @param   n_threads   number of threads to use in OpenMP.
 * @param   grad        a dense vector that represents the r - A * beta
 *                      right __before__ the first value in lmdas (going downwards)
 *                      that fails the KKT check.
 *                      If KKT fails at the first lambda, grad is unchanged.
 * @param   grad_next   a dense vector that represents r - A * beta
 *                      right __at__ the first value in lmdas (going downwards)
 *                      that fails the KKT check.
 *                      This is really used for optimizing memory allocation.
 *                      User should not need to access this directly.
 *                      It is undefined-behavior accessing this after the call.
 *                      It just has to be initialized to the same size as grad.
 * @param   abs_grad    similar to grad but represents ||grad_i||_2 for each group i.
 * @param   abs_grad_next   similar to grad_next, but for abs_grad.
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
    GradType& abs_grad_next
)
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
        {
            size_t inner_idx = 0;
            for (size_t k = 0; k < groups.size(); ++k) {
                if (inner_idx == beta_i_nnz) break;
                const auto gk = groups[k];
                const auto gk_size = group_sizes[k];
                while ((inner_idx < beta_i_nnz) && (beta_i_inner[inner_idx] < gk + gk_size)) {
                    auto mat = A.block(0, gk, A.rows(), gk_size);
                    grad_next -= mat.col(beta_i_inner[inner_idx] - gk) * beta_i_value[inner_idx];
                    ++inner_idx;
                }
            }
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
    explicit GroupBasilCheckpoint(
        const BasilStateType& bs
    )
        : is_initialized(true),
          strong_set(bs.strong_set),
          strong_g1(bs.strong_g1),
          strong_g2(bs.strong_g2),
          strong_begins(bs.strong_begins),
          strong_order(bs.strong_order),
          strong_beta(bs.strong_beta),
          strong_grad(bs.strong_grad),
          strong_A_diag(bs.strong_A_diag),
          active_set(bs.active_set),
          active_g1(bs.active_g1),
          active_g2(bs.active_g2),
          active_begins(bs.active_begins),
          active_order(bs.active_order),
          is_active(bs.is_active),
          grad(bs.grad),
          abs_grad(bs.abs_grad),
          rsq(bs.rsq_prev_valid)
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
        // TODO: I'm pretty sure this is not necessary, but just to be safe...
        for (size_t i = 0; i < groups.size(); ++i) {
            const auto k = groups[i];
            const auto size_k = group_sizes[i];
            abs_grad[i] = grad.segment(k, size_k).norm();
        }

        /* initialize strong_set */
        // only add variables that have no l1 penalty
        if (alpha <= 0) {
            strong_set.resize(groups.size());
            std::iota(strong_set.begin(), strong_set.end(), 0);
        } else {
            strong_set.reserve(initial_size_groups); 
            for (size_t i = 0; i < penalty.size(); ++i) {
                if (penalty[i] <= 0.0) {
                    strong_set.push_back(i);        
                }
            }
        }
        
        /* initialize g1/g2 parts */
        update_strong_g1_g2(0, initial_size_groups);
        
        /* initialize strong_begins */
        const auto total_strong_size = update_strong_begins(0, initial_size_groups);

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
            Eigen::Map<vec_value_t> sg_map(strong_grad.data(), strong_grad.size());
            sg_map.segment(begin_i, size_i) = grad.segment(groups[strong_set[i]], size_i);
        }

        /* initialize strong_A_diag */
        update_strong_A_diag(0, total_strong_size, initial_size);

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
        {
            size_t inner_idx = 0;
            for (size_t i = 0; i < groups.size(); ++i) {
                if (inner_idx == beta_nnz) break;
                const auto gi = groups[i];
                const auto gi_size = group_sizes[i];
                while ((inner_idx < beta_nnz) && (beta_inner[inner_idx] < gi + gi_size)) {
                    auto mat = A.block(0, gi, A.rows(), gi_size);
                    grad -= mat.col(beta_inner[inner_idx] - gi) * beta_value[inner_idx];
                    ++inner_idx;
                }
            }
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

        group_lasso::screen(abs_grad, lmda_prev_valid, lmda_next, 
            alpha, penalty, [&](auto i) { return is_strong(i); }, 
            delta_strong_size, strong_set, do_strong_rule);

        new_strong_added = (old_strong_set_size < strong_set.size());

        const auto strong_set_new_begin = std::next(strong_set.begin(), old_strong_set_size);
        strong_hashset.insert(strong_set_new_begin, strong_set.end());
        
        // update strong_begins
        const auto new_total_strong_size = update_strong_begins(old_total_strong_size);

        // update strong_g1/g2 
        update_strong_g1_g2(old_strong_set_size);

        // Note: DO NOT UPDATE strong_order YET!
        // Updating previously valid beta requires the old order.
        
        // only need to update on the new strong variables
        update_strong_A_diag(old_strong_set_size, new_total_strong_size);

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
            Eigen::Map<vec_value_t> sg_map(strong_grad.data(), strong_grad.size());
            sg_map.segment(sbegin_k, size_k) = grad.segment(begin_k, size_k);
        }

        // At this point, strong_set is ordered for all old variables
        // and unordered for the last few (new variables).
        // But all referencing quantities (strong_beta, strong_grad, is_active, strong_A_diag)
        // match up in size and positions with strong_set.
        // Note: strong_grad has been updated properly to previous valid version in all cases.

        // create dense viewers of old strong betas
        Eigen::Map<vec_value_t> old_strong_beta_view(
                strong_beta.data(), old_total_strong_size);
        Eigen::Map<vec_value_t> strong_beta_prev_valid_view(
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
            Eigen::Map<const vec_index_t> ss_map(
                strong_set.data(), old_strong_set_size
            );
            Eigen::Map<const vec_index_t> so_map(
                strong_order.data(), old_strong_set_size
            );
            Eigen::Map<const vec_index_t> sb_map(
                strong_begins.data(), old_strong_set_size
            );
            last_valid_strong_beta(
                old_strong_beta_view, last_valid_sol,
                groups, group_sizes, ss_map, so_map, sb_map
            );
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
    void update_strong_g1_g2(
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
            size_t old_strong_set_size,
            size_t new_size,
            size_t capacity=0)
    {
        assert(old_strong_set_size <= strong_set.size());
        
        auto old_size = strong_A_diag.size();
        
        // subsequent calls does not affect capacity
        strong_A_diag.reserve(capacity);
        strong_A_diag.resize(new_size);

        for (size_t i = old_strong_set_size; i < strong_set.size(); ++i) {
            const auto k = strong_set[i];
            const auto begin_k = groups[k];
            const auto size_k = group_sizes[k];
            Eigen::Map<vec_value_t> sad_map(strong_A_diag.data(), strong_A_diag.size());
            sad_map.segment(old_size, size_k) = A_diag.segment(begin_k, size_k);
            old_size += size_k;
        }
        assert(old_size == new_size);
    }
};    

struct GroupBasilDiagnostic
{
    using value_t = double;
    using index_t = int;
    using bool_t = int;
    using dyn_vec_value_t = std::vector<value_t>;
    using dyn_vec_index_t = std::vector<index_t>;
    using dyn_vec_bool_t = std::vector<bool_t>;
    using dyn_vec_basil_state_t = std::vector<
        GroupBasilCheckpoint<value_t, index_t, bool_t>
    >;
    using dyn_vec_group_lasso_diagnostic_t = std::vector<GroupLassoDiagnostic>;

    dyn_vec_index_t strong_sizes;
    dyn_vec_index_t active_sizes;
    dyn_vec_bool_t used_strong_rule;
    dyn_vec_index_t n_cds;
    dyn_vec_index_t n_lambdas_proc;
    dyn_vec_basil_state_t checkpoints;
    dyn_vec_value_t time_init;
    dyn_vec_value_t time_init_fit;
    dyn_vec_value_t time_screen;
    dyn_vec_value_t time_fit;
    dyn_vec_value_t time_kkt;
    dyn_vec_value_t time_transform;
    dyn_vec_value_t time_untransform;
    dyn_vec_group_lasso_diagnostic_t time_group_lasso;
};

/**
 * @brief Solves the lasso objective for a sequence of \f$\lambda\f$ values.
 *
 * @param   A   covariance matrix.
 * @param   r   covariance between covariates and response.
 * @param   alpha   elastic net proportion.
 * @param   penalty penalty factor for each coefficient.
 * @param   user_lmdas      user provided lambda sequence.
 *                          Assumes it is in decreasing order.
 *                          If empty, then lambda sequence will be generated.
 * @param   max_n_lambdas   max number of lambdas to compute solutions for.
 *                          If user_lmdas is non-empty, it will be internally
 *                          reset to user_lmdas.size().
 *                          Assumes it is > 0.
 * @param   n_lambdas_iter  number of lambdas per BASIL iteration for fitting lasso on strong set.
 *                          Internally, it is capped at max_n_lambdas.
 *                          Assumes it is > 0.
 * @param   use_strong_rule     if true then use strong rule as the primary way of discarding variables.
 *                              Original incremental method with delta_strong_size is used when
 *                              the strong rule fails to get enough variables and KKT fails at
 *                              the first lambda in the sub-sequence at every BASIL iteration.
 * @param   delta_strong_size   number of variables to add to strong set 
 *                              at every BASIL iteration.
 *                              Internally, it is capped at number of non-strong variables
 *                              at every BASIL iteration.
 *                              Assumes it is > 0.
 * @param   max_strong_size     max number of strong set variables.
 *                              Internally, it is capped at number of features.
 *                              Assumes it is > 0.
 * @param   max_n_cds           maximum number of coordinate descent per BASIL iteration.
 * @param   thr                 convergence threshold for coordinate descent.
 * @param   betas               vector of sparse vectors to store a list of solutions.
 * @param   lmdas               vector of values to store a list of lambdas
 *                              corresponding to the solutions in betas:
 *                              lmdas[i] is a lambda corresponding to the solution
 *                              at betas[i].
 * @param   rsqs                vector of values to store the list of (unnormalized) R^2 values.
 *                              rsqs[i] is the R^2 at lmdas[i] and betas[i].
 */
template <class AType, class RType, class GroupsType, class GroupSizesType,
          class ADiagType, class ValueType, class PenaltyType, class ULmdasType,
          class BetasType, class LmdasType, class RsqsType,
          class UpdateCoefficientsType,
          class CheckpointType = GroupBasilCheckpoint<ValueType, int, int>,
          class DiagnosticType = GroupBasilDiagnostic,
          class CUIType = util::no_op>
inline void group_basil(
        AType& A,
        const RType& r,
        const GroupsType& groups,
        const GroupSizesType& group_sizes,
        const ADiagType& A_diag,
        ValueType alpha,
        const PenaltyType& penalty,
        const ULmdasType& user_lmdas,
        size_t max_n_lambdas,
        size_t n_lambdas_iter,
        bool use_strong_rule,
        bool do_early_exit,
        bool verbose_diagnostic,
        size_t delta_strong_size,
        size_t max_strong_size,
        size_t max_n_cds,
        ValueType thr,
        ValueType cond_0_thresh,
        ValueType cond_1_thresh,
        ValueType newton_tol,
        size_t newton_max_iters,
        ValueType min_ratio,
        size_t n_threads,
        BetasType& betas_out,
        LmdasType& lmdas,
        RsqsType& rsqs_out,
        UpdateCoefficientsType update_coefficients_f,
        CheckpointType&& checkpoint = CheckpointType(),
        DiagnosticType&& diagnostic = DiagnosticType(),
        CUIType check_user_interrupt = CUIType())
{
    using A_t = std::decay_t<AType>;
    using value_t = ValueType;
    using index_t = int;
    using bool_t = index_t;
    using basil_state_t = GroupBasilState<A_t, value_t, index_t, bool_t>;
    using vec_value_t = typename basil_state_t::vec_value_t;
    using lasso_pack_t = GroupLassoParamPack<A_t, value_t, index_t, bool_t>;
    using sw_t = util::Stopwatch;

    // clear the output vectors to guarantee that it only contains produced results
    betas_out.clear();
    lmdas.clear();
    rsqs_out.clear();
    
    const size_t n_features = r.size();
    max_strong_size = std::min(max_strong_size, n_features);

    // Initialize current state to consider non-penalized variables
    // with 0 coefficient everywhere if checkpoint is not initialized.
    // Otherwise, use checkpoint to initialize the state.
    diagnostic.time_init.push_back(0);
    std::unique_ptr<basil_state_t> basil_state_ptr;
    {
        sw_t stopwatch(diagnostic.time_init.back());
        basil_state_ptr = (
            checkpoint.is_initialized ? 
            std::make_unique<basil_state_t>(A, r, groups, group_sizes, A_diag, alpha, penalty, checkpoint) :
            std::make_unique<basil_state_t>(A, r, groups, group_sizes, A_diag, alpha, penalty)
        );
    }
    auto& basil_state = *basil_state_ptr;

    if (verbose_diagnostic) {
        diagnostic.checkpoints.emplace_back(basil_state);
    }

    const auto& strong_set = basil_state.strong_set;
    const auto& strong_g1 = basil_state.strong_g1;
    const auto& strong_g2 = basil_state.strong_g2;
    const auto& strong_begins = basil_state.strong_begins;
    const auto& strong_A_diag = basil_state.strong_A_diag;
    const auto& rsq_prev_valid = basil_state.rsq_prev_valid;
    auto& grad = basil_state.grad;
    auto& grad_next = basil_state.grad_next;
    auto& abs_grad = basil_state.abs_grad;
    auto& abs_grad_next = basil_state.abs_grad_next;
    auto& strong_beta = basil_state.strong_beta;
    auto& strong_grad = basil_state.strong_grad;
    auto& active_set = basil_state.active_set;
    auto& active_g1 = basil_state.active_g1;
    auto& active_g2 = basil_state.active_g2;
    auto& active_begins = basil_state.active_begins;
    auto& active_order = basil_state.active_order;
    auto& is_active = basil_state.is_active;
    auto& betas_curr = basil_state.betas_curr;
    auto& rsqs_curr = basil_state.rsqs_curr;
    auto& betas = basil_state.betas;
    auto& rsqs = basil_state.rsqs;
    
    // check strong set (all features) size
    if (strong_beta.size() > max_strong_size) throw util::max_basil_strong_set();

    // current sequence of lambdas for basil iteration.
    // If checkpoint is not provided, can keep it uninitialized.
    vec_value_t lmdas_curr(1);

    lasso_pack_t fit_pack(
        A, groups, group_sizes, alpha, penalty, strong_set, 
        strong_g1, strong_g2, strong_begins, strong_A_diag,
        lmdas_curr, max_n_cds, thr, cond_0_thresh, cond_1_thresh, newton_tol, newton_max_iters, 0,
        strong_beta, strong_grad,
        active_set, active_g1, active_g2, active_begins, active_order,
        is_active, betas_curr, rsqs_curr, 0, 0
    );

    // fit only on the non-penalized variables
    diagnostic.time_init_fit.push_back(0);
    {
        sw_t stopwatch(diagnostic.time_init_fit.back());
        if (!checkpoint.is_initialized) {
            assert(betas_curr.size() == 1);
            assert(rsqs_curr.size() == 1);

            lmdas_curr[0] = std::numeric_limits<value_t>::max();

            fit(fit_pack, update_coefficients_f, check_user_interrupt);

            // update state after fitting on non-penalized variables
            basil_state.update_after_initial_fit(fit_pack.rsq);
        }     
    }

    // full lambda sequence 
    vec_value_t lmda_seq;
    const bool use_user_lmdas = user_lmdas.size() != 0;
    if (!use_user_lmdas) {
        const auto lmda_max = lambda_max(abs_grad, alpha, penalty);
        generate_lambdas(max_n_lambdas, min_ratio, lmda_max, lmda_seq);
    } else {
        lmda_seq = user_lmdas;
        max_n_lambdas = user_lmdas.size();
    }
    n_lambdas_iter = std::min(n_lambdas_iter, max_n_lambdas);

    // number of remaining lambdas to process
    size_t n_lambdas_rem = max_n_lambdas;

    // first index of KKT failure
    size_t idx = 1;         

    const auto tidy_up = [&]() {
        // Last screening to ensure that the basil state is at the previously valid state.
        // We force non-strong rule and add 0 new variables to preserve the state.
        basil_state.screen(
            0, 0, false, idx == 0, idx < lmdas_curr.size(),
            lmdas.back(), lmdas.back() 
        );

        betas_out = std::move(betas);
        rsqs_out = std::move(rsqs);
        checkpoint = std::move(basil_state);
    };

    // update diagnostic for initialization
    diagnostic.strong_sizes.push_back(strong_set.size());
    diagnostic.active_sizes.push_back(active_set.size());
    diagnostic.used_strong_rule.push_back(false);
    diagnostic.n_cds.push_back(fit_pack.n_cds);
    diagnostic.n_lambdas_proc.push_back(1);

    while (1) 
    {
        // check early termination 
        if (do_early_exit && (rsqs.size() >= 3)) {
            const auto rsq_u = rsqs[rsqs.size()-1];
            const auto rsq_m = rsqs[rsqs.size()-2];
            const auto rsq_l = rsqs[rsqs.size()-3];
            if (lasso::check_early_stop_rsq(rsq_l, rsq_m, rsq_u, cond_0_thresh, cond_1_thresh)) break;
        }

        // finish if no more lambdas to process finished
        if (n_lambdas_rem == 0) break;

        /* Update lambda sequence */
        const bool some_lambdas_failed = idx < lmdas_curr.size();

        // if some lambdas have valid solutions, shift the next lambda sequence
        if (idx > 0) {
            lmdas_curr.resize(std::min(n_lambdas_iter, n_lambdas_rem));
            auto begin = std::next(lmda_seq.data(), lmda_seq.size()-n_lambdas_rem);
            auto end = std::next(begin, lmdas_curr.size());
            std::copy(begin, end, lmdas_curr.data());
        }

        /* Screening */
        const auto lmda_prev_valid = (lmdas.size() == 0) ? lmdas_curr[0] : lmdas.back(); 
        const auto lmda_next = lmdas_curr[0]; // well-defined
        const bool do_strong_rule = use_strong_rule && (idx != 0);
        diagnostic.time_screen.push_back(0);
        {
            sw_t stopwatch(diagnostic.time_screen.back());
            basil_state.screen(
                lmdas_curr.size(), delta_strong_size,
                do_strong_rule, idx == 0, some_lambdas_failed,
                lmda_prev_valid, lmda_next 
            );
        }

        if (verbose_diagnostic) {
            diagnostic.checkpoints.emplace_back(basil_state);
        }

        if (strong_beta.size() > max_strong_size) throw util::max_basil_strong_set();

        /* Fit lasso */
        lasso_pack_t fit_pack(
            A, groups, group_sizes, alpha, penalty, strong_set, 
            strong_g1, strong_g2, strong_begins, strong_A_diag,
            lmdas_curr, max_n_cds, thr, cond_0_thresh, cond_1_thresh, newton_tol, newton_max_iters, rsq_prev_valid,
            strong_beta, strong_grad,
            active_set, active_g1, active_g2, active_begins, active_order,
            is_active, betas_curr, rsqs_curr, 0, 0
        );
        diagnostic.time_fit.push_back(0);
        try {
            sw_t stopwatch(diagnostic.time_fit.back());
            fit(fit_pack, update_coefficients_f, check_user_interrupt);
        } catch (const std::exception& e) {
            tidy_up();
            throw util::propagator_error(e.what());
        }
        diagnostic.time_group_lasso.emplace_back(std::move(fit_pack.diagnostic));

        const auto& n_lmdas = fit_pack.n_lmdas;

        /* Checking KKT */

        // Get index of lambda of first failure.
        // grad will be the corresponding gradient vector at the returned lmdas_curr[index-1] if index >= 1,
        // and if idx <= 0, then grad is unchanged.
        // In any case, grad corresponds to the first smallest lambda where KKT check passes.
        diagnostic.time_kkt.push_back(0);
        {
            sw_t stopwatch(diagnostic.time_kkt.back());
            idx = check_kkt(
                A, r, groups, group_sizes, alpha, penalty, lmdas_curr.head(n_lmdas), betas_curr.head(n_lmdas), 
                [&](auto i) { return basil_state.is_strong(i); }, n_threads, grad, grad_next, abs_grad, abs_grad_next
            );
        }

        // decrement number of remaining lambdas
        n_lambdas_rem -= idx;

        /* Save output and check for any early stopping */

        // if first failure is not at the first lambda, save all previous solutions.
        for (size_t i = 0; i < idx; ++i) {
            betas.emplace_back(std::move(betas_curr[i]));
            rsqs.emplace_back(std::move(rsqs_curr[i]));
            lmdas.emplace_back(std::move(lmdas_curr[i]));
        }

        // update diagnostic 
        diagnostic.strong_sizes.push_back(strong_set.size());
        diagnostic.active_sizes.push_back(active_set.size());
        diagnostic.used_strong_rule.push_back(do_strong_rule);
        diagnostic.n_cds.push_back(fit_pack.n_cds);
        diagnostic.n_lambdas_proc.push_back(idx);
    }

    tidy_up();
}

/*
 * Small wrapper for individual-level data.
 */
template <class XType, class YType, class GroupsType, class GroupSizesType,
          class ValueType, class PenaltyType, class ULmdasType,
          class BetasType, class LmdasType, class RsqsType,
          class UpdateCoefficientsType,
          class CheckpointType = GroupBasilCheckpoint<ValueType, int, int>,
          class DiagnosticType = GroupBasilDiagnostic,
          class CUIType = util::no_op>
inline void group_basil(
    const XType& X,
    const YType& y,
    const GroupsType& groups,
    const GroupSizesType& group_sizes,
    ValueType alpha,
    const PenaltyType& penalty,
    const ULmdasType& user_lmdas,
    size_t max_n_lambdas,
    size_t n_lambdas_iter,
    bool use_strong_rule,
    bool do_early_exit,
    bool verbose_diagnostic,
    size_t delta_strong_size,
    size_t max_strong_size,
    size_t max_n_cds,
    ValueType thr,
    ValueType cond_0_thresh,
    ValueType cond_1_thresh,
    ValueType newton_tol,
    size_t newton_max_iters,
    ValueType min_ratio,
    size_t n_threads,
    BetasType& betas_out,
    LmdasType& lmdas,
    RsqsType& rsqs_out,
    UpdateCoefficientsType update_coefficients_f,
    CheckpointType&& checkpoint = CheckpointType(),
    DiagnosticType&& diagnostic = DiagnosticType(),
    CUIType check_user_interrupt = CUIType()
)
{
    using X_t = std::decay_t<XType>;
    using value_t = ValueType;
    using vec_t = util::vec_type<value_t>;
    using mat_t = util::mat_type<value_t>;
    using sw_t = util::Stopwatch;

    mat_t X_trans = X;
    vec_t A_diag(X.cols());
    diagnostic.time_transform.push_back(0);
    {
        sw_t stopwatch(diagnostic.time_transform.back());
        transform_data(X_trans, groups, group_sizes, n_threads, A_diag);    
    }
    vec_t r = X_trans.transpose() * y;
    CovCache<X_t, value_t> A(X_trans);
    
    const auto tidy_up = [&]() {
        diagnostic.time_untransform.push_back(0);
        sw_t stopwatch(diagnostic.time_untransform.back());
        untransform_solutions(X, groups, group_sizes, betas_out, n_threads);
    };
    
    try {
        group_basil(
            A, r, groups, group_sizes, A_diag, alpha, penalty, 
            user_lmdas, max_n_lambdas, n_lambdas_iter,
            use_strong_rule, do_early_exit, verbose_diagnostic, delta_strong_size,
            max_strong_size, max_n_cds, thr, cond_0_thresh, cond_1_thresh, newton_tol, newton_max_iters,
            min_ratio, n_threads, betas_out, lmdas, rsqs_out,
            update_coefficients_f,
            checkpoint, diagnostic, check_user_interrupt
        );
        tidy_up();
    } catch (const std::exception& e) {
        tidy_up();
        throw util::propagator_error(e.what());
    }
}

} // namespace cov
} // namespace group_lasso 
} // namespace ghostbasil