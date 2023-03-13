#pragma once
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <ghostbasil/util/algorithm.hpp>
#include <ghostbasil/util/exceptions.hpp>
#include <ghostbasil/util/macros.hpp>
#include <ghostbasil/util/types.hpp>
#include <ghostbasil/optimization/lasso.hpp>
#include <atomic>
#include <cmath>
#include <unordered_set>
#include <vector>

namespace ghostbasil {
namespace lasso {

/**
 * Append at most max_size number of (first) elements
 * that achieve the maximum absolute value in out.
 * If there are at least max_size number of such elements,
 * exactly max_size will be added.
 * Otherwise, all such elements will be added, but no more.
 */
template <class AbsGradType, class ValueType, class PenaltyType, class ISType, class SSType>
GHOSTBASIL_STRONG_INLINE 
void screen(
        const AbsGradType& abs_grad,
        ValueType lmda_prev,
        ValueType lmda_next,
        ValueType alpha,
        const PenaltyType& penalty,
        const ISType& is_strong,
        size_t size,
        SSType& strong_set,
        bool do_strong_rule)
{
    using value_t = ValueType;

    assert(strong_set.size() <= abs_grad.size());
    if (!do_strong_rule) {
        size_t rem_size = abs_grad.size() - strong_set.size();
        size_t size_capped = std::min(size, rem_size);
        size_t old_strong_size = strong_set.size();
        strong_set.insert(strong_set.end(), size_capped, 0);
        const auto abs_grad_p = util::vec_type<value_t>::NullaryExpr(
            abs_grad.size(), [&](auto i) {
                return (penalty[i] <= 0) ? 0.0 : abs_grad[i] / penalty[i];
            }
        ) / std::max(alpha, 1e-3);
        util::k_imax(abs_grad_p, is_strong, size_capped, 
                std::next(strong_set.begin(), old_strong_size));
        return;
    }
    
    const auto strong_rule_lmda = (2 * lmda_next - lmda_prev) * alpha;
    for (size_t i = 0; i < abs_grad.size(); ++i) {
        if (is_strong(i)) continue;
        if (abs_grad[i] > strong_rule_lmda * penalty[i]) {
            strong_set.push_back(i);
        }
    }
}

/**
 * Checks the KKT condition on the sequence of lambdas and the fitted coefficients.
 *
 * @param   A       covariance matrix.
 * @param   r       correlation vector.
 * @param   s       regularization parameter of A towards identity.
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
template <class AType, class RType, class ValueType, class PenaltyType,
          class LmdasType, class BetasType, class ISType, class GradType>
GHOSTBASIL_STRONG_INLINE 
auto check_kkt(
    const AType& A, 
    const RType& r,
    ValueType alpha,
    const PenaltyType& penalty,
    const LmdasType& lmdas, 
    const BetasType& betas,
    const ISType& is_strong,
    size_t n_threads,
    GradType& grad,
    GradType& grad_next)
{
    assert(r.size() == grad.size());
    assert(grad.size() == grad_next.size());
    assert(betas.size() == lmdas.size());

    size_t i = 0;
    auto alpha_c = 1 - alpha;

    if (lmdas.size() == 0) return i;

    for (; i < lmdas.size(); ++i) {
        const auto& beta_i = betas[i];
        const auto lmda = lmdas[i];

        std::atomic<bool> kkt_fail(false);

#pragma omp parallel for schedule(static) num_threads(n_threads)
        for (size_t k = 0; k < r.size(); ++k) {
            // Just omit the KKT check for strong variables.
            // If KKT failed previously, just do a no-op until loop finishes.
            // This is because OpenMP doesn't allow break statements.
            bool kkt_fail_raw = kkt_fail.load(std::memory_order_relaxed);
            if (kkt_fail_raw) continue;

            // we still need to save the gradients including strong variables
            auto gk = r[k] - A.col_dot(k, beta_i);
            grad_next[k] = gk;

            const auto pk = penalty[k];
            if (is_strong(k) || 
                (std::abs(gk - lmda * alpha_c * pk * beta_i.coeff(k)) <= 
                    lmda * alpha * pk)) continue;
            
            kkt_fail.store(true, std::memory_order_relaxed);
        }

        if (kkt_fail.load(std::memory_order_relaxed)) break; 
        else grad.swap(grad_next);
    }

    return i;
}

/**
 * @brief 
 * Checkpoint class for basil routine.
 * This class contains the minimal state variables that must be provided
 * to generate a full basil state object.
 * 
 * @tparam ValueType    float type.
 * @tparam IndexType    index type.
 * @tparam BoolType     boolean type.
 */
template <class ValueType,
          class IndexType,
          class BoolType>
struct BasilCheckpoint
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
    dyn_vec_index_t strong_order;
    dyn_vec_value_t strong_beta;
    dyn_vec_value_t strong_grad;
    dyn_vec_value_t strong_A_diag;
    dyn_vec_index_t active_set;
    dyn_vec_index_t active_order;
    dyn_vec_index_t active_set_ordered;
    dyn_vec_bool_t is_active;
    vec_value_t grad;
    value_t rsq;

    explicit BasilCheckpoint() =default;

    template <class VecIndexType, class VecValueType, 
              class VecBoolType, class GradType>
    explicit BasilCheckpoint(
        const VecIndexType& strong_set_,
        const VecIndexType& strong_order_,
        const VecValueType& strong_beta_,
        const VecValueType& strong_grad_,
        const VecValueType& strong_A_diag_,
        const VecIndexType& active_set_,
        const VecIndexType& active_order_,
        const VecIndexType& active_set_ordered_,
        const VecBoolType& is_active_,
        const GradType& grad_,
        value_t rsq_
    )
        : is_initialized(true),
          strong_set(strong_set_),
          strong_order(strong_order_),
          strong_beta(strong_beta_),
          strong_grad(strong_grad_),
          strong_A_diag(strong_A_diag_),
          active_set(active_set_),
          active_order(active_order_),
          active_set_ordered(active_set_ordered_),
          is_active(is_active_),
          grad(grad_),
          rsq(rsq_)
    {}

    template <class BasilStateType>
    BasilCheckpoint& operator=(BasilStateType&& bs)
    {
        is_initialized = true;
        strong_set = std::move(bs.strong_set);
        strong_order = std::move(bs.strong_order);
        strong_beta = std::move(bs.strong_beta);
        strong_grad = std::move(bs.strong_grad);
        strong_A_diag = std::move(bs.strong_A_diag);
        active_set = std::move(bs.active_set);
        active_order = std::move(bs.active_order);
        active_set_ordered = std::move(bs.active_set_ordered);
        is_active = std::move(bs.is_active);
        grad = std::move(bs.grad);
        rsq = std::move(bs.rsq_prev_valid);
        return *this;
    }
};

/**
 * @brief 
 * State class for the basil routine.
 * This class contains the full state variables that describes the state of the basil algorithm.
 * 
 * @tparam AType        matrix A type.
 * @tparam ValueType    float type.
 * @tparam IndexType    index type.
 * @tparam BoolType     boolean type.
 */
template <class AType,
          class ValueType,
          class IndexType,
          class BoolType>
struct BasilState
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

    template <class T>
    using dyn_vec_t = std::vector<T>;
    using dyn_vec_value_t = dyn_vec_t<value_t>;
    using dyn_vec_index_t = dyn_vec_t<index_t>;
    using dyn_vec_bool_t = dyn_vec_t<bool_t>;
    using dyn_vec_sp_vec_value_t = dyn_vec_t<sp_vec_value_t>;

    const size_t initial_size;
    const value_t alpha;
    const map_cvec_value_t penalty;
    const AType& A;
    const map_cvec_value_t r;

    std::unordered_set<index_t> strong_hashset;
    dyn_vec_index_t strong_set; 
    dyn_vec_index_t strong_order;
    dyn_vec_value_t strong_beta;
    dyn_vec_value_t strong_beta_prev_valid;
    dyn_vec_value_t strong_grad;
    dyn_vec_value_t strong_A_diag;
    dyn_vec_index_t active_set;
    dyn_vec_index_t active_order;
    dyn_vec_index_t active_set_ordered;
    dyn_vec_bool_t is_active;
    vec_value_t grad;
    vec_value_t grad_next;
    util::vec_type<sp_vec_value_t> betas_curr;
    vec_value_t rsqs_curr;
    value_t rsq_prev_valid = 0;
    dyn_vec_sp_vec_value_t betas;
    dyn_vec_value_t rsqs;

    template <class RType, class PenaltyType>
    explicit BasilState(
        const AType& A_,
        const RType& r_,
        value_t alpha_,
        const PenaltyType& penalty_,
        const BasilCheckpoint<value_t, index_t, bool_t>& bc
    )
        : initial_size(std::min(static_cast<size_t>(r_.size()), 1uL << 20)),
          alpha(alpha_),
          penalty(penalty_.data(), penalty_.size()),
          A(A_),
          r(r_.data(), r_.size()),
          strong_hashset(bc.strong_set.begin(), bc.strong_set.end()),
          strong_set(bc.strong_set),
          strong_order(bc.strong_order),
          strong_beta(bc.strong_beta),
          strong_beta_prev_valid(strong_beta),
          strong_grad(bc.strong_grad),
          strong_A_diag(bc.strong_A_diag),
          active_set(bc.active_set),
          active_order(bc.active_order),
          active_set_ordered(bc.active_set_ordered),
          is_active(bc.is_active),
          grad(bc.grad),
          grad_next(bc.grad.size()),
          rsq_prev_valid(bc.rsq)
    {}

    template <class RType, class PenaltyType>
    explicit BasilState(
        const AType& A_,
        const RType& r_,
        value_t alpha_,
        const PenaltyType& penalty_
    )
        : initial_size(std::min(static_cast<size_t>(r_.size()), 1uL << 20)),
          alpha(alpha_),
          penalty(penalty_.data(), penalty_.size()),
          A(A_),
          r(r_.data(), r_.size()),
          grad(r_),
          grad_next(r_.size()),
          betas_curr(1),
          rsqs_curr(1)
    { 
        /* initialize strong_set */
        // if no L1 penalty, every variable is active
        if (alpha <= 0) {
            strong_set.resize(penalty.size());
            std::iota(strong_set.begin(), strong_set.end(), 0);
        // otherwise, add all non-penalized variables
        } else {
            strong_set.reserve(initial_size); 
            for (size_t i = 0; i < penalty.size(); ++i) {
                if (penalty[i] <= 0.0) {
                    strong_set.push_back(i);        
                }
            }
        }

        /* initialize strong_hashset */
        strong_hashset.insert(strong_set.begin(), strong_set.end());

        /* initialize strong_order */
        update_strong_order(0, initial_size);

        /* initialize strong_grad */
        strong_grad.reserve(initial_size);
        strong_grad.resize(strong_set.size());
        for (size_t i = 0; i < strong_set.size(); ++i) {
            strong_grad[i] = grad[strong_set[i]];
        }

        /* initialize strong_A_diag */
        update_strong_A_diag(0, strong_set.size(), initial_size);

        /* initialize active_set */
        active_set.reserve(initial_size); 

        /* initialize active_order */
        active_order.reserve(initial_size);

        /* initialize active_set_ordered */
        active_set_ordered.reserve(initial_size);

        /* initialize is_active */
        is_active.reserve(initial_size);
        is_active.resize(strong_set.size(), false);

        /* initialize strong_beta */
        strong_beta.reserve(initial_size);
        strong_beta.resize(strong_set.size(), 0);

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
        for (size_t i = 0; i < strong_set.size(); ++i) {
            grad[strong_set[i]] = strong_grad[i];
        }
        for (size_t i = 0; i < grad.size(); ++i) {
            if (is_strong(i)) continue;
            grad[i] -= A.col_dot(i, betas_curr[0]);
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

        lasso::screen(grad.array().abs(), lmda_prev_valid, lmda_next, 
            alpha, penalty, [&](auto i) { return is_strong(i); }, 
            delta_strong_size, strong_set, do_strong_rule);

        new_strong_added = (old_strong_set_size < strong_set.size());

        const auto strong_set_new_begin = std::next(strong_set.begin(), old_strong_set_size);
        strong_hashset.insert(strong_set_new_begin, strong_set.end());

        // Note: DO NOT UPDATE strong_order YET!
        // Updating previously valid beta requires the old order.
        
        // only need to update on the new strong variables
        update_strong_A_diag(old_strong_set_size, strong_set.size());

        // updates on these will be done later!
        strong_beta.resize(strong_set.size(), 0);
        strong_beta_prev_valid.resize(strong_set.size(), 0);

        // update is_active to set the new strong variables to false
        is_active.resize(strong_set.size(), false);

        // update strong grad to last valid gradient
        strong_grad.resize(strong_set.size());
        for (size_t i = 0; i < strong_grad.size(); ++i) {
            strong_grad[i] = grad[strong_set[i]];
        }

        // At this point, strong_set is ordered for all old variables
        // and unordered for the last few (new variables).
        // But all referencing quantities (strong_beta, strong_grad, is_active, strong_A_diag)
        // match up in size and positions with strong_set.
        // Note: strong_grad has been updated properly to previous valid version in all cases.

        // create dense viewers of old strong betas
        Eigen::Map<util::vec_type<value_t>> old_strong_beta_view(
                strong_beta.data(), old_strong_set_size);
        Eigen::Map<util::vec_type<value_t>> strong_beta_prev_valid_view(
                strong_beta_prev_valid.data(),
                old_strong_set_size);

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
            size_t capacity=0)
    {
        assert((begin <= end) && (end <= strong_set.size()));

        // subsequent calls does not affect capacity
        strong_A_diag.reserve(capacity);
        strong_A_diag.resize(strong_set.size());

        for (size_t i = begin; i < end; ++i) {
            auto k = strong_set[i];
            strong_A_diag[i] = A.coeff(k, k);
        }
    }
};    

struct BasilDiagnostic
{
    using index_t = int;
    using bool_t = int;
    using dyn_vec_index_t = std::vector<index_t>;
    using dyn_vec_bool_t = std::vector<bool_t>;

    dyn_vec_index_t strong_sizes;
    dyn_vec_index_t active_sizes;
    dyn_vec_bool_t used_strong_rule;
    dyn_vec_index_t n_cds;
    dyn_vec_index_t n_lambdas_proc;
};

template <class ValueType, class GradType, class PenaltyType>
GHOSTBASIL_STRONG_INLINE
auto lambda_max(
    const GradType& grad,
    ValueType alpha,
    const PenaltyType& penalty
)
{
    using value_t = ValueType;
    using vec_value_t = util::vec_type<value_t>;
    return vec_value_t::NullaryExpr(
        grad.size(), [&](auto i) {
            return (penalty[i] <= 0.0) ? 0.0 : std::abs(grad[i]) / penalty[i];
        }
    ).maxCoeff() / std::max(alpha, 1e-3);
}

template <class ValueType, class OutType>
GHOSTBASIL_STRONG_INLINE
void generate_lambdas(
    size_t max_n_lambdas,
    ValueType min_ratio,
    ValueType lmda_max,
    OutType& out
)
{
    using value_t = ValueType;
    using vec_value_t = util::vec_type<value_t>;

    // lmda_seq = [l_max, l_max * f, l_max * f^2, ..., l_max * f^(max_n_lambdas-1)]
    // l_max is the smallest lambda such that the penalized features (penalty > 0)
    // have 0 coefficients (assuming alpha > 0). The logic is still coherent when alpha = 0.
    auto log_factor = std::log(min_ratio) * static_cast<value_t>(1.0)/(max_n_lambdas-1);
    out.array() = lmda_max * (
        log_factor * vec_value_t::LinSpaced(max_n_lambdas, 0, max_n_lambdas-1)
    ).array().exp();
}

/**
 * Solves the lasso objective for a sequence of \f$\lambda\f$ values.
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
 *
 * TODO:
 * The algorithm stops at a $\lambda$ value where the 
 * pseudo-validation loss stops decreasing.
 * For now, we return a vector of vector of lambdas, 
 * and vector of coefficient (sparse) matrices 
 * corresponding to each vector of lambdas.
 */
template <class AType, class RType, class ValueType,
          class PenaltyType, class ULmdasType,
          class BetasType, class LmdasType, class RsqsType,
          class CheckpointType = BasilCheckpoint<ValueType, int, int>,
          class DiagnosticType = BasilDiagnostic,
          class CUIType = util::no_op>
inline void basil(
        const AType& A,
        const RType& r,
        ValueType alpha,
        const PenaltyType& penalty,
        const ULmdasType& user_lmdas,
        size_t max_n_lambdas,
        size_t n_lambdas_iter,
        bool use_strong_rule,
        bool do_early_exit,
        size_t delta_strong_size,
        size_t max_strong_size,
        size_t max_n_cds,
        ValueType thr,
        ValueType min_ratio,
        size_t n_threads,
        BetasType& betas_out,
        LmdasType& lmdas,
        RsqsType& rsqs_out,
        CheckpointType&& checkpoint = CheckpointType(),
        DiagnosticType&& diagnostic = DiagnosticType(),
        CUIType check_user_interrupt = CUIType())
{
    using A_t = std::decay_t<AType>;
    using value_t = ValueType;
    using index_t = int;
    using bool_t = index_t;
    using basil_state_t = BasilState<A_t, value_t, index_t, bool_t>;
    using vec_value_t = typename basil_state_t::vec_value_t;
    using lasso_pack_t = LassoParamPack<A_t, value_t, index_t, bool_t>;

    // clear the output vectors to guarantee that it only contains produced results
    betas_out.clear();
    lmdas.clear();
    rsqs_out.clear();
    
    const size_t n_features = r.size();
    max_strong_size = std::min(max_strong_size, n_features);

    // Initialize current state to consider non-penalized variables
    // with 0 coefficient everywhere if checkpoint is not initialized.
    // Otherwise, use checkpoint to initialize the state.
    std::unique_ptr<basil_state_t> basil_state_ptr(
        checkpoint.is_initialized ? 
        new basil_state_t(A, r, alpha, penalty, checkpoint) :
        new basil_state_t(A, r, alpha, penalty)
    );
    auto& basil_state = *basil_state_ptr;

    const auto& strong_set = basil_state.strong_set;
    const auto& strong_order = basil_state.strong_order;
    const auto& strong_A_diag = basil_state.strong_A_diag;
    const auto& rsq_prev_valid = basil_state.rsq_prev_valid;
    auto& grad = basil_state.grad;
    auto& grad_next = basil_state.grad_next;
    auto& strong_beta = basil_state.strong_beta;
    auto& strong_grad = basil_state.strong_grad;
    auto& active_set = basil_state.active_set;
    auto& active_order = basil_state.active_order;
    auto& active_set_ordered = basil_state.active_set_ordered;
    auto& is_active = basil_state.is_active;
    auto& betas_curr = basil_state.betas_curr;
    auto& rsqs_curr = basil_state.rsqs_curr;
    auto& betas = basil_state.betas;
    auto& rsqs = basil_state.rsqs;
    
    // check strong set size
    if (strong_set.size() > max_strong_size) throw util::max_basil_strong_set();

    // current sequence of lambdas for basil iteration.
    // If checkpoint is not provided, can keep it uninitialized.
    vec_value_t lmdas_curr(1);

    lasso_pack_t fit_pack(
        A, alpha, penalty, strong_set, strong_order, strong_A_diag,
        lmdas_curr, max_n_cds, thr, 0, strong_beta, strong_grad,
        active_set, active_order, active_set_ordered,
        is_active, betas_curr, rsqs_curr, 0, 0, false       
    );

    // fit only on the non-penalized variables
    if (!checkpoint.is_initialized) {
        assert(betas_curr.size() == 1);
        assert(rsqs_curr.size() == 1);

        lmdas_curr[0] = std::numeric_limits<value_t>::max();

        fit(fit_pack, check_user_interrupt);

        // update state after fitting on non-penalized variables
        basil_state.update_after_initial_fit(fit_pack.rsq);
    }     

    // full lambda sequence 
    vec_value_t lmda_seq;
    const bool use_user_lmdas = user_lmdas.size() != 0;
    if (!use_user_lmdas) {
        const auto lmda_max = lambda_max(grad, alpha, penalty);
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
            if (check_early_stop_rsq(rsq_l, rsq_m, rsq_u)) break;
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
        basil_state.screen(
            lmdas_curr.size(), delta_strong_size,
            do_strong_rule, idx == 0, some_lambdas_failed,
            lmda_prev_valid, lmda_next 
        );

        if (strong_set.size() > max_strong_size) throw util::max_basil_strong_set();

        /* Fit lasso */
        LassoParamPack<
            AType, value_t, index_t, bool_t
        > fit_pack(
            A, alpha, penalty, strong_set, strong_order, strong_A_diag,
            lmdas_curr, max_n_cds, thr, rsq_prev_valid, strong_beta, strong_grad,
            active_set, active_order, active_set_ordered,
            is_active, betas_curr, rsqs_curr, 0, 0, false      
        );
        try {
            fit(fit_pack, check_user_interrupt);
        } catch (const std::exception& e) {
            tidy_up();
            throw util::propagator_error(e.what());
        }

        const auto& n_lmdas = fit_pack.n_lmdas;

        /* Checking KKT */

        // Get index of lambda of first failure.
        // grad will be the corresponding gradient vector at the returned lmdas_curr[index-1] if index >= 1,
        // and if idx <= 0, then grad is unchanged.
        // In any case, grad corresponds to the first smallest lambda where KKT check passes.
        idx = check_kkt(
            A, r, alpha, penalty, lmdas_curr.head(n_lmdas), betas_curr.head(n_lmdas), 
            [&](auto i) { return basil_state.is_strong(i); }, n_threads, grad, grad_next
        );

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

/**
 * Specialized routine for block matrix A.
 */
template <class AType, class RType, class ValueType,
          class PenaltyType, class ULmdasType,
          class BetasType, class LmdasType, class RsqsType,
          class VecCheckpointType = std::vector<BasilCheckpoint<ValueType, int, int>>,
          class DiagnosticType = BasilDiagnostic,
          class CUIType = util::no_op>
inline void basil(
        const BlockMatrix<AType>& A,
        const RType& r,
        ValueType alpha,
        const PenaltyType& penalty,
        const ULmdasType& user_lmdas,
        size_t max_n_lambdas,
        size_t n_lambdas_iter,
        bool use_strong_rule,
        bool do_early_exit,
        size_t delta_strong_size,
        size_t max_strong_size,
        size_t max_n_cds,
        ValueType thr,
        ValueType min_ratio,
        size_t n_threads,
        BetasType& betas_out,
        LmdasType& lmdas,
        RsqsType& rsqs_out,
        VecCheckpointType&& checkpoints = VecCheckpointType(),
        DiagnosticType&& diagnostic = DiagnosticType(),
        CUIType check_user_interrupt = CUIType())
{
    using A_t = std::decay_t<AType>;
    using value_t = ValueType;
    using index_t = int;
    using bool_t = index_t;
    using basil_state_t = BasilState<A_t, value_t, index_t, bool_t>;
    using vec_value_t = typename basil_state_t::vec_value_t;
    using vec_index_t = typename basil_state_t::vec_index_t;
    using sp_vec_value_t = typename basil_state_t::sp_vec_value_t;
    using lasso_pack_t = LassoParamPack<A_t, value_t, index_t, bool_t>;

    betas_out.clear();
    lmdas.clear();
    rsqs_out.clear();
    
    const size_t n_features = r.size();
    max_strong_size = std::min(max_strong_size, n_features);
    
    const size_t n_blocks = A.n_blocks();
    const auto block_ptr = A.blocks();
    const auto& strides = A.strides();

    // either checkpoints is empty (no checkpoints are initialized)
    // or all checkpoints are given for each block.
    assert((checkpoints.size() == 0) ||
            (checkpoints.size() == n_blocks));

    // Extra information is needed to properly update grad state.
    // This serves as an extra buffer.
    // See the basil loop below.
    vec_value_t grad_last_valid(n_features);

    // Buffer to store n_cds per block
    vec_index_t n_cds_block(n_blocks);

    // Buffer to store first KKT failure index for each block
    vec_index_t indices_block(n_blocks);

    // Buffer to store exceptions
    std::vector<std::pair<util::propagator_error, bool>> exceptions(
        n_blocks, {{}, false} 
    );
    std::atomic<bool> has_exception(false);

    // initialize current state to consider non-penalized variables
    // with 0 coefficient everywhere for each block.
    std::vector<basil_state_t> basil_states;
    basil_states.reserve(n_blocks);
    for (size_t i = 0; i < n_blocks; ++i) {
        const auto& A_block = block_ptr[i];
        const auto begin = strides[i];
        const auto size = strides[i+1] - begin;
        const auto r_block = r.segment(begin, size);
        const auto penalty_block = penalty.segment(begin, size);
        if (checkpoints.size() && checkpoints[i].is_initialized) {
            basil_states.emplace_back(A_block, r_block, alpha, penalty_block, checkpoints[i]);
        } else {
            basil_states.emplace_back(A_block, r_block, alpha, penalty_block);
            assert(basil_states.back().betas_curr.size() == 1);
            assert(basil_states.back().rsqs_curr.size() == 1);
        }
    }

    const auto get_strong_set_tot_size = [&]() {
        return util::vec_type<size_t>::NullaryExpr(
            basil_states.size(), [&](auto i) {
                const auto& basil_state = basil_states[i];
                return basil_state.strong_set.size();
            }
        ).sum();
    };

    const auto get_active_set_tot_size = [&]() {
        return util::vec_type<size_t>::NullaryExpr(
            basil_states.size(), [&](auto i) {
                const auto& basil_state = basil_states[i];
                return basil_state.active_set.size();
            }
        ).sum();
    };
    
    const auto check_exceptions = [&]() {
        for (size_t i = 0; i < exceptions.size(); ++i) {
            if (std::get<1>(exceptions[i])) {
                throw std::get<0>(exceptions[i]);
            }
        }
    };

    const auto strong_set_tot_size = get_strong_set_tot_size();

    if (strong_set_tot_size > max_strong_size) throw util::max_basil_strong_set();

    // current lambda sequence
    vec_value_t lmdas_curr(1);
    lmdas_curr[0] = std::numeric_limits<value_t>::max();

    // fit only on the non-penalized variables
#pragma omp parallel for schedule(static) num_threads(n_threads)
    for (size_t i = 0; i < n_blocks; ++i) {
        if (checkpoints.size() && checkpoints[i].is_initialized) {
            continue;
        }
        auto& basil_state = basil_states[i];

        const auto& A_block = basil_state.A;
        const auto& penalty_block = basil_state.penalty;
        const auto& strong_set = basil_state.strong_set;
        const auto& strong_order = basil_state.strong_order;
        const auto& strong_A_diag = basil_state.strong_A_diag;
        auto& strong_beta = basil_state.strong_beta;
        auto& strong_grad = basil_state.strong_grad;
        auto& active_set = basil_state.active_set;
        auto& active_order = basil_state.active_order;
        auto& active_set_ordered = basil_state.active_set_ordered;
        auto& is_active = basil_state.is_active;
        auto& betas_curr = basil_state.betas_curr;
        auto& rsqs_curr = basil_state.rsqs_curr;

        lasso_pack_t fit_pack(
            A_block, alpha, penalty_block, strong_set, strong_order, strong_A_diag,
            lmdas_curr, max_n_cds, thr, 0, strong_beta, strong_grad,
            active_set, active_order, active_set_ordered,
            is_active, betas_curr, rsqs_curr, 0, 0, false
        );
        
        try {
            fit(fit_pack);
        } catch (const std::exception& e) {
            auto& ei = exceptions[i];
            std::get<0>(ei) = util::propagator_error(e.what());
            std::get<1>(ei) = true;
            has_exception.store(true, std::memory_order_relaxed);
        }

        n_cds_block[i] = fit_pack.n_cds;

        // update state after fitting on non-penalized variables
        basil_state.update_after_initial_fit(fit_pack.rsq);
    }
    
    if (has_exception) check_exceptions(); 

    check_user_interrupt(0);

    // compute max cds
    const auto max_n_cds_block = n_cds_block.maxCoeff();

    // lambda sequence in each basil iteration
    vec_value_t lmda_seq;
    const bool use_user_lmdas = user_lmdas.size() != 0;
    if (!use_user_lmdas) {
        const auto lmda_max = vec_value_t::NullaryExpr(
            n_blocks, [&](auto i) {
                const auto& basil_state = basil_states[i];
                const auto& grad = basil_state.grad;
                const auto& penalty = basil_state.penalty;
                return lambda_max(grad, alpha, penalty);
            }
        ).maxCoeff();
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
        checkpoints.resize(basil_states.size());
#pragma omp parallel for schedule(static) num_threads(n_threads)
        for (size_t i = 0; i < basil_states.size(); ++i) {
            auto& basil_state = basil_states[i];
            basil_state.screen(
                0, 0, false, idx == 0, idx < lmdas_curr.size(),
                lmdas.back(), lmdas.back() 
            );
            checkpoints[i] = std::move(basil_states[i]);
        }
    };

    diagnostic.strong_sizes.push_back(strong_set_tot_size);
    diagnostic.active_sizes.push_back(get_active_set_tot_size());
    diagnostic.used_strong_rule.push_back(false);
    diagnostic.n_cds.push_back(max_n_cds_block);
    diagnostic.n_lambdas_proc.push_back(1);

    while (1) 
    {
        // check early termination 
        if (do_early_exit && (rsqs_out.size() >= 3)) {
            const auto rsq_u = rsqs_out[rsqs_out.size()-1];
            const auto rsq_m = rsqs_out[rsqs_out.size()-2];
            const auto rsq_l = rsqs_out[rsqs_out.size()-3];
            if (check_early_stop_rsq(rsq_l, rsq_m, rsq_u)) break;
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
#pragma omp parallel for schedule(static) num_threads(n_threads)
        for (size_t i = 0; i < n_blocks; ++i) {
            auto& basil_state = basil_states[i];
            basil_state.screen(
                lmdas_curr.size(), delta_strong_size,
                do_strong_rule, idx == 0, some_lambdas_failed,
                lmda_prev_valid, lmda_next 
            );
        }
        
        const auto strong_set_tot_size = get_strong_set_tot_size();

        if (strong_set_tot_size > max_strong_size) throw util::max_basil_strong_set();

#pragma omp parallel for schedule(static) num_threads(n_threads)
        for (size_t i = 0; i < n_blocks; ++i) {
            auto& basil_state = basil_states[i];

            const auto& A_block = basil_state.A;
            const auto& r_block = basil_state.r;
            const auto& penalty_block = basil_state.penalty;
            const auto& strong_set = basil_state.strong_set;
            const auto& strong_order = basil_state.strong_order;
            const auto& strong_A_diag = basil_state.strong_A_diag;
            const auto& rsq_prev_valid = basil_state.rsq_prev_valid;
            auto& strong_beta = basil_state.strong_beta;
            auto& strong_grad = basil_state.strong_grad;
            auto& active_set = basil_state.active_set;
            auto& active_order = basil_state.active_order;
            auto& active_set_ordered = basil_state.active_set_ordered;
            auto& is_active = basil_state.is_active;
            auto& betas_curr = basil_state.betas_curr;
            auto& rsqs_curr = basil_state.rsqs_curr;
            auto& grad = basil_state.grad;
            auto& grad_next = basil_state.grad_next;

            /* Fit lasso */
            lasso_pack_t fit_pack(
                A_block, alpha, penalty_block, strong_set, strong_order, strong_A_diag,
                lmdas_curr, max_n_cds, thr, rsq_prev_valid, strong_beta, strong_grad,
                active_set, active_order, active_set_ordered,
                is_active, betas_curr, rsqs_curr, 0, 0, false
            );

            try {
                fit(fit_pack);
            } catch (const std::exception& e) {
                auto& ei = exceptions[i];
                std::get<0>(ei) = util::propagator_error(e.what());
                std::get<1>(ei) = true;
                has_exception.store(true, std::memory_order_relaxed);
            }

            const auto& n_lmdas = fit_pack.n_lmdas;

            n_cds_block[i] = fit_pack.n_cds;

            // save last valid gradient
            Eigen::Map<vec_value_t> grad_last_valid_block(
                grad_last_valid.data() + strides[i], strides[i+1] - strides[i]
            );
            grad_last_valid_block = grad;
            
            /* Checking KKT */
            indices_block[i] = check_kkt(
                A_block, r_block, alpha, penalty_block, 
                lmdas_curr.head(n_lmdas), betas_curr.head(n_lmdas), 
                [&](auto j) { return basil_state.is_strong(j); }, 1, grad, grad_next
            );
        }
        
        if (has_exception) {
            tidy_up();
            check_exceptions();
        }
        
        idx = indices_block.minCoeff();
        const auto max_n_cds_block = n_cds_block.maxCoeff();

        // MUST reset grad for each block to the one at idx-1 (if idx > 0)
        // and grad_last_valid if idx == 0.
#pragma omp parallel for schedule(static) num_threads(n_threads)
        for (size_t i = 0; i < n_blocks; ++i) {
            auto& basil_state = basil_states[i];
            auto& grad = basil_state.grad;
            if (idx == 0) {
                Eigen::Map<vec_value_t> grad_last_valid_block(
                    grad_last_valid.data() + strides[i], strides[i+1] - strides[i]
                );
                grad = grad_last_valid_block;
            } else if (idx < lmdas_curr.size()){
                const auto& A_block = basil_state.A;
                const auto& r_block = basil_state.r;
                const auto& beta_valid = basil_state.betas_curr[idx-1];
                for (size_t j = 0; j < grad.size(); ++j) {
                    grad[j] = r_block[j] - A_block.col_dot(j, beta_valid);
                }
            }
        }

        check_user_interrupt(0);

        // decrement number of remaining lambdas
        n_lambdas_rem -= idx;

        /* Save output and check for any early stopping */

        // if first failure is not at the first lambda, save all previous solutions.
        std::vector<index_t> active_indices;
        std::vector<value_t> active_values;
        active_indices.reserve(strong_set_tot_size);
        active_values.reserve(strong_set_tot_size);
        for (size_t i = 0; i < idx; ++i) {
            lmdas.emplace_back(lmdas_curr[i]);
            
            active_indices.clear();
            active_values.clear();
            value_t rsq_tot = 0;
            for (size_t j = 0; j < n_blocks; ++j) {
                auto& basil_state = basil_states[j];
                const auto& betas_curr = basil_state.betas_curr;
                const auto& rsqs_curr = basil_state.rsqs_curr;
                auto& betas = basil_state.betas;
                auto& rsqs = basil_state.rsqs;

                // Append jth block of active coefficients in the concatenated coeff vector
                // Assumes that blocks are ordered in terms of coefficient indices
                // (block 1: [0, n_1), block 2: [n_1, n_2) ... )
                const size_t nzn = betas_curr[i].nonZeros();
                const auto inner = betas_curr[i].innerIndexPtr();
                const auto vals = betas_curr[i].valuePtr();
                const auto shift = strides[j];
                for (size_t k = 0; k < nzn; ++k) {
                    active_indices.push_back(shift + inner[k]);
                    active_values.push_back(vals[k]);
                }

                rsq_tot += rsqs_curr[i];

                betas.emplace_back(std::move(betas_curr[i]));
                rsqs.emplace_back(std::move(rsqs_curr[i]));
            }

            Eigen::Map<sp_vec_value_t> beta_concat(
                n_features,
                active_indices.size(),
                active_indices.data(),
                active_values.data()
            );
            betas_out.emplace_back(beta_concat);

            rsqs_out.emplace_back(rsq_tot);
        }

        // update diagnostic
        diagnostic.strong_sizes.push_back(strong_set_tot_size);
        diagnostic.active_sizes.push_back(get_active_set_tot_size());
        diagnostic.used_strong_rule.push_back(do_strong_rule);
        diagnostic.n_cds.push_back(max_n_cds_block);
        diagnostic.n_lambdas_proc.push_back(idx);
    }

    tidy_up();
}

} // namespace lasso
} // namespace ghostbasil
