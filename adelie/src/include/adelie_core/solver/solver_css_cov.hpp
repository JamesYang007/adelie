#pragma once
#include <atomic>
#include <unordered_set>
#include <vector>
#include <Eigen/Cholesky>
#include <adelie_core/configs.hpp>
#include <adelie_core/matrix/utils.hpp>
#include <adelie_core/util/exceptions.hpp>
#include <adelie_core/util/functional.hpp>
#include <adelie_core/util/macros.hpp>
#include <adelie_core/util/stopwatch.hpp>

namespace adelie_core {
namespace solver {
namespace css {
namespace cov {

template <class ValueType, class IndexType>
ADELIE_CORE_STRONG_INLINE
bool compute_least_squares_scores(
    const std::unordered_set<IndexType>& subset_set,
    const Eigen::Ref<const util::colmat_type<ValueType>>& S,
    Eigen::Ref<util::rowvec_type<ValueType>> out,
    size_t n_threads
)
{
    const auto p = S.cols();
    const auto routine = [&](auto j) {
        const auto S_jj = S(j, j);
        if (
            (subset_set.find(j) != subset_set.end()) ||
            (S_jj <= 0)
        ) {
            out[j] = 0;
            return;
        }
        out[j] = (S.row(j).head(j).squaredNorm() + S.col(j).tail(p-j).squaredNorm()) / S_jj;
    };
    util::omp_parallel_for(routine, 0, p, n_threads);
    return false;
}

template <class ValueType, class IndexType>
ADELIE_CORE_STRONG_INLINE
bool compute_subset_factor_scores(
    const std::unordered_set<IndexType>& subset_set,
    int j_to_swap,
    const Eigen::Ref<const util::colmat_type<ValueType>>& S,
    Eigen::Ref<util::rowvec_type<ValueType>> out,
    size_t n_threads
)
{
    using value_t = ValueType;
    constexpr value_t inf = std::numeric_limits<value_t>::infinity();
    constexpr value_t eps = 1e-10;
    static_assert(eps < 1, "eps must be strictly smaller than 1.");

    const auto S_diag = S.diagonal().transpose().array();
    const auto p = S.cols();

    // IMPORTANT: r_j contains RSS of regressing X_i ~ X_j.
    // If the RSS is small for any i not in subset. 
    // this means the objective would be minimized to -inf as soon as j enters the set.
    // All such columns are given maximum score.
    // NOTE: we skip computation if either i or j is in U or i == j.
    // Hence, this check effectively only applies to i != j both not in U.
    std::atomic_bool early_exit = false;
    const auto routine = [&](auto j) {
        if (
            (
                (j != j_to_swap) && 
                early_exit.load(std::memory_order_relaxed) 
            ) ||
            (subset_set.find(j) != subset_set.end())
        ) {
            return;
        }
        const auto S_jj = S_diag[j];
        if (S_jj <= 0) {
            out[j] = inf;
            early_exit = true;
            return;
        }
        value_t sum = -std::log(S_jj);
        for (int i = 0; i < p; ++i) {
            if ((subset_set.find(i) != subset_set.end()) || (i == j)) continue;
            const auto is_i_leq_j = i <= j;
            const auto ij_min = is_i_leq_j ? i : j;
            const auto ij_max = is_i_leq_j ? j : i;
            const auto S_ij = S(ij_max, ij_min);
            const auto r_ij = S_diag[i] - S_ij * S_ij / S_jj;
            if (r_ij <= eps) {
                sum = inf;
                early_exit = true;
                break;
            }
            sum -= std::log(r_ij);
        }
        out[j] = sum;
    };

    // Initialize all scores to be lowest.
    out = -inf;

    // First, attempt to solve for the j_to_swap entry.
    // If this entry is already infinite score, we don't need to compute for other entries.
    if (j_to_swap >= 0) {
        routine(j_to_swap);
        if (out[j_to_swap] == inf) return true; 
    }

    util::omp_parallel_for(routine, 0, p, n_threads);

    return early_exit;
}

template <class ValueType, class IndexType>
ADELIE_CORE_STRONG_INLINE
bool compute_min_det_scores(
    const std::unordered_set<IndexType>& subset_set,
    const Eigen::Ref<const util::colmat_type<ValueType>>& S,
    Eigen::Ref<util::rowvec_type<ValueType>> out
)
{
    using value_t = ValueType;
    constexpr value_t eps = 1e-10;
    const auto p = S.cols();
    out = -S.diagonal().transpose().array().max(0);
    for (int j = 0; j < p; ++j) {
        if (subset_set.find(j) != subset_set.end()) continue;
        if (out[j] >= -eps) {
            out[j] = 0;
            return true;
        }
    }
    return false;
}

template <
    class SType, 
    class ValueType
>
ADELIE_CORE_STRONG_INLINE
void update_cov_resid_fwd(
    SType& S,
    size_t i,
    util::rowvec_type<ValueType>& buff
)
{
    const auto S_ii = S(i, i);
    if (S_ii <= 0) return;
    const auto p = S.cols();
    auto beta = buff.head(p).matrix().transpose();
    beta.head(i) = S.row(i).head(i).transpose();
    beta.tail(p-i) = S.col(i).tail(p-i);
    S.template selfadjointView<Eigen::Lower>().rankUpdate(beta, -1 / S_ii);
}

template <
    class StateType,
    class ComputeScoresType,
    class CheckUserInterruptType
>
inline void solve_greedy(
    StateType& state,
    ComputeScoresType compute_scores,
    CheckUserInterruptType check_user_interrupt
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;

    constexpr value_t neg_inf = -std::numeric_limits<value_t>::infinity();

    const auto& S = state.S;
    const auto subset_size = state.subset_size;
    auto& subset_set = state.subset_set;
    auto& subset = state.subset;
    auto& S_resid = state.S_resid;

    const auto p = S.cols();

    vec_value_t scores(p);

    subset_set.clear();
    subset.clear();
    subset.reserve(p);
    // NOTE: S_resid is only well-defined on the lower triangular part!
    S_resid.resize(p, p);
    S_resid.template triangularView<Eigen::Lower>() = S;

    for (size_t t = 0; t < subset_size; ++t) 
    {
        check_user_interrupt();

        compute_scores(subset_set, -1, S_resid, scores);

        Eigen::Index i_star;
        vec_value_t::NullaryExpr(p, [&](auto i) {
            return (subset_set.find(i) != subset_set.end()) ? neg_inf : scores[i];
        }).maxCoeff(&i_star);

        subset_set.insert(i_star);
        subset.push_back(i_star);

        update_cov_resid_fwd(S_resid, i_star, scores);
    }
}

template <
    class StateType,
    class ComputeScoresType,
    class CheckUserInterruptType
>
inline void solve_swapping(
    StateType& state,
    ComputeScoresType compute_scores,
    CheckUserInterruptType check_user_interrupt
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;
    using colmat_value_t = typename state_t::colmat_value_t;

    constexpr value_t neg_inf = -std::numeric_limits<value_t>::infinity();
    constexpr value_t eps = 1e-10;

    const auto& S = state.S;
    const auto max_iters = state.max_iters;
    const auto n_threads = state.n_threads;
    auto& subset_set = state.subset_set;
    auto& subset = state.subset;
    auto& S_resid = state.S_resid;
    auto& L_T = state.L_T;

    const size_t p = S.cols();
    const size_t k = subset.size();
    
    if (k <= 0 || k >= p) return;

    vec_value_t scores(p);

#ifdef ADELIE_CORE_CSS_COV_DEBUG
    util::Stopwatch sw;

    sw.start();
#endif

    // initialize residual covariance w.r.t. T
    // NOTE: S_resid is only well-defined on the lower triangular part!
    S_resid.resize(p, p);
    S_resid.template triangularView<Eigen::Lower>() = S;
    for (size_t jj = 0; jj < subset.size(); ++jj) {
        const auto j = subset[jj];
        if (S_resid(j, j) <= eps) {
            throw util::adelie_core_error(
                "Initial subset are not linearly independent columns. "
            );
        }
        update_cov_resid_fwd(S_resid, j, scores /* buffer */);
    }

    // initialize L_T
    colmat_value_t S_T(k, k);
    for (size_t i1 = 0; i1 < subset.size(); ++i1) {
        for (size_t i2 = 0; i2 <= i1; ++i2) {
            S_T(i1, i2) = S(subset[i1], subset[i2]);
        }
    }
    S_T.template triangularView<Eigen::Upper>() = S_T.transpose();
    Eigen::LLT<colmat_value_t> S_T_llt(S_T);
    L_T = S_T_llt.matrixL();

    if ((L_T.diagonal().array() <= eps).any()) {
        throw util::adelie_core_error(
            "Initial subset are not linearly independent columns. "
        );
    }

    // initialize Sigma_{.T}
    colmat_value_t S_T_full(p, k);
    for (size_t jj = 0; jj < subset.size(); ++jj) {
        S_T_full.col(jj) = S.col(subset[jj]);
    }

    // initialize L_U (empty)
    colmat_value_t L_U(k-1, k-1);

    // extra buffer
    vec_value_t buff(2*k+p-1);

    // convergence metric
    size_t n_consec_keep = 0;
    
#ifdef ADELIE_CORE_CSS_COV_DEBUG
    state.benchmark_init = sw.elapsed();
#endif

    for (size_t iters = 0; iters < max_iters; ++iters) 
    {
        // cycle through each selected feature and try swapping
        for (size_t jj = 0; jj < k; ++jj)
        {
            check_user_interrupt();

            // NOTE: 
            // T = [subset[jj], ..., subset[k-1], subset[0], ..., subset[jj-1]]
            // U = T[1:]

            const auto j = subset[jj];

            // compute L_U
            {
#ifdef ADELIE_CORE_CSS_COV_DEBUG
                sw.start();
#endif
                const auto L_0 = L_T.bottomRightCorner(k-1, k-1).template triangularView<Eigen::Lower>();
                const auto v_0 = L_T.col(0).tail(k-1);
                auto _L_U = L_U.template triangularView<Eigen::Lower>();
                _L_U = L_0;
                Eigen::internal::llt_rank_update_lower(L_U, v_0, 1.0);
#ifdef ADELIE_CORE_CSS_COV_DEBUG
                state.benchmark_L_U.push_back(sw.elapsed());
#endif
            }

            // compute S_resid w.r.t. U
            {
#ifdef ADELIE_CORE_CSS_COV_DEBUG
                sw.start();
#endif
                auto buff_ptr = buff.data();

                // compute Sigma_{U,j}
                Eigen::Map<vec_value_t> S_Uj(buff_ptr, k-1); buff_ptr += k-1;
                S_Uj = vec_value_t::NullaryExpr(S_Uj.size(), [&](auto i) {
                    return S(subset[(jj + 1 + i) % k], j);
                });

                // compute v = Sigma_U^{-1} Sigma_{U,j}
                L_U.template triangularView<Eigen::Lower>().solveInPlace(
                    S_Uj.matrix().transpose()
                );
                L_U.transpose().template triangularView<Eigen::Upper>().solveInPlace(
                    S_Uj.matrix().transpose()
                );

                // NOTE: S_T_full matches the ordering of subset, not T, (as an optimization)!
                // Equivalently, we can swap the entries of v to match the ordering of subset.
                Eigen::Map<vec_value_t> v(buff_ptr, k); buff_ptr += k;
                v.tail(k-jj-1) = S_Uj.head(k-jj-1);
                v[jj] = 0;
                v.head(jj) = S_Uj.tail(jj);

                // compute Sigma_{.j} - Sigma_{.U} v
                Eigen::Map<vec_value_t> beta(buff_ptr, p); buff_ptr += p;
                auto beta_m = beta.matrix();
                matrix::dgemv(
                    S_T_full.transpose(),
                    v.matrix(),
                    n_threads,
                    buff /* unused */,
                    beta_m
                );
                beta_m = S.col(j).transpose() - beta_m;

                // update S_resid
                const auto beta_j = beta[j];
                // numerically unstable, so no swapping will be accurate (just terminate)
                if (beta_j <= 0) return;
                S_resid.template selfadjointView<Eigen::Lower>().rankUpdate(beta.matrix().transpose(), 1/beta_j);
#ifdef ADELIE_CORE_CSS_COV_DEBUG
                state.benchmark_S_resid.push_back(sw.elapsed());
#endif
            }

            // remove the current index to swap
            subset_set.erase(j);

            // compute scores using current residual covariance
#ifdef ADELIE_CORE_CSS_COV_DEBUG
            sw.start();
#endif
            const bool early_exit = compute_scores(subset_set, j, S_resid, scores);
#ifdef ADELIE_CORE_CSS_COV_DEBUG
            state.benchmark_scores.push_back(sw.elapsed());
#endif

            // compute max score outside U
            Eigen::Index j_star;
            vec_value_t::NullaryExpr(p, [&](auto i) {
                return (subset_set.find(i) != subset_set.end()) ? neg_inf : scores[i];
            }).maxCoeff(&j_star);

            // if swapping strictly improves based on the scores
            if (scores[j] < scores[j_star]) {
                // perform the swap
                subset[jj] = j_star;

                // compute S_T_full
                S_T_full.col(jj) = S.col(j_star);

                // reset the counter
                n_consec_keep = 0;
            } else {
                ++n_consec_keep;
            }

            // add the updated index
            subset_set.insert(subset[jj]);

            // compute L_T with the new T for the next iteration
            {
#ifdef ADELIE_CORE_CSS_COV_DEBUG
                sw.start();
#endif
                const auto j = subset[jj];
                const auto _L_U = L_U.template triangularView<Eigen::Lower>();
                L_T.topLeftCorner(k-1, k-1).template triangularView<Eigen::Lower>() = _L_U;
                auto v = L_T.row(k-1).head(k-1);
                v = vec_value_t::NullaryExpr(k-1, [&](auto i) {
                    return S(subset[(jj + 1 + i) % k], j);
                });
                _L_U.solveInPlace(v.matrix().transpose());
                const auto v_sq_norm = v.squaredNorm();
                L_T(k-1, k-1) = std::sqrt(std::max<value_t>(S(j, j) - v_sq_norm, 0));
#ifdef ADELIE_CORE_CSS_COV_DEBUG
                state.benchmark_L_T.push_back(sw.elapsed());
#endif
            }

            // compute S_resid w.r.t. T
#ifdef ADELIE_CORE_CSS_COV_DEBUG
            sw.start();
#endif
            update_cov_resid_fwd(S_resid, subset[jj], scores /* buffer */);
#ifdef ADELIE_CORE_CSS_COV_DEBUG
            state.benchmark_resid_fwd.push_back(sw.elapsed());
#endif

            // check for convergence or early exit or invariance that S_T is still invertible
            if (n_consec_keep >= k || early_exit || L_T(k-1, k-1) <= eps) return;
        }
    }

    throw util::adelie_core_solver_error("Maximum swapping cycles reached!");
}

template <
    class StateType,
    class CheckUserInterruptType=util::no_op
>
inline void solve(
    StateType&& state,
    CheckUserInterruptType check_user_interrupt=CheckUserInterruptType()
)
{
    using state_t = std::decay_t<StateType>;
    using matrix_t = typename state_t::matrix_t;
    using index_t = typename state_t::index_t;
    using vec_value_t = typename state_t::vec_value_t;
    using score_func_t = std::function<
        bool(
            const std::unordered_set<index_t>&,
            index_t,
            const Eigen::Ref<const matrix_t>&,
            Eigen::Ref<vec_value_t>
        )
    >;

    const auto method = state.method;
    const auto loss = state.loss;
    const auto n_threads = state.n_threads;

    score_func_t score_func = [&]() -> score_func_t {
        switch (loss) {
            case util::css_loss_type::_least_squares: {
                return [&](
                    const std::unordered_set<index_t>& subset_set,
                    index_t,
                    const Eigen::Ref<const matrix_t>& S,
                    Eigen::Ref<vec_value_t> out
                ) {
                    return compute_least_squares_scores(subset_set, S, out, n_threads);
                };
                break;
            }
            case util::css_loss_type::_subset_factor: {
                return [&](
                    const std::unordered_set<index_t>& subset_set,
                    index_t j_to_swap,
                    const Eigen::Ref<const matrix_t>& S,
                    Eigen::Ref<vec_value_t> out 
                ) {
                    return compute_subset_factor_scores(subset_set, j_to_swap, S, out, n_threads);
                };
                break;
            }
            case util::css_loss_type::_min_det: {
                return [&](
                    const std::unordered_set<index_t>& subset_set,
                    index_t,
                    const Eigen::Ref<const matrix_t>& S,
                    Eigen::Ref<vec_value_t> out
                ) {
                    return compute_min_det_scores(subset_set, S, out);
                };
                break;
            }
        }

        throw util::adelie_core_solver_error("Unrecognized loss type!");

        // dummy output (only to appease compiler)
        return [&](
            const std::unordered_set<index_t>&,
            index_t,
            const Eigen::Ref<const matrix_t>&,
            Eigen::Ref<vec_value_t> 
        ) {
            return false;
        };
    }();

    switch (method) {
        case util::css_method_type::_greedy: {
            solve_greedy(state, score_func, check_user_interrupt);
            break;
        }
        case util::css_method_type::_swapping: {
            solve_swapping(state, score_func, check_user_interrupt);
            break;
        }
    }
}

} // namespace cov
} // namespace css
} // namespace solver
} // namespace adelie_core