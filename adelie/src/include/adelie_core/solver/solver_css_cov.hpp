#pragma once
#include <unordered_set>
#include <vector>
#include <Eigen/Cholesky>
#include <adelie_core/configs.hpp>
#include <adelie_core/matrix/utils.hpp>
#include <adelie_core/util/exceptions.hpp>
#include <adelie_core/util/functional.hpp>
#include <adelie_core/util/macros.hpp>
#include <adelie_core/util/stopwatch.hpp>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace adelie_core {
namespace solver {
namespace css {
namespace cov {

template <class ValueType>
struct BufferPack
{
    using value_t = ValueType;

    util::colmat_type<value_t> S_resid;
    util::rowvec_type<value_t> scores;

    explicit BufferPack(
        size_t p
    ):
        S_resid(p, p),
        scores(p)
    {}
};

template <class ValueType>
ADELIE_CORE_STRONG_INLINE
void compute_least_squares_scores(
    const Eigen::Ref<const util::colmat_type<ValueType>>& S,
    Eigen::Ref<util::rowvec_type<ValueType>> out,
    size_t n_threads
)
{
    using value_t = ValueType;
    matrix::sq_norm(S, out, n_threads);
    const auto S_diag = S.diagonal().transpose().array();
    const auto mask = (S_diag <= 0).template cast<value_t>();
    out = out / (S_diag + mask) * (1-mask);
}

template <class ValueType, class IndexType>
ADELIE_CORE_STRONG_INLINE
void compute_subset_factor_scores(
    const std::unordered_set<IndexType>& subset_set,
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
    const auto routine = [&](auto j) {
        if (subset_set.find(j) != subset_set.end()) {
            out[j] = -inf;
            return;
        }
        const auto S_j = S.col(j).transpose().array();
        const auto S_jj = S_j[j];
        const auto r_j = S_diag - S_j.square() * ((S_jj > 0) / (S_jj + (S_jj <= 0)));
        if (S_jj <= 0) {
            out[j] = inf;
            return;
        }
        value_t sum = -std::log(S_jj);
        for (int i = 0; i < p; ++i) {
            if ((subset_set.find(i) != subset_set.end()) || (i == j)) continue;
            const auto r_ij = r_j[i];
            if (r_ij <= eps) {
                sum = inf;
                break;
            }
            sum -= std::log(r_ij);
        }
        out[j] = sum;
    };
    if (n_threads <= 1) {
        for (int j = 0; j < p; ++j) routine(j);
    } else {
        #pragma omp parallel for schedule(static) num_threads(n_threads)
        for (int j = 0; j < p; ++j) routine(j);
    }
}

template <class ValueType>
ADELIE_CORE_STRONG_INLINE
void compute_min_det_scores(
    const Eigen::Ref<const util::colmat_type<ValueType>>& S,
    Eigen::Ref<util::rowvec_type<ValueType>> out
)
{
    out = -S.diagonal().transpose().array().max(0);
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
    const auto p = S.cols();
    auto beta = buff.head(p).matrix().transpose();
    beta = S.col(i);
    const auto beta_i = beta[i];
    if (beta_i <= 0) return;
    S.template selfadjointView<Eigen::Lower>().rankUpdate(beta, -1/beta_i);
    S.template triangularView<Eigen::Upper>() = S.transpose();
}

template <
    class StateType,
    class BufferPackType,
    class ComputeScoresType,
    class CheckUserInterruptType
>
inline void solve_greedy(
    StateType& state,
    BufferPackType& buffer,
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

    auto& S_resid = buffer.S_resid;
    auto& scores = buffer.scores;

    const auto p = S.cols();

    subset_set.clear();
    subset.clear();
    subset.reserve(p);
    S_resid = S;

    for (size_t t = 0; t < subset_size; ++t) 
    {
        check_user_interrupt();

        compute_scores(subset_set, S_resid, scores);

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
    class BufferPackType,
    class ComputeScoresType,
    class CheckUserInterruptType
>
inline void solve_swapping(
    StateType& state,
    BufferPackType& buffer,
    ComputeScoresType compute_scores,
    CheckUserInterruptType check_user_interrupt
)
{
    using state_t = std::decay_t<StateType>;
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;
    using colmat_value_t = util::colmat_type<value_t>;

    constexpr value_t neg_inf = -std::numeric_limits<value_t>::infinity();
    constexpr value_t eps = 1e-10;

    const auto& S = state.S;
    const auto max_iters = state.max_iters;
    const auto n_threads = state.n_threads;
    auto& subset_set = state.subset_set;
    auto& subset = state.subset;

    const size_t p = S.cols();
    const size_t k = subset.size();
    
    if (k <= 0 || k >= p) return;

    auto& S_resid = buffer.S_resid;
    auto& scores = buffer.scores;

    // initialize residual covariance w.r.t. T
    S_resid = S;
    for (size_t jj = 0; jj < subset.size(); ++jj) {
        const auto j = subset[jj];
        if (S_resid(j, j) <= eps) {
            throw util::adelie_core_solver_error(
                "Initial subset are not linearly independent columns. "
            );
        }
        update_cov_resid_fwd(S_resid, j, scores /* buffer */);
    }

    // initialize L_T
    colmat_value_t S_T(k, k);
    for (size_t i1 = 0; i1 < subset.size(); ++i1) {
        for (size_t i2 = 0; i2 < subset.size(); ++i2) {
            S_T(i1, i2) = S(subset[i1], subset[i2]);
        }
    }
    Eigen::LLT<colmat_value_t> S_T_llt(S_T);
    colmat_value_t L_T = S_T_llt.matrixL();

    if ((L_T.diagonal().array() <= eps).any()) {
        throw util::adelie_core_solver_error(
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
                const auto L_0 = L_T.bottomRightCorner(k-1, k-1).template triangularView<Eigen::Lower>();
                const auto v_0 = L_T.col(0).tail(k-1);
                auto _L_U = L_U.template triangularView<Eigen::Lower>();
                _L_U = L_0;
                Eigen::internal::llt_rank_update_lower(L_U, v_0, 1.0);
            }

            // compute S_resid w.r.t. U
            {
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
                S_resid.template triangularView<Eigen::Upper>() = S_resid.transpose();
            }

            // remove the current index to swap
            subset_set.erase(j);

            // compute scores using current residual covariance
            compute_scores(subset_set, S_resid, scores);

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

            // compute L_T
            {
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
            }

            // compute S_resid w.r.t. T
            update_cov_resid_fwd(S_resid, subset[jj], scores /* buffer */);

            // check invariance that S_T is still invertible
            // or check for convergence
            // or check for early exit criterion
            if (
                (L_T(k-1, k-1) <= eps) || 
                (n_consec_keep >= k)
            ) return;
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
    using value_t = typename state_t::value_t;
    using vec_value_t = typename state_t::vec_value_t;
    using score_func_t = std::function<
        void(
            const std::unordered_set<index_t>&,
            const Eigen::Ref<const matrix_t>&,
            Eigen::Ref<vec_value_t>
        )
    >;

    const auto method = state.method;
    const auto loss = state.loss;
    const auto n_threads = state.n_threads;
    const auto p = state.S.cols();

    BufferPack<value_t> buffer(p);

    score_func_t score_func = [&]() -> score_func_t {
        switch (loss) {
            case util::css_loss_type::_least_squares: {
                return [&](
                    const std::unordered_set<index_t>&,
                    const Eigen::Ref<const matrix_t>& S,
                    Eigen::Ref<vec_value_t> out
                ) {
                    return compute_least_squares_scores(S, out, n_threads);
                };
                break;
            }
            case util::css_loss_type::_subset_factor: {
                return [&](
                    const std::unordered_set<index_t>& subset_set,
                    const Eigen::Ref<const matrix_t>& S,
                    Eigen::Ref<vec_value_t> out 
                ) {
                    return compute_subset_factor_scores(subset_set, S, out, n_threads);
                };
                break;
            }
            case util::css_loss_type::_min_det: {
                return [&](
                    const std::unordered_set<index_t>&,
                    const Eigen::Ref<const matrix_t>& S,
                    Eigen::Ref<vec_value_t> out
                ) {
                    return compute_min_det_scores(S, out);
                };
                break;
            }
        }
    }();

    switch (method) {
        case util::css_method_type::_greedy: {
            solve_greedy(state, buffer, score_func, check_user_interrupt);
            break;
        }
        case util::css_method_type::_swapping: {
            solve_swapping(state, buffer, score_func, check_user_interrupt);
            break;
        }
    }
}

} // namespace cov
} // namespace css
} // namespace solver
} // namespace adelie_core