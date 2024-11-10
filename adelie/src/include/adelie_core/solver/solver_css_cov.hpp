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
    using vec_value_t = util::rowvec_type<value_t>;
    constexpr value_t eps = 1e-100;
    const value_t stable_cap = 32;
    const auto S_diag = S.diagonal().transpose().array();
    const auto p = S.cols();
    out = -S_diag.max(eps).log();
    const auto routine = [&](auto i) {
        const auto S_ii = S_diag[i];
        const auto S_i = S.col(i).transpose().array();
        const auto r_i = (S_diag - S_i.square() * ((S_ii > 0) / (S_ii + (S_ii <= 0)))).max(eps).log();
        out[i] -= vec_value_t::NullaryExpr(p, [&](auto j) {
            return ((subset_set.find(j) != subset_set.end()) || (j == i)) ? 0 : r_i[j];
        }).sum();
    };
    if (n_threads <= 1) {
        for (int i = 0; i < p; ++i) routine(i);
    } else {
        #pragma omp parallel for schedule(static) num_threads(n_threads)
        for (int i = 0; i < p; ++i) routine(i);
    }
    const auto mask = (out <= stable_cap).template cast<value_t>();
    out = out * mask + std::numeric_limits<value_t>::max() * (1-mask);
}

template <class ValueType>
ADELIE_CORE_STRONG_INLINE
void compute_min_det_scores(
    const Eigen::Ref<const util::colmat_type<ValueType>>& S,
    Eigen::Ref<util::rowvec_type<ValueType>> out
)
{
    out = -S.diagonal().transpose().array();
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

        compute_scores(subset_set, subset, S_resid, scores);

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
    constexpr value_t eps = 1e-9;

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

    // initialize Sigma_{T}
    colmat_value_t S_T(k, k);
    for (size_t i1 = 0; i1 < subset.size(); ++i1) {
        for (size_t i2 = 0; i2 < subset.size(); ++i2) {
            S_T(i1, i2) = S(subset[i1], subset[i2]);
        }
    }

    // initialize Sigma_{T}^{-1}
    Eigen::LLT<colmat_value_t> S_T_llt(S_T);
    colmat_value_t S_T_inv = S_T_llt.solve(
        colmat_value_t::Identity(k, k)
    );

    // initialize Sigma_{.T}
    colmat_value_t S_T_full(p, k);
    for (size_t jj = 0; jj < subset.size(); ++jj) {
        S_T_full.col(jj) = S.col(subset[jj]);
    }

    // initialize Sigma_{U}^{-1} (empty)
    colmat_value_t S_U_inv(k-1, k-1);

    // extra buffer
    vec_value_t buff(2*k+p-1);

    for (size_t iters = 0; iters < max_iters; ++iters) 
    {
        bool converged = true;

        // cycle through each selected feature and try swapping
        for (size_t jj = 0; jj < k; ++jj)
        {
            check_user_interrupt();

            // swap so that U is exactly the first k-1 entries of subset
            // NOTE: all invariants must be swapped also
            std::swap(subset[jj], subset[k-1]);
            S_T.col(jj).swap(S_T.col(k-1));
            S_T.row(jj).swap(S_T.row(k-1));
            S_T_inv.col(jj).swap(S_T_inv.col(k-1));
            S_T_inv.row(jj).swap(S_T_inv.row(k-1));

            // column index to swap
            const auto j = subset[k-1];

            // compute S_U_inv where U = T[:-1]
            {
                S_U_inv = S_T_inv.topLeftCorner(k-1, k-1);
                const auto b_tilde = S_T_inv.col(k-1).head(k-1);
                const auto c_tilde = std::max<value_t>(S_T_inv(k-1, k-1), 1e-24);
                S_U_inv.template selfadjointView<Eigen::Lower>().rankUpdate(b_tilde, -1/c_tilde);
                S_U_inv.template triangularView<Eigen::Upper>() = S_U_inv.transpose();
            }

            // compute S_resid w.r.t. U
            {
                auto buff_ptr = buff.data();

                // compute Sigma_{U,j}
                Eigen::Map<vec_value_t> S_Uj(buff_ptr, k-1); buff_ptr += k-1;
                S_Uj = vec_value_t::NullaryExpr(S_Uj.size(), [&](auto i) {
                    return S(subset[i], j);
                });

                // compute v = Sigma_U^{-1} Sigma_{U,j}
                Eigen::Map<vec_value_t> v(buff_ptr, k); buff_ptr += k;
                auto vm = v.head(k-1).matrix();
                matrix::dgemv(
                    S_U_inv,
                    S_Uj.matrix(),
                    n_threads,
                    buff /* unused */,
                    vm
                );
                v[k-1] = 0;

                // NOTE: S_T_full is not swapped before (as an optimization)!
                // Equivalently, we can swap the entries of v.
                std::swap(v[jj], v[k-1]);

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

            // remove the current index to swap (now at the end)
            subset_set.erase(j);

            // compute scores using current residual covariance
            compute_scores(subset_set, subset, S_resid, scores);

            // compute max score outside U
            Eigen::Index j_star;
            vec_value_t::NullaryExpr(p, [&](auto i) {
                return (subset_set.find(i) != subset_set.end()) ? neg_inf : scores[i];
            }).maxCoeff(&j_star);

            // flag to check if the updated T is collinear
            bool is_T_collinear = false;

            // if swapping strictly improves based on the scores
            if (scores[j] < scores[j_star]) {
                // perform the swap
                subset[k-1] = j_star;

                // compute S_T
                {
                    S_T.col(k-1).transpose() = vec_value_t::NullaryExpr(k, [&](auto i) {
                        return S(subset[i], j_star);
                    });
                    S_T.row(k-1) = S_T.col(k-1).transpose();
                }

                // compute S_T_inv
                {
                    auto b_tilde = S_T_inv.col(k-1).head(k-1);
                    const auto b = S_T.col(k-1).head(k-1);
                    auto b_tilde_m = b_tilde.transpose();
                    matrix::dgemv(
                        S_U_inv.transpose(),
                        b.transpose(),
                        n_threads,
                        buff /* unused */,
                        b_tilde_m
                    );
                    const auto c = S_T(k-1, k-1);
                    const auto denom = c - b.dot(b_tilde);

                    // check whether T is collinear along the way
                    is_T_collinear = (denom <= 1e-7) || (S_resid(j_star, j_star) <= eps);

                    const auto c_tilde = 1 / denom;
                    b_tilde *= -c_tilde;
                    S_T_inv.row(k-1).head(k-1) = b_tilde.transpose();
                    S_T_inv(k-1, k-1) = c_tilde;
                    auto A_tilde = S_T_inv.topLeftCorner(k-1, k-1);
                    A_tilde = S_U_inv;
                    A_tilde.template selfadjointView<Eigen::Lower>().rankUpdate(b_tilde, 1/c_tilde);
                    A_tilde.template triangularView<Eigen::Upper>() = A_tilde.transpose();
                }

                // compute S_T_full
                S_T_full.col(jj) = S.col(j_star);

                // since a swapping occurred, we did not converge
                converged = false;
            }

            // add the updated index
            subset_set.insert(subset[k-1]);

            // compute S_resid w.r.t. T
            update_cov_resid_fwd(S_resid, subset[k-1], scores /* buffer */);

            // swap back to the original order for the next iteration
            std::swap(subset[jj], subset[k-1]);
            S_T.col(jj).swap(S_T.col(k-1));
            S_T.row(jj).swap(S_T.row(k-1));
            S_T_inv.col(jj).swap(S_T_inv.col(k-1));
            S_T_inv.row(jj).swap(S_T_inv.row(k-1));

            // check invariance that S_T is still invertible
            if (is_T_collinear) return;
        }

        if (converged) return;
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
            const std::vector<index_t>&,
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
                    const std::vector<index_t>&,
                    const Eigen::Ref<const matrix_t>& S,
                    Eigen::Ref<vec_value_t> out
                ) {
                    compute_least_squares_scores(S, out, n_threads);
                };
                break;
            }
            case util::css_loss_type::_subset_factor: {
                return [&](
                    const std::unordered_set<index_t>& subset_set,
                    const std::vector<index_t>&,
                    const Eigen::Ref<const matrix_t>& S,
                    Eigen::Ref<vec_value_t> out 
                ) {
                    compute_subset_factor_scores(subset_set, S, out, n_threads);
                };
                break;
            }
            case util::css_loss_type::_min_det: {
                return [&](
                    const std::unordered_set<index_t>&,
                    const std::vector<index_t>&,
                    const Eigen::Ref<const matrix_t>& S,
                    Eigen::Ref<vec_value_t> out
                ) {
                    compute_min_det_scores(S, out);
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