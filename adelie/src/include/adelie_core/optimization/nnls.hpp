#pragma once
#include <adelie_core/util/macros.hpp>
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace optimization {

template <class MatrixType,
          bool path=false,
          class IndexType=Eigen::Index,
          class DynVecIndexType=std::vector<IndexType>&>
struct StateNNLS
{
    using matrix_t = MatrixType;
    using value_t = typename MatrixType::value_t;
    using index_t = IndexType;
    using dyn_vec_index_t = DynVecIndexType;
    using vec_value_t = util::rowvec_type<value_t>;
    using vec_bool_t = util::rowvec_type<bool>;
    using map_vec_value_t = Eigen::Map<vec_value_t>;
    using map_vec_bool_t = Eigen::Map<vec_bool_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;

    matrix_t* XT;
    const map_cvec_value_t X_vars;

    const size_t max_iters;
    const value_t tol;

    size_t iters = 0;
    dyn_vec_index_t active_set;
    map_vec_bool_t is_active;
    map_vec_value_t beta;
    map_vec_value_t resid;
    value_t loss;

    double time_elapsed = 0;

    explicit StateNNLS(
        matrix_t& XT,
        const Eigen::Ref<const vec_value_t>& X_vars,
        size_t max_iters,
        value_t tol,
        dyn_vec_index_t& active_set,
        Eigen::Ref<vec_bool_t> is_active,
        Eigen::Ref<vec_value_t> beta,
        Eigen::Ref<vec_value_t> resid,
        value_t loss
    ):
        XT(&XT),
        X_vars(X_vars.data(), X_vars.size()),
        max_iters(max_iters),
        tol(tol),
        active_set(active_set),
        is_active(is_active.data(), is_active.size()),
        beta(beta.data(), beta.size()),
        resid(resid.data(), resid.size()),
        loss(loss)
    {
        const auto n = XT.cols();
        const auto p = XT.rows();

        if (X_vars.size() != p) {
            throw util::adelie_core_solver_error(
                "X_vars must be (p,) where XT is (p, n). "
            );
        }
        if (tol < 0) {
            throw util::adelie_core_solver_error(
                "tol must be >= 0."
            );
        }
        if (beta.size() != p) {
            throw util::adelie_core_solver_error(
                "beta must be (p,) where XT is (p, n). "
            );
        }
        if (resid.size() != n) {
            throw util::adelie_core_solver_error(
                "resid must be (n,) where XT is (p, n). "
            );
        }
    }

    template <class EarlyExitType, class LowerType, class UpperType>
    void solve(
        EarlyExitType early_exit_f,
        LowerType lower,
        UpperType upper
    )
    {
        const auto add_active = [&](int i) {
            if (!is_active[i]) {
                active_set.push_back(i);
                is_active[i] = true;
            }
        };
        const auto prune = [&]() {
            size_t n_active = 0;
            for (size_t i = 0; i < active_set.size(); ++i) {
                const auto k = active_set[i];
                const auto bk = beta[k];
                if (std::abs(bk) <= 1e-16) {
                    is_active[k] = false;
                    continue;
                }
                active_set[n_active] = k;
                ++n_active;
            }
            active_set.erase(
                std::next(active_set.begin(), n_active),
                active_set.end()
            );
        };

        iters = 0;
        const auto n = XT->cols();
        const auto p = XT->rows();

        while (1) {
            while (1) {
                ++iters;
                value_t convg_measure = 0;
                for (size_t i = 0; i < active_set.size(); ++i) {
                    if (early_exit_f()) return;
                    const auto k = active_set[i];
                    const auto lk = lower(k);
                    const auto uk = upper(k);
                    const auto vk = X_vars[k];
                    const auto gk = XT->rvmul(k, resid);
                    auto& bk = beta[k];
                    const auto bk_old = bk;
                    const auto step = (vk <= 0) ? 0 : (gk / vk);
                    const auto bk_cand = bk + step;
                    bk = std::min<value_t>(std::max<value_t>(bk_cand, lk), uk);
                    if (bk == bk_old) continue;
                    const auto del = bk - bk_old;
                    const auto scaled_del_sq = vk * del * del; 
                    convg_measure = std::max<value_t>(convg_measure, scaled_del_sq);
                    loss -= del * gk - 0.5 * scaled_del_sq;
                    XT->rvtmul(k, -del, resid);
                }

                if (iters >= max_iters) {
                    throw util::adelie_core_solver_error(
                        "StateNNLS: max iterations reached!"
                    );
                }
                if (convg_measure <= n * tol) break;
            }

            prune();

            ++iters;
            value_t convg_measure = 0;
            for (Eigen::Index k = 0; k < p; ++k) {
                if (early_exit_f()) return;
                const auto lk = lower(k);
                const auto uk = upper(k);
                const auto vk = X_vars[k];
                const auto gk = XT->rvmul(k, resid);
                auto& bk = beta[k];
                const auto bk_old = bk;
                const auto step = (vk <= 0) ? 0 : (gk / vk);
                const auto bk_cand = bk + step;
                bk = std::min<value_t>(std::max<value_t>(bk_cand, lk), uk);
                if (bk == bk_old) continue;
                const auto del = bk - bk_old;
                const auto scaled_del_sq = vk * del * del; 
                convg_measure = std::max<value_t>(convg_measure, scaled_del_sq);
                loss -= del * gk - 0.5 * scaled_del_sq;
                XT->rvtmul(k, -del, resid);
                add_active(k);
            }

            if (iters >= max_iters) {
                throw util::adelie_core_solver_error(
                    "StateNNLS: max iterations reached!"
                );
            }
            if (convg_measure <= n * tol) break;
        } // end while(1)
    }
};

template <class MatrixType,
          class IndexType,
          class DynVecIndexType>
struct StateNNLS<MatrixType, true, IndexType, DynVecIndexType>
{
    using matrix_t = MatrixType;
    using value_t = typename MatrixType::value_t;
    using index_t = IndexType;
    using dyn_vec_index_t = DynVecIndexType;
    using vec_value_t = util::rowvec_type<value_t>;
    using vec_bool_t = util::rowvec_type<bool>;
    using map_vec_value_t = Eigen::Map<vec_value_t>;
    using map_vec_bool_t = Eigen::Map<vec_bool_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;

    matrix_t* XT;
    const map_cvec_value_t X_vars;

    const size_t max_iters;
    const value_t tol;
    const size_t lmda_path_size;
    const value_t min_ratio;
    const value_t path_tol;

    size_t iters = 0;
    dyn_vec_index_t active_set;
    map_vec_bool_t is_active;
    map_vec_value_t beta;
    map_vec_value_t resid;
    map_vec_value_t grad;
    value_t loss;

    double time_elapsed = 0;

    explicit StateNNLS(
        matrix_t& XT,
        const Eigen::Ref<const vec_value_t>& X_vars,
        size_t max_iters,
        value_t tol,
        size_t lmda_path_size,
        value_t min_ratio,
        value_t path_tol,
        dyn_vec_index_t& active_set,
        Eigen::Ref<vec_bool_t> is_active,
        Eigen::Ref<vec_value_t> beta,
        Eigen::Ref<vec_value_t> resid,
        Eigen::Ref<vec_value_t> grad,
        value_t loss
    ):
        XT(&XT),
        X_vars(X_vars.data(), X_vars.size()),
        max_iters(max_iters),
        tol(tol),
        lmda_path_size(lmda_path_size),
        min_ratio(min_ratio),
        path_tol(path_tol),
        active_set(active_set),
        is_active(is_active.data(), is_active.size()),
        beta(beta.data(), beta.size()),
        resid(resid.data(), resid.size()),
        grad(grad.data(), grad.size()),
        loss(loss)
    {
        const auto n = XT.cols();
        const auto p = XT.rows();

        if (X_vars.size() != p) {
            throw util::adelie_core_solver_error(
                "X_vars must be (p,) where XT is (p, n). "
            );
        }
        if (tol < 0) {
            throw util::adelie_core_solver_error(
                "tol must be >= 0."
            );
        }
        if (beta.size() != p) {
            throw util::adelie_core_solver_error(
                "beta must be (p,) where XT is (p, n). "
            );
        }
        if (resid.size() != n) {
            throw util::adelie_core_solver_error(
                "resid must be (n,) where XT is (p, n). "
            );
        }
    }

    template <class LowerType, class UpperType>
    void solve(
        LowerType lower,
        UpperType upper
    )
    {
        const auto add_active = [&](int i) {
            if (!is_active[i]) {
                active_set.push_back(i);
                is_active[i] = true;
            }
        };
        const auto prune = [&]() {
            size_t n_active = 0;
            for (size_t i = 0; i < active_set.size(); ++i) {
                const auto k = active_set[i];
                const auto bk = beta[k];
                if (std::abs(bk) <= 1e-16) {
                    is_active[k] = false;
                    continue;
                }
                active_set[n_active] = k;
                ++n_active;
            }
            active_set.erase(
                std::next(active_set.begin(), n_active),
                active_set.end()
            );
        };

        iters = 0;
        const auto n = XT->cols();
        const auto p = XT->rows();

        // compute lmda_max
        XT->tmul(resid, grad);
        const value_t lmda_max = grad.abs().maxCoeff();

        // compute factor to decrease lambda
        const value_t factor = std::pow(min_ratio, 1.0 / (lmda_path_size - 1));

        value_t lmda = lmda_max;

        for (size_t i = 0; i < lmda_path_size; ++i) {
            lmda *= factor;

            prune();

            // coordinate-descent on all variables once
            ++iters;
            value_t convg_measure = 0;
            for (Eigen::Index k = 0; k < p; ++k) {
                const auto lk = lower(k);
                const auto uk = upper(k);
                const auto vk = X_vars[k];
                const auto gk = XT->rvmul(k, resid);
                auto& bk = beta[k];
                const auto bk_old = bk;
                const auto u = gk + vk * bk_old;
                const auto v = std::abs(u) - lmda;
                const auto bk_cand = (v > 0.0) ? (std::copysign(v,u)/vk) : 0;
                bk = std::min<value_t>(std::max<value_t>(bk_cand, lk), uk);
                if (bk == bk_old) continue;
                const auto del = bk - bk_old;
                const auto scaled_del_sq = vk * del * del; 
                convg_measure = std::max<value_t>(convg_measure, scaled_del_sq);
                loss -= del * gk - 0.5 * scaled_del_sq;
                XT->rvtmul(k, -del, resid);
                add_active(k);
            }

            if (iters >= max_iters) {
                throw util::adelie_core_solver_error(
                    "StateNNLS: max iterations reached!"
                );
            }

            PRINT(active_set.size());

            // solve carefully on the active set until convergence
            while (1) {
                ++iters;
                value_t convg_measure = 0;
                for (size_t i = 0; i < active_set.size(); ++i) {
                    const auto k = active_set[i];
                    const auto lk = lower(k);
                    const auto uk = upper(k);
                    const auto vk = X_vars[k];
                    const auto gk = XT->rvmul(k, resid);
                    auto& bk = beta[k];
                    const auto bk_old = bk;
                    const auto u = gk + vk * bk_old;
                    const auto v = std::abs(u) - lmda;
                    const auto bk_cand = (v > 0.0) ? (std::copysign(v,u)/vk) : 0;
                    bk = std::min<value_t>(std::max<value_t>(bk_cand, lk), uk);
                    if (bk == bk_old) continue;
                    const auto del = bk - bk_old;
                    const auto scaled_del_sq = vk * del * del; 
                    convg_measure = std::max<value_t>(convg_measure, scaled_del_sq);
                    loss -= del * gk - 0.5 * scaled_del_sq;
                    XT->rvtmul(k, -del, resid);
                }

                if (iters >= max_iters) {
                    throw util::adelie_core_solver_error(
                        "StateNNLS: max iterations reached!"
                    );
                }
                if (convg_measure <= n * path_tol) break;
            }
            PRINT(iters);
        }

        StateNNLS<matrix_t, false, index_t, dyn_vec_index_t> state_nnls(
            *XT, X_vars, max_iters, tol, 
            active_set, is_active, beta, resid, loss
        );
        state_nnls.solve(
            []() { return false; },
            lower,
            upper
        );
        loss = state_nnls.loss;
    }
};

} // namespace optimization
} // namespace adelie_core