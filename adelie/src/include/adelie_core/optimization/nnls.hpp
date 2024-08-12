#pragma once
#include <adelie_core/util/macros.hpp>
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace optimization {

template <class MatrixType,
          class IndexType=Eigen::Index,
          class DynVecIndexType=std::vector<IndexType>&>
struct StateNNLS
{
    using matrix_t = MatrixType;
    using value_t = typename MatrixType::Scalar;
    using index_t = IndexType;
    using dyn_vec_index_t = DynVecIndexType;
    using vec_value_t = util::rowvec_type<value_t>;
    using vec_bool_t = util::rowvec_type<bool>;
    using map_vec_value_t = Eigen::Map<vec_value_t>;
    using map_vec_bool_t = Eigen::Map<vec_bool_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;
    using map_cmatrix_t = Eigen::Map<const matrix_t>;

    const map_cmatrix_t X;
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
        const Eigen::Ref<const matrix_t>& X,
        const Eigen::Ref<const vec_value_t>& X_vars,
        size_t max_iters,
        value_t tol,
        dyn_vec_index_t& active_set,
        Eigen::Ref<vec_bool_t> is_active,
        Eigen::Ref<vec_value_t> beta,
        Eigen::Ref<vec_value_t> resid,
        value_t loss
    ):
        X(X.data(), X.rows(), X.cols()),
        X_vars(X_vars.data(), X_vars.size()),
        max_iters(max_iters),
        tol(tol),
        active_set(active_set),
        is_active(is_active.data(), is_active.size()),
        beta(beta.data(), beta.size()),
        resid(resid.data(), resid.size()),
        loss(loss)
    {
        const auto n = X.rows();
        const auto p = X.cols();

        if (X_vars.size() != p) {
            throw util::adelie_core_solver_error(
                "X_vars must be (p,) where X is (n, p). "
            );
        }
        if (tol < 0) {
            throw util::adelie_core_solver_error(
                "tol must be >= 0."
            );
        }
        if (beta.size() != p) {
            throw util::adelie_core_solver_error(
                "beta must be (p,) where X is (n, p). "
            );
        }
        if (resid.size() != n) {
            throw util::adelie_core_solver_error(
                "resid must be (n,) where X is (n, p). "
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
        const auto n = X.rows();
        const auto p = X.cols();

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
                    const auto gk = X.col(k).dot(resid.matrix());
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
                    resid -= del * X.col(k).array();
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
                const auto gk = X.col(k).dot(resid.matrix());
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
                resid -= del * X.col(k).array();
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

} // namespace optimization
} // namespace adelie_core