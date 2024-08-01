#pragma once
#include <adelie_core/matrix/utils.hpp>
#include <adelie_core/util/types.hpp>
#include <adelie_core/util/macros.hpp>

namespace adelie_core {
namespace optimization {

template <class ValueType, 
          class IndexType=Eigen::Index,
          class DynVecIndexType=std::vector<IndexType>&,
          class DynVecValueType=std::vector<ValueType>&>
struct StateHingeLowRank
{
    using value_t = ValueType;
    using index_t = IndexType;
    using dyn_vec_index_t = DynVecIndexType;
    using dyn_vec_value_t = DynVecValueType;
    using vec_index_t = util::rowvec_type<index_t>;
    using vec_value_t = util::rowvec_type<value_t>;
    using colmat_value_t = util::colmat_type<value_t>;
    using rowmat_value_t = util::rowmat_type<value_t>;
    using map_vec_index_t = Eigen::Map<vec_index_t>;
    using map_vec_value_t = Eigen::Map<vec_value_t>;
    using map_rowmat_value_t = Eigen::Map<rowmat_value_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;
    using map_ccolmat_value_t = Eigen::Map<const colmat_value_t>;
    using map_crowmat_value_t = Eigen::Map<const rowmat_value_t>;

    const map_ccolmat_value_t quad;
    const map_crowmat_value_t A;
    const map_cvec_value_t penalty_neg;
    const map_cvec_value_t penalty_pos;

    const size_t batch_size;
    const size_t max_iters;
    const value_t tol;
    const size_t n_threads;

    size_t iters = 0;
    dyn_vec_index_t active_set;
    dyn_vec_value_t active_value;
    map_vec_value_t active_vars;
    map_rowmat_value_t active_AQ;
    map_vec_value_t resid;
    map_vec_value_t grad;

    double time_elapsed = 0;

    explicit StateHingeLowRank(
        const Eigen::Ref<const colmat_value_t>& quad,
        const Eigen::Ref<const rowmat_value_t>& A,
        const Eigen::Ref<const vec_value_t>& penalty_neg,
        const Eigen::Ref<const vec_value_t>& penalty_pos,
        size_t batch_size,
        size_t max_iters,
        value_t tol,
        size_t n_threads,
        dyn_vec_index_t& active_set,
        dyn_vec_value_t& active_value,
        Eigen::Ref<vec_value_t> active_vars,
        Eigen::Ref<rowmat_value_t> active_AQ,
        Eigen::Ref<vec_value_t> resid,
        Eigen::Ref<vec_value_t> grad
    ):
        quad(quad.data(), quad.rows(), quad.cols()),
        A(A.data(), A.rows(), A.cols()),
        penalty_neg(penalty_neg.data(), penalty_neg.size()),
        penalty_pos(penalty_pos.data(), penalty_pos.size()),
        batch_size(batch_size),
        max_iters(max_iters),
        tol(tol),
        n_threads(n_threads),
        active_set(active_set),
        active_value(active_value),
        active_vars(active_vars.data(), active_vars.size()),
        active_AQ(active_AQ.data(), active_AQ.rows(), active_AQ.cols()),
        resid(resid.data(), resid.size()),
        grad(grad.data(), grad.size())
    {
        const auto m = A.rows();
        const auto d = A.cols();

        if (quad.rows() != d || quad.cols() != d) {
            throw util::adelie_core_solver_error(
                "quad must be (d, d) where A is (m, d). "
            );
        }
        if (m < d && n_threads > 1) {
            throw util::adelie_core_error(
                "A must be (m, d) where m >= d if n_threads > 1. "
            );
        }
        if (penalty_neg.size() != m) {
            throw util::adelie_core_solver_error(
                "penalty_neg must be (m,) where A is (m, d). "
            );
        }
        if (penalty_pos.size() != m) {
            throw util::adelie_core_solver_error(
                "penalty_pos must be (m,) where A is (m, d). "
            );
        }
        if (tol < 0) {
            throw util::adelie_core_solver_error(
                "tol must be >= 0."
            );
        }
        if (resid.size() != d) {
            throw util::adelie_core_solver_error(
                "resid must be (d,) where A is (m, d). "
            );
        }
        if (active_vars.size() != m) {
            throw util::adelie_core_solver_error(
                "active_vars must be (m,) where A is (m, d). "
            );
        }
        if (active_AQ.rows() != m || active_AQ.cols() != d) {
            throw util::adelie_core_solver_error(
                "active_AQ must be (m, d) where A is (m, d). "
            );
        }
        if (grad.size() != m) {
            throw util::adelie_core_solver_error(
                "grad must be (m,) where A is (m, d). "
            );
        }
    }

    void solve()
    {
        const auto compute_grad = [&]() {
            auto grad_m = grad.matrix();
            matrix::dgemv(
                A.transpose(),
                resid.matrix(),
                n_threads,
                grad_m /* unused because A.rows() >= A.cols() */,
                grad_m
            );

            // measure KKT violation and enforce active coefficients to never violate
            grad = (grad - penalty_pos).max(-penalty_neg-grad);
            for (size_t i = 0; i < active_set.size(); ++i) {
                const auto k = active_set[i];
                grad[k] = -std::numeric_limits<value_t>::infinity();
            }
        };

        const auto add_active = [&](int i) {
            const auto active_size = active_set.size();
            active_set.push_back(i);
            active_value.push_back(0);
            const auto Ai = A.row(i);
            active_AQ.row(active_size) = Ai * quad;
            active_vars[active_size] = std::max<value_t>(
                active_AQ.row(active_size).dot(Ai),
                1e-14
            );
        };

        iters = 0;

        while (1) {
            /* Screening step */
            if (iters)
            {
                const auto active_size = active_set.size();
                const size_t max_n_new_active = std::min<size_t>(batch_size, A.rows()-active_size);
                if (max_n_new_active <= 0) return;

                size_t n_new_active = 0;

                // check if any violations exist and append to active set
                for (Eigen::Index i = 0; i < grad.size(); ++i) {
                    if (grad[i] <= 0) continue;

                    add_active(i);
                    ++n_new_active;

                    if (n_new_active >= max_n_new_active) break;
                }
                if (n_new_active <= 0) return;
            }

            /* Fit step */

            while (1) {
                ++iters;
                value_t convg_measure = 0;
                for (size_t i = 0; i < active_set.size(); ++i) {
                    const auto k = active_set[i];
                    const auto vk = active_vars[i];
                    const auto lk = penalty_neg[k];
                    const auto uk = penalty_pos[k];
                    const auto Ak = A.row(k);
                    const auto QAk = active_AQ.row(i);
                    const auto gk = Ak.dot(resid.matrix());
                    auto& xk = active_value[i];
                    const auto xk_old = xk;
                    const auto gk0 = gk + vk * xk_old;
                    const auto gk0_lk = gk0 + lk;
                    xk = std::copysign(
                        std::max<value_t>(std::max<value_t>(
                            -gk0_lk, gk0-uk
                        ), 0),
                        gk0_lk 
                    ) / vk;

                    if (xk == xk_old) continue;

                    const auto del = xk - xk_old;
                    convg_measure = std::max<value_t>(
                        convg_measure,
                        vk * del * del
                    );
                    resid.matrix() -= del * QAk;
                }

                if (iters >= max_iters) {
                    throw util::adelie_core_solver_error(
                        "StateHingeLowRank: max iterations reached!"
                    );
                }
                if (convg_measure <= A.cols() * tol) break;
            }

            /* Invariance step */

            compute_grad();
        } // end while(1)
    }
};

} // namespace optimization
} // namespace adelie_core