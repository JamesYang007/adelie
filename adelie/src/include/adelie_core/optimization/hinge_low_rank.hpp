#pragma once
#include <adelie_core/matrix/utils.hpp>
#include <adelie_core/util/types.hpp>
#include <adelie_core/util/macros.hpp>

namespace adelie_core {
namespace optimization {

template <class ValueType, class IndexType=Eigen::Index>
struct StateHingeLowRank
{
    using value_t = ValueType;
    using index_t = IndexType;
    using vec_index_t = util::rowvec_type<index_t>;
    using vec_value_t = util::rowvec_type<value_t>;
    using colmat_value_t = util::colmat_type<value_t>;
    using rowmat_value_t = util::rowmat_type<value_t>;
    using map_vec_index_t = Eigen::Map<vec_index_t>;
    using map_vec_value_t = Eigen::Map<vec_value_t>;
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
    size_t active_size = 0;
    map_vec_value_t x;
    map_vec_value_t resid;
    map_vec_index_t active_set;
    map_vec_value_t active_vars;
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
        Eigen::Ref<vec_value_t> x,
        Eigen::Ref<vec_value_t> resid,
        Eigen::Ref<vec_index_t> active_set,
        Eigen::Ref<vec_value_t> active_vars,
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
        x(x.data(), x.size()),
        resid(resid.data(), resid.size()),
        active_set(active_set.data(), active_set.size()),
        active_vars(active_vars.data(), active_vars.size()),
        grad(grad.data(), grad.size())
    {
        if (A.rows() < A.cols()) {
            throw util::adelie_core_error(
                "Constraint matrix must be tall (number of rows at least as large as number of columns)."
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
                grad_m /* unused */,
                grad_m
            );
        };

        const auto add_active = [&](int i) {
            active_set[active_size] = i;
            const auto Ai = A.row(i);
            active_vars[active_size] = std::max<value_t>(
                Ai * quad * Ai.transpose(),
                1e-14
            );
            ++active_size;
        };

        iters = 0;
        active_size = 0;
        for (Eigen::Index i = 0; i < x.size(); ++i) {
            if (x[i] == 0) continue;
            add_active(i);
        }
        compute_grad();

        while (1) {
            /* Screening step */

            // if no new active variables allowed just finish
            const size_t max_n_new_active = std::min<size_t>(batch_size, A.rows()-active_size);
            if (max_n_new_active <= 0) return;

            // measure KKT violation and enforce active coefficients to never violate
            grad = (grad - penalty_pos).max(-penalty_neg-grad);
            for (size_t i = 0; i < active_size; ++i) {
                const auto k = active_set[i];
                grad[k] = -std::numeric_limits<value_t>::infinity();
            }

            // check if any violations exist and append to active set
            size_t n_new_active = 0;
            for (Eigen::Index i = 0; i < grad.size(); ++i) {
                if (grad[i] <= 0) continue;

                add_active(i);
                ++n_new_active;

                if (n_new_active >= max_n_new_active) break;
            }
            if (n_new_active <= 0) return;

            /* Fit step */

            while (1) {
                ++iters;
                value_t convg_measure = 0;
                for (size_t i = 0; i < active_size; ++i) {
                    const auto k = active_set[i];
                    const auto vk = active_vars[i];
                    const auto lk = penalty_neg[k];
                    const auto uk = penalty_pos[k];
                    const auto Ak = A.row(k);
                    const auto gk = Ak.dot(resid.matrix());
                    auto& xk = x[k];

                    const auto xk_old = xk;
                    const auto gk0 = gk + vk * xk_old;
                    xk = std::copysign(
                        std::max<value_t>(std::max<value_t>(
                            -lk-gk0, gk0-uk
                        ), 0),
                        gk0+lk 
                    ) / vk;

                    if (xk == xk_old) continue;

                    const auto del = xk - xk_old;
                    convg_measure = std::max<value_t>(
                        convg_measure,
                        vk * del * del
                    );
                    resid.matrix() -= del * (Ak * quad);
                }

                if (iters >= max_iters) {
                    throw util::adelie_core_solver_error(
                        "StateHingeLowRank: max iterations reached!"
                    );
                }
                if (convg_measure <= tol) break;
            }

            /* Invariance step */

            compute_grad();

        } // end while(1)
    }
};

} // namespace optimization
} // namespace adelie_core