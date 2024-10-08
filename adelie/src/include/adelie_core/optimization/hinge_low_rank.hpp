#pragma once
#include <adelie_core/matrix/utils.hpp>
#include <adelie_core/util/types.hpp>
#include <adelie_core/util/macros.hpp>

namespace adelie_core {
namespace optimization {

template <class MatrixType, 
          class IndexType=Eigen::Index,
          class DynVecIndexType=std::vector<IndexType>&,
          class DynVecValueType=std::vector<typename MatrixType::value_t>&>
struct StateHingeLowRank
{
    using matrix_t = MatrixType;
    using value_t = typename matrix_t::value_t;
    using index_t = IndexType;
    using dyn_vec_index_t = DynVecIndexType;
    using dyn_vec_value_t = DynVecValueType;
    using vec_value_t = util::rowvec_type<value_t>;
    using vec_index_t = util::rowvec_type<index_t>;
    using colmat_value_t = util::colmat_type<value_t>;
    using rowmat_value_t = util::rowmat_type<value_t>;
    using map_vec_value_t = Eigen::Map<vec_value_t>;
    using map_rowmat_value_t = Eigen::Map<rowmat_value_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;
    using map_ccolmat_value_t = Eigen::Map<const colmat_value_t>;

    const map_ccolmat_value_t quad;
    matrix_t* A;
    const map_cvec_value_t penalty_neg;
    const map_cvec_value_t penalty_pos;
    const value_t y_var;

    const size_t batch_size;
    const size_t max_iters;
    const value_t tol;

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
        matrix_t& A,
        const Eigen::Ref<const vec_value_t>& penalty_neg,
        const Eigen::Ref<const vec_value_t>& penalty_pos,
        value_t y_var,
        size_t batch_size,
        size_t max_iters,
        value_t tol,
        dyn_vec_index_t& active_set,
        dyn_vec_value_t& active_value,
        Eigen::Ref<vec_value_t> active_vars,
        Eigen::Ref<rowmat_value_t> active_AQ,
        Eigen::Ref<vec_value_t> resid,
        Eigen::Ref<vec_value_t> grad
    ):
        quad(quad.data(), quad.rows(), quad.cols()),
        A(&A),
        penalty_neg(penalty_neg.data(), penalty_neg.size()),
        penalty_pos(penalty_pos.data(), penalty_pos.size()),
        y_var(y_var),
        batch_size(batch_size),
        max_iters(max_iters),
        tol(tol),
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
            A->tmul(resid, grad);

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
            auto AiQ = active_AQ.row(active_size);
            A->rmmul(i, quad, AiQ);
            active_vars[active_size] = std::max<value_t>(A->rvmul(i, AiQ), 0);
        };

        const auto m = A->rows();
        const auto d = A->cols();
        vec_index_t viols_order = vec_index_t::LinSpaced(m, 0, m-1);
        iters = 0;

        while (1) {
            while (1) {
                ++iters;
                value_t convg_measure = 0;
                for (size_t i = 0; i < active_set.size(); ++i) {
                    const auto k = active_set[i];
                    const auto vk = active_vars[i];
                    const auto lk = penalty_neg[k];
                    const auto uk = penalty_pos[k];
                    const auto QAk = active_AQ.row(i);
                    const auto gk = A->rvmul(k, resid);
                    auto& xk = active_value[i];
                    const auto xk_old = xk;
                    const auto gk0 = gk + vk * xk_old;
                    const auto gk0_lk = gk0 + lk;
                    xk = (vk <= 0) ? xk_old : std::copysign(
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
                if (convg_measure <= y_var * tol) break;
            }

            compute_grad();
            auto& viols = grad;
            std::sort(
                viols_order.data(),
                viols_order.data() + m,
                [&](auto i, auto j) { return viols[i] > viols[j]; }
            );

            const auto active_size_old = active_set.size();
            bool kkt_passed = true;

            // check if any violations exist and append to active set
            for (Eigen::Index i = 0; i < m; ++i) {
                const auto k = viols_order[i];
                const auto vk = viols[k];
                if (vk <= tol) continue;
                kkt_passed = false;
                if (active_set.size() >= active_size_old + batch_size) break;
                add_active(k);
            }

            if (kkt_passed) return;
        } // end while(1)
    }
};

} // namespace optimization
} // namespace adelie_core