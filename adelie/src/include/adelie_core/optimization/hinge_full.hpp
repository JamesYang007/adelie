#pragma once
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace optimization {

template <class MatrixType>
struct StateHingeFull
{
    using matrix_t = MatrixType;
    using value_t = typename std::decay_t<MatrixType>::Scalar;
    using vec_value_t = util::rowvec_type<value_t>;
    using map_vec_value_t = Eigen::Map<vec_value_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;
    using map_cmatrix_t = Eigen::Map<const matrix_t>;

    const map_cmatrix_t quad;
    const map_cvec_value_t penalty_pos;
    const map_cvec_value_t penalty_neg;

    const size_t max_iters;
    const value_t tol;

    size_t iters = 0;
    map_vec_value_t x;      
    map_vec_value_t grad;

    double time_elapsed = 0;

    explicit StateHingeFull(
        const Eigen::Ref<const matrix_t>& quad,
        const Eigen::Ref<const vec_value_t>& penalty_pos,
        const Eigen::Ref<const vec_value_t>& penalty_neg,
        size_t max_iters,
        value_t tol,
        Eigen::Ref<vec_value_t> x,
        Eigen::Ref<vec_value_t> grad
    ):
        quad(quad.data(), quad.rows(), quad.cols()),
        penalty_pos(penalty_pos.data(), penalty_pos.size()),
        penalty_neg(penalty_neg.data(), penalty_neg.size()),
        max_iters(max_iters),
        tol(tol),
        x(x.data(), x.size()),
        grad(grad.data(), grad.size())
    {}

    void solve()
    {
        const auto n = x.size();

        iters = 0;

        while (iters < max_iters) {
            value_t convg_measure = 0;
            ++iters;
            for (int i = 0; i < n; ++i) {
                const auto qii = quad(i,i);
                const auto ppi = penalty_pos[i];
                const auto pni = penalty_neg[i];
                auto& xi = x[i];
                const auto gi = grad[i];
                const auto xi_old = xi;
                const auto gi0 = gi + qii * xi_old;
                xi = std::copysign(
                    std::max<value_t>(std::max<value_t>(-pni-gi0, gi0-ppi), 0),
                    gi0 + pni
                ) / qii;
                const auto del = xi - xi_old;
                if (del == 0) continue;
                const auto scaled_del_sq = qii * del * del; 
                convg_measure = std::max<value_t>(convg_measure, scaled_del_sq);
                if constexpr (matrix_t::IsRowMajor) {
                    grad -= del * quad.array().row(i);
                } else {
                    grad -= del * quad.array().col(i);
                }
            }
            if (convg_measure < tol) return;
        }

        throw util::adelie_core_solver_error(
            "StateHingeFull: max iterations reached!"
        );
    }
};

} // namespace optimization
} // namespace adelie_core