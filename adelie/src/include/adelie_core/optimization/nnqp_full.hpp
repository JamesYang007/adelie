#pragma once
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace optimization {

template <class MatrixType, bool _sign=false>
struct StateNNQPFull
{
    using matrix_t = MatrixType;
    using value_t = typename std::decay_t<MatrixType>::Scalar;
    using vec_value_t = util::rowvec_type<value_t>;
    using map_vec_value_t = Eigen::Map<vec_value_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;
    using map_cmatrix_t = Eigen::Map<const matrix_t>;

    static constexpr bool sign = false;

    const map_cmatrix_t quad;

    const size_t max_iters;
    const value_t tol;

    size_t iters = 0;
    map_vec_value_t x;      
    map_vec_value_t grad;

    double time_elapsed = 0;

    explicit StateNNQPFull(
        const Eigen::Ref<const matrix_t>& quad,
        size_t max_iters,
        value_t tol,
        Eigen::Ref<vec_value_t> x,
        Eigen::Ref<vec_value_t> grad
    ):
        quad(quad.data(), quad.rows(), quad.cols()),
        max_iters(max_iters),
        tol(tol),
        x(x.data(), x.size()),
        grad(grad.data(), grad.size())
    {
        const auto d = quad.rows(); 

        if (quad.cols() != d) {
            throw util::adelie_core_solver_error(
                "quad must be (d, d). "
            );
        }
        if (tol < 0) {
            throw util::adelie_core_solver_error(
                "tol must be >= 0."
            );
        }
        if (x.size() != d) {
            throw util::adelie_core_solver_error(
                "x must be (d,) where quad is (d, d). "
            );
        }
        if (grad.size() != d) {
            throw util::adelie_core_solver_error(
                "grad must be (d,) where quad is (d, d). "
            );
        }
    }

    void solve()
    {
        const auto n = x.size();

        iters = 0;

        while (iters < max_iters) {
            value_t convg_measure = 0;
            ++iters;
            for (int i = 0; i < n; ++i) {
                const auto qii = quad(i,i);
                const auto gi = grad[i];
                auto& xi = x[i];
                const auto xi_old = xi;
                const auto step = (qii <= 0) ? 0 : (gi / qii);
                xi = std::max<value_t>(xi + step, 0);
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
            if (convg_measure < quad.cols() * tol) return;
        }

        throw util::adelie_core_solver_error(
            "StateNNQPFull: max iterations reached!"
        );
    }
};

template <class MatrixType>
struct StateNNQPFull<MatrixType, true>
{
    using matrix_t = MatrixType;
    using value_t = typename std::decay_t<MatrixType>::Scalar;
    using vec_value_t = util::rowvec_type<value_t>;
    using map_vec_value_t = Eigen::Map<vec_value_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;
    using map_cmatrix_t = Eigen::Map<const matrix_t>;

    static constexpr bool sign = true;

    const map_cvec_value_t sgn;
    const map_cmatrix_t quad;
    const value_t y_var;

    const size_t max_iters;
    const value_t tol;

    size_t iters = 0;
    map_vec_value_t x;      
    map_vec_value_t grad;

    double time_elapsed = 0;

    explicit StateNNQPFull(
        const Eigen::Ref<const vec_value_t>& sgn,
        const Eigen::Ref<const matrix_t>& quad,
        value_t y_var,
        size_t max_iters,
        value_t tol,
        Eigen::Ref<vec_value_t> x,
        Eigen::Ref<vec_value_t> grad
    ):
        sgn(sgn.data(), sgn.size()),
        quad(quad.data(), quad.rows(), quad.cols()),
        y_var(y_var),
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
                const auto si = sgn[i];
                const auto qii = quad(i,i);
                const auto gi = grad[i];
                auto& xi = x[i];
                const auto xi_old = xi;
                const auto step = (qii <= 0) ? 0 : (gi / qii);
                xi = (si > 0) ? std::max<value_t>(xi + step, 0) : std::min<value_t>(xi + step, 0);
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
            if (convg_measure < y_var * tol) return;
        }

        throw util::adelie_core_solver_error(
            "StateNNQPFull: max iterations reached!"
        );
    }
};

} // namespace optimization
} // namespace adelie_core