#pragma once
#include <Eigen/Cholesky>
#include <adelie_core/util/types.hpp>
#include <adelie_core/util/macros.hpp>

namespace adelie_core {
namespace optimization {

template <class MatrixType>
struct StateLinQPFull
{
    using matrix_t = MatrixType;
    using value_t = typename std::decay_t<MatrixType>::Scalar;
    using vec_value_t = util::rowvec_type<value_t>;
    using map_vec_value_t = Eigen::Map<vec_value_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;
    using map_cmatrix_t = Eigen::Map<const matrix_t>;

    const map_cmatrix_t quad;
    const map_cvec_value_t linear;
    const map_cmatrix_t A;
    const map_cvec_value_t lower;
    const map_cvec_value_t upper;

    const size_t max_iters;
    const value_t tol;
    const value_t slack;

    size_t iters = 0;
    map_vec_value_t x;   

    double time_elapsed = 0;

    explicit StateLinQPFull(
        const Eigen::Ref<const matrix_t>& quad,
        const Eigen::Ref<const vec_value_t>& linear,
        const Eigen::Ref<const matrix_t>& A,
        const Eigen::Ref<const vec_value_t>& lower,
        const Eigen::Ref<const vec_value_t>& upper,
        size_t max_iters,
        value_t tol,
        value_t slack,
        Eigen::Ref<vec_value_t> x
    ):
        quad(quad.data(), quad.rows(), quad.cols()),
        linear(linear.data(), linear.size()),
        A(A.data(), A.rows(), A.cols()),
        lower(lower.data(), lower.size()),
        upper(upper.data(), upper.size()),
        max_iters(max_iters),
        tol(tol),
        slack(slack),
        x(x.data(), x.size())
    {}

    size_t buffer_size() const
    {
        const auto m = A.rows();
        const auto d = A.cols();
        return d * (3 + d) + 3 * m; 
    }

    void solve(
        Eigen::Ref<vec_value_t> buff
    ) 
    {
        const auto m = A.rows();
        const auto d = A.cols();

        auto buff_ptr = buff.data();
        Eigen::Map<vec_value_t> x_prev(buff_ptr, d); buff_ptr += d;
        Eigen::Map<vec_value_t> Ax(buff_ptr, m); buff_ptr += m;
        Eigen::Map<vec_value_t> Ax_prev(buff_ptr, m); buff_ptr += m;
        Eigen::Map<vec_value_t> grad(buff_ptr, d); buff_ptr += d;
        Eigen::Map<vec_value_t> grad_prev(buff_ptr, d); buff_ptr += d;
        Eigen::Map<vec_value_t> D(buff_ptr, m); buff_ptr += m;
        Eigen::Map<matrix_t> hess(buff_ptr, d, d); buff_ptr += d * d;

        iters = 0;

        Ax.matrix() = x.matrix() * A.transpose();

        while (iters < max_iters) {
            // compute gradient
            grad = (1 / (upper - Ax) - 1 / (Ax + lower)).matrix() * A;
            grad += t * ((x.matrix() * quad).array() - linear);

            if (iters) {
                if (std::abs(((x-x_prev) * (grad - grad_prev)).mean()) <= tol) return;
            }

            // save previous 
            x_prev = x;
            Ax_prev = Ax;
            grad_prev = grad;

            // compute hessian
            D = 1 / (upper - Ax).square() + 1 / (Ax + lower).square();
            hess.noalias() = A.transpose() * D.matrix().asDiagonal() * A;
            hess += t * quad;

            // compute Newton update
            Eigen::LLT<Eigen::Ref<matrix_t>> hess_chol(hess);
            x.matrix() -= hess_chol.solve(grad.matrix().transpose());

            Ax.matrix() = x.matrix() * A.transpose();

            // backtrack for feasibility
            value_t step_size = -1;
            do {
                step_size = (1-slack) * std::max<value_t>(std::min<value_t>(
                    ((upper - Ax_prev).min(lower + Ax_prev) / (Ax - Ax_prev).abs().max(tol)).minCoeff(),
                    1
                ), 0);
                Ax = Ax_prev + step_size * (Ax - Ax_prev);
            } while (
                (Ax >= upper).any() ||
                (Ax <= -lower).any()
            );
            x = x_prev + step_size * (x - x_prev);

            ++iters;
        }
    }
};

} // namespace optimization
} // namespace adelie_core