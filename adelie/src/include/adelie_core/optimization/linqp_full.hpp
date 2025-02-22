#pragma once
#include <Eigen/Cholesky>
#include <adelie_core/util/types.hpp>

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
    const value_t relaxed_tol;
    const value_t tol;
    const value_t slack;
    const value_t lmda_max;
    const value_t lmda_min;
    const size_t lmda_path_size;

    size_t iters = 0;
    size_t backtrack_iters = 0;
    map_vec_value_t x;   

    double time_elapsed = 0;

    explicit StateLinQPFull(
        const Eigen::Ref<const matrix_t>& quad,
        const Eigen::Ref<const vec_value_t>& linear,
        const Eigen::Ref<const matrix_t>& A,
        const Eigen::Ref<const vec_value_t>& lower,
        const Eigen::Ref<const vec_value_t>& upper,
        size_t max_iters,
        value_t relaxed_tol,
        value_t tol,
        value_t slack,
        value_t lmda_max,
        value_t lmda_min,
        size_t lmda_path_size,
        Eigen::Ref<vec_value_t> x
    ):
        quad(quad.data(), quad.rows(), quad.cols()),
        linear(linear.data(), linear.size()),
        A(A.data(), A.rows(), A.cols()),
        lower(lower.data(), lower.size()),
        upper(upper.data(), upper.size()),
        max_iters(max_iters),
        relaxed_tol(relaxed_tol),
        tol(tol),
        slack(slack),
        lmda_max(lmda_max),
        lmda_min(lmda_min),
        lmda_path_size(lmda_path_size),
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

        Ax.matrix() = x.matrix() * A.transpose();

        const value_t min_ratio = lmda_min / lmda_max;
        const value_t lmda_factor = std::pow(min_ratio, 1.0 / (lmda_path_size-1));
        value_t lmda = lmda_max;

        iters = 0;
        backtrack_iters = 0;

        for (size_t i = 0; i < lmda_path_size; ++i) {
            const value_t _tol = (i+1 < lmda_path_size) ? relaxed_tol : tol;

            bool is_prev_valid = false;

            while (iters < max_iters) {
                // compute gradient
                grad = (1 / (upper - Ax) - 1 / (Ax + lower)).matrix() * A;
                grad = (x.matrix() * quad).array() - linear + (lmda / m) * grad;

                if (is_prev_valid) {
                    if (std::abs(((x-x_prev) * (grad - grad_prev)).mean()) <= _tol) break;
                }

                // save previous 
                x_prev = x;
                Ax_prev = Ax;
                grad_prev = grad;
                is_prev_valid = true;

                // compute hessian
                D = (lmda / m) * (1 / (upper - Ax).square() + 1 / (Ax + lower).square());
                hess.noalias() = A.transpose() * D.matrix().asDiagonal() * A;
                hess += quad;

                // compute Newton update
                Eigen::LLT<Eigen::Ref<matrix_t>> hess_chol(hess);
                x.matrix() -= hess_chol.solve(grad.matrix().transpose());

                Ax.matrix() = x.matrix() * A.transpose();

                // backtrack for feasibility
                value_t step_size = -1;
                do {
                    step_size = (1-slack) * std::max<value_t>(std::min<value_t>(
                        ((upper - Ax_prev).min(lower + Ax_prev) / (Ax - Ax_prev).abs().max(_tol)).minCoeff(),
                        1
                    ), 0);
                    Ax = Ax_prev + step_size * (Ax - Ax_prev);
                    ++backtrack_iters;
                } while (
                    (Ax >= upper).any() ||
                    (Ax <= -lower).any()
                );
                x = x_prev + step_size * (x - x_prev);

                ++iters;
            }

            if (iters >= max_iters) {
                throw util::adelie_core_solver_error(
                    "StateLinQPFull: maximum iterations reached!"
                );
            }

            lmda *= lmda_factor;
        }
    }
};

} // namespace optimization
} // namespace adelie_core