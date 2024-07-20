#pragma once
#include <adelie_core/bcd/unconstrained/newton.hpp>
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace bcd {
namespace constrained {

template <class QuadCType, class QCType,
          class AQCType, class QTVCType, class AType, class BType,
          class ValueType, class OutType, class BuffType>
void admm_solver(
    const QuadCType& quad_c,
    ValueType l1,
    ValueType l2,
    const QCType& Q_c,
    const AQCType& AQ_c,
    const QTVCType& QTv_c,
    const AType& A,
    const BType& b,
    ValueType rho,
    size_t max_iters,
    ValueType tol_abs,
    ValueType tol_rel,
    OutType& x,
    OutType& z,
    OutType& u,
    size_t& iters,
    BuffType& buff
)
{
    using value_t = ValueType;
    using vec_value_t = util::rowvec_type<value_t>;

    const auto m = A.rows();
    const auto d = A.cols();

    iters = 0;
    while (iters < max_iters) {
        // increment counter no matter what
        ++iters;

        // x-update
        Eigen::Map<vec_value_t> zmu(buff.data(), m);
        Eigen::Map<vec_value_t> linear_curr(buff.data() + m, d);
        Eigen::Map<vec_value_t> QTx(buff.data() + m+d, d);
        Eigen::Map<vec_value_t> x_buffer1(buff.data() + m+2*d, d);
        Eigen::Map<vec_value_t> x_buffer2(buff.data() + m+3*d, d);
        zmu = z - u;
        linear_curr.matrix() = zmu.matrix() * AQ_c;
        linear_curr = QTv_c + rho * linear_curr;
        size_t x_iters;
        unconstrained::newton_solver(
            quad_c,
            linear_curr,
            l1,
            l2,
            1e-12,
            1000,
            QTx,
            x_iters,
            x_buffer1,
            x_buffer2
        );
        x.matrix() = QTx.matrix() * Q_c.transpose();

        // z-update
        Eigen::Map<vec_value_t> Ax(buff.data(), m);
        Eigen::Map<vec_value_t> z_prev(buff.data()+m, m);
        Ax.matrix() = x.matrix() * A.transpose();
        z_prev = z;
        z = (u + Ax).min(b);

        // u-update
        Eigen::Map<vec_value_t> r(buff.data()+2*m, m);
        r = Ax - z;
        u += r;

        // check convergence
        Eigen::Map<vec_value_t> s(buff.data()+3*m, d);
        s.matrix() = (z - z_prev).matrix() * A;
        const auto eps_pri = std::sqrt(m) * tol_abs + tol_rel * std::max<value_t>(
            Ax.matrix().norm(), z.matrix().norm()
        );
        Eigen::Map<vec_value_t> ATu(buff.data()+3*m+d, d);
        ATu.matrix() = u.matrix() * A;
        const auto eps_dual = std::sqrt(d) * tol_abs + tol_rel * rho * ATu.matrix().norm();
        if (
            (r.square().sum() <= eps_pri * eps_pri) &&
            (rho * rho * s.square().sum() <= eps_dual * eps_dual)
        ) break;
    }
}

} // namespace constrained
} // namespace bcd
} // namespace adelie_core