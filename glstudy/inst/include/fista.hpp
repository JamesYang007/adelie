#pragma once

namespace glstudy {
    
template <class LType, class VType, class ValueType, class XType>
inline
auto fista_solver(
    const LType& L,
    const VType& v,
    ValueType l1,
    ValueType l2,
    ValueType tol,
    size_t max_iters,
    XType& x_sol,
    size_t& iters
)
{
    const auto lip = L.maxCoeff();
    const auto p = L.size();
    Eigen::VectorXd x_diff(p);
    Eigen::VectorXd x_old(p);
    Eigen::VectorXd x(p); x.setZero();
    Eigen::VectorXd y(p); y = x;
    Eigen::VectorXd v_tilde(p);
    ValueType t = 1;

    /*
     * p_L(y) = (L+l_2)^{-1} (1-l_1 / ||v_tilde||_2)_+ v_tilde
     * v_tilde = L (y - 1/L df(y))
     */ 
    iters = 0;
    for (; iters < max_iters; ++iters) {
        x_old.swap(x);
        v_tilde.array() = lip * (y.array() - 1./lip * (L.array() * y.array() - v.array()));
        const auto v_tilde_l2 = v_tilde.norm();
        if (v_tilde_l2 <= l1) {
            x.setZero();
        } else {
            x = 1./(lip + l2) * (1 - l1 / v_tilde_l2) * v_tilde;
        }
        const auto t_old = t;
        t = (1 + std::sqrt(1 + 4 * t * t)) * 0.5;
        x_diff = x - x_old;
        y = x + (t_old - 1) / t * x_diff;
        
        if (x_diff.norm() < tol) break;
    }
    
    x_sol = x;
}

} // namespace glstudy