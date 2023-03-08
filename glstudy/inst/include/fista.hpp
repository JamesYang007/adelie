#pragma once
#include <cstddef>

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
    const auto nu = 1. / lip;
    const auto p = L.size();
    Eigen::VectorXd x_diff(p);
    Eigen::VectorXd x_old(p);
    Eigen::VectorXd x(p); x.setZero();
    Eigen::VectorXd y(p); y = x;
    Eigen::VectorXd v_tilde(p);
    ValueType t = 1;

    iters = 0;
    for (; iters < max_iters; ++iters) {
        x_old.swap(x);
        v_tilde.array() = y.array() - nu * (L.array() * y.array() - v.array());
        const auto v_tilde_l2 = v_tilde.norm();
        if (v_tilde_l2 <= l1 * nu) {
            x.setZero();
        } else {
            x = (lip /(lip + l2)) * (1 - (l1 * nu) / v_tilde_l2) * v_tilde;
        }
        const auto t_old = t;
        t = (1 + std::sqrt(1 + 4 * t * t)) * 0.5;
        x_diff = x - x_old;
        y = x + (t_old - 1) / t * x_diff;
        
        if ((x_diff.array().abs() < tol * x.array().abs()).all()) break;
    }
    
    x_sol = x;
}

/*
 * FISTA with adaptive restart.
 */
template <class LType, class VType, class ValueType, class XType>
inline
auto fista_adares_solver(
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
    const auto nu = 1. / lip;
    const auto p = L.size();
    Eigen::VectorXd x_diff(p);
    Eigen::VectorXd x_old(p);
    Eigen::VectorXd x(p); x.setZero();
    Eigen::VectorXd y(p); y = x;
    Eigen::VectorXd v_tilde(p);
    Eigen::VectorXd df(p);
    ValueType t = 1;

    iters = 0;
    for (; iters < max_iters; ++iters) {
        x_old.swap(x);
        df.array() = (L.array() * y.array() - v.array());
        v_tilde.array() = y.array() - nu * df.array();
        const auto v_tilde_l2 = v_tilde.norm();
        if (v_tilde_l2 <= l1 * nu) {
            x.setZero();
        } else {
            x = (lip /(lip + l2)) * (1 - (l1 * nu) / v_tilde_l2) * v_tilde;
        }
        x_diff = x - x_old;
        
        // accelerated
        if (v_tilde.dot(x_diff) > 0) {
            y = x;
            t = 1;
        } else {
            const auto t_old = t;
            t = (1 + std::sqrt(1 + 4 * t * t)) * 0.5;
            y = x + (t_old - 1) / t * x_diff;
        }

        if ((x_diff.array().abs() < tol * x.array().abs()).all()) break;
    }
    
    x_sol = x;
}

} // namespace glstudy