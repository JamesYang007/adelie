#pragma once
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace bcd {
namespace unconstrained {

template <class LType, class VType, class ValueType, class XType>
inline
void ista_solver(
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
    using value_t = ValueType;
    const auto lip = L.maxCoeff();
    const auto nu = 1. / lip;
    const auto p = L.size();
    util::rowvec_type<value_t> x_diff(p);
    util::rowvec_type<value_t> x_old(p);
    util::rowvec_type<value_t> x(p); x.setZero();
    util::rowvec_type<value_t> v_tilde(p);

    iters = 0;
    for (; iters < max_iters; ++iters) {
        x_old.swap(x);
        v_tilde = x_old - nu * (L * x_old - v);
        const auto v_tilde_l2 = v_tilde.matrix().norm();
        if (v_tilde_l2 <= l1 * nu) {
            x.setZero();
        } else {
            x = ((lip /(lip + l2)) * (1 - (l1 * nu) / v_tilde_l2)) * v_tilde;
        }
        x_diff = x - x_old;
        if ((x_diff.abs() <= (tol * x.abs())).all()) break;
    }
    
    x_sol = x;
}
    
template <class LType, class VType, class ValueType, class XType>
inline
void fista_solver(
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
    using value_t = ValueType;
    const auto lip = L.maxCoeff();
    const auto nu = 1. / lip;
    const auto p = L.size();
    util::rowvec_type<value_t> x_diff(p);
    util::rowvec_type<value_t> x_old(p);
    util::rowvec_type<value_t> x(p); x.setZero();
    util::rowvec_type<value_t> y(p); y = x;
    util::rowvec_type<value_t> v_tilde(p);
    value_t t = 1;

    iters = 0;
    for (; iters < max_iters; ++iters) {
        x_old.swap(x);
        v_tilde = y - nu * (L * y - v);
        const auto v_tilde_l2 = v_tilde.matrix().norm();
        if (v_tilde_l2 <= l1 * nu) {
            x.setZero();
        } else {
            x = (lip /(lip + l2)) * (1 - (l1 * nu) / v_tilde_l2) * v_tilde;
        }
        const auto t_old = t;
        t = (1 + std::sqrt(1 + 4 * t * t)) * 0.5;
        x_diff = x - x_old;
        y = x + (t_old - 1) / t * x_diff;
        
        if ((x_diff.abs() <= (tol * x.abs())).all()) break;
    }
    
    x_sol = x;
}

template <class LType, class VType, class ValueType, class XType>
inline
void fista_adares_solver(
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
    using value_t = ValueType;
    const auto lip = L.maxCoeff();
    const auto nu = 1. / lip;
    const auto p = L.size();
    util::rowvec_type<value_t> x_diff(p);
    util::rowvec_type<value_t> x_old(p);
    util::rowvec_type<value_t> x(p); x.setZero();
    util::rowvec_type<value_t> y(p); y = x;
    util::rowvec_type<value_t> v_tilde(p);
    util::rowvec_type<value_t> df(p);
    value_t t = 1;

    iters = 0;
    for (; iters < max_iters; ++iters) {
        x_old.swap(x);
        df = (L * y - v);
        v_tilde = y - nu * df;
        const auto v_tilde_l2 = v_tilde.matrix().norm();
        if (v_tilde_l2 <= l1 * nu) {
            x.setZero();
        } else {
            x = (lip /(lip + l2)) * (1 - (l1 * nu) / v_tilde_l2) * v_tilde;
        }
        x_diff = x - x_old;
        
        // accelerated
        if (v_tilde.matrix().dot(x_diff.matrix()) > 0) {
            y = x;
            t = 1;
        } else {
            const auto t_old = t;
            t = (1 + std::sqrt(1 + 4 * t * t)) * 0.5;
            y = x + (t_old - 1) / t * x_diff;
        }

        if ((x_diff.abs() <= (tol * x.abs())).all()) break;
    }
    
    x_sol = x;
}

} // namespace unconstrained
} // namespace bcd
} // namespace adelie_core