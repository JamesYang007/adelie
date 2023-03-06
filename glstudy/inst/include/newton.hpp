#pragma once 
#include <cstddef>

namespace glstudy {

/**
 * Solves the solution for the equation (w.r.t. \f$x\f$):
 * \f[
 *      minimize \frac{1}{2} x^\top L x - x^\top v 
 *          + l_1 ||x||_2 + \frac{l_2}{2} ||x||_2^2
 * \f]
 *      
 * @param   L       vector representing a diagonal PSD matrix.
 *                  Must have max(L + s) > 0. 
 *                  L.size() <= buffer1.size().
 * @param   v       any vector.  
 * @param   l1      L2-norm penalty. Must be >= 0.
 * @param   l2      L2 penalty. Must be >= 0.
 * @param   tol         Newton's method tolerance of closeness to 0.
 * @param   max_iters   maximum number of iterations of Newton's method.
 * @param   x           solution vector.
 * @param   iters       number of Newton's method iterations taken.
 * @param   buffer1     any vector with L.size() <= buffer1.size().
 * @param   buffer2     any vector with L.size() <= buffer2.size().
 */
template <class LType, class VType, class ValueType, 
          class XType, class BufferType>
inline
void newton_solver(
    const LType& L,
    const VType& v,
    ValueType l1,
    ValueType l2,
    ValueType tol,
    size_t max_iters,
    XType& x,
    size_t& iters,
    BufferType& buffer1,
    BufferType& buffer2
)
{
    using value_t = ValueType;

    iters = 0;

    // Easy case: ||v||_2 <= l1 -> x = 0
    const auto v_l2 = v.norm();
    if (v_l2 <= l1) {
        x.setZero();
        return;
    }
    
    // Difficult case: ||v||_2 > l1
    if (l1 <= 0.0) {
        x.array() = v.array() / (L.array() + l2);
        return;
    }
    
    // First solve for h := ||x||_2
    auto vbuffer1 = buffer1.head(L.size());
    auto vbuffer2 = buffer2.head(L.size());

    // Find good initialization
    // The following helps tremendously if the resulting h > 0.
    vbuffer1.array() = (L.array() + l2);
    const value_t b = l1 * vbuffer1.sum();
    const value_t a = vbuffer1.squaredNorm();
    const value_t v_l1 = v.template lpNorm<1>();
    const value_t c = l1 * l1 * L.size() - v_l1 * v_l1;
    const value_t zero = 0.0;
    const value_t discr = b*b - a*c;
    auto h = (discr > -1e-12) ? 
        (-b + std::sqrt(std::max(discr, zero))) / a : 0.0;
    
    // Otherwise, if h <= 0, we know at least 0 is a reasonable solution.
    // The only case h <= 0 is when 0 is already close to the solution.
    h = std::max(h, zero);

    // Newton method
    value_t fh;
    value_t dfh;

    const auto newton_update = [&]() {
        vbuffer2.array() = vbuffer1.array() * h + l1;
        x.array() = (v.array() / vbuffer2.array()).square();
        fh = x.sum() - 1;
        dfh = -2 * (
            x.array() * (vbuffer1.array() / vbuffer2.array())
        ).sum();
    };
    
    newton_update();

    while (std::abs(fh) > tol && iters < max_iters) {
        // Newton update 
        h -= fh / dfh;
        newton_update();
        ++iters;
    }
    
    // numerical stability
    h = std::max(h, 1e-14);

    // final solution
    x.array() = h * v.array() / vbuffer2.array();
}

} // namespace glstudy