#pragma once 
#include <cstddef>
#include <bisect.hpp>
#include <objective.hpp>

namespace glstudy {

/*
 * Vanilla Newton solver.
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
    constexpr value_t zero = 0;

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

    vbuffer1.array() = (L.array() + l2);

    value_t h = 0;

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
    h = std::max(h, zero);

    // final solution
    x.array() = h * v.array() / vbuffer2.array();
}

/*
 * Newton + Brent solver.
 */
template <class LType, class VType, class ValueType, 
          class XType, class BufferType>
inline
void newton_brent_solver(
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
    constexpr value_t zero = 0;

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

    vbuffer1.array() = (L.array() + l2);

    const value_t h_min = compute_h_min(vbuffer1, v, l1);
    const value_t h_max_out = compute_h_max(vbuffer1, v, l1);
    const value_t h_max = std::get<0>(h_max_out);
    value_t h;
    size_t iters_brent = 0;
    brent(
        [&](auto x) { return block_update_objective(h, vbuffer1, v, l1); },
        1e-8, 
        max_iters,
        h_min,
        h_max,
        h,
        iters_brent
    );
    iters += iters_brent;

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
    h = std::max(h, zero);

    // final solution
    x.array() = h * v.array() / vbuffer2.array();
}

/*
 * Newton-ABS solver
 */
template <class LType, class VType, class ValueType, 
          class XType, class BufferType>
inline
void newton_abs_solver(
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
    constexpr value_t zero = 0;

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

    // pre-compute L + l2
    vbuffer1.array() = (L.array() + l2);
    
    // Find good initialization
    const value_t h_min = compute_h_min(vbuffer1, v, l1);
    const auto h_max_out = compute_h_max(vbuffer1, v, l1);
    const value_t h_max = std::get<0>(h_max_out);
    const value_t vbuffer1_min_nzn = std::get<1>(h_max_out);

    value_t h = 0; // norm solution

    const auto tidy_up = [&]() {
        // numerical stability
        h = std::max(h, zero);
        // final solution
        x.array() = h * v.array() / vbuffer2.array();
    };
    
    // If range is very small, just set h = h_min and Newton.
    // NOTE: this accounts for when h_max <= h_min as well on purpose!
    // The numerically stable way of computing h_max may lead to h_max <= h_min.
    // This is an indication that the correct range was small to begin with.
    if (h_max - h_min <= 1e-1) {
        h = h_min;
    } else {
        //// NEW METHOD: bisection
        value_t h_cand = h_max;
        value_t w = -1;
        value_t fh = -1;

        const auto ada_bisect = [&]() {
            // enforce some movement towards h_min for safety.
            w = std::max(l1 / (vbuffer1_min_nzn * h_cand + l1), 0.05);
            h_cand = w * h_min + (1-w) * h_cand;
            fh = block_norm_objective(h_cand, vbuffer1, v, l1);
        };
        
        ada_bisect();
        size_t bisect_iters = 1;
        while ((fh < 0) && (std::abs(fh) > tol)) {
            ada_bisect();
            ++bisect_iters;
        }
        
        // save candidate solution
        h = h_cand;

        // add 1/2 the iterations of bisection since it is about
        // half the cost of a newton step
        iters += static_cast<size_t>(bisect_iters * 0.5);

        if (std::abs(fh) <= tol) {
            tidy_up();
            return;
        }        
    }

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
    
    tidy_up();
}

/**
 * Same as newton_solver above but returns way more information.
 *      
 * @param   init        initial point if used.
 * @param   x           matrix of iterate vectors (column-wise).
 * @param   buffer1     any vector with L.size() <= buffer1.size().
 * @param   buffer2     any vector with L.size() <= buffer2.size().
 */
template <class LType, class VType, class ValueType, 
          class XType, class ItersType, class BufferType>
inline
void newton_solver_debug(
    const LType& L,
    const VType& v,
    ValueType l1,
    ValueType l2,
    ValueType tol,
    size_t max_iters,
    bool smart_init,
    ValueType& h_min,
    ValueType& h_max,
    XType& x,
    ItersType& iters,
    ItersType& smart_iters,
    BufferType& buffer1,
    BufferType& buffer2
)
{
    using value_t = ValueType;
    
    const value_t zero = 0.0;
    value_t h = 0;

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

    // pre-compute L + l2
    vbuffer1.array() = (L.array() + l2);
    
    // Find good initialization
    // The following helps tremendously if the resulting h > 0.
    if (smart_init) {
        // compute h_min
        {
            const value_t b = l1 * vbuffer1.sum();
            const value_t a = vbuffer1.squaredNorm();
            const value_t v_l1 = v.template lpNorm<1>();
            const value_t c = l1 * l1 * L.size() - v_l1 * v_l1;
            const value_t discr = b*b - a*c;
            h_min = (discr > -1e-12) ? 
                (-b + std::sqrt(std::max(discr, zero))) / a : 0.0;
            
            // Otherwise, if h <= 0, we know at least 0 is a reasonable solution.
            // The only case h <= 0 is when 0 is already close to the solution.
            h_min = std::max(h_min, zero);
        }
        
        // compute h_max
        const value_t vbuffer1_min = vbuffer1.minCoeff();
        value_t vbuffer1_min_nzn = vbuffer1_min;

        // If L+l2 have entries <= threshold,
        // find h_max with more numerically-stable routine.
        // There is NO guarantee that f(h_max) <= 0, 
        // but we will use this to bisect and find an h where f(h) >= 0,
        // so we don't necessarily need h_max to be f(h_max) <= 0.
        if (vbuffer1_min <= 1e-10) {
            vbuffer1_min_nzn = std::numeric_limits<value_t>::infinity();
            h_max = 0;
            value_t denom = 0;
            for (int i = 0; i < vbuffer1.size(); ++i) {
                const bool is_nonzero = vbuffer1[i] > 1e-10;
                const auto vi2 = v[i] * v[i];
                h_max += is_nonzero ? vi2 / (vbuffer1[i] * vbuffer1[i]) : 0;
                denom += is_nonzero ? 0 : vi2; 
                vbuffer1_min_nzn = is_nonzero ? std::min(vbuffer1_min_nzn, vbuffer1[i]) : vbuffer1_min_nzn;
            }
            h_max = std::sqrt(std::abs(h_max / (1.0 - denom / (l1 * l1))));
        } else {
            h_max = (v.array() / vbuffer1.array()).matrix().norm();
        }
        
        // If range is very small, just set h = h_min and Newton.
        // NOTE: this accounts for when h_max <= h_min as well on purpose!
        // The numerically stable way of computing h_max may lead to h_max <= h_min.
        // This is an indication that the correct range was small to begin with.
        if (h_max - h_min <= 1e-1) {
            h = h_min;
        } else {
            //// NEW METHOD: bisection
            // Adaptive method enforces some movement towards h_min for safety.

            value_t w = std::max(l1 / (vbuffer1_min_nzn * h_max + l1), 0.05);
            h = w * h_min + (1-w) * h_max;
            value_t fh = (v.array() / (vbuffer1.array() * h + l1)).matrix().squaredNorm() - 1;
            
            smart_iters.push_back(h);

            while ((fh < 0) && std::abs(fh) > tol) {
                w = std::max(l1 / (vbuffer1_min_nzn * h + l1), 0.05);
                h = w * h_min + (1-w) * h;
                fh = (v.array() / (vbuffer1.array() * h + l1)).matrix().squaredNorm() - 1;
                smart_iters.push_back(h);
            }

            if (std::abs(fh) <= tol) {
                h = std::max(h, zero);
                x.array() = h * v.array() / vbuffer2.array();
                return;
            }
        }
    }

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

    size_t i = 0;
    while (std::abs(fh) > tol && i < max_iters) {
        // Newton update 
        h -= fh / dfh;
        iters.push_back(h);
        newton_update();
        ++i;
    }
    
    // numerical stability
    h = std::max(h, zero);

    // final solution
    x.array() = h * v.array() / vbuffer2.array();
}


} // namespace glstudy