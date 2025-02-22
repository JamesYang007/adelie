#pragma once
#include <adelie_core/bcd/utils.hpp>
#include <adelie_core/optimization/bisect.hpp>
#include <adelie_core/optimization/newton.hpp>
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace bcd {
namespace unconstrained {

/**
 * @brief Base Newton solver for the block update norm objective.
 * IMPORTANT: This solver assumes that v is of the form sqrt(L) * u.
 * 
 * @tparam LType        vector type.
 * @tparam VType        vector type.
 * @tparam ValueType    float type.
 * @tparam XType        vector type.
 * @tparam BufferType   vector type.
 * @tparam InitialType  see newton_root_find.
 * @param L             vector with non-negative entries.
 * @param v             any vector same length as L.
 * @param l1            l1 regularization.
 * @param l2            l2 regularization.
 * @param tol           see newton_root_find.
 * @param max_iters     see newton_root_find.
 * @param x             solution vector.
 *                      Also used as buffer to store (v / buffer2).square() 
 *                      in each Newton step.
 * @param iters         number of iterations taken. Set to 0 before anything.
 * @param buffer1       buffer to store L + l2.
 * @param buffer2       buffer to store buffer1 * h + l1.
 * @param initial_f     see newton_root_find.
 */
template <
    class LType, 
    class VType, 
    class ValueType, 
    class XType, 
    class BufferType, 
    class InitialType
>
inline
void newton_solver_base(
    const LType& L,
    const VType& v,
    ValueType l1,
    ValueType l2,
    ValueType tol,
    size_t max_iters,
    InitialType initial_f,
    XType& x,
    size_t& iters,
    BufferType& buffer1,
    BufferType& buffer2
)
{
    using value_t = ValueType;
    iters = 0;

    // Easy case: ||v||_2 <= l1 -> x = 0
    const auto v_l2 = v.matrix().norm();
    if (v_l2 <= l1) {
        x.setZero();
        return;
    }
    
    // Difficult case: ||v||_2 > l1
    
    // Still easy if l1 == 0.0
    // IMPORTANT: user's responsibility that L + l2 does not have any zeros.
    if (l1 <= 0.0) {
        x = v / (L + l2);
        return;
    }
    
    // First solve for h := ||x||_2
    auto vbuffer1 = buffer1.head(L.size());
    auto vbuffer2 = buffer2.head(L.size());

    vbuffer1 = (L + l2);

    const auto step_f = [&](auto h) {
        vbuffer2 = 1 / (vbuffer1 * h + l1);
        x = (v * vbuffer2).square();
        const auto t = x.sum();
        const auto sqrt_t = std::sqrt(t);
        const auto fh = t - 1.0;
        const auto dfh = -(
            x * vbuffer1 * vbuffer2
        ).sum() * (1 + sqrt_t) / t;
        return std::make_pair(fh, dfh);
    };

    const auto project_f = [&](auto h) {
        return std::max<value_t>(h, 0.0);
    };

    const auto root_find_state = optimization::newton_root_find(
        initial_f,
        step_f,
        project_f,
        tol, 
        max_iters
    );

    const auto h = std::get<0>(root_find_state);

    x = h * v * vbuffer2;
    iters = std::get<3>(root_find_state);
}

/*
 * Vanilla Newton solver.
 */
template <
    class LType, 
    class VType, 
    class ValueType, 
    class XType, 
    class BufferType
>
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
    const auto initial_f = [](){ return std::make_pair(0.0, 0); };
    newton_solver_base(
        L, v, l1, l2, tol, max_iters, initial_f,
        x, iters, buffer1, buffer2
    );
}

/*
 * Newton + Brent solver.
 */
template <
    class LType, 
    class VType, 
    class ValueType, 
    class XType, 
    class BufferType
>
inline
void newton_brent_solver(
    const LType& L,
    const VType& v,
    ValueType l1,
    ValueType l2,
    ValueType tol,
    ValueType brent_tol,
    size_t max_iters,
    XType& x,
    size_t& iters,
    BufferType& buffer1,
    BufferType& buffer2
)
{
    using value_t = ValueType;
    auto vbuffer1 = buffer1.head(L.size());

    const auto initial_f = [&]() {
        const value_t h_min = root_lower_bound(vbuffer1, v, l1);
        const auto h_max_out = root_upper_bound(vbuffer1, v, l1, 0.0); // IMPORTANT: NEEDS GUARANTEE
        const value_t h_max = std::get<0>(h_max_out);

        value_t h = 0;
        size_t iters_brent;
        optimization::brent(
            [&](auto x) { return root_function(x, vbuffer1, v, l1); },
            tol,
            brent_tol, 
            max_iters,
            h_min,
            h_min,
            h_max,
            [](auto, auto, auto, auto) { 
                return std::make_tuple(false, 0.0);
            },
            h,
            iters_brent
        );
        return std::make_pair(h, 0); 
    };

    newton_solver_base(
        L, v, l1, l2, tol, max_iters, initial_f,
        x, iters, buffer1, buffer2
    );
}

/*
 * Newton-ABS solver
 */
template <
    class LType, 
    class VType, 
    class ValueType, 
    class XType, 
    class BufferType
>
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
    auto vbuffer1 = buffer1.head(L.size());

    const auto initial_f = [&]() {
        const value_t h_min = root_lower_bound(vbuffer1, v, l1);
        const auto h_max_out = root_upper_bound(vbuffer1, v, l1);
        const value_t h_max = std::get<0>(h_max_out);
        const value_t vbuffer1_min_nnz = std::get<1>(h_max_out);

        value_t h;

        // If range is very small, just set h = h_min and Newton.
        // NOTE: this accounts for when h_max <= h_min as well on purpose!
        // The numerically stable way of computing h_max may lead to h_max <= h_min.
        // This is an indication that the correct range was small to begin with.
        if (h_max - h_min <= 1e-1) {
            h = h_min;
        } else {
            // NEW METHOD: adaptive bisection
            value_t h_cand = h_max;
            value_t w;  // prior
            value_t fh; // current function value

            const auto ada_bisect = [&]() {
                // enforce some movement towards h_min for safety.
                w = std::max<value_t>(l1 / (vbuffer1_min_nnz * h_cand + l1), 0.05);
                h_cand = w * h_min + (1-w) * h_cand;
                fh = root_function(h_cand, vbuffer1, v, l1);
            };
            
            ada_bisect();
            while ((fh < 0) && (std::abs(fh) > tol)) { 
                ada_bisect();
            }
            
            // save candidate solution
            h = h_cand;
        }

        return std::make_pair(h, 0);
    };

    newton_solver_base(
        L, v, l1, l2, tol, max_iters, initial_f,
        x, iters, buffer1, buffer2
    );
}

/**
 * Same as newton_solver above but returns way more information.
 *      
 * @param   init        initial point if used.
 * @param   x           matrix of iterate vectors (column-wise).
 * @param   buffer1     any vector with L.size() <= buffer1.size().
 * @param   buffer2     any vector with L.size() <= buffer2.size().
 */
template <
    class LType, 
    class VType, 
    class ValueType, 
    class XType, 
    class ItersType, 
    class BufferType
>
inline
void newton_abs_debug_solver(
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
    const auto v_l2 = v.matrix().norm();
    if (v_l2 <= l1) {
        x.setZero();
        return;
    }
    
    // Difficult case: ||v||_2 > l1
    if (l1 <= 0.0) {
        x = v / (L + l2);
        return;
    }
    
    // First solve for h := ||x||_2
    auto vbuffer1 = buffer1.head(L.size());
    auto vbuffer2 = buffer2.head(L.size());

    // pre-compute L + l2
    vbuffer1 = (L + l2);
    
    // Find good initialization
    // The following helps tremendously if the resulting h > 0.
    if (smart_init) {
        // compute h_min
        {
            const value_t b = l1 * vbuffer1.sum();
            const value_t a = vbuffer1.matrix().squaredNorm();
            const value_t v_l1 = v.matrix().template lpNorm<1>();
            const value_t c = l1 * l1 * L.size() - v_l1 * v_l1;
            const value_t discr = b*b - a*c;
            h_min = (discr > -1e-12) ? 
                (-b + std::sqrt(std::max<value_t>(discr, zero))) / a : 0.0;
            
            // Otherwise, if h <= 0, we know at least 0 is a reasonable solution.
            // The only case h <= 0 is when 0 is already close to the solution.
            h_min = std::max<value_t>(h_min, zero);
        }
        
        // compute h_max
        const value_t vbuffer1_min = vbuffer1.minCoeff();
        value_t vbuffer1_min_nnz = vbuffer1_min;

        // If L+l2 have entries <= threshold,
        // find h_max with more numerically-stable routine.
        // There is NO guarantee that f(h_max) <= 0, 
        // but we will use this to bisect and find an h where f(h) >= 0,
        // so we don't necessarily need h_max to be f(h_max) <= 0.
        if (vbuffer1_min <= 1e-10) {
            vbuffer1_min_nnz = std::numeric_limits<value_t>::infinity();
            h_max = 0;
            for (int i = 0; i < vbuffer1.size(); ++i) {
                const bool is_nonzero = vbuffer1[i] > 1e-10;
                const auto vi2 = v[i] * v[i];
                h_max += is_nonzero ? vi2 / (vbuffer1[i] * vbuffer1[i]) : 0;
                vbuffer1_min_nnz = is_nonzero ? std::min<value_t>(vbuffer1_min_nnz, vbuffer1[i]) : vbuffer1_min_nnz;
            }
            h_max = std::sqrt(h_max);
        } else {
            h_max = (v / vbuffer1).matrix().norm();
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

            value_t w = std::max<value_t>(l1 / (vbuffer1_min_nnz * h_max + l1), 0.05);
            h = w * h_min + (1-w) * h_max;
            value_t fh = (v / (vbuffer1 * h + l1)).matrix().squaredNorm() - 1;
            
            smart_iters.push_back(h);

            while ((fh < 0) && std::abs(fh) > tol) {
                w = std::max<value_t>(l1 / (vbuffer1_min_nnz * h + l1), 0.05);
                h = w * h_min + (1-w) * h;
                fh = (v / (vbuffer1 * h + l1)).matrix().squaredNorm() - 1;
                smart_iters.push_back(h);
            }

            if (std::abs(fh) <= tol) {
                h = std::max<value_t>(h, zero);
                x = h * v / vbuffer2;
                return;
            }
        }
    }

    // Newton method
    value_t fh;
    value_t dfh;

    const auto newton_update = [&]() {
        vbuffer2 = vbuffer1 * h + l1;
        x = (v / vbuffer2).square();
        const auto t = x.sum();
        const auto sqrt_t = std::sqrt(t);
        fh = t - 1;
        dfh = -(x * (vbuffer1 / vbuffer2)).sum() * (1 + sqrt_t) / t;
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
    h = std::max<value_t>(h, zero);

    // final solution
    x = h * v / vbuffer2;
}

} // namespace unconstrained
} // namespace bcd
} // namespace adelie_core