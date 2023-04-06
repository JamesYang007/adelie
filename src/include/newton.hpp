#pragma once 
#include <cstddef>
#include <bisect.hpp>
#include <objective.hpp>

namespace glstudy {
    
/**
 * @brief General Newton root-finder for one-dimensional functions.
 * 
 * @tparam InitialType  Functor type.
 * @tparam StepType     Functor type.
 * @tparam ProjectType  Functor type.
 * @tparam ValueType    float type.
 * @param initial_f     function with no arguments that returns the initial point.
 *                      Guaranteed to be called first.
 * @param step_f        function with one argument (current value) that returns
 *                      the function and derivative values.
 *                      Guaranteed to be called after initial_f and every iteration
 *                      after project_f is called so that the current value is feasible.
 * @param project_f     function with one argument that projects the current value
 *                      to the constrained set.
 *                      Guaranteed to be called after the current value takes a step
 *                      based on the current step_f call.
 * @param tol           tolerance for convergence. Used to check if function value is close to 0.
 * @param max_iters     max number of iterations
 * @return (x, i, e) where x is the solution, i is the number of iterations, and e is the error.
 */
template <class InitialType, class StepType, 
          class ProjectType, class ValueType>
inline
auto newton_root_find(
    InitialType initial_f,
    StepType step_f,
    ProjectType project_f,
    ValueType tol,
    size_t max_iters
)
{
    using value_t = ValueType;

    const auto initial_pack = initial_f();
    value_t h = std::get<0>(initial_pack);    // solution candidate
    size_t iters = std::get<1>(initial_pack); // number of iterations
    value_t fh; // function value at h
    value_t dfh; // derivative at h

    const auto step_pack = step_f(h);
    fh = std::get<0>(step_pack);
    dfh = std::get<1>(step_pack);

    while ((std::abs(fh) > tol) && (iters < max_iters)) {
        h -= fh / dfh;
        h = project_f(h);
        const auto step_pack = step_f(h);
        fh = std::get<0>(step_pack);
        dfh = std::get<1>(step_pack);
        ++iters;
    }

    return std::make_tuple(h, fh, dfh, iters); 
}
    
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
template <class LType, class VType, class ValueType, 
          class XType, class BufferType, class InitialType>
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
    iters = 0;

    // Easy case: ||v||_2 <= l1 -> x = 0
    const auto v_l2 = v.norm();
    if (v_l2 <= l1) {
        x.setZero();
        return;
    }
    
    // Difficult case: ||v||_2 > l1
    
    // Still easy if l1 == 0.0
    // IMPORTANT: user's responsibility that L + l2 does not have any zeros.
    if (l1 <= 0.0) {
        x.array() = v.array() / (L.array() + l2);
        return;
    }
    
    // First solve for h := ||x||_2
    auto vbuffer1 = buffer1.head(L.size());
    auto vbuffer2 = buffer2.head(L.size());

    vbuffer1.array() = (L.array() + l2);

    const auto step_f = [&](auto h) {
        vbuffer2.array() = vbuffer1.array() * h + l1;
        x.array() = (v.array() / vbuffer2.array()).square();
        auto fh = x.sum() - 1.0;
        auto dfh = -2.0 * (
            x.array() * (vbuffer1.array() / vbuffer2.array())
        ).sum();
        return std::make_pair(fh, dfh);
    };

    const auto project_f = [&](auto h) {
        return std::max(h, 0.0);
    };

    const auto root_find_pack = newton_root_find(
        initial_f,
        step_f,
        project_f,
        tol, 
        max_iters
    );

    const auto h = std::get<0>(root_find_pack);

    x.array() = h * v.array() / vbuffer2.array();
    iters = std::get<3>(root_find_pack);
}

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
    const auto initial_f = [](){ return std::make_pair(0.0, 0); };
    newton_solver_base(
        L, v, l1, l2, tol, max_iters, initial_f,
        x, iters, buffer1, buffer2
    );
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
        const value_t h_min = compute_h_min(vbuffer1, v, l1);
        const auto h_max_out = compute_h_max(vbuffer1, v, 0.0); // IMPORTANT: NEEDS GUARANTEE
        const value_t h_max = std::get<0>(h_max_out);

        value_t h;
        size_t iters_brent;
        brent(
            [&](auto x) { return block_norm_objective(x, vbuffer1, v, l1); },
            tol,
            brent_tol, 
            max_iters,
            h_min,
            h_min,
            h_max,
            [](auto a, auto fa, auto b, auto fb) { 
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
    auto vbuffer1 = buffer1.head(L.size());

    const auto initial_f = [&]() {
        const value_t h_min = compute_h_min(vbuffer1, v, l1);
        const auto h_max_out = compute_h_max(vbuffer1, v, l1);
        const value_t h_max = std::get<0>(h_max_out);
        const value_t vbuffer1_min_nzn = std::get<1>(h_max_out);

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
                w = std::max(l1 / (vbuffer1_min_nzn * h_cand + l1), 0.05);
                h_cand = w * h_min + (1-w) * h_cand;
                fh = block_norm_objective(h_cand, vbuffer1, v, l1);
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
            for (int i = 0; i < vbuffer1.size(); ++i) {
                const bool is_nonzero = vbuffer1[i] > 1e-10;
                const auto vi2 = v[i] * v[i];
                h_max += is_nonzero ? vi2 / (vbuffer1[i] * vbuffer1[i]) : 0;
                vbuffer1_min_nzn = is_nonzero ? std::min(vbuffer1_min_nzn, vbuffer1[i]) : vbuffer1_min_nzn;
            }
            h_max = std::sqrt(h_max);
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