#pragma once 
#include <cstddef>

namespace adelie_core {
    
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
    
} // namespace adelie_core