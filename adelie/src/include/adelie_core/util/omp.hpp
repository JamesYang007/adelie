#pragma once
#include <adelie_core/util/types.hpp>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace adelie_core {
namespace util {

inline int omp_get_thread_num()
{
#if defined(_OPENMP)
    return ::omp_get_thread_num();
#else
    return 0;
#endif
}

inline bool omp_in_parallel() 
{
#if defined(_OPENMP)
    return ::omp_in_parallel();
#else
    return false;
#endif
}

template <
    util::omp_schedule_type schedule_type=util::omp_schedule_type::_static, 
    class F
>
inline void omp_parallel_for(
    F f,
    Eigen::Index begin,
    Eigen::Index end,
    size_t n_threads
)
{
    if (n_threads <= 1 || omp_in_parallel()) {
        for (Eigen::Index i = begin; i < end; ++i) f(i);
    } else {
        if constexpr (schedule_type == util::omp_schedule_type::_static) {
            #pragma omp parallel for schedule(static) num_threads(n_threads)
            for (Eigen::Index i = begin; i < end; ++i) f(i);
        } else if constexpr (schedule_type == util::omp_schedule_type::_dynamic) {
            #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
            for (Eigen::Index i = begin; i < end; ++i) f(i);
        } else if constexpr (schedule_type == util::omp_schedule_type::_guided) {
            #pragma omp parallel for schedule(guided) num_threads(n_threads)
            for (Eigen::Index i = begin; i < end; ++i) f(i);
        } else if constexpr (schedule_type == util::omp_schedule_type::_runtime) {
            #pragma omp parallel for schedule(runtime) num_threads(n_threads)
            for (Eigen::Index i = begin; i < end; ++i) f(i);
        } else {
            // dummy check since we cannot put "false" (early compiler error)
            static_assert(schedule_type == util::omp_schedule_type::_static, "Unrecognized schedule type.");
        }
    }
}

} // namespace util
} // namespace adelie_core