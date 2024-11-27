#pragma once
#include <adelie_core/util/exceptions.hpp>
#include <adelie_core/util/macros.hpp>
#include <adelie_core/util/omp.hpp>
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace io {

template <class MType, class OutType>
ADELIE_CORE_STRONG_INLINE
void compute_column_mean(
    const MType& m,
    OutType& out,
    size_t n_threads
)
{
    const auto n = m.rows();
    const int p = m.cols();

    const auto routine = [&](auto j) {
        uint64_t sum = 0;
        uint64_t n_miss = 0;
        for (int k = 0; k < n; ++k) {
            if (m(k,j) > 0) sum += m(k,j);
            else if (m(k,j) < 0) ++n_miss;
        }
        out[j] = static_cast<double>(sum) / std::max<uint64_t>(n - n_miss, 1);
    };
    util::omp_parallel_for(routine, 0, p, n_threads);
}

ADELIE_CORE_STRONG_INLINE
void compute_nnm(
    const Eigen::Ref<const util::colarr_type<int8_t>>& m,
    Eigen::Ref<util::rowvec_type<uint64_t>> out,
    size_t n_threads
)
{
    const auto n = m.rows();
    const int p = m.cols();
    const auto routine = [&](auto j) {
        uint64_t n_miss = 0;
        for (int k = 0; k < n; ++k) {
            if (m(k,j) < 0) ++n_miss;
        }
        out[j] = n - n_miss;
    };
    util::omp_parallel_for(routine, 0, p, n_threads);
}

template <class MType>
ADELIE_CORE_STRONG_INLINE
void compute_nnz(
    const MType& m,
    Eigen::Ref<util::rowvec_type<uint64_t>> out,
    size_t n_threads
)
{
    const auto n = m.rows();
    const int p = m.cols();
    const auto routine = [&](auto j) {
        uint64_t nnz = 0;
        for (int k = 0; k < n; ++k) if (m(k,j) != 0) ++nnz;
        out[j] = nnz;
    };
    util::omp_parallel_for(routine, 0, p, n_threads);
}

ADELIE_CORE_STRONG_INLINE
void compute_impute(
    const Eigen::Ref<const util::colarr_type<int8_t>>& m,
    util::impute_method_type impute_method,
    Eigen::Ref<util::rowvec_type<double>> impute,
    size_t n_threads
)
{
    const auto p = m.cols();
    if (impute.size() != p) {
        throw util::adelie_core_error(
            "impute must have length equal to the number of columns of the matrix."
        );
    } 

    switch (impute_method) {
        case util::impute_method_type::_user: break;
        case util::impute_method_type::_mean: {
            compute_column_mean(m, impute, n_threads);
            break;
        }
        default: {
            throw util::adelie_core_error(
                "Unrecognized impute_method!"
            );
        }
    };
}

} // namespace io
} // namespace adelie_core