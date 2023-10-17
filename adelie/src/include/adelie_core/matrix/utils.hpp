#pragma once
#include <cstddef>
#include <adelie_core/util/macros.hpp>

namespace adelie_core {
namespace matrix {

template <class X1Type, class X2Type>
ADELIE_CORE_STRONG_INLINE
void dvsubi(
    X1Type& x1,
    const X2Type& x2,
    size_t n_threads
)
{
    assert(n_threads > 0);
    const size_t n = x1.size();
    const int n_blocks = std::min(n_threads, n);
    const int block_size = n / n_blocks;
    const int remainder = n % n_blocks;

    #pragma omp parallel for schedule(static) num_threads(n_blocks)
    for (int t = 0; t < n_blocks; ++t) {
        const auto begin = (
            std::min<int>(t, remainder) * (block_size + 1) 
            + std::max<int>(t-remainder, 0) * block_size
        );
        const auto size = block_size + (t < remainder);
        x1.segment(begin, size) -= x2.segment(begin, size);
    }
}

template <class X1Type, class X2Type>
ADELIE_CORE_STRONG_INLINE
void dmvsubi(
    X1Type& x1,
    const X2Type& x2,
    size_t n_threads
)
{
    assert(n_threads > 0);
    const size_t n = x1.rows();
    const int n_blocks = std::min(n_threads, n);
    const int block_size = n / n_blocks;
    const int remainder = n % n_blocks;

    #pragma omp parallel for schedule(static) num_threads(n_blocks)
    for (int t = 0; t < n_blocks; ++t) {
        const auto begin = (
            std::min<int>(t, remainder) * (block_size + 1) 
            + std::max<int>(t-remainder, 0) * block_size
        );
        const auto size = block_size + (t < remainder);
        x1.middleRows(begin, size).rowwise() -= x2;
    }
}

template <class X1Type, class X2Type>
ADELIE_CORE_STRONG_INLINE
void dmmeq(
    X1Type& x1,
    const X2Type& x2,
    size_t n_threads
)
{
    assert(n_threads > 0);
    const size_t n = x1.rows();
    const int n_blocks = std::min(n_threads, n);
    const int block_size = n / n_blocks;
    const int remainder = n % n_blocks;

    #pragma omp parallel for schedule(static) num_threads(n_blocks)
    for (int t = 0; t < n_blocks; ++t) {
        const auto begin = (
            std::min<int>(t, remainder) * (block_size + 1) 
            + std::max<int>(t-remainder, 0) * block_size
        );
        const auto size = block_size + (t < remainder);
        x1.middleRows(begin, size) = x2.middleRows(begin, size);
    }
}

template <class X1Type, class X2Type>
ADELIE_CORE_STRONG_INLINE
auto ddot(
    const X1Type& x1, 
    const X2Type& x2, 
    size_t n_threads
)
{
    using value_t = typename std::decay_t<X1Type>::Scalar;

    assert(n_threads > 0);
    const size_t n = x1.size();
    const int n_blocks = std::min(n_threads, n);
    const int block_size = n / n_blocks;
    const int remainder = n % n_blocks;

    value_t out = 0;
    #pragma omp parallel for schedule(static) num_threads(n_blocks) reduction(+:out)
    for (int t = 0; t < n_blocks; ++t)
    {
        const auto begin = (
            std::min<int>(t, remainder) * (block_size + 1) 
            + std::max<int>(t-remainder, 0) * block_size
        );
        const auto size = block_size + (t < remainder);
        out += x1.segment(begin, size).dot(x2.segment(begin, size));
    }
    return out;
}

template <class ValueType, class XType, class OutType>
ADELIE_CORE_STRONG_INLINE
void dax(
    ValueType a,
    const XType& x, 
    size_t n_threads,
    OutType& out
)
{
    assert(n_threads > 0);
    const size_t n = x.size();
    const int n_blocks = std::min(n_threads, n);
    const int block_size = n / n_blocks;
    const int remainder = n % n_blocks;
    #pragma omp parallel for schedule(static) num_threads(n_blocks)
    for (int t = 0; t < n_blocks; ++t)
    {
        const auto begin = (
            std::min<int>(t, remainder) * (block_size + 1) 
            + std::max<int>(t-remainder, 0) * block_size
        );
        const auto size = block_size + (t < remainder);
        out.segment(begin, size) = a * x.segment(begin, size);
    }
}

template <class MType, class VType, class BuffType, class OutType>
ADELIE_CORE_STRONG_INLINE
void dgemv(
    const MType& m,
    const VType& v,
    size_t n_threads,
    BuffType& buff,
    OutType& out
)
{
    assert(n_threads > 0);
    const size_t n = m.rows();
    const size_t p = m.cols();
    const size_t max_np = std::max(n, p);
    const int n_blocks = std::min(n_threads, max_np);
    const int block_size = max_np / n_blocks;
    const int remainder = max_np % n_blocks;

    if (n <= p) {
        #pragma omp parallel for schedule(static) num_threads(n_blocks)
        for (int t = 0; t < n_blocks; ++t)
        {
            const auto begin = (
                std::min<int>(t, remainder) * (block_size + 1) 
                + std::max<int>(t-remainder, 0) * block_size
            );
            const auto size = block_size + (t < remainder);
            out.segment(begin, size).noalias() = v * m.middleCols(begin, size);
        }
    } else {
        assert(buff.rows() >= n_blocks);
        assert(buff.cols() >= p);
        #pragma omp parallel for schedule(static) num_threads(n_blocks)
        for (int t = 0; t < n_blocks; ++t)
        {
            const auto begin = (
                std::min<int>(t, remainder) * (block_size + 1) 
                + std::max<int>(t-remainder, 0) * block_size
            );
            const auto size = block_size + (t < remainder);
            buff.row(t).head(p).noalias() = (
                v.segment(begin, size) * m.middleRows(begin, size)
            );
        }
        out = buff.block(0, 0, n_blocks, p).colwise().sum();
    }
}

} // namespace matrix
} // namespace adelie_core