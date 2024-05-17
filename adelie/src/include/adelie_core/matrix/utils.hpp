#pragma once
#include <cstddef>
#include <adelie_core/util/macros.hpp>
#include <adelie_core/configs.hpp>

namespace adelie_core {
namespace matrix {

template <class X1Type, class X2Type>
ADELIE_CORE_STRONG_INLINE
void dvaddi(
    X1Type& x1,
    const X2Type& x2,
    size_t n_threads
)
{
    const size_t n = x1.size();
    if (n_threads <= 1 || 2 * n <= Configs::min_flops) { 
        x1 += x2; 
        return; 
    }
    const int n_blocks = std::min(n_threads, n);
    const int block_size = n / n_blocks;
    const int remainder = n % n_blocks;

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int t = 0; t < n_blocks; ++t) {
        const auto begin = (
            std::min<int>(t, remainder) * (block_size + 1) 
            + std::max<int>(t-remainder, 0) * block_size
        );
        const auto size = block_size + (t < remainder);
        x1.segment(begin, size) += x2.segment(begin, size);
    }
}

template <class X1Type, class X2Type>
ADELIE_CORE_STRONG_INLINE
void dvsubi(
    X1Type& x1,
    const X2Type& x2,
    size_t n_threads
)
{
    const size_t n = x1.size();
    if (n_threads <= 1 || 2 * n <= Configs::min_flops) { 
        x1 -= x2;
        return; 
    }
    const int n_blocks = std::min(n_threads, n);
    const int block_size = n / n_blocks;
    const int remainder = n % n_blocks;

    #pragma omp parallel for schedule(static) num_threads(n_threads)
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
void dvveq(
    X1Type& x1,
    const X2Type& x2,
    size_t n_threads
)
{
    const size_t n = x1.size();
    if (n_threads <= 1 || n <= Configs::min_flops) { 
        x1 = x2;
        return; 
    }
    const int n_blocks = std::min(n_threads, n);
    const int block_size = n / n_blocks;
    const int remainder = n % n_blocks;

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int t = 0; t < n_blocks; ++t) {
        const auto begin = (
            std::min<int>(t, remainder) * (block_size + 1) 
            + std::max<int>(t-remainder, 0) * block_size
        );
        const auto size = block_size + (t < remainder);
        x1.segment(begin, size) = x2.segment(begin, size);
    }
}

template <class OutType>
ADELIE_CORE_STRONG_INLINE
void dvzero(
    OutType& out,
    size_t n_threads
)
{
    const size_t n = out.size();
    if (n_threads <= 1 || n <= 2 * Configs::min_flops) { 
        out.setZero(); 
        return; 
    }
    const int n_blocks = std::min(n_threads, n);
    const int block_size = n / n_blocks;
    const int remainder = n % n_blocks;

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int t = 0; t < n_blocks; ++t) 
    {
        const auto begin = (
            std::min<int>(t, remainder) * (block_size + 1) 
            + std::max<int>(t-remainder, 0) * block_size
        );
        const auto size = block_size + (t < remainder);
        out.segment(begin, size).setZero();
    }
}

template <class X1Type, class X2Type, class BuffType>
ADELIE_CORE_STRONG_INLINE
typename std::decay_t<X1Type>::Scalar ddot(
    const X1Type& x1, 
    const X2Type& x2, 
    size_t n_threads,
    BuffType& buff
)
{
    const size_t n = x1.size();
    if (n_threads <= 1 || 2 * n <= Configs::min_flops) { 
        return x1.dot(x2); 
    }
    const int n_blocks = std::min(n_threads, n);
    const int block_size = n / n_blocks;
    const int remainder = n % n_blocks;

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int t = 0; t < n_blocks; ++t)
    {
        const auto begin = (
            std::min<int>(t, remainder) * (block_size + 1) 
            + std::max<int>(t-remainder, 0) * block_size
        );
        const auto size = block_size + (t < remainder);
        buff[t] = x1.segment(begin, size).dot(x2.segment(begin, size));
    }
    return buff.head(n_blocks).sum();
}

template <class X1Type, class X2Type>
ADELIE_CORE_STRONG_INLINE
void dmmeq(
    X1Type& x1,
    const X2Type& x2,
    size_t n_threads
)
{
    const size_t n = x1.rows();
    // NOTE: multiplier of 4 from experimentation
    if (n_threads <= 1 || 4 * n * x1.cols() <= Configs::min_flops) { 
        x1 = x2; 
        return; 
    }
    const int n_blocks = std::min(n_threads, n);
    const int block_size = n / n_blocks;
    const int remainder = n % n_blocks;

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int t = 0; t < n_blocks; ++t) {
        const auto begin = (
            std::min<int>(t, remainder) * (block_size + 1) 
            + std::max<int>(t-remainder, 0) * block_size
        );
        const auto size = block_size + (t < remainder);
        x1.middleRows(begin, size) = x2.middleRows(begin, size);
    }
}

template <util::operator_type op=util::operator_type::_eq, 
          class MType, class VType, class BuffType, class OutType>
ADELIE_CORE_STRONG_INLINE
void dgemv(
    const MType& m,
    const VType& v,
    size_t n_threads,
    BuffType& buff,
    OutType& out
)
{
    const size_t n = m.rows();
    const size_t p = m.cols();
    const size_t max_np = std::max(n, p);
    if (n_threads <= 1 || n * p <= Configs::min_flops) { 
        if constexpr (op == util::operator_type::_eq) {
            out = v * m; 
        } else if constexpr (op == util::operator_type::_add) {
            out += v * m; 
        } else {
            static_assert("Bad operator type!");
        }
        return; 
    }
    const int n_blocks = std::min(n_threads, max_np);
    const int block_size = max_np / n_blocks;
    const int remainder = max_np % n_blocks;

    if (n <= p) {
        #pragma omp parallel for schedule(static) num_threads(n_threads)
        for (int t = 0; t < n_blocks; ++t)
        {
            const auto begin = (
                std::min<int>(t, remainder) * (block_size + 1) 
                + std::max<int>(t-remainder, 0) * block_size
            );
            const auto size = block_size + (t < remainder);
            if constexpr (op == util::operator_type::_eq) {
                out.segment(begin, size) = v * m.middleCols(begin, size);
            } else if constexpr (op == util::operator_type::_add) {
                out.segment(begin, size) += v * m.middleCols(begin, size);
            } else {
                static_assert("Bad operator type!");
            }
        }
    } else {
        assert(buff.rows() >= n_blocks);
        assert(buff.cols() >= p);
        #pragma omp parallel for schedule(static) num_threads(n_threads)
        for (int t = 0; t < n_blocks; ++t)
        {
            const auto begin = (
                std::min<int>(t, remainder) * (block_size + 1) 
                + std::max<int>(t-remainder, 0) * block_size
            );
            const auto size = block_size + (t < remainder);
            buff.row(t).head(p) = (
                v.segment(begin, size) * m.middleRows(begin, size)
            );
        }
        if constexpr (op == util::operator_type::_eq) {
            out = buff.block(0, 0, n_blocks, p).colwise().sum();
        } else if constexpr (op == util::operator_type::_add) {
            out += buff.block(0, 0, n_blocks, p).colwise().sum();
        } else {
            static_assert("Bad operator type!");
        }
    }
}

template <class InnerType, class ValueType, class WeightsType>
ADELIE_CORE_STRONG_INLINE
auto svsvwdot(
    const InnerType& inner_1,
    const ValueType& value_1,
    const InnerType& inner_2,
    const ValueType& value_2,
    const WeightsType& weights
)
{
    using value_t = typename std::decay_t<WeightsType>::Scalar;

    int i1 = 0;
    int i2 = 0;
    value_t sum = 0;
    while (
        (i1 < inner_1.size()) &&
        (i2 < inner_2.size())
    ) {
        while ((i1 < inner_1.size()) && (inner_1[i1] < inner_2[i2])) ++i1;
        if (i1 == inner_1.size()) break;
        while ((i2 < inner_2.size()) && (inner_2[i2] < inner_1[i1])) ++i2;
        if (i2 == inner_2.size()) break;
        while (
            (i1 < inner_1.size()) &&
            (i2 < inner_2.size()) &&
            (inner_1[i1] == inner_2[i2])
        ) {
            sum += value_1[i1] * value_2[i2] * weights[inner_1[i1]];
            ++i1;
            ++i2;
        }
    }
    return sum;
}

template <class Inner1Type, class Value1Type,
          class Inner2Type, class Value2Type>
ADELIE_CORE_STRONG_INLINE
auto svsvdot(
    const Inner1Type& inner_1,
    const Value1Type& value_1,
    const Inner2Type& inner_2,
    const Value2Type& value_2
)
{
    using value_t = typename std::decay_t<Value1Type>::Scalar;

    int i1 = 0;
    int i2 = 0;
    value_t sum = 0;
    while (
        (i1 < inner_1.size()) &&
        (i2 < inner_2.size())
    ) {
        while ((i1 < inner_1.size()) && (inner_1[i1] < inner_2[i2])) ++i1;
        if (i1 == inner_1.size()) break;
        while ((i2 < inner_2.size()) && (inner_2[i2] < inner_1[i1])) ++i2;
        if (i2 == inner_2.size()) break;
        while (
            (i1 < inner_1.size()) &&
            (i2 < inner_2.size()) &&
            (inner_1[i1] == inner_2[i2])
        ) {
            sum += value_1[i1] * value_2[i2];
            ++i1;
            ++i2;
        }
    }
    return sum;
}

template <class InnerType, class ValueType, class DenseType, class BuffType>
ADELIE_CORE_STRONG_INLINE
auto spddot(
    const InnerType& inner, 
    const ValueType& value,
    const DenseType& x,
    size_t n_threads,
    BuffType& buff
)
{
    using value_t = typename std::decay_t<DenseType>::Scalar;

    const size_t nnz = inner.size();
    // NOTE: multiplier of 8 from experimentation
    if (n_threads <= 1 || 8 * nnz <= Configs::min_flops) {
        value_t sum = 0;
        for (int i = 0; i < inner.size(); ++i) {
            sum += x[inner[i]] * value[i];
        }
        return sum;
    }
    const int n_blocks = std::min(n_threads, nnz);
    const int block_size = nnz / n_blocks;
    const int remainder = nnz % n_blocks;

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int t = 0; t < n_blocks; ++t)
    {
        const auto begin = (
            std::min<int>(t, remainder) * (block_size + 1) 
            + std::max<int>(t-remainder, 0) * block_size
        );
        const auto size = block_size + (t < remainder);
        value_t sum = 0;
        for (int i = begin; i < begin+size; ++i) {
            sum += x[inner[i]] * value[i];
        }
        buff[t] = sum;
    }
    return buff.head(n_blocks).sum();
}

template <class InnerType, class ValueType, class T, class OutType>
ADELIE_CORE_STRONG_INLINE
void spaxi(
    const InnerType& inner, 
    const ValueType& value,
    T v,
    OutType& out
)
{
    for (int i = 0; i < inner.size(); ++i) {
        out[inner[i]] += v * value[i];
    }
}

} // namespace matrix
} // namespace adelie_core