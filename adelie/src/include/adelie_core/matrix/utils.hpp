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
    const int n_blocks = std::min(n_threads, n);
    const int block_size = n / n_blocks;
    if (n_threads <= 1 || block_size <= Configs::min_flops_per_thread) { 
        x1 += x2; 
        return; 
    }
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
    assert(n_threads > 0);
    if (n_threads <= 1) { x1 -= x2; return; }
    const size_t n = x1.size();
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
    assert(n_threads > 0);
    if (n_threads <= 1) { x1 = x2; return; }
    const size_t n = x1.size();
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

template <class X1Type, class X2Type>
ADELIE_CORE_STRONG_INLINE
void dmmeq(
    X1Type& x1,
    const X2Type& x2,
    size_t n_threads
)
{
    assert(n_threads > 0);
    if (n_threads <= 1) { x1 = x2; return; }
    const size_t n = x1.rows();
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

template <class X1Type, class X2Type, class BuffType>
ADELIE_CORE_STRONG_INLINE
typename std::decay_t<X1Type>::Scalar ddot(
    const X1Type& x1, 
    const X2Type& x2, 
    size_t n_threads,
    BuffType& buff
)
{
    assert(n_threads > 0);
    if (n_threads <= 1) { return x1.dot(x2); }
    const size_t n = x1.size();
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
    if (n_threads <= 1) { out = a * x; return; }
    const size_t n = x.size();
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
        out.segment(begin, size) = a * x.segment(begin, size);
    }
}

template <class OutType>
ADELIE_CORE_STRONG_INLINE
void dvzero(
    OutType& out,
    size_t n_threads
)
{
    assert(n_threads > 0);
    if (n_threads <= 1) { out.setZero(); return; }
    const size_t n = out.size();
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
    assert(n_threads > 0);
    if (n_threads <= 1) { 
        if constexpr (op == util::operator_type::_eq) {
            out.noalias() = v * m; 
        } else if constexpr (op == util::operator_type::_add) {
            out += v * m; 
        } else {
            static_assert("Bad operator type!");
        }
        return; 
    }
    const size_t n = m.rows();
    const size_t p = m.cols();
    const size_t max_np = std::max(n, p);
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
                out.segment(begin, size).noalias() = v * m.middleCols(begin, size);
            } else if constexpr (op == util::operator_type::_add) {
                out.segment(begin, size).noalias() += v * m.middleCols(begin, size);
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
            buff.row(t).head(p).noalias() = (
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

template <class InnerType, class ValueType, class DenseType>
ADELIE_CORE_STRONG_INLINE
auto spddot(
    const InnerType& inner, 
    const ValueType& value,
    const DenseType& x
)
{
    using value_t = typename std::decay_t<DenseType>::Scalar;

    value_t sum = 0;
    for (int i = 0; i < inner.size(); ++i) {
        sum += x[inner[i]] * value[i];
    }
    return sum;
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