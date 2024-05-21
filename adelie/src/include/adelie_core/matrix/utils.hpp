#pragma once
#include <cstddef>
#include <adelie_core/configs.hpp>
#include <adelie_core/util/macros.hpp>
#include <adelie_core/util/types.hpp>

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
    using value_t = typename std::decay_t<X1Type>::Scalar;
    const size_t n = x1.size();
    const size_t n_bytes = (2 * sizeof(value_t)) * n;
    if (n_threads <= 1 || n_bytes <= Configs::min_bytes) { 
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
    using value_t = typename std::decay_t<X1Type>::Scalar;
    const size_t n = x1.size();
    const size_t n_bytes = (2 * sizeof(value_t)) * n;
    if (n_threads <= 1 || n_bytes <= Configs::min_bytes) { 
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
    using value_t = typename std::decay_t<X1Type>::Scalar;
    const size_t n = x1.size();
    const size_t n_bytes = (1.5 * sizeof(value_t)) * n;
    if (n_threads <= 1 || n_bytes <= Configs::min_bytes) { 
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
    using value_t = typename std::decay_t<OutType>::Scalar;
    const size_t n = out.size();
    const size_t n_bytes = sizeof(value_t) * n;
    if (n_threads <= 1 || n_bytes <= 2 * Configs::min_bytes) { 
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
    using value_t = typename std::decay_t<X1Type>::Scalar;
    const size_t n = x1.size();
    const size_t n_bytes = (2 * sizeof(value_t)) * n;
    if (n_threads <= 1 || n_bytes <= Configs::min_bytes) { 
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
    using value_t = typename std::decay_t<X1Type>::Scalar;
    const size_t n = x1.rows();
    // NOTE: multiplier from experimentation
    const size_t n_bytes = (4 * sizeof(value_t)) * n * x1.cols();
    if (n_threads <= 1 || n_bytes <= Configs::min_bytes) { 
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
    using value_t = typename std::decay_t<MType>::Scalar;
    const size_t n = m.rows();
    const size_t p = m.cols();
    const size_t max_np = std::max(n, p);
    const size_t n_bytes = sizeof(value_t) * n * (p + 1);
    if (n_threads <= 1 || n_bytes <= Configs::min_bytes) { 
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
    // NOTE: multiplier from experimentation
    const size_t n_bytes = (16 * sizeof(value_t)) * nnz;
    if (n_threads <= 1 || n_bytes <= Configs::min_bytes) {
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
    OutType& out,
    size_t n_threads
)
{
    using value_t = typename std::decay_t<ValueType>::Scalar;
    const size_t nnz = inner.size();
    // NOTE: multiplier from experimentation
    const size_t n_bytes = (16 * sizeof(value_t)) * nnz;
    if (n_threads <= 1 || n_bytes <= Configs::min_bytes) {
        for (int i = 0; i < nnz; ++i) {
            out[inner[i]] += v * value[i];
        }
        return;
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
        for (int i = begin; i < begin+size; ++i) {
            out[inner[i]] += v * value[i];
        }
    }
}

template <class UnaryType, class IOType, class VType, class BuffType>
ADELIE_CORE_STRONG_INLINE
auto snp_unphased_dot(
    const UnaryType& unary,
    const IOType& io,
    int j, 
    const VType& v,
    size_t n_threads,
    BuffType& buff
)
{
    using io_t = std::decay_t<IOType>;
    using value_t = typename std::decay_t<VType>::Scalar;

    const auto nnz = io.nnz()[j];
    const value_t imp = io.impute()[j];
    // NOTE: multiplier from experimentation
    const size_t n_bytes = (16 * sizeof(value_t)) * nnz;
    if (n_threads <= 1 || n_bytes <= Configs::min_bytes) {
        value_t sum = 0;
        for (int c = 0; c < io_t::n_categories; ++c) {
            auto it = io.begin(j, c);
            const auto end = io.end(j, c);
            const value_t val = (c == 0) ? imp : c;
            value_t curr_sum = 0;
            for (; it != end; ++it) {
                const auto idx = *it;
                curr_sum += v[idx]; 
            }
            sum += curr_sum * unary(val);
        }
        return sum;
    }

    auto vbuff = buff.head(n_threads);
    vbuff.setZero();

    #pragma omp parallel num_threads(n_threads)
    {
        for (int c = 0; c < io_t::n_categories; ++c) {
            const size_t n_chunks = io.n_chunks(j, c);
            const int n_blocks = std::min(n_threads, n_chunks);
            const int block_size = n_chunks / n_blocks;
            const int remainder = n_chunks % n_blocks;
            const value_t val = (c == 0) ? imp : c;

            #pragma omp for schedule(static) nowait
            for (int t = 0; t < n_blocks; ++t) {
                const auto begin = (
                    std::min<int>(t, remainder) * (block_size + 1) 
                    + std::max<int>(t-remainder, 0) * block_size
                );
                const auto size = block_size + (t < remainder);
                auto it = io.begin(j, c, begin);
                const auto end = io.begin(j, c, begin + size);

                value_t sum = 0;
                for (; it != end; ++it) {
                    const auto idx = *it;
                    sum += v[idx]; 
                }
                vbuff[t] += sum * unary(val);
            }
        }
    }
    return vbuff.sum();
}

template <class IOType, class ValueType, class OutType>
ADELIE_CORE_STRONG_INLINE
void snp_unphased_axi(
    const IOType& io,
    int j, 
    ValueType v,
    OutType& out,
    size_t n_threads
)
{
    using io_t = std::decay_t<IOType>;
    using value_t = ValueType;

    const auto nnz = io.nnz()[j];
    const value_t imp = io.impute()[j];
    // NOTE: multiplier from experimentation
    const size_t n_bytes = (16 * sizeof(value_t)) * nnz;
    if (n_threads <= 1 || n_bytes <= Configs::min_bytes) {
        for (int c = 0; c < io_t::n_categories; ++c) {
            auto it = io.begin(j, c);
            const auto end = io.end(j, c);
            const value_t curr_val = v * ((c == 0) ? imp : c);
            for (; it != end; ++it) {
                const auto idx = *it;
                out[idx] += curr_val; 
            }
        }
        return;
    }

    #pragma omp parallel num_threads(n_threads)
    {
        for (int c = 0; c < io_t::n_categories; ++c) {
            const size_t n_chunks = io.n_chunks(j, c);
            const int n_blocks = std::min(n_threads, n_chunks);
            const int block_size = n_chunks / n_blocks;
            const int remainder = n_chunks % n_blocks;
            const value_t curr_val = v * ((c == 0) ? imp : c);

            #pragma omp for schedule(static) nowait
            for (int t = 0; t < n_blocks; ++t) {
                const auto begin = (
                    std::min<int>(t, remainder) * (block_size + 1) 
                    + std::max<int>(t-remainder, 0) * block_size
                );
                const auto size = block_size + (t < remainder);
                auto it = io.begin(j, c, begin);
                const auto end = io.begin(j, c, begin + size);

                for (; it != end; ++it) {
                    out[*it] += curr_val; 
                }
            }
        }
    }
}

template <class IOType, class VType, class BuffType>
auto snp_phased_ancestry_dot(
    const IOType& io,
    int j, 
    const VType& v,
    size_t n_threads,
    BuffType& buff
)
{
    using io_t = std::decay_t<IOType>;
    using value_t = typename std::decay_t<VType>::Scalar;

    const auto A = io.ancestries();
    const auto snp = j / A;
    const auto anc = j % A;
    const auto nnz = io.nnz0()[j] + io.nnz1()[j];
    // NOTE: multiplier from experimentation
    const size_t n_bytes = (16 * sizeof(value_t)) * nnz;
    if (n_threads <= 1 || n_bytes <= Configs::min_bytes) {
        value_t sum = 0;
        for (int hap = 0; hap < io_t::n_haps; ++hap) {
            auto it = io.begin(snp, anc, hap);
            const auto end = io.end(snp, anc, hap);
            for (; it != end; ++it) {
                sum += v[*it];
            }
        }
        return sum;
    }

    auto vbuff = buff.head(n_threads);
    vbuff.setZero();

    #pragma omp parallel num_threads(n_threads)
    {
        for (int hap = 0; hap < io_t::n_haps; ++hap) {
            const size_t n_chunks = io.n_chunks(snp, anc, hap);
            const int n_blocks = std::min(n_threads, n_chunks);
            const int block_size = n_chunks / n_blocks;
            const int remainder = n_chunks % n_blocks;

            #pragma omp for schedule(static) nowait
            for (int t = 0; t < n_blocks; ++t) {
                const auto begin = (
                    std::min<int>(t, remainder) * (block_size + 1) 
                    + std::max<int>(t-remainder, 0) * block_size
                );
                const auto size = block_size + (t < remainder);
                auto it = io.begin(snp, anc, hap, begin);
                const auto end = io.begin(snp, anc, hap, begin + size);

                value_t sum = 0;
                for (; it != end; ++it) {
                    sum += v[*it];
                }
                vbuff[t] += sum;
            }
        }
    }
    return vbuff.sum();
}

template <class IOType, class ValueType, class OutType>
auto snp_phased_ancestry_axi(
    const IOType& io,
    int j, 
    ValueType v,
    OutType& out,
    size_t n_threads
)
{
    using io_t = std::decay_t<IOType>;
    using value_t = ValueType;

    const auto A = io.ancestries();
    const auto snp = j / A;
    const auto anc = j % A;
    const auto nnz = io.nnz0()[j] + io.nnz1()[j];
    // NOTE: multiplier from experimentation
    const size_t n_bytes = (8 * sizeof(value_t)) * nnz;
    if (n_threads <= 1 || n_bytes <= Configs::min_bytes) {
        for (int hap = 0; hap < io_t::n_haps; ++hap) {
            auto it = io.begin(snp, anc, hap);
            const auto end = io.end(snp, anc, hap);
            for (; it != end; ++it) {
                out[*it] += v;
            }
        }
        return;
    }

    for (int hap = 0; hap < io_t::n_haps; ++hap) {
        const size_t n_chunks = io.n_chunks(snp, anc, hap);
        const int n_blocks = std::min(n_threads, n_chunks);
        const int block_size = n_chunks / n_blocks;
        const int remainder = n_chunks % n_blocks;

        #pragma omp parallel for schedule(static) num_threads(n_threads)
        for (int t = 0; t < n_blocks; ++t) {
            const auto begin = (
                std::min<int>(t, remainder) * (block_size + 1) 
                + std::max<int>(t-remainder, 0) * block_size
            );
            const auto size = block_size + (t < remainder);
            auto it = io.begin(snp, anc, hap, begin);
            const auto end = io.begin(snp, anc, hap, begin + size);

            for (; it != end; ++it) {
                out[*it] += v;
            }
        }
    }
}

} // namespace matrix
} // namespace adelie_core