#pragma once
#include <adelie_core/io/io_snp_base.hpp>

namespace adelie_core {
namespace io {

class IOSNPUnphased : public IOSNPBase
{
public:
    using base_t = IOSNPBase;
    using outer_t = uint64_t;
    using inner_t = uint32_t;
    using value_t = int8_t;
    using vec_outer_t = util::rowvec_type<outer_t>;
    using vec_inner_t = util::rowvec_type<inner_t>;
    using vec_value_t = util::rowvec_type<value_t>;
    using rowarr_value_t = util::rowarr_type<value_t>;
    using colarr_value_t = util::colarr_type<value_t>;

protected:
    static constexpr size_t _multiplier = (
        sizeof(inner_t) + 
        sizeof(value_t)
    );

    using base_t::_buffer;
    using base_t::_filename;
    using base_t::_is_read;

public:
    using base_t::base_t;
    using base_t::read;

    inner_t rows() const {
        if (!_is_read) throw_no_read();
        return reinterpret_cast<const inner_t&>(_buffer[sizeof(bool_t)]);
    }

    inner_t snps() const {
        if (!_is_read) throw_no_read();
        return reinterpret_cast<const inner_t&>(_buffer[sizeof(bool_t) + sizeof(inner_t)]);
    }

    inner_t cols() const { return snps(); }

    Eigen::Ref<const vec_outer_t> outer() const
    {
        if (!_is_read) throw_no_read();
        return Eigen::Map<const vec_outer_t>(
            reinterpret_cast<const outer_t*>(&_buffer[sizeof(bool_t) + 2 * sizeof(inner_t)]),
            cols() + 1
        );
    }

    inner_t nnz(int j) const
    {
        if (!_is_read) throw_no_read();
        const auto _outer = outer();
        return (_outer[j+1] - _outer[j]) / _multiplier;
    }

    Eigen::Ref<const vec_inner_t> inner(int j) const 
    {
        if (!_is_read) throw_no_read();
        const auto _outer = outer();
        return Eigen::Map<const vec_inner_t>(
            reinterpret_cast<const inner_t*>(&_buffer[_outer[j]]),
            nnz(j)
        );
    }

    Eigen::Ref<const vec_value_t> value(int j) const 
    {
        if (!_is_read) throw_no_read();
        const auto _outer = outer();
        const auto _nnz = nnz(j);
        return Eigen::Map<const vec_value_t>(
            reinterpret_cast<const value_t*>(&_buffer[_outer[j] + sizeof(inner_t) * _nnz]),
            _nnz
        );
    }

    rowarr_value_t to_dense(
        size_t n_threads
    ) const
    {
        if (!_is_read) throw_no_read();
        const auto n = rows();
        const auto p = cols();
        rowarr_value_t dense(n, p);

        #pragma omp parallel for schedule(auto) num_threads(n_threads)
        for (inner_t j = 0; j < p; ++j) {
            const auto _inner = inner(j);
            const auto _value = value(j);
            auto dense_j = dense.col(j);
            dense_j.setZero();
            for (inner_t i = 0; i < _inner.size(); ++i) {
                dense_j[_inner[i]] = _value[i];
            }
        }

        return dense;
    }

    size_t write(
        const Eigen::Ref<const colarr_value_t>& calldata,
        size_t n_threads
    )
    {
        const bool_t endian = is_big_endian();
        const inner_t n = calldata.rows();
        const inner_t p = calldata.cols();

        // outer[i] = number of bytes to jump from beginning of file 
        // to start reading column i information.
        // outer[i+1] - outer[i] = total number of bytes for column i. 
        vec_outer_t outer(p+1); 
        outer[0] = 0;
        outer.tail(p) = (calldata != 0).colwise().count().template cast<outer_t>();
        for (int i = 1; i < outer.size(); ++i) {
            outer[i] += outer[i-1];
        }
        outer *= _multiplier;
        outer += (
            sizeof(bool_t) +
            2 * sizeof(inner_t) +
            outer.size() * sizeof(outer_t)
        );

        auto& buffer = _buffer;
        buffer.resize(outer[p]);

        size_t idx = 0;
        reinterpret_cast<bool_t&>(buffer[idx]) = endian; idx += sizeof(bool_t);
        reinterpret_cast<inner_t&>(buffer[idx]) = n; idx += sizeof(inner_t);
        reinterpret_cast<inner_t&>(buffer[idx]) = p; idx += sizeof(inner_t);
        Eigen::Map<vec_outer_t>(
            reinterpret_cast<outer_t*>(&buffer[idx]),
            outer.size()
        ) = outer;

        #pragma omp parallel for schedule(auto) num_threads(n_threads)
        for (inner_t j = 0; j < p; ++j) {
            const auto col_j = calldata.col(j);
            const auto nnz_bytes = outer[j+1] - outer[j];
            const auto nnz = nnz_bytes / _multiplier;
            Eigen::Map<vec_inner_t> inner(
                reinterpret_cast<inner_t*>(&buffer[outer[j]]),
                nnz
            );
            Eigen::Map<vec_value_t> value(
                reinterpret_cast<value_t*>(&buffer[outer[j] + sizeof(inner_t) * nnz]),
                nnz
            );

            size_t count = 0;
            for (inner_t i = 0; i < n; ++i) {
                if (col_j[i] == 0) continue;
                inner[count] = i;
                value[count] = col_j[i];
                ++count;
            }
        }

        auto file_ptr = fopen_safe(_filename.c_str(), "wb");
        auto fp = file_ptr.get();
        auto total_bytes = std::fwrite(buffer.data(), sizeof(char), buffer.size(), fp);
        if (total_bytes != static_cast<size_t>(buffer.size())) {
            throw std::runtime_error(
                "Could not write the full buffer."
            );
        }

        return total_bytes;
    }
};

} // namespace io
} // namespace adelie_core