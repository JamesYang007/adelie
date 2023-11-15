#pragma once
#include <adelie_core/matrix/matrix_naive_base.hpp>

namespace adelie_core {
namespace matrix {

class MatrixNaiveSNPBase
{
public:
    using vec_index_t = util::rowvec_type<int>;
    using string_t = std::string;
    using dyn_vec_string_t = std::vector<string_t>;

protected:
    const dyn_vec_string_t _filenames;  // (F,) array of file names
    const size_t _n_threads;            // number of threads

    template <class VecIOType>
    static auto init_ios(
        const dyn_vec_string_t& filenames
    )
    {
        VecIOType ios;
        ios.reserve(filenames.size());
        for (int i = 0; i < filenames.size(); ++i) {
            ios.emplace_back(filenames[i]);
            ios.back().read();
        }
        return ios;
    }

    template <class VecIOType>
    static auto init_snps(
        const VecIOType& ios
    )
    {
        size_t snps = 0;
        for (const auto& io : ios) {
            snps += io.snps();
        }
        return snps;
    }

    template <class VecIOType>
    static auto init_io_slice_map(
        const VecIOType& ios,
        size_t snps
    )
    {
        vec_index_t io_slice_map(snps);
        size_t begin = 0;
        for (int i = 0; i < ios.size(); ++i) {
            const auto& io = ios[i];
            const auto si = io.snps();
            for (int j = 0; j < si; ++j) {
                io_slice_map[begin + j] = i;
            }
            begin += si;
        } 
        return io_slice_map;
    }

    template <class VecIOType>
    static auto init_io_index_map(
        const VecIOType& ios,
        size_t snps
    )
    {
        vec_index_t io_index_map(snps);
        size_t begin = 0;
        for (int i = 0; i < ios.size(); ++i) {
            const auto& io = ios[i];
            const auto si = io.snps();
            for (int j = 0; j < si; ++j) {
                io_index_map[begin + j] = j;
            }
            begin += si;
        } 
        return io_index_map;
    }

public:
    MatrixNaiveSNPBase(
        const dyn_vec_string_t& filenames,
        size_t n_threads
    ):
        _filenames(filenames),
        _n_threads(n_threads)
    {
        if (filenames.size() == 0) {
            throw std::runtime_error(
                "filenames must be non-empty!"
            );
        }
    }
};

} // namespace matrix
} // namespace adelie_core