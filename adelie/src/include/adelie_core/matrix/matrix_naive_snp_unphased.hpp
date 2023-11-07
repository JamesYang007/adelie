#pragma once
#include <cstdio>
#include <string>
#include <vector>
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

template <class ValueType>
class MatrixNaiveSNPUnphased : public MatrixNaiveBase<ValueType>
{
public:
    using base_t = MatrixNaiveBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using typename base_t::rowmat_value_t;
    using typename base_t::sp_mat_value_t;
    using vec_vec_index_t = util::rowvec_type<vec_index_t>;
    using dyn_vec_string_t = std::vector<std::string>;
    
protected:
    const dyn_vec_string_t _filenames;  // (F,) array of file names
    const vec_index_t _file_begins;     // (F+1,) array of file begin indices.
                                        // _file_begins[i] == starting feature index for file i.
     
    const size_t _n_threads;

    auto init_file_begins(
        const dyn_vec_string_t& filenames
    )
    {
        vec_index_t file_begins(filenames.size() + 1);
        file_begins[0] = 0;
        for (int i = 1; i < file_begins.size(); ++i) {
            const auto& name = filenames[i-1];
            FILE* fp = fopen(name.c_str(), "rb");
            if (!fp) {
                throw std::runtime_error("Cannot open file " + name);
            }
            int32_t n_cols;
            size_t status = fread(&n_cols, sizeof(int32_t), 1, fp);
            if (status < 1) {
                throw std::runtime_error("Could not read the first byte of file " + name);
            }
            file_begins[i] = n_cols;
            fclose(fp);
        }

        for (int i = 1; i < file_begins.size(); ++i) {
            file_begins[i] += file_begins[i-1];
        }
        return file_begins;
    }

public:
    MatrixNaiveSNPUnphased(
        const dyn_vec_string_t& filenames,
        size_t n_threads
    ): 
        _filenames(filenames),
        _file_begins(init_file_begins(filenames)),
        _n_threads(n_threads)
    {}

    value_t cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v
    ) const override
    {
        if (is_cached(j)) {

        }
    }

    bool is_cached(int j) const
    {

    }

    void cache()
    {

    }
};

} // namespace matrix
} // namespace adelie_core