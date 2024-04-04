#pragma once
#include <algorithm>
#include <adelie_core/matrix/matrix_cov_base.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

template <class ValueType>
class MatrixCovBlockDiag: public MatrixCovBase<ValueType>
{
public:
    using base_t = MatrixCovBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_index_t;
    using typename base_t::vec_value_t;
    using typename base_t::colmat_value_t;
    
private:
    const std::vector<base_t*> _mat_list;   // (L,) list of covariance matrices
    const vec_index_t _mat_size_cumsum;     // (L+1,) list of matrix size cumulative sum.
    const size_t _cols;                     // number of columns
    const vec_index_t _slice_map;           // (p,) array mapping to matrix slice
    const size_t _n_threads;                // number of threads
    vec_index_t _ibuff;
    vec_value_t _vbuff;

    static inline auto init_mat_size_cumsum(
        const std::vector<base_t*>& mat_list
    )
    {
        vec_index_t mat_size_cumsum(mat_list.size() + 1);
        mat_size_cumsum[0] = 0;
        for (int i = 0; i < mat_list.size(); ++i) {
            mat_size_cumsum[i+1] = mat_size_cumsum[i] + mat_list[i]->cols();
        }
        return mat_size_cumsum;
    }

    static inline auto init_cols(
        const std::vector<base_t*>& mat_list
    )
    {
        size_t p = 0;
        for (auto mat : mat_list) p += mat->cols();
        return p;
    }

    static inline auto init_slice_map(
        const std::vector<base_t*>& mat_list,
        size_t p
    )
    {
        vec_index_t slice_map(p);
        size_t begin = 0;
        for (size_t i = 0; i < mat_list.size(); ++i) {
            const auto& mat = *mat_list[i];
            const auto pi = mat.cols();
            for (int j = 0; j < pi; ++j) {
                slice_map[begin + j] = i;
            }
            begin += pi;
        } 
        return slice_map;
    }

public:
    explicit MatrixCovBlockDiag(
        const std::vector<base_t*>& mat_list,
        size_t n_threads
    ): 
        _mat_list(mat_list),
        _mat_size_cumsum(init_mat_size_cumsum(mat_list)),
        _cols(init_cols(mat_list)),
        _slice_map(init_slice_map(mat_list, _cols)),
        _n_threads(n_threads),
        _ibuff(_cols),
        _vbuff(_cols) // just optimization
    {
        if (mat_list.size() <= 0) {
            throw util::adelie_core_error("mat_list must be non-empty.");
        }
        if (n_threads < 1) {
            throw util::adelie_core_error("n_threads must be >= 1.");
        }
    }

    using base_t::rows;
    
    void bmul(
        const Eigen::Ref<const vec_index_t>& subset,
        const Eigen::Ref<const vec_index_t>& indices,
        const Eigen::Ref<const vec_value_t>& values,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_bmul(subset.size(), indices.size(), values.size(), out.size(), rows(), cols());
        int n_processed = 0;
        int begin = 0;  // just a hint
        while (n_processed < subset.size()) {
            const auto j = subset[n_processed];
            auto& mat = *_mat_list[_slice_map[j]];
            const auto mat_pos = _mat_size_cumsum[_slice_map[j]];
            const auto subset_end = std::lower_bound(
                subset.data() + n_processed,
                subset.data() + subset.size(),
                mat_pos + mat.cols()
            ) - subset.data();
            const auto indices_begin = std::lower_bound(
                indices.data() + begin,
                indices.data() + indices.size(),
                mat_pos
            ) - indices.data();
            const auto indices_end = std::lower_bound(
                indices.data() + indices_begin,
                indices.data() + indices.size(),
                mat_pos + mat.cols()
            ) - indices.data();
            const auto new_subset_size = subset_end-n_processed;
            const auto new_indices_size = indices_end-indices_begin;
            Eigen::Map<vec_index_t> new_subset(
                _ibuff.data(), new_subset_size
            );
            Eigen::Map<vec_index_t> new_indices(
                _ibuff.data() + new_subset_size, new_indices_size
            );
            const Eigen::Map<const vec_value_t> new_values(
                values.data() + indices_begin, new_indices_size
            );
            Eigen::Map<vec_value_t> new_out(
                out.data() + n_processed, new_subset_size
            );
            new_subset = Eigen::Map<const vec_index_t>(
                subset.data() + n_processed, new_subset_size
            ) - mat_pos;
            new_indices = Eigen::Map<const vec_index_t>(
                indices.data() + indices_begin, new_indices_size
            ) - mat_pos;
            mat.bmul(new_subset, new_indices, new_values, new_out);
            n_processed += new_subset_size;
            begin = indices_end;
        }
    }

    void mul(
        const Eigen::Ref<const vec_index_t>& indices,
        const Eigen::Ref<const vec_value_t>& values,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_mul(indices.size(), values.size(), out.size(), rows(), cols());
        const auto routine = [&](int i) {
            auto& mat = *_mat_list[i];
            const auto mat_pos = _mat_size_cumsum[i];
            const auto begin = std::lower_bound(
                indices.data(),
                indices.data() + indices.size(),
                mat_pos
            ) - indices.data();
            const auto end = std::lower_bound(
                indices.data() + begin,
                indices.data() + indices.size(),
                mat_pos + mat.cols()
            ) - indices.data();
            const auto new_indices_size = end-begin;
            Eigen::Map<vec_index_t> new_indices(
                _ibuff.data() + mat_pos, new_indices_size
            );
            const Eigen::Map<const vec_value_t> new_values(
                values.data() + begin, new_indices_size
            );
            Eigen::Map<vec_value_t> new_out(
                out.data() + mat_pos, mat.cols()
            );
            new_indices = Eigen::Map<const vec_index_t>(
                indices.data() + begin, new_indices_size
            ) - mat_pos;
            mat.mul(new_indices, new_values, new_out);
        };
        if (_n_threads <= 1) {
            for (int i = 0; i < _mat_list.size(); ++i) routine(i);
        } else {
            #pragma omp parallel for schedule(static) num_threads(_n_threads)
            for (int i = 0; i < _mat_list.size(); ++i) routine(i);
        }
    } 

    void to_dense(
        int i, int p,
        Eigen::Ref<colmat_value_t> out
    ) override
    {
        base_t::check_to_dense(i, p, out.rows(), out.cols(), rows(), cols());
        out.setZero();
        int n_processed = 0;
        while (n_processed < p) {
            const auto j = i + n_processed;
            auto& mat = *_mat_list[_slice_map[j]];
            const auto mat_pos = _mat_size_cumsum[_slice_map[j]];
            const auto new_i = j - mat_pos;
            const auto new_p = std::min<size_t>(mat.cols()-new_i, p-n_processed);
            const auto new_p_sq = new_p * new_p;
            if (_vbuff.size() < new_p_sq) {
                _vbuff.resize(new_p_sq);
            }
            Eigen::Map<colmat_value_t> new_out(
                _vbuff.data(), new_p, new_p
            );
            mat.to_dense(new_i, new_p, new_out);
            out.block(n_processed, n_processed, new_p, new_p) = new_out;
            n_processed += new_p;
        }
    }

    int cols() const override
    {
        return _cols;
    }
};

} // namespace matrix
} // namespace adelie_core