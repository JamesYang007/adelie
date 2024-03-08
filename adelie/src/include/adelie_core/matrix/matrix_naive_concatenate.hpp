#pragma once
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

template <class ValueType>
class MatrixNaiveConcatenate: public MatrixNaiveBase<ValueType>
{
public:
    using base_t = MatrixNaiveBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using typename base_t::rowmat_value_t;
    using typename base_t::sp_mat_value_t;
    
private:
    const std::vector<base_t*> _mat_list;   // (L,) list of naive matrices
    const size_t _rows;                     // number of rows
    const size_t _cols;                     // number of columns
    const vec_index_t _slice_map;           // (p,) array mapping to matrix slice
    const vec_index_t _index_map;           // (p,) array mapping to (relative) index of the slice
    const size_t _n_threads;                // number of threads
    vec_value_t _buff;                      // (n,) buffer

    static inline auto init_rows(
        const std::vector<base_t*>& mat_list
    )
    {
        if (mat_list.size() == 0) {
            throw std::runtime_error("List must be non-empty.");
        }

        const auto n = mat_list[0]->rows();
        for (auto mat : mat_list) {
            if (n != mat->rows()) {
                throw std::runtime_error("All matrices must have the same number of rows.");
            }
        }

        return n;
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

    static inline auto init_index_map(
        const std::vector<base_t*>& mat_list,
        size_t p
    )
    {
        vec_index_t index_map(p);
        size_t begin = 0;
        for (size_t i = 0; i < mat_list.size(); ++i) {
            const auto& mat = *mat_list[i];
            const auto pi = mat.cols();
            for (int j = 0; j < pi; ++j) {
                index_map[begin + j] = j;
            }
            begin += pi;
        } 
        return index_map;
    }

public:
    explicit MatrixNaiveConcatenate(
        const std::vector<base_t*>& mat_list,
        size_t n_threads
    ): 
        _mat_list(mat_list),
        _rows(init_rows(mat_list)),
        _cols(init_cols(mat_list)),
        _slice_map(init_slice_map(mat_list, _cols)),
        _index_map(init_index_map(mat_list, _cols)),
        _n_threads(n_threads),
        _buff(_rows)
    {}

    value_t cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v,
        const Eigen::Ref<const vec_value_t>& weights
    ) override
    {
        base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
        const auto slice = _slice_map[j];
        auto& mat = *_mat_list[slice];
        const auto index = _index_map[j];
        return mat.cmul(index, v, weights);
    }

    void ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_ctmul(j, out.size(), rows(), cols());
        const auto slice = _slice_map[j];
        auto& mat = *_mat_list[slice];
        const auto index = _index_map[j];
        mat.ctmul(index, v, out);
    }

    void bmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
        int n_processed = 0;
        while (n_processed < q) {
            const auto j_curr = j + n_processed;
            const auto slice = _slice_map[j_curr];
            auto& mat = *_mat_list[slice];
            const auto index = _index_map[j_curr];
            const int q_curr = std::min(mat.cols()-index, q-n_processed);
            mat.bmul(index, q_curr, v, weights, out.segment(n_processed, q_curr));
            n_processed += q_curr;
        }
    }

    void btmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_btmul(j, q, v.size(), out.size(), rows(), cols());
        dvzero(out, _n_threads);
        int n_processed = 0;
        while (n_processed < q) {
            const auto j_curr = j + n_processed;
            const auto slice = _slice_map[j_curr];
            auto& mat = *_mat_list[slice];
            const auto index = _index_map[j_curr];
            const int q_curr = std::min(mat.cols()-index, q-n_processed);
            mat.btmul(index, q_curr, v.segment(n_processed, q_curr), _buff);
            dvaddi(out, _buff, _n_threads);
            n_processed += q_curr;
        }
    }

    void mul(
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        bmul(0, cols(), v, weights, out);
    }

    int rows() const override
    {
        return _rows;
    }
    
    int cols() const override
    {
        return _cols;
    }

    void cov(
        int j, int q,
        const Eigen::Ref<const vec_value_t>& sqrt_weights,
        Eigen::Ref<colmat_value_t> out,
        Eigen::Ref<colmat_value_t> buffer
    ) override
    {
        base_t::check_cov(
            j, q, sqrt_weights.size(), 
            out.rows(), out.cols(), buffer.rows(), buffer.cols(), 
            rows(), cols()
        );

        const auto slice = _slice_map[j]; 
        auto& mat = *_mat_list[slice];
        const auto index = _index_map[j];

        // check that the block is fully contained in one matrix
        if (slice != _slice_map[j+q-1]) {
            throw std::runtime_error(
                "MatrixNaiveConcatenate::cov() only allows the block to be fully contained in one of the matrices in the list."
            );
        }

        mat.cov(index, q, sqrt_weights, out, buffer);
    }

    void sp_btmul(
        const sp_mat_value_t& v, 
        Eigen::Ref<rowmat_value_t> out
    ) override
    {
        base_t::check_sp_btmul(
            v.rows(), v.cols(), out.rows(), out.cols(), rows(), cols()
        );
        out.setZero();
        rowmat_value_t buff(out.rows(), out.cols());
        int n_processed = 0;
        for (size_t i = 0; i < _mat_list.size(); ++i) {
            auto& mat = *_mat_list[i];
            const auto q_curr = mat.cols();
            mat.sp_btmul(v.middleCols(n_processed, q_curr), buff);
            out += buff;
            n_processed += q_curr;
        }
    }
};

} // namespace matrix 
} // namespace adelie_core