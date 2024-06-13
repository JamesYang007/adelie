#pragma once
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

template <class ValueType, class IndexType=Eigen::Index>
class MatrixNaiveCConcatenate: public MatrixNaiveBase<ValueType, IndexType>
{
public:
    using base_t = MatrixNaiveBase<ValueType, IndexType>;
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
    vec_value_t _buff;                      // (n,) buffer

    static inline auto init_rows(
        const std::vector<base_t*>& mat_list
    )
    {
        if (mat_list.size() == 0) {
            throw util::adelie_core_error("List must be non-empty.");
        }

        const auto n = mat_list[0]->rows();
        for (auto mat : mat_list) {
            if (n != mat->rows()) {
                throw util::adelie_core_error("All matrices must have the same number of rows.");
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
    explicit MatrixNaiveCConcatenate(
        const std::vector<base_t*>& mat_list
    ): 
        _mat_list(mat_list),
        _rows(init_rows(mat_list)),
        _cols(init_cols(mat_list)),
        _slice_map(init_slice_map(mat_list, _cols)),
        _index_map(init_index_map(mat_list, _cols)),
        _buff(_rows)
    {
        if (mat_list.size() <= 0) {
            throw util::adelie_core_error("mat_list must be non-empty.");
        }
    }

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
            const int q_curr = std::min<int>(mat.cols()-index, q-n_processed);
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
        int n_processed = 0;
        while (n_processed < q) {
            const auto j_curr = j + n_processed;
            const auto slice = _slice_map[j_curr];
            auto& mat = *_mat_list[slice];
            const auto index = _index_map[j_curr];
            const int q_curr = std::min<int>(mat.cols()-index, q-n_processed);
            mat.btmul(index, q_curr, v.segment(n_processed, q_curr), out);
            n_processed += q_curr;
        }
    }

    void mul(
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        int n_processed = 0;
        for (size_t i = 0; i < _mat_list.size(); ++i) {
            auto& mat = *_mat_list[i];
            const auto p = mat.cols();
            mat.mul(v, weights, out.segment(n_processed, p));
            n_processed += p;
        }
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
            throw util::adelie_core_error(
                "MatrixNaiveCConcatenate::cov() only allows the block to be fully contained in one of the matrices in the list."
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

template <class ValueType, class IndexType=Eigen::Index>
class MatrixNaiveRConcatenate: public MatrixNaiveBase<ValueType, IndexType>
{
public:
    using base_t = MatrixNaiveBase<ValueType, IndexType>;
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
    vec_value_t _buff;                      // (p,) buffer

    static inline auto init_rows(
        const std::vector<base_t*>& mat_list
    )
    {
        size_t n = 0;
        for (auto mat : mat_list) n += mat->rows();
        return n;
    }

    static inline auto init_cols(
        const std::vector<base_t*>& mat_list
    )
    {
        if (mat_list.size() == 0) {
            throw util::adelie_core_error("List must be non-empty.");
        }

        const auto p = mat_list[0]->cols();
        for (auto mat : mat_list) {
            if (p != mat->cols()) {
                throw util::adelie_core_error("All matrices must have the same number of columns.");
            }
        }

        return p;
    }

public:
    explicit MatrixNaiveRConcatenate(
        const std::vector<base_t*>& mat_list
    ): 
        _mat_list(mat_list),
        _rows(init_rows(mat_list)),
        _cols(init_cols(mat_list)),
        _buff(_cols)
    {
        if (mat_list.size() <= 0) {
            throw util::adelie_core_error("mat_list must be non-empty.");
        }
    }

    value_t cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v,
        const Eigen::Ref<const vec_value_t>& weights
    ) override
    {
        base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
        size_t begin = 0;
        value_t sum = 0;
        for (size_t i = 0; i < _mat_list.size(); ++i) {
            auto& mat = *_mat_list[i];
            const auto rows_curr = mat.rows();
            const Eigen::Map<const vec_value_t> v_curr(
                v.data() + begin, rows_curr
            );
            const Eigen::Map<const vec_value_t> weights_curr(
                weights.data() + begin, rows_curr
            );
            sum += mat.cmul(j, v_curr, weights_curr);
            begin += rows_curr;
        }
        return sum;
    }

    void ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_ctmul(j, out.size(), rows(), cols());
        size_t begin = 0;
        for (size_t i = 0; i < _mat_list.size(); ++i) {
            auto& mat = *_mat_list[i];
            const auto rows_curr = mat.rows();
            Eigen::Map<vec_value_t> out_curr(
                out.data() + begin, rows_curr
            );
            mat.ctmul(j, v, out_curr);
            begin += rows_curr;
        }
    }

    void bmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
        size_t begin = 0;
        out.setZero();
        Eigen::Map<vec_value_t> buff(_buff.data(), q);
        for (size_t i = 0; i < _mat_list.size(); ++i) {
            auto& mat = *_mat_list[i];
            const auto rows_curr = mat.rows();
            const Eigen::Map<const vec_value_t> v_curr(
                v.data() + begin, rows_curr
            );
            const Eigen::Map<const vec_value_t> weights_curr(
                weights.data() + begin, rows_curr
            );
            mat.bmul(j, q, v_curr, weights_curr, buff);
            out += buff;
            begin += rows_curr;
        }
    }

    void btmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_btmul(j, q, v.size(), out.size(), rows(), cols());
        size_t begin = 0;
        for (size_t i = 0; i < _mat_list.size(); ++i) {
            auto& mat = *_mat_list[i];
            const auto rows_curr = mat.rows();
            Eigen::Map<vec_value_t> out_curr(
                out.data() + begin, rows_curr
            );
            mat.btmul(j, q, v, out_curr);
            begin += rows_curr;
        }
    }

    void mul(
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        size_t begin = 0;
        out.setZero();
        Eigen::Map<vec_value_t> buff(_buff.data(), out.size());
        for (size_t i = 0; i < _mat_list.size(); ++i) {
            auto& mat = *_mat_list[i];
            const auto rows_curr = mat.rows();
            const Eigen::Map<const vec_value_t> v_curr(
                v.data() + begin, rows_curr
            );
            const Eigen::Map<const vec_value_t> weights_curr(
                weights.data() + begin, rows_curr
            );
            mat.mul(v_curr, weights_curr, buff);
            out += buff;
            begin += rows_curr;
        }
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

        if (_buff.size() < q * q) _buff.resize(q * q);

        size_t begin = 0;
        out.setZero();
        for (size_t i = 0; i < _mat_list.size(); ++i) {
            auto& mat = *_mat_list[i];
            const auto rows_curr = mat.rows();
            const Eigen::Map<const vec_value_t> sqrt_weights_curr(
                sqrt_weights.data() + begin, rows_curr
            );
            Eigen::Map<colmat_value_t> out_curr(
                _buff.data(), q, q
            );
            Eigen::Map<colmat_value_t> buffer_curr(
                buffer.data(), rows_curr, q
            );
            mat.cov(j, q, sqrt_weights_curr, out_curr, buffer_curr);
            out += out_curr;
            begin += rows_curr;
        }
    }

    void sp_btmul(
        const sp_mat_value_t& v, 
        Eigen::Ref<rowmat_value_t> out
    ) override
    {
        base_t::check_sp_btmul(
            v.rows(), v.cols(), out.rows(), out.cols(), rows(), cols()
        );
        vec_value_t buff;

        const auto L = v.rows();
        size_t begin = 0;
        for (size_t i = 0; i < _mat_list.size(); ++i) {
            auto& mat = *_mat_list[i];
            const auto rows_curr = mat.rows();
            if (buff.size() < L * rows_curr) buff.resize(L * rows_curr);
            Eigen::Map<rowmat_value_t> out_curr(
                buff.data(), L, rows_curr
            );
            mat.sp_btmul(v, out_curr);
            out.middleCols(begin, rows_curr) = out_curr;
            begin += rows_curr;
        }
    }
};

} // namespace matrix 
} // namespace adelie_core