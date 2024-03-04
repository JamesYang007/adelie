#pragma once
#include <adelie_core/matrix/matrix_cov_base.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

template <class DenseType>
class MatrixCovDense: public MatrixCovBase<typename DenseType::Scalar>
{
public:
    using base_t = MatrixCovBase<typename DenseType::Scalar>;
    using dense_t = DenseType;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::colmat_value_t;
    
private:
    const Eigen::Map<const dense_t> _mat;   // underlying dense matrix
    const size_t _n_threads;                // number of threads
    util::rowmat_type<value_t> _buff;
    
public:
    explicit MatrixCovDense(
        const Eigen::Ref<const dense_t>& mat,
        size_t n_threads
    ): 
        _mat(mat.data(), mat.rows(), mat.cols()),
        _n_threads(n_threads),
        _buff(_n_threads, mat.cols())
    {
        if (mat.rows() != mat.cols()) {
            throw std::runtime_error("Matrix must be square!");
        }
    }

    using base_t::rows;
    
    void bmul(
        const Eigen::Ref<const vec_index_t>& groups,
        const Eigen::Ref<const vec_index_t>& group_sizes,
        const Eigen::Ref<const vec_index_t>& row_indices,
        const Eigen::Ref<const vec_index_t>& col_indices,
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_bmul(
            groups.size(), 
            group_sizes.size(), 
            row_indices.size(), 
            col_indices.size(), 
            v.size(), 
            out.size(), 
            rows(), 
            cols()
        );
        auto _vbuff = _buff.row(0).head(out.size());
        out.setZero();
        int rpos = 0;
        for (int i = 0; i < row_indices.size(); ++i) {
            const auto r = row_indices[i];
            const auto rg = groups[r];
            const auto rgs = group_sizes[r];
            const auto rv = v.segment(rpos, rgs);

            int cpos = 0;
            for (int j = 0; j < col_indices.size(); ++j) {
                const auto c = col_indices[j];
                const auto cg = groups[c];
                const auto cgs = group_sizes[c];
                _vbuff.segment(cpos, cgs).noalias() = rv.matrix() * _mat.block(rg, cg, rgs, cgs);
                cpos += cgs;
            }
            dvaddi(out, _vbuff, _n_threads);

            rpos += rgs;
        }
    }

    void mul(
        int i, int p,
        const Eigen::Ref<const vec_value_t>& v,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_mul(i, p, v.size(), out.size(), rows(), cols());
        auto outm = out.matrix();
        dgemv(
            _mat.middleCols(i, p).transpose(),
            v.matrix(),
            _n_threads,
            _buff,
            outm
        );
    } 

    void to_dense(
        int i, int p,
        Eigen::Ref<colmat_value_t> out
    ) override
    {
        base_t::check_to_dense(i, p, out.rows(), out.cols(), rows(), cols());
        out = _mat.block(i, i, p, p);
    }

    int cols() const override
    {
        return _mat.cols();
    }
};

} // namespace matrix
} // namespace adelie_core