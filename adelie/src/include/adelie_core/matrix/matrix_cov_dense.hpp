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
    using typename base_t::vec_index_t;
    using typename base_t::vec_value_t;
    using typename base_t::colmat_value_t;
    
private:
    const Eigen::Map<const dense_t> _mat;   // underlying dense matrix
    const size_t _n_threads;                // number of threads
    
public:
    explicit MatrixCovDense(
        const Eigen::Ref<const dense_t>& mat,
        size_t n_threads
    ): 
        _mat(mat.data(), mat.rows(), mat.cols()),
        _n_threads(n_threads)
    {
        if (mat.rows() != mat.cols()) {
            throw std::runtime_error("Matrix must be square!");
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
        out.setZero();
        for (int j_idx = 0; j_idx < subset.size(); ++j_idx) {
            const auto j = subset[j_idx];
            for (int i_idx = 0; i_idx < indices.size(); ++i_idx) {
                const auto i = indices[i_idx];
                const auto v = values[i_idx];
                out[j_idx] += v * _mat(i, j);
            }
        }
    }

    void mul(
        const Eigen::Ref<const vec_index_t>& indices,
        const Eigen::Ref<const vec_value_t>& values,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_mul(indices.size(), values.size(), out.size(), rows(), cols());
        out.setZero();
        for (int i_idx = 0; i_idx < indices.size(); ++i_idx) {
            const auto i = indices[i_idx];
            const auto v = values[i_idx];
            if constexpr (dense_t::IsRowMajor) {
                dvaddi(out, v * _mat.row(i).array(), _n_threads);
            } else {
                dvaddi(out, v * _mat.col(i).array(), _n_threads);
            }
        }
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