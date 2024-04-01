#pragma once
#include <adelie_core/matrix/matrix_cov_base.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

template <class SparseType>
class MatrixCovSparse: public MatrixCovBase<typename SparseType::Scalar>
{
public:
    using base_t = MatrixCovBase<typename SparseType::Scalar>;
    using sparse_t = SparseType;
    using typename base_t::value_t;
    using typename base_t::vec_index_t;
    using typename base_t::vec_value_t;
    using typename base_t::colmat_value_t;
    using vec_sp_value_t = vec_value_t;
    using vec_sp_index_t = util::rowvec_type<typename sparse_t::StorageIndex>;

    static_assert(!sparse_t::IsRowMajor, "MatrixCovSparse: only column-major allowed!");
    
private:
    const Eigen::Map<const sparse_t> _mat;  // underlying sparse matrix
    const size_t _n_threads;                // number of threads
    
public:
    explicit MatrixCovSparse(
        size_t rows,
        size_t cols,
        size_t nnz,
        const Eigen::Ref<const vec_sp_index_t>& outer,
        const Eigen::Ref<const vec_sp_index_t>& inner,
        const Eigen::Ref<const vec_sp_value_t>& value,
        size_t n_threads
    ): 
        _mat(rows, cols, nnz, outer.data(), inner.data(), value.data()),
        _n_threads(n_threads)
    {
        if (n_threads < 1) {
            throw std::runtime_error("n_threads must be >= 1.");
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
            const auto outer = _mat.outerIndexPtr()[j];
            const auto size = _mat.outerIndexPtr()[j+1] - outer;
            const Eigen::Map<const vec_sp_index_t> inner(
                _mat.innerIndexPtr() + outer, size
            );
            const Eigen::Map<const vec_sp_value_t> value(
                _mat.valuePtr() + outer, size
            );
            out[j_idx] = svsvdot(indices, values, inner, value);
        }
    }

    void mul(
        const Eigen::Ref<const vec_index_t>& indices,
        const Eigen::Ref<const vec_value_t>& values,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_mul(indices.size(), values.size(), out.size(), rows(), cols());
        #pragma omp parallel for schedule(static) num_threads(_n_threads)
        for (int j = 0; j < _mat.cols(); ++j) {
            const auto outer = _mat.outerIndexPtr()[j];
            const auto size = _mat.outerIndexPtr()[j+1] - outer;
            const Eigen::Map<const vec_sp_index_t> inner(
                _mat.innerIndexPtr() + outer, size
            );
            const Eigen::Map<const vec_sp_value_t> value(
                _mat.valuePtr() + outer, size
            );
            out[j] = svsvdot(indices, values, inner, value);
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