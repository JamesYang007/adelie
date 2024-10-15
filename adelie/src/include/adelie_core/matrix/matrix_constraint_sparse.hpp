#pragma once
#include <adelie_core/matrix/matrix_constraint_base.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

template <class SparseType, class IndexType=Eigen::Index>
class MatrixConstraintSparse: public MatrixConstraintBase<typename SparseType::Scalar, IndexType>
{
public:
    using base_t = MatrixConstraintBase<typename SparseType::Scalar, IndexType>;
    using sparse_t = SparseType;
    using typename base_t::value_t;
    using typename base_t::index_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using vec_sp_value_t = vec_value_t;
    using vec_sp_index_t = util::rowvec_type<typename sparse_t::StorageIndex>;

    static_assert(sparse_t::IsRowMajor, "MatrixConstraintSparse: only row-major allowed!");
    
private:
    const Eigen::Map<const sparse_t> _mat;  // underlying sparse matrix
    const size_t _n_threads;                // number of threads
    vec_value_t _buff;
    
public:
    explicit MatrixConstraintSparse(
        size_t rows,
        size_t cols,
        size_t nnz,
        const Eigen::Ref<const vec_sp_index_t>& outer,
        const Eigen::Ref<const vec_sp_index_t>& inner,
        const Eigen::Ref<const vec_sp_value_t>& value,
        size_t n_threads
    ): 
        _mat(rows, cols, nnz, outer.data(), inner.data(), value.data()),
        _n_threads(n_threads),
        _buff(_n_threads)
    {
        if (n_threads < 1) {
            throw util::adelie_core_error("n_threads must be >= 1.");
        }
    }

    void rmmul(
        int j,
        const Eigen::Ref<const colmat_value_t>& Q,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        out.matrix() = _mat.row(j) * Q;
    }

    value_t rvmul(
        int j,
        const Eigen::Ref<const vec_value_t>& v
    ) override
    {
        return _mat.row(j).dot(v.matrix());
    }

    void rvtmul(
        int j,
        value_t v,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        out.matrix() += v * _mat.row(j);
    }

    void mul(
        const Eigen::Ref<const vec_value_t>& v,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        out.matrix() = v.matrix() * _mat;
    }

    void tmul(
        const Eigen::Ref<const vec_value_t>& v,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        const auto routine = [&](int k) {
            out[k] = _mat.row(k).dot(v.matrix());
        };
        if (_n_threads <= 1) {
            for (int k = 0; k < out.size(); ++k) routine(k);
        } else {
            #pragma omp parallel for schedule(static) num_threads(_n_threads)
            for (int k = 0; k < out.size(); ++k) routine(k);
        }
    }

    void cov(
        const Eigen::Ref<const colmat_value_t>& Q,
        Eigen::Ref<colmat_value_t> out
    ) override
    {
        Eigen::setNbThreads(_n_threads);
        out.noalias() = _mat * Q * _mat.transpose();
    }

    int rows() const override
    {
        return _mat.rows();
    }
    
    int cols() const override
    {
        return _mat.cols();
    }

    void sp_mul(
        const Eigen::Ref<const vec_index_t>& indices,
        const Eigen::Ref<const vec_value_t>& values,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        out.setZero();
        for (Eigen::Index i = 0; i < indices.size(); ++i) {
            out.matrix() += values[i] * _mat.row(indices[i]);
        }
    }
};

} // namespace matrix
} // namespace adelie_core