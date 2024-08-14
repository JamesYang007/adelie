#pragma once
#include <adelie_core/matrix/matrix_constraint_base.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

template <class DenseType, class IndexType=Eigen::Index>
class MatrixConstraintDense: public MatrixConstraintBase<typename DenseType::Scalar, IndexType>
{
public:
    using base_t = MatrixConstraintBase<typename DenseType::Scalar, IndexType>;
    using dense_t = DenseType;
    using typename base_t::value_t;
    using typename base_t::index_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using rowmat_value_t = util::rowmat_type<value_t>;
    
private:
    const Eigen::Map<const dense_t> _mat;   // underlying dense matrix
    const size_t _n_threads;                // number of threads
    rowmat_value_t _buff;
    
public:
    explicit MatrixConstraintDense(
        const Eigen::Ref<const dense_t>& mat,
        size_t n_threads
    ): 
        _mat(mat.data(), mat.rows(), mat.cols()),
        _n_threads(n_threads),
        _buff(_n_threads, std::min<size_t>(mat.rows(), mat.cols()))
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
        auto out_m = out.matrix();
        dgemv(
            _mat,
            v.matrix(),
            _n_threads,
            _buff,
            out_m
        );
    }

    void tmul(
        const Eigen::Ref<const vec_value_t>& v,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        auto out_m = out.matrix();
        dgemv(
            _mat.transpose(),
            v.matrix(),
            _n_threads,
            _buff,
            out_m
        );
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
            out += values[i] * _mat.row(indices[i]).array();
        }
    }
};

} // namespace matrix
} // namespace adelie_core