#pragma once
#include <adelie_core/matrix/matrix_base.hpp>

namespace adelie_core {
namespace matrix {
    
template <class DenseType>
class MatrixDense: public MatrixBase<typename DenseType::Scalar>
{
public:
    using base_t = MatrixBase<typename DenseType::Scalar>;
    using dense_t = DenseType;
    using typename base_t::value_t;
    using typename base_t::rowvec_t;
    
private:
    const Eigen::Map<const dense_t> _mat;
    size_t _n_threads;
    
public:
    MatrixDense(
        const Eigen::Ref<const dense_t>& mat,
        size_t n_threads
    )
        : _mat(mat.data(), mat.rows(), mat.cols()),
          _n_threads(n_threads)
    {}
    
    value_t cmul(
        int j, 
        const Eigen::Ref<const rowvec_t>& v
    ) const override
    {
        const auto c = _mat.col(j);
        const size_t n_threads_cap = std::min<size_t>(_n_threads, v.size());
        const int n_blocks = std::max<int>(n_threads_cap, 1);
        const int block_size = c.size() / n_blocks;
        const int remainder = c.size() % n_blocks;
        value_t out = 0;
        #pragma omp parallel for schedule(static) num_threads(n_threads_cap) reduction(+:out)
        for (int t = 0; t < n_blocks; ++t)
        {
            const auto begin = (
                std::min<int>(t, remainder) * (block_size + 1) 
                + std::max<int>(t-remainder, 0) * block_size
            );
            const auto size = block_size + (t < remainder);
            out += c.segment(begin, size).dot(v.matrix().segment(begin, size));
        }
        return out;
    }

    void ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<rowvec_t> out
    ) const override
    {
        const auto c = _mat.col(j);
        const size_t n_threads_cap = std::min<size_t>(_n_threads, out.size());
        const int n_blocks = std::max<int>(n_threads_cap, 1);
        const int block_size = out.size() / n_blocks;
        const int remainder = out.size() % n_blocks;
        #pragma omp parallel for schedule(static) num_threads(n_threads_cap)
        for (int t = 0; t < n_blocks; ++t)
        {
            const auto begin = (
                std::min<int>(t, remainder) * (block_size + 1) 
                + std::max<int>(t-remainder, 0) * block_size
            );
            const auto size = block_size + (t < remainder);
            out.matrix().segment(begin, size) = v * c.segment(begin, size);
        }
    }

    void bmul(
        int i, int j, int p, int q, 
        const Eigen::Ref<const rowvec_t>& v, 
        Eigen::Ref<rowvec_t> out
    ) const override
    {
        // TODO: might be better to chunk the dot-product into two loops and collapse
        // - independent jobs over columns
        // - dot-product split as in cmul for each column
        //const size_t n_threads_cap = std::min<size_t>(_n_threads, q);
        //const int n_blocks = std::max<int>(n_threads_cap, 1);
        //const int block_size = out.size() / n_blocks;
        //const int remainder = out.size() % n_blocks;
        out.matrix().noalias() = v.matrix() * _mat.block(i, j, p, q);
    }

    void btmul(
        int i, int j, int p, int q, 
        const Eigen::Ref<const rowvec_t>& v, 
        Eigen::Ref<rowvec_t> out
    ) const override
    {
        out.matrix().noalias() = v.matrix() * _mat.block(i, j, p, q).transpose();
    }

    value_t cnormsq(int j) const override
    {
        return _mat.col(j).squaredNorm();
    }

    int rows() const override
    {
        return _mat.rows();
    }
    
    int cols() const override
    {
        return _mat.cols();
    }
};

} // namespace matrix
} // namespace adelie_core