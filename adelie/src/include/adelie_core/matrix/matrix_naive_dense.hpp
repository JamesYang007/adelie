#pragma once
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

template <class DenseType>
class MatrixNaiveDense: public MatrixNaiveBase<typename DenseType::Scalar>
{
public:
    using base_t = MatrixNaiveBase<typename DenseType::Scalar>;
    using dense_t = DenseType;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    
private:
    const Eigen::Map<const dense_t> _mat;   // underlying dense matrix
    const size_t _n_threads;                // number of threads
    util::rowmat_type<value_t> _buff;
    
public:
    MatrixNaiveDense(
        const Eigen::Ref<const dense_t>& mat,
        size_t n_threads
    ): 
        _mat(mat.data(), mat.rows(), mat.cols()),
        _n_threads(n_threads),
        _buff(_n_threads, std::min(mat.rows(), mat.cols()))
    {}
    
    value_t cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v
    ) const override
    {
        return ddot(_mat.col(j).matrix(), v.matrix(), _n_threads);
    }

    void ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out
    ) const override
    {
        dax(v, _mat.col(j), _n_threads, out);
    }

    void bmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        auto outm = out.matrix();
        dgemv(
            _mat.middleCols(j, q),
            v.matrix(),
            _n_threads,
            _buff,
            outm
        );
    }

    void btmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        auto outm = out.matrix();
        dgemv(
            _mat.middleCols(j, q).transpose(),
            v.matrix(),
            _n_threads,
            _buff,
            outm
        );
    }

    void to_dense(
        int j, int q,
        Eigen::Ref<colmat_value_t> out
    ) const override
    {
        dmmeq(
            out,
            _mat.middleCols(j, q),
            _n_threads
        );
    }

    void means(
        Eigen::Ref<vec_value_t> out
    ) const override
    {
        const size_t p = _mat.cols();
        const int n_blocks = std::min<int>(_n_threads, p);
        const int block_size = p / n_blocks;
        const int remainder = p % n_blocks;
        #pragma omp parallel for schedule(static) num_threads(n_blocks)
        for (int t = 0; t < n_blocks; ++t) {
            const auto begin = (
                std::min<int>(t, remainder) * (block_size + 1) 
                + std::max<int>(t-remainder, 0) * block_size
            );
            const auto size = block_size + (t < remainder);
            out.segment(begin, size) = _mat.middleCols(begin, size).colwise().mean();
        }
    }

    void group_norms(
        const Eigen::Ref<const vec_index_t>& groups,
        const Eigen::Ref<const vec_index_t>& group_sizes,
        const Eigen::Ref<const vec_value_t>& means,
        bool center,
        Eigen::Ref<vec_value_t> out
    ) const override
    {
        const auto n = _mat.rows();
        const auto G = groups.size();
        const auto n_threads_capped = std::min<int>(_n_threads, G);
        #pragma omp parallel for schedule(static) num_threads(n_threads_capped)
        for (int i = 0; i < G; ++i) {
            const auto g = groups[i];
            const auto gs = group_sizes[i];
            auto Xi_fro = _mat.middleCols(g, gs).squaredNorm();
            if (center) Xi_fro -= n * means.segment(g, gs).matrix().squaredNorm();
            out[i] = std::sqrt(Xi_fro);
        }
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