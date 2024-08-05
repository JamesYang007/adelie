#pragma once
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

template <class SparseType, class IndexType=Eigen::Index>
class MatrixNaiveSparse: public MatrixNaiveBase<typename SparseType::Scalar, IndexType>
{
public:
    using base_t = MatrixNaiveBase<typename SparseType::Scalar, IndexType>;
    using sparse_t = SparseType;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using typename base_t::rowmat_value_t;
    using typename base_t::sp_mat_value_t;
    using vec_sp_value_t = vec_value_t;
    using vec_sp_index_t = util::rowvec_type<typename sparse_t::StorageIndex>;

    static_assert(!sparse_t::IsRowMajor, "MatrixNaiveSparse: only column-major allowed!");
    
private:
    const Eigen::Map<const sparse_t> _mat;  // underlying sparse matrix
    const size_t _n_threads;                // number of threads
    vec_value_t _buff;

    ADELIE_CORE_STRONG_INLINE
    value_t _cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v,
        const Eigen::Ref<const vec_value_t>& weights,
        size_t n_threads
    ) 
    {
        const auto outer = _mat.outerIndexPtr()[j];
        const auto size = _mat.outerIndexPtr()[j+1] - outer;
        const Eigen::Map<const vec_sp_index_t> inner(
            _mat.innerIndexPtr() + outer, size
        );
        const Eigen::Map<const vec_sp_value_t> value(
            _mat.valuePtr() + outer, size
        );
        return spddot(inner, value, v * weights, n_threads, _buff);
    }

    ADELIE_CORE_STRONG_INLINE
    void _ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out,
        size_t n_threads
    ) const
    {
        const auto outer = _mat.outerIndexPtr()[j];
        const auto size = _mat.outerIndexPtr()[j+1] - outer;
        const Eigen::Map<const vec_sp_index_t> inner(
            _mat.innerIndexPtr() + outer, size
        );
        const Eigen::Map<const vec_sp_value_t> value(
            _mat.valuePtr() + outer, size
        );
        spaxi(inner, value, v, out, n_threads);
    }
    
public:
    explicit MatrixNaiveSparse(
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
        _buff(n_threads)
    {
        if (n_threads < 1) {
            throw util::adelie_core_error("n_threads must be >= 1.");
        }
    }

    value_t cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v,
        const Eigen::Ref<const vec_value_t>& weights
    ) override
    {
        base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
        return _cmul(j, v, weights, _n_threads);
    }

    void ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_ctmul(j, out.size(), rows(), cols());
        _ctmul(j, v, out, _n_threads);
    }

    void bmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
        for (int k = 0; k < q; ++k) {
            out[k] = _cmul(j+k, v, weights, _n_threads);
        }
    }

    void btmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_btmul(j, q, v.size(), out.size(), rows(), cols());
        for (int k = 0; k < q; ++k) {
            _ctmul(j+k, v[k], out, _n_threads);
        }
    }

    void mul(
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        const auto routine = [&](int k) {
            out[k] = _cmul(k, v, weights, 1);
        };
        if (_n_threads <= 1) {
            for (int k = 0; k < out.size(); ++k) routine(k);
        } else {
            #pragma omp parallel for schedule(static) num_threads(_n_threads)
            for (int k = 0; k < out.size(); ++k) routine(k);
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
        const auto routine = [&](int i1) {
            const auto index_1 = j+i1;
            const auto outer_1 = _mat.outerIndexPtr()[index_1];
            const auto size_1 = _mat.outerIndexPtr()[index_1+1] - outer_1;
            const Eigen::Map<const vec_sp_index_t> inner_1(
                _mat.innerIndexPtr() + outer_1, size_1
            );
            const Eigen::Map<const vec_sp_value_t> value_1(
                _mat.valuePtr() + outer_1, size_1
            );
            for (int i2 = 0; i2 <= i1; ++i2) {
                const auto index_2 = j+i2;
                const auto outer_2 = _mat.outerIndexPtr()[index_2];
                const auto size_2 = _mat.outerIndexPtr()[index_2+1] - outer_2;
                const Eigen::Map<const vec_sp_index_t> inner_2(
                    _mat.innerIndexPtr() + outer_2, size_2
                );
                const Eigen::Map<const vec_sp_value_t> value_2(
                    _mat.valuePtr() + outer_2, size_2
                );

                out(i1, i2) = svsvwdot(
                    inner_1, value_1,
                    inner_2, value_2,
                    sqrt_weights.square()
                );
            }
        };
        if (_n_threads <= 1) {
            for (int i1 = 0; i1 < q; ++i1) routine(i1);
        } else {
            #pragma omp parallel for schedule(static) num_threads(_n_threads)
            for (int i1 = 0; i1 < q; ++i1) routine(i1);
        }
        for (int i1 = 0; i1 < q; ++i1) {
            for (int i2 = i1+1; i2 < q; ++i2) {
                out(i1, i2) = out(i2, i1);
            }
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
        const auto outer = v.outerIndexPtr();
        const auto inner = v.innerIndexPtr();
        const auto value = v.valuePtr();

        const auto routine = [&](auto k) {
            const Eigen::Map<const sp_mat_value_t> vk(
                1,
                v.cols(),
                outer[k+1] - outer[k],
                outer + k,
                inner,
                value
            );
            auto out_k = out.row(k);
            out_k = vk * _mat.transpose();
        };
        if (_n_threads <= 1) {
            for (int k = 0; k < v.outerSize(); ++k) routine(k);
        } else {
            #pragma omp parallel for schedule(static) num_threads(_n_threads)
            for (int k = 0; k < v.outerSize(); ++k) routine(k);
        }
    }
};

} // namespace matrix
} // namespace adelie_core