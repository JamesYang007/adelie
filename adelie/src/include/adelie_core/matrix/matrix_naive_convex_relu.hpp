#pragma once
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

template <class DenseType, 
          class MaskType,
          class IndexType=Eigen::Index>
class MatrixNaiveConvexReluDense: public MatrixNaiveBase<typename DenseType::Scalar, IndexType>
{
public:
    using base_t = MatrixNaiveBase<typename DenseType::Scalar, IndexType>;
    using typename base_t::value_t;
    using typename base_t::index_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using typename base_t::rowmat_value_t;
    using typename base_t::sp_mat_value_t;
    using dense_t = DenseType;
    using mask_t = MaskType;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;

private:
    const Eigen::Map<const dense_t> _mat;      // (n, d) underlying matrix
    const Eigen::Map<const mask_t> _mask;      // (n, m) mask matrix
    const size_t _n_threads;
    vec_value_t _buff;

    ADELIE_CORE_STRONG_INLINE
    void _ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out,
        size_t n_threads
    ) 
    {
        const auto d = _mat.cols();
        const auto m = _mask.cols();
        const auto j_sgn = j / (m * d);
        j -= j_sgn * m * d;
        const auto j_m = j / d;
        j -= j_m * d;
        const auto j_d = j;
        dvaddi(
            out, 
            (v * (1-2*j_sgn)) * _mat.col(j_d).cwiseProduct(
                _mask.col(j_m).template cast<value_t>()
            ).array(),
            n_threads
        );
    }

public:
    explicit MatrixNaiveConvexReluDense(
        const Eigen::Ref<const dense_t>& mat,
        const Eigen::Ref<const mask_t>& mask,
        size_t n_threads
    ):
        _mat(mat.data(), mat.rows(), mat.cols()),
        _mask(mask.data(), mask.rows(), mask.cols()),
        _n_threads(n_threads),
        _buff(n_threads * std::min<size_t>(mat.rows(), mat.cols()) + mat.rows())
    {
        const auto n = mat.rows();
        const auto d = mat.cols();
        const auto m = mask.cols();

        if (mask.rows() != n) {
            throw util::adelie_core_error("mask must be (n, m) where mat is (n, d).");
        }
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
        const auto d = _mat.cols();
        const auto m = _mask.cols();
        const auto j_sgn = j / (m * d);
        j -= j_sgn * m * d;
        const auto j_m = j / d;
        j -= j_m * d;
        const auto j_d = j;
        return (1-2*j_sgn) * ddot(
            _mat.col(j_d).cwiseProduct(_mask.col(j_m).template cast<value_t>()),
            (v * weights).matrix(),
            _n_threads,
            _buff
        );
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
        const auto d = _mat.cols();
        const auto m = _mask.cols();
        int n_processed = 0;
        while (n_processed < q) {
            auto k = j + n_processed;
            const auto k_sgn = k / (m * d);
            k -= k_sgn * m * d;
            const auto k_m = k / d;
            k -= k_m * d;
            const auto k_d = k;
            const auto size = std::min<int>(d-k_d, q-n_processed);
            auto out_m = out.segment(n_processed, size).matrix();
            Eigen::Map<rowmat_value_t> buff(_buff.data(), _n_threads, d);
            dgemv(
                _mat.middleCols(k_d, size),
                (1-2*k_sgn) * _mask.col(k_m).transpose().template cast<value_t>().cwiseProduct((v * weights).matrix()),
                _n_threads,
                buff,
                out_m
            );
            n_processed += size;
        }
    }

    void btmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_btmul(j, q, v.size(), out.size(), rows(), cols());
        const auto n = _mat.rows();
        const auto d = _mat.cols();
        const auto m = _mask.cols();
        int n_processed = 0;
        while (n_processed < q) {
            auto k = j + n_processed;
            const auto k_sgn = k / (m * d);
            k -= k_sgn * m * d;
            const auto k_m = k / d;
            k -= k_m * d;
            const auto k_d = k;
            const auto size = std::min<int>(d-k_d, q-n_processed);
            Eigen::Map<vec_value_t> Xv(_buff.data(), n);
            Eigen::Map<rowmat_value_t> buff(_buff.data() + n, _n_threads, n);
            auto Xv_m = Xv.matrix();
            dgemv(
                _mat.middleCols(k_d, size).transpose(),
                v.segment(n_processed, size).matrix(),
                _n_threads,
                buff,
                Xv_m
            );
            dvaddi(
                out, 
                (1-2*k_sgn) * Xv * _mask.col(k_m).transpose().template cast<value_t>().array(), 
                _n_threads
            );
            n_processed += size;
        }
    }

    void mul(
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        const auto d = _mat.cols();
        const auto m = _mask.cols();
        const auto routine = [&](auto i) {
            const auto i_sgn = i / m;
            const auto i_m = i - i_sgn * m;
            auto out_m = out.segment(i * d, d).matrix();
            Eigen::Map<rowmat_value_t> buff(_buff.data(), _n_threads, d);
            dgemv(
                _mat,
                (1-2*i_sgn) * _mask.col(i_m).transpose().template cast<value_t>().cwiseProduct((v * weights).matrix()),
                1,
                buff /* unused */,
                out_m
            );
        };
        if (_n_threads <= 1) {
            for (int i = 0; i < 2 * m; ++i) routine(i);
        } else {
            #pragma omp parallel for schedule(static) num_threads(_n_threads)
            for (int i = 0; i < 2 * m; ++i) routine(i);
        }
    }

    int rows() const override
    {
        return _mat.rows();
    }
    
    int cols() const override
    {
        return _mat.cols() * _mask.cols() * 2;
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

        const auto d = _mat.cols();
        const auto m = _mask.cols();

        Eigen::setNbThreads(_n_threads);

        int n_processed = 0;
        while (n_processed < q) {
            auto k = j + n_processed;
            const auto k_sgn = k / (m * d);
            k -= k_sgn * m * d;
            const auto k_m = k / d;
            k -= k_m * d;
            const auto k_d = k;
            const auto size = std::min<int>(d-k_d, q-n_processed);
            const auto mat = _mat.middleCols(k_d, size);
            const auto mask = _mask.col(k_m);

            auto curr_block = buffer.middleCols(n_processed, size).array();
            curr_block.array() = (1-2*k_sgn) * (
                mat.array().colwise() * 
                mask.template cast<value_t>().cwiseProduct(sqrt_weights.matrix().transpose()).array()
            );
            n_processed += size;
        }

        out = buffer.transpose() * buffer;
    }

    void sp_btmul(
        const sp_mat_value_t& v, 
        Eigen::Ref<rowmat_value_t> out
    ) override
    {
        base_t::check_sp_btmul(
            v.rows(), v.cols(), out.rows(), out.cols(), rows(), cols()
        );
        const auto routine = [&](int k) {
            typename sp_mat_value_t::InnerIterator it(v, k);
            auto out_k = out.row(k);
            out_k.setZero();
            for (; it; ++it) {
                _ctmul(it.index(), it.value(), out_k, 1);
            }
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