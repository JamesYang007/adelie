#pragma once
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

template <class ValueType, class IndexType=Eigen::Index>
class MatrixNaiveStandardize: public MatrixNaiveBase<ValueType, IndexType>
{
public:
    using base_t = MatrixNaiveBase<ValueType, IndexType>;
    using typename base_t::value_t;
    using typename base_t::index_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using typename base_t::rowmat_value_t;
    using typename base_t::sp_mat_value_t;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;

private:
    base_t* _mat;
    const map_cvec_value_t _centers;
    const map_cvec_value_t _scales;
    const size_t _n_threads;
    vec_value_t _buff;

public:
    explicit MatrixNaiveStandardize(
        base_t& mat,
        const Eigen::Ref<const vec_value_t>& centers,
        const Eigen::Ref<const vec_value_t>& scales,
        size_t n_threads
    ):
        _mat(&mat),
        _centers(centers.data(), centers.size()),
        _scales(scales.data(), scales.size()),
        _n_threads(n_threads),
        _buff(std::max<size_t>(mat.cols(), n_threads))
    {
        const auto p = mat.cols();

        if (centers.size() != p) {
            throw util::adelie_core_error(
                "centers must have the same length as the number of columns of mat."
            );
        }
        if (scales.size() != p) {
            throw util::adelie_core_error(
                "scales must have the same length as the number of columns of mat."
            );
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
        const auto c = _centers[j];
        const auto vw_sum = (
            (c == 0) ? 0 : ddot(v.matrix(), weights.matrix(), _n_threads, _buff)
        );
        return (_mat->cmul(j, v, weights) - c * vw_sum) / _scales[j];
    }

    void ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_ctmul(j, out.size(), rows(), cols());
        const auto vs = v / _scales[j];
        _mat->ctmul(j, vs, out);
        const auto vsc = _centers[j] * vs;
        if (!vsc) return;
        dvsubi(
            out, 
            vec_value_t::NullaryExpr(out.size(), [&](auto i) {
                return vsc;
            }),
            _n_threads
        );
    }

    void bmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
        _mat->bmul(j, q, v, weights, out);
        const auto c = _centers.segment(j, q);
        const auto vw_sum = (
            (c == 0).all() ? 0 : ddot(v.matrix(), weights.matrix(), _n_threads, _buff)
        );
        const auto s = _scales.segment(j, q);
        dvveq(out, (out - vw_sum * c) / s, _n_threads);
    }

    void btmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_btmul(j, q, v.size(), out.size(), rows(), cols());
        auto vs = _buff.segment(0, q);
        const auto s = _scales.segment(j, q);
        dvveq(vs, v / s, _n_threads);
        _mat->btmul(j, q, vs, out);

        const auto vsc = ddot(
            _centers.segment(j, q).matrix(),
            vs.matrix(),
            _n_threads,
            _buff
        );
        if (!vsc) return;
        dvsubi(
            out, 
            vec_value_t::NullaryExpr(out.size(), [&](auto) { return vsc; }),
            _n_threads
        );
    }

    void mul(
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        _mat->mul(v, weights, out);
        const auto vw_sum = ddot(v.matrix(), weights.matrix(), _n_threads, _buff);
        dvveq(out, (out - vw_sum * _centers) / _scales, _n_threads);
    }

    int rows() const override
    {
        return _mat->rows();
    }
    
    int cols() const override
    {
        return _mat->cols();
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

        _mat->cov(j, q, sqrt_weights, out, buffer);

        const auto centers = _centers.segment(j, q);
        const auto scales = _scales.segment(j, q);

        if ((centers != 0).any()) {
            auto out_lower = out.template selfadjointView<Eigen::Lower>();
            auto means = _buff.segment(j, q);
            _mat->bmul(j, q, sqrt_weights, sqrt_weights, means);
            out_lower.rankUpdate(centers.matrix().transpose(), means.matrix().transpose(), -1);
            out_lower.rankUpdate(centers.matrix().transpose(), sqrt_weights.square().sum());
            out.template triangularView<Eigen::Upper>() = out.transpose();
        }

        out.array().rowwise() /= scales;
        out.array().colwise() /= scales.matrix().transpose().array();
    }

    void sp_btmul(
        const sp_mat_value_t& v, 
        Eigen::Ref<rowmat_value_t> out
    ) override
    {
        base_t::check_sp_btmul(
            v.rows(), v.cols(), out.rows(), out.cols(), rows(), cols()
        );
        sp_mat_value_t vs = v;
        if (!vs.isCompressed()) vs.makeCompressed();
        for (int k = 0; k < vs.outerSize(); ++k) {
            const auto outer = vs.outerIndexPtr();
            const auto inner = vs.innerIndexPtr() + outer[k];
            const auto value = vs.valuePtr() + outer[k];
            const auto size = outer[k+1] - outer[k];
            for (int i = 0; i < size; ++i) {
                value[i] /= _scales[inner[i]];
            }
        }
        _mat->sp_btmul(vs, out);

        const auto routine = [&](auto k) {
            typename sp_mat_value_t::InnerIterator it(vs, k);
            auto out_k = out.row(k);
            value_t vsc = 0;
            for (; it; ++it) {
                const auto idx = it.index();
                vsc += it.value() * _centers[idx];
            }
            if (vsc) out_k.array() -= vsc;
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