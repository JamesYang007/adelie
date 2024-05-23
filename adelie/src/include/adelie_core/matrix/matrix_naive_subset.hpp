#pragma once
#include <adelie_core/matrix/matrix_naive_base.hpp>

namespace adelie_core {
namespace matrix {

template <class ValueType>
class MatrixNaiveCSubset: public MatrixNaiveBase<ValueType>
{
public:
    using base_t = MatrixNaiveBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::index_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using typename base_t::rowmat_value_t;
    using typename base_t::sp_mat_value_t;
    using dyn_vec_index_t = std::vector<index_t>;
    using map_cvec_index_t = Eigen::Map<const vec_index_t>;
    
private:
    base_t* _mat;               // underlying matrix
    const map_cvec_index_t _subset;  // column subset
    const std::tuple<
        vec_index_t,
        dyn_vec_index_t
    > _subset_cinfo;            // 1) number of elements left
                                // in the contiguous sub-chunk
                                // starting at _subset[i]
                                // 2) beginning index to each
                                // contiguous sub-chunk
    const size_t _n_threads;

    static auto init_subset_cinfo(
        const Eigen::Ref<const vec_index_t>& subset
    )
    {
        if (subset.size() == 0) {
            throw util::adelie_core_error(
                "subset must be non-empty."
            );
        }

        vec_index_t subset_csize(subset.size());
        dyn_vec_index_t subset_cbegin;
        subset_cbegin.reserve(subset.size());

        size_t count = 1;
        size_t begin = 0;
        for (size_t i = 1; i < subset.size(); ++i) {
            if (subset[i] == subset[i-1] + 1) {
                ++count;
                continue;
            }
            for (size_t j = 0; j < count; ++j) {
                subset_csize[begin+j] = count - j;
            }
            subset_cbegin.push_back(begin);
            begin += count;
            count = 1;
        }
        if (begin != subset.size()) {
            for (size_t j = 0; j < count; ++j) {
                subset_csize[begin+j] = count - j;
            }
            subset_cbegin.push_back(begin);
        }
        return std::make_tuple(subset_csize, subset_cbegin);
    }

public:
    explicit MatrixNaiveCSubset(
        base_t* mat,
        const Eigen::Ref<const vec_index_t>& subset,
        size_t n_threads
    ): 
        _mat(mat),
        _subset(subset.data(), subset.size()),
        _subset_cinfo(init_subset_cinfo(subset)),
        _n_threads(n_threads)
    {
        if ((subset.minCoeff() < 0) || (subset.maxCoeff() >= mat->cols())) {
            throw util::adelie_core_error(
                "subset must contain unique values in the range [0, p) "
                "where p is the number of columns."
            ) ;
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
        return _mat->cmul(_subset[j], v, weights);
    }

    void ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_ctmul(j, out.size(), rows(), cols());
        _mat->ctmul(_subset[j], v, out);
    }

    void bmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
        const auto& _subset_csize = std::get<0>(_subset_cinfo);
        size_t n_processed = 0;
        while (n_processed < q) {
            const auto k = j + n_processed;
            const auto size = std::min<size_t>(_subset_csize[k], q-n_processed);
            if (size == 1) {
                out[n_processed] = _mat->cmul(_subset[k], v, weights);
            } else {
                auto curr_out = out.segment(n_processed, size);
                _mat->bmul(_subset[k], size, v, weights, curr_out);
            }
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
        const auto& _subset_csize = std::get<0>(_subset_cinfo);
        size_t n_processed = 0;
        while (n_processed < q) {
            const auto k = j + n_processed;
            const auto size = std::min<size_t>(_subset_csize[k], q-n_processed);
            if (size == 1) {
                _mat->ctmul(_subset[k], v[n_processed], out);
            } else {
                const auto curr_v = v.segment(n_processed, size);
                _mat->btmul(_subset[k], size, curr_v, out);
            }
            n_processed += size;
        }
    }

    void mul(
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        const auto& _subset_csize = std::get<0>(_subset_cinfo);
        const auto& _subset_cbegin = std::get<1>(_subset_cinfo);

        const auto routine = [&](auto t) {
            const auto subset_idx = _subset_cbegin[t];
            const auto j = _subset[subset_idx];
            const auto q = _subset_csize[subset_idx];
            auto curr_out = out.segment(subset_idx, q);
            _mat->bmul(j, q, v, weights, curr_out);
        };
        if (_n_threads <= 1) {
            for (int t = 0; t < _subset_cbegin.size(); ++t) routine(t);
        } else {
            #pragma omp parallel for schedule(static) num_threads(_n_threads)
            for (int t = 0; t < _subset_cbegin.size(); ++t) routine(t);
        }
    }

    int rows() const override
    {
        return _mat->rows();
    }
    
    int cols() const override
    {
        return _subset.size();
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
        const auto& _subset_csize = std::get<0>(_subset_cinfo);
        if (_subset_csize[j] < q) {
            throw util::adelie_core_error(
                "MatrixNaiveCSubset::cov() is not implemented when "
                "subset[j:j+q] is not contiguous. "
            );
        }
        _mat->cov(_subset[j], q, sqrt_weights, out, buffer);
    }

    void sp_btmul(
        const sp_mat_value_t& v, 
        Eigen::Ref<rowmat_value_t> out
    ) override
    {
        base_t::check_sp_btmul(
            v.rows(), v.cols(), out.rows(), out.cols(), rows(), cols()
        );

        const auto routine = [&](auto k) {
            typename sp_mat_value_t::InnerIterator it(v, k);
            auto out_k = out.row(k);
            out_k.setZero();
            for (; it; ++it) {
                _mat->ctmul(_subset[it.index()], it.value(), out_k);
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

template <class ValueType>
class MatrixNaiveRSubset: public MatrixNaiveBase<ValueType>
{
public:
    using base_t = MatrixNaiveBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using typename base_t::rowmat_value_t;
    using typename base_t::sp_mat_value_t;
    using vec_bool_t = util::rowvec_type<bool>;
    using map_cvec_index_t = Eigen::Map<const vec_index_t>;
    
private:
    base_t* _mat;               // underlying matrix
    const map_cvec_index_t _subset;
    const vec_value_t _mask;  
    const size_t _n_threads;
    vec_value_t _buffer;
    vec_value_t _cov_buffer;

    static auto init_mask(
        size_t n,
        const Eigen::Ref<const vec_index_t>& subset
    )
    {
        if (subset.size() == 0) {
            throw util::adelie_core_error(
                "subset must be non-empty."
            );
        }

        vec_value_t mask(n);
        mask.setZero();
        for (int i = 0; i < subset.size(); ++i) {
            mask[subset[i]] = true;
        } 
        return mask;
    }
    
public:
    explicit MatrixNaiveRSubset(
        base_t* mat,
        const Eigen::Ref<const vec_index_t>& subset,
        size_t n_threads
    ): 
        _mat(mat),
        _subset(subset.data(), subset.size()),
        _mask(init_mask(mat->rows(), subset)),
        _n_threads(n_threads),
        _buffer(mat->rows())
    {
        if ((subset.minCoeff() < 0) || (subset.maxCoeff() >= mat->rows())) {
            throw util::adelie_core_error(
                "subset must contain unique values in the range [0, n) "
                "where n is the number of rows."
            ) ;
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
        _buffer.setZero();
        for (int i = 0; i < _subset.size(); ++i) {
            _buffer[_subset[i]] = v[i] * weights[i];
        }
        return _mat->cmul(j, _mask, _buffer);
    }

    void ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_ctmul(j, out.size(), rows(), cols());
        _buffer.setZero();
        _mat->ctmul(j, v, _buffer);
        for (int i = 0; i < _subset.size(); ++i) {
            out[i] += _buffer[_subset[i]];
        }
    }

    void bmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
        _buffer.setZero();
        for (int i = 0; i < _subset.size(); ++i) {
            _buffer[_subset[i]] = v[i] * weights[i];
        }
        _mat->bmul(j, q, _mask, _buffer, out);
    }

    void btmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_btmul(j, q, v.size(), out.size(), rows(), cols());
        _buffer.setZero();
        _mat->btmul(j, q, v, _buffer);
        for (int i = 0; i < _subset.size(); ++i) {
            out[i] += _buffer[_subset[i]];
        }
    }

    void mul(
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        _buffer.setZero();
        for (int i = 0; i < _subset.size(); ++i) {
            _buffer[_subset[i]] = v[i] * weights[i];
        }
        _mat->mul(_mask, _buffer, out);
    }

    int rows() const override
    {
        return _subset.size();
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
        _buffer.setZero();
        for (int i = 0; i < _subset.size(); ++i) {
            _buffer[_subset[i]] = sqrt_weights[i];
        }
        if (_cov_buffer.size() < _mat->rows() * q) {
            _cov_buffer.resize(_mat->rows() * q);
        }
        Eigen::Map<colmat_value_t> cov_buffer(
            _cov_buffer.data(),
            _mat->rows(),
            q
        );
        _mat->cov(j, q, _buffer, out, cov_buffer);
    }

    void sp_btmul(
        const sp_mat_value_t& v, 
        Eigen::Ref<rowmat_value_t> out
    ) override
    {
        base_t::check_sp_btmul(
            v.rows(), v.cols(), out.rows(), out.cols(), rows(), cols()
        );
        rowmat_value_t _out(out.rows(), _mat->rows());
        _mat->sp_btmul(v, _out);
        for (int i = 0; i < _subset.size(); ++i) {
            out.col(i) = _out.col(_subset[i]);
        }
    }
};

} // namespace matrix
} // namespace adelie_core