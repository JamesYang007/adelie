#pragma once
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

template <class DenseType>
class MatrixNaiveOneHotDense: public MatrixNaiveBase<typename DenseType::Scalar>
{
public:
    using base_t = MatrixNaiveBase<typename DenseType::Scalar>;
    using typename base_t::value_t;
    using typename base_t::index_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using typename base_t::rowmat_value_t;
    using typename base_t::sp_mat_value_t;
    using dense_t = DenseType;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;

private:
    const Eigen::Map<const dense_t> _mat;   // (n, d) underlying matrix
    const Eigen::Map<const vec_index_t> _levels;  // (d,) number of levels
    const vec_index_t _outer;               // (d+1,) outer vector
    const size_t _cols;                     // number of columns (p)
    const vec_index_t _slice_map;           // (p,) array mapping to matrix slice
    const vec_index_t _index_map;           // (p,) array mapping to (relative) index of the slice
    const size_t _n_threads;                // number of threads

    static inline auto init_outer(
        const Eigen::Ref<const vec_index_t>& levels
    )
    {
        vec_index_t outer(levels.size() + 1);
        outer[0] = 0;
        for (int i = 0; i < levels.size(); ++i) {
            const auto li = levels[i];
            const auto size = std::max<size_t>(li, 1);
            outer[i+1] = outer[i] + size; 
        }
        return outer;
    }

    static inline auto init_slice_map(
        const Eigen::Ref<const vec_index_t>& levels,
        size_t cols
    )
    {
        vec_index_t slice_map(cols);
        size_t begin = 0;
        for (int i = 0; i < levels.size(); ++i) {
            const auto li = levels[i];
            const auto block_size = std::max<size_t>(li, 1);
            for (int j = 0; j < block_size; ++j) {
                slice_map[begin + j] = i;
            }
            begin += block_size;
        }
        return slice_map;
    }

    static inline auto init_index_map(
        const Eigen::Ref<const vec_index_t>& levels,
        size_t cols
    )
    {
        vec_index_t index_map(cols);
        size_t begin = 0;
        for (int i = 0; i < levels.size(); ++i) {
            const auto li = levels[i];
            const auto block_size = std::max<size_t>(li, 1);
            for (int j = 0; j < block_size; ++j) {
                index_map[begin + j] = j;
            }
            begin += block_size;
        }
        return index_map;
    }

    value_t _cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v,
        const Eigen::Ref<const vec_value_t>& weights
    )
    {
        const auto& w = weights;
        const auto slice = _slice_map[j];
        const auto index = _index_map[j];
        const auto level = std::max<size_t>(_levels[slice], 0);

        switch (level) {
            case 0: {
                return (v * w * _mat.col(slice).transpose().array()).sum();
                break;
            }
            case 1: {
                return (v * w).sum();
                break;
            }
            default: {
                value_t sum = 0;
                for (int i = 0; i < _mat.rows(); ++i) {
                    if (_mat(i, slice) != index) continue;
                    sum += v[i] * w[i];
                }
                return sum;
                break;
            }
        }
    } 

    void _ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out
    ) 
    {
        const auto slice = _slice_map[j];
        const auto index = _index_map[j];
        const auto level = std::max<size_t>(_levels[slice], 0);

        switch (level) {
            case 0: {
                out += v * _mat.col(slice).transpose().array();
                break;
            }
            case 1: {
                out += v;
                break;
            }
            default: {
                for (int i = 0; i < _mat.rows(); ++i) {
                    if (_mat(i, slice) != index) continue;
                    out[i] += v;
                }
                break;
            }
        }
    }

    void _bmul(
        int begin,
        int slice,
        int index,
        int level,
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    )
    {
        const auto size = out.size();
        const auto full_size = std::max<size_t>(level, 1);
        if (index != 0 || size != full_size) {
            for (int l = 0; l < size; ++l) {
                out[l] = _cmul(begin+l, v, weights);
            }
            return;
        }
        const auto& w = weights;
        level = std::max<size_t>(level, 0);
        switch (level) {
            case 0: 
            case 1: {
                out[0] = _cmul(begin, v, weights);
                break;
            }
            default: {
                out.setZero();
                for (int i = 0; i < _mat.rows(); ++i) {
                    const auto val = v[i] * w[i];
                    const int k = _mat(i, slice);
                    out[k] += val;
                }
                break;
            }
        }
    }

    void _btmul(
        int begin,
        int slice,
        int index,
        int level,
        int size,
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    )
    {
        const auto full_size = std::max<size_t>(level, 1);
        if (index != 0 || size != full_size) {
            for (int l = 0; l < size; ++l) {
                _ctmul(begin+l, v[l], out);
            }
            return;
        }
        level = std::max(level, 0);
        switch (level) {
            case 0: {
                out += v[0] * _mat.col(slice).transpose().array();
                break;
            }
            case 1: {
                out += v[0];
                break;
            }
            default: {
                for (int i = 0; i < _mat.rows(); ++i) {
                    const int k = _mat(i, slice);
                    out[i] += v[k];
                }
                break;
            }
        }
    }

public:
    explicit MatrixNaiveOneHotDense(
        const Eigen::Ref<const dense_t>& mat,
        const Eigen::Ref<const vec_index_t>& levels,
        size_t n_threads
    ):
        _mat(mat.data(), mat.rows(), mat.cols()),
        _levels(levels.data(), levels.size()),
        _outer(init_outer(levels)),
        _cols(_outer[_outer.size()-1]),
        _slice_map(init_slice_map(levels, _cols)),
        _index_map(init_index_map(levels, _cols)),
        _n_threads(n_threads)
    {
        const auto d = mat.cols();

        if (levels.size() != d) {
            throw util::adelie_core_error("levels must be of shape (d,) where mat is (n, d).");
        }
        if (n_threads < 1) {
            throw util::adelie_core_error("n_threads must be >= 1.");
        }
    }

    vec_index_t groups() const 
    {
        const size_t G = _outer.size() - 1;
        return _outer.head(G);
    }

    vec_index_t group_sizes() const
    {
        const size_t G = _outer.size() - 1;
        return _outer.tail(G) - _outer.head(G);
    }

    value_t cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v,
        const Eigen::Ref<const vec_value_t>& weights
    ) override
    {
        base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
        return _cmul(j, v, weights);
    }

    void ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_ctmul(j, out.size(), rows(), cols());
        _ctmul(j, v, out);
    }

    void bmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
        int n_processed = 0;
        while (n_processed < q) {
            const auto jj = j + n_processed;
            const auto slice = _slice_map[jj];
            const auto index = _index_map[jj];
            const auto level = _levels[slice];
            const auto full_size = std::max<size_t>(level, 1);
            const auto size = std::min<size_t>(full_size - index, q - n_processed);
            auto out_curr = out.segment(n_processed, size);
            _bmul(jj, slice, index, level, v, weights, out_curr);
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
        int n_processed = 0;
        while (n_processed < q) {
            const auto jj = j + n_processed;
            const auto slice = _slice_map[jj];
            const auto index = _index_map[jj];
            const auto level = _levels[slice];
            const auto full_size = std::max<size_t>(level, 1);
            const auto size = std::min<size_t>(full_size - index, q - n_processed);
            const auto v_curr = v.segment(n_processed, size);
            _btmul(jj, slice, index, level, size, v_curr, out);
            n_processed += size;
        }
    }

    void mul(
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        const auto routine = [&](auto g) {
            const auto j = _outer[g];
            const auto level = _levels[g];
            const auto full_size = std::max<size_t>(level, 1);
            auto out_curr = out.segment(j, full_size);
            _bmul(j, g, 0, level, v, weights, out_curr);
        };
        if (_n_threads <= 1) {
            for (int g = 0; g < _mat.cols(); ++g) routine(g);
        } else {
            #pragma omp parallel for schedule(static) num_threads(_n_threads)
            for (int g = 0; g < _mat.cols(); ++g) routine(g);
        }
    }

    int rows() const override
    {
        return _mat.rows();
    }
    
    int cols() const override
    {
        return _cols;
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
    }

    void sp_btmul(
        const sp_mat_value_t& v, 
        Eigen::Ref<rowmat_value_t> out
    ) override
    {
        base_t::check_sp_btmul(
            v.rows(), v.cols(), out.rows(), out.cols(), rows(), cols()
        );
    }
};

} // namespace matrix
} // namespace adelie_core 