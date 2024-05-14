#pragma once
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

template <class DenseType>
class MatrixNaiveInteractionDense: public MatrixNaiveBase<typename DenseType::Scalar>
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
    using rowarr_index_t = util::rowarr_type<index_t>;

private:
    static constexpr size_t _n_levels_cont = 2;

    const Eigen::Map<const dense_t> _mat;   // (n, d) underlying matrix
    const Eigen::Map<const rowarr_index_t> _pairs;  // (G, 2) pair matrix
    const Eigen::Map<const vec_index_t> _levels;  // (d,) number of levels
    const vec_index_t _outer;               // (G+1,) outer vector
    const size_t _cols;                     // number of columns (p)
    const vec_value_t _centers;             // (p,) centers
    const vec_value_t _scales;              // (p,) scales
    const vec_index_t _slice_map;           // (p,) array mapping to matrix slice
    const vec_index_t _index_map;           // (p,) array mapping to (relative) index of the slice
    const size_t _n_threads;                // number of threads

    static inline auto init_outer(
        const Eigen::Ref<const rowarr_index_t>& pairs,
        const Eigen::Ref<const vec_index_t>& levels
    )
    {
        vec_index_t outer(pairs.rows() + 1);
        outer[0] = 0;
        for (int i = 0; i < pairs.rows(); ++i) {
            auto l0 = levels[pairs(i, 0)];
            auto l1 = levels[pairs(i, 1)];
            const auto both_cont = (l0 <= 0) & (l1 <= 0);
            l0 = (l0 <= 0) ? _n_levels_cont : l0;
            l1 = (l1 <= 0) ? _n_levels_cont : l1;
            outer[i+1] = outer[i] + (l0 * l1 - both_cont); 
        }
        return outer;
    }

    static inline vec_value_t init_centers(
        const Eigen::Ref<const dense_t> mat,
        const Eigen::Ref<const rowarr_index_t> pairs,
        const Eigen::Ref<const vec_index_t> levels,
        const Eigen::Ref<const vec_value_t>& centers_map,
        size_t cols 
    )
    {
        if (centers_map.size()) {
            if (centers_map.size() != cols) {
                throw util::adelie_core_error(
                    "If centers is provided by the user, it must be of length p."
                );
            }
            return centers_map;
        }

        vec_value_t centers(cols);
        size_t pos = 0;
        for (int i = 0; i < pairs.rows(); ++i) {
            const auto i0 = pairs(i, 0);
            const auto i1 = pairs(i, 1);
            auto l0 = levels[i0];
            auto l1 = levels[i1];
            const auto both_cont = (l0 <= 0) & (l1 <= 0);
            l0 = (l0 <= 0) ? _n_levels_cont : l0;
            l1 = (l1 <= 0) ? _n_levels_cont : l1;
            const auto size = l0 * l1 - both_cont;
            if (both_cont) {
                centers.segment(pos, size-1).setZero();
                centers[pos+size-1] = (mat.col(i0).array() * mat.col(i1).array()).mean();
            } else {
                centers.segment(pos, size).setZero();
            }
            pos += size;
        }
        return centers;
    }

    static inline vec_value_t init_scales(
        const Eigen::Ref<const dense_t> mat,
        const Eigen::Ref<const rowarr_index_t> pairs,
        const Eigen::Ref<const vec_index_t> levels,
        const Eigen::Ref<const vec_value_t>& centers,
        const Eigen::Ref<const vec_value_t>& scales_map,
        size_t cols 
    )
    {
        if (scales_map.size()) {
            if (scales_map.size() != cols) {
                throw util::adelie_core_error(
                    "If scales is provided by the user, it must be of length p."
                );
            }
            return scales_map;
        }

        vec_value_t scales(cols);
        size_t pos = 0;
        for (int i = 0; i < pairs.rows(); ++i) {
            const auto i0 = pairs(i, 0);
            const auto i1 = pairs(i, 1);
            auto l0 = levels[i0];
            auto l1 = levels[i1];
            const auto both_cont = (l0 <= 0) & (l1 <= 0);
            l0 = (l0 <= 0) ? _n_levels_cont : l0;
            l1 = (l1 <= 0) ? _n_levels_cont : l1;
            const auto size = l0 * l1 - both_cont;
            if (both_cont) {
                scales.segment(pos, size-1).setOnes();
                const auto c = centers[pos+size-1];
                const auto x = mat.col(i0).array() * mat.col(i1).array();
                const auto m = x.mean();
                const auto n = mat.rows();
                scales[pos+size-1] = std::sqrt(
                    x.square().mean() - c * (2 * m - c)
                );
            } else {
                scales.segment(pos, size).setOnes();
            }
            pos += size;
        }
        return scales;
    }

    static inline auto init_slice_map(
        const Eigen::Ref<const rowarr_index_t>& pairs,
        const Eigen::Ref<const vec_index_t>& levels,
        size_t cols
    )
    {
        vec_index_t slice_map(cols);
        size_t begin = 0;
        for (int i = 0; i < pairs.rows(); ++i) {
            auto l0 = levels[pairs(i, 0)];
            auto l1 = levels[pairs(i, 1)];
            const auto both_cont = (l0 <= 0) & (l1 <= 0);
            l0 = (l0 <= 0) ? _n_levels_cont : l0;
            l1 = (l1 <= 0) ? _n_levels_cont : l1;
            const auto block_size = l0 * l1 - both_cont;
            for (int j = 0; j < block_size; ++j) {
                slice_map[begin + j] = i;
            }
            begin += block_size;
        }
        return slice_map;
    }

    static inline auto init_index_map(
        const Eigen::Ref<const rowarr_index_t>& pairs,
        const Eigen::Ref<const vec_index_t>& levels,
        size_t cols
    )
    {
        vec_index_t index_map(cols);
        size_t begin = 0;
        for (int i = 0; i < pairs.rows(); ++i) {
            auto l0 = levels[pairs(i, 0)];
            auto l1 = levels[pairs(i, 1)];
            const auto both_cont = (l0 <= 0) & (l1 <= 0);
            l0 = (l0 <= 0) ? _n_levels_cont : l0;
            l1 = (l1 <= 0) ? _n_levels_cont : l1;
            const auto block_size = l0 * l1 - both_cont;
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
        const auto pair = _pairs.row(slice);
        const auto i0 = pair[0];
        const auto i1 = pair[1];
        const auto l0 = _levels[i0];
        const auto l1 = _levels[i1];
        const auto l0_exp = l0 <= 0 ? _n_levels_cont : l0;
        const auto k1 = index / l0_exp;
        const auto k0 = index - l0_exp * k1;
        const auto _case = static_cast<int>(l0 > 0) | static_cast<int>(l1 > 0 ? _n_levels_cont : 0);
        value_t dot = 0;
        switch (_case) {
            case 0: {
                switch (index) {
                    case 0: {
                        dot = (v * w * _mat.col(i0).transpose().array()).sum();
                        break;
                    }
                    case 1: {
                        dot = (v * w * _mat.col(i1).transpose().array()).sum();
                        break;
                    }
                    case 2: {
                        dot = (v * w * _mat.col(i0).transpose().array() * _mat.col(i1).transpose().array()).sum();
                        break;
                    }
                }
                break;
            }
            case 1: {
                if (k1 == 0) {
                    value_t sum = 0;
                    for (int i = 0; i < _mat.rows(); ++i) {
                        if (_mat(i, i0) != k0) continue;
                        sum += v[i] * w[i];
                    }
                    dot = sum;
                } else {
                    value_t sum = 0;
                    for (int i = 0; i < _mat.rows(); ++i) {
                        if (_mat(i, i0) != k0) continue;
                        sum += v[i] * w[i] * _mat(i, i1);
                    }
                    dot = sum;
                }
                break;
            }
            case 2: {
                if (k0 == 0) {
                    value_t sum = 0;
                    for (int i = 0; i < _mat.rows(); ++i) {
                        if (_mat(i, i1) != k1) continue;
                        sum += v[i] * w[i];
                    }
                    dot = sum;
                } else {
                    value_t sum = 0;
                    for (int i = 0; i < _mat.rows(); ++i) {
                        if (_mat(i, i1) != k1) continue;
                        sum += v[i] * w[i] * _mat(i, i0);
                    }
                    dot = sum;
                }
                break;
            }
            case 3: {
                value_t sum = 0;
                for (int i = 0; i < _mat.rows(); ++i) {
                    if (_mat(i, i0) != k0 || _mat(i, i1) != k1) continue;
                    sum += v[i] * w[i];
                }
                dot = sum;
                break;
            }
        }
        const auto center_term = (_centers[j] == 0) ? 0 : (_centers[j] * (v * w).sum());
        const auto scale = _scales[j];
        return (dot - center_term) / scale;
    }

    void _ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out
    )
    {
        const auto slice = _slice_map[j];
        const auto index = _index_map[j];
        const auto pair = _pairs.row(slice);
        const auto i0 = pair[0];
        const auto i1 = pair[1];
        const auto l0 = _levels[i0];
        const auto l1 = _levels[i1];
        const auto l0_exp = l0 <= 0 ? _n_levels_cont : l0;
        const auto k1 = index / l0_exp;
        const auto k0 = index - l0_exp * k1;
        const auto _case = static_cast<int>(l0 > 0) | static_cast<int>(l1 > 0 ? _n_levels_cont : 0);
        const auto center = _centers[j];
        const auto scale = _scales[j];
        const auto vs = v / scale;
        switch (_case) {
            case 0: {
                switch (index) {
                    case 0: {
                        out += vs * (_mat.col(i0).transpose().array() - center);
                        break;
                    }
                    case 1: {
                        out += vs * (_mat.col(i1).transpose().array() - center);
                        break;
                    }
                    case 2: {
                        out += vs * (_mat.col(i0).transpose().array() * _mat.col(i1).transpose().array() - center);
                        break;
                    }
                }
                break;
            }
            case 1: {
                if (k1 == 0) {
                    for (int i = 0; i < _mat.rows(); ++i) {
                        if (_mat(i, i0) != k0) continue;
                        out[i] += vs;
                    }
                } else {
                    for (int i = 0; i < _mat.rows(); ++i) {
                        if (_mat(i, i0) != k0) continue;
                        out[i] += vs * _mat(i, i1);
                    }
                }
                const auto vsc = vs * center;
                if (vsc) out -= vsc;
                break;
            }
            case 2: {
                if (k0 == 0) {
                    for (int i = 0; i < _mat.rows(); ++i) {
                        if (_mat(i, i1) != k1) continue;
                        out[i] += vs;
                    }
                } else {
                    for (int i = 0; i < _mat.rows(); ++i) {
                        if (_mat(i, i1) != k1) continue;
                        out[i] += vs * _mat(i, i0);
                    }
                }
                const auto vsc = vs * center;
                if (vsc) out -= vsc;
                break;
            }
            case 3: {
                for (int i = 0; i < _mat.rows(); ++i) {
                    if (_mat(i, i0) != k0 || _mat(i, i1) != k1) continue;
                    out[i] += vs;
                }
                const auto vsc = vs * center;
                if (vsc) out -= vsc;
                break;
            }
        }
    }

    template <bool do_standardize>
    void _bmul(
        int begin,
        int i0, int i1,
        int l0, int l1,
        int index,
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    )
    {
        const auto size = out.size();
        const auto both_cont = (l0 <= 0) & (l1 <= 0);
        const auto l0_exp = l0 <= 0 ? _n_levels_cont : l0;
        const auto l1_exp = l1 <= 0 ? _n_levels_cont : l1;
        const auto full_size = l0_exp * l1_exp - both_cont;
        // not a full-block
        if (index != 0 || size != full_size) {
            for (int l = 0; l < size; ++l) {
                out[l] = _cmul(begin+l, v, weights);
            }
            return;
        }
        const auto& w = weights;
        const auto _case = static_cast<int>(l0 > 0) | static_cast<int>(l1 > 0 ? _n_levels_cont : 0);
        switch (_case) {
            case 0: {
                out[0] = (v * w * _mat.col(i0).transpose().array()).sum();
                out[1] = (v * w * _mat.col(i1).transpose().array()).sum();
                out[2] = (v * w * _mat.col(i0).transpose().array() * _mat.col(i1).transpose().array()).sum();
                break;
            }
            case 1: {
                out.setZero();
                for (int i = 0; i < _mat.rows(); ++i) {
                    const auto val = v[i] * w[i];
                    const int k0 = _mat(i, i0);
                    out[k0] += val;
                    out[l0 + k0] += val * _mat(i, i1);
                }
                break;
            }
            case 2: {
                out.setZero();
                for (int i = 0; i < _mat.rows(); ++i) {
                    const auto val = v[i] * w[i];
                    const int k1 = _mat(i, i1);
                    const auto b = _n_levels_cont * k1;
                    out[b] += val;
                    out[b + 1] += val * _mat(i, i0);
                }
                break;
            }
            case 3: {
                out.setZero();
                for (int i = 0; i < _mat.rows(); ++i) {
                    const int k0 = _mat(i, i0);
                    const int k1 = _mat(i, i1);
                    out[k1 * l0 + k0] += v[i] * w[i];
                }
                break;
            }
        }

        if constexpr (do_standardize) {
            const auto centers = _centers.segment(begin, size);
            const auto scales = _scales.segment(begin, size);
            const auto vwsum = (centers == 0).all() ? 0 : (v * w).sum();
            out = (out - vwsum * centers) / scales;
        }
    }

    void _btmul(
        int begin,
        int i0, int i1,
        int l0, int l1,
        int index,
        int size,
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    )
    {
        const auto both_cont = (l0 <= 0) & (l1 <= 0);
        const auto l0_exp = l0 <= 0 ? _n_levels_cont : l0;
        const auto l1_exp = l1 <= 0 ? _n_levels_cont : l1;
        const auto full_size = l0_exp * l1_exp - both_cont;
        // not a full-block
        if (index != 0 || size != full_size) {
            for (int l = 0; l < size; ++l) {
                _ctmul(begin+l, v[l], out);
            }
            return;
        }
        const auto _case = static_cast<int>(l0 > 0) | static_cast<int>(l1 > 0 ? _n_levels_cont : 0);
        const auto centers = _centers.segment(begin, size);
        const auto scales = _scales.segment(begin, size);
        const auto vs = v / scales;
        switch (_case) {
            case 0: {
                const auto mi0 = _mat.col(i0).transpose().array();
                const auto mi1 = _mat.col(i1).transpose().array();
                const auto vs0 = v[0] / scales[0];
                const auto vs1 = v[1] / scales[1];
                const auto vs2 = v[2] / scales[2];
                out += (
                    vs0 * mi0 +
                    mi1 * (vs1 + vs2 * mi0) - 
                    (vs0 * centers[0] + vs1 * centers[1] + vs2 * centers[2])
                );
                break;
            }
            case 1: {
                for (int i = 0; i < _mat.rows(); ++i) {
                    const int k0 = _mat(i, i0);
                    out[i] += vs[k0] + vs[l0 + k0] * _mat(i, i1);
                }
                const auto vsc = (vs * centers).sum();
                if (vsc) out -= vsc;
                break;
            }
            case 2: {
                for (int i = 0; i < _mat.rows(); ++i) {
                    const int k1 = _mat(i, i1);
                    const auto b = _n_levels_cont * k1;
                    out[i] += vs[b] + vs[b+1] * _mat(i, i0);
                }
                const auto vsc = (vs * centers).sum();
                if (vsc) out -= vsc;
                break;
            }
            case 3: {
                for (int i = 0; i < _mat.rows(); ++i) {
                    const int k0 = _mat(i, i0);
                    const int k1 = _mat(i, i1);
                    out[i] += vs[k1 * l0 + k0];
                }
                const auto vsc = (vs * centers).sum();
                if (vsc) out -= vsc;
                break;
            }
        }
    }

public:
    explicit MatrixNaiveInteractionDense(
        const Eigen::Ref<const dense_t>& mat,
        const Eigen::Ref<const rowarr_index_t>& pairs,
        const Eigen::Ref<const vec_index_t>& levels,
        const Eigen::Ref<const vec_value_t>& centers,
        const Eigen::Ref<const vec_value_t>& scales,
        size_t n_threads
    ):
        _mat(mat.data(), mat.rows(), mat.cols()),
        _pairs(pairs.data(), pairs.rows(), pairs.cols()),
        _levels(levels.data(), levels.size()),
        _outer(init_outer(pairs, levels)),
        _cols(_outer[_outer.size()-1]),
        _centers(init_centers(mat, pairs, levels, centers, _cols)),
        _scales(init_scales(mat, pairs, levels, _centers, scales, _cols)),
        _slice_map(init_slice_map(pairs, levels, _cols)),
        _index_map(init_index_map(pairs, levels, _cols)),
        _n_threads(n_threads)
    {
        const auto d = _mat.cols();
        if (pairs.cols() != 2) {
            throw util::adelie_core_error("pairs must be of shape (G, 2).");
        }
        if (levels.size() != d) {
            throw util::adelie_core_error("levels must be of shape (d,) where mat is (n, d).");
        }
        if ((_scales <= 0).any()) {
            throw util::adelie_core_error("scales must all be positive.");
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

    vec_value_t centers() const { return _centers; }
    vec_value_t scales() const { return _scales; }

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
            const auto pair = _pairs.row(slice);
            const auto i0 = pair[0];
            const auto i1 = pair[1];
            const auto l0 = _levels[i0];
            const auto l1 = _levels[i1];
            const auto both_cont = (l0 <= 0) & (l1 <= 0);
            const auto l0_exp = (l0 <= 0) ? _n_levels_cont : l0;
            const auto l1_exp = (l1 <= 0) ? _n_levels_cont : l1;
            const auto full_size = l0_exp * l1_exp - both_cont;
            const auto size = std::min<size_t>(full_size - index, q - n_processed);
            auto out_curr = out.segment(n_processed, size);
            _bmul<true>(jj, i0, i1, l0, l1, index, v, weights, out_curr);
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
            const auto pair = _pairs.row(slice);
            const auto i0 = pair[0];
            const auto i1 = pair[1];
            const auto l0 = _levels[i0];
            const auto l1 = _levels[i1];
            const auto both_cont = (l0 <= 0) & (l1 <= 0);
            const auto l0_exp = (l0 <= 0) ? _n_levels_cont : l0;
            const auto l1_exp = (l1 <= 0) ? _n_levels_cont : l1;
            const auto full_size = l0_exp * l1_exp - both_cont;
            const auto size = std::min<size_t>(full_size - index, q - n_processed);
            const auto v_curr = v.segment(n_processed, size);
            _btmul(jj, i0, i1, l0, l1, index, size, v_curr, out);
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
            const auto pair = _pairs.row(g);
            const auto i0 = pair[0];
            const auto i1 = pair[1];
            const auto l0 = _levels[i0];
            const auto l1 = _levels[i1];
            const auto both_cont = (l0 <= 0) & (l1 <= 0);
            const auto l0_exp = (l0 <= 0) ? _n_levels_cont : l0;
            const auto l1_exp = (l1 <= 0) ? _n_levels_cont : l1;
            const auto full_size = l0_exp * l1_exp - both_cont;
            auto out_curr = out.segment(j, full_size);
            _bmul<true>(j, i0, i1, l0, l1, 0, v, weights, out_curr);
        };
        if (_n_threads <= 1) {
            for (int g = 0; g < _outer.size()-1; ++g) routine(g);
        } else {
            #pragma omp parallel for schedule(static) num_threads(_n_threads)
            for (int g = 0; g < _outer.size()-1; ++g) routine(g);
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

        const auto slice = _slice_map[j];
        const auto index = _index_map[j];
        const auto outer = _outer[slice];
        if ((index != 0) || (_outer[slice+1] - outer != q)) {
            throw util::adelie_core_error(
                "MatrixNaiveInteractionDense::cov() not implemented for ranges that contain multiple blocks. "
                "If triggered from a solver, this error is usually because "
                "the groups argument is inconsistent with the implicit group structure "
                "of the matrix. "
            );
        }

        const auto& sqrt_w = sqrt_weights;
        const auto pair = _pairs.row(slice);
        const auto i0 = pair[0];
        const auto i1 = pair[1];
        const auto l0 = _levels[i0];
        const auto l1 = _levels[i1];
        const auto _case = static_cast<int>(l0 > 0) | static_cast<int>(l1 > 0 ? _n_levels_cont : 0);

        switch (_case) {
            case 0: {
                const auto mi0 = _mat.col(i0).array();
                const auto mi1 = _mat.col(i1).array();
                auto w = buffer.col(0).array();
                w = sqrt_w.square();
                out(0, 0) = (w * mi0.square()).sum();
                out(1, 0) = (w * mi0 * mi1).sum();
                out(1, 1) = (w * mi1.square()).sum();
                out(2, 0) = (w * mi0.square() * mi1).sum();
                out(2, 1) = (w * mi1.square() * mi0).sum();
                out(2, 2) = (w * (mi0 * mi1).square()).sum();
                for (int i0 = 0; i0 < q; ++i0) {
                    for (int i1 = i0+1; i1 < q; ++i1) {
                        out(i0, i1) = out(i1, i0);
                    }
                }
                break;
            }
            case 1: {
                out.setZero();
                auto out_11 = out.block(0, 0, l0, l0);
                auto out_21 = out.block(l0, 0, l0, l0);
                auto out_22 = out.block(l0, l0, l0, l0);
                for (int i = 0; i < _mat.rows(); ++i) {
                    const int k0 = _mat(i, i0);
                    const auto wi = sqrt_w[i] * sqrt_w[i];
                    const auto m = _mat(i, i1);
                    const auto mwi = m * wi;
                    out_11(k0, k0) += wi;
                    out_21(k0, k0) += mwi;
                    out_22(k0, k0) += mwi * m;
                }
                auto out_12 = out.block(0, l0, l0, l0);
                out_12.diagonal() = out_21.diagonal();
                break;
            }
            case 2: {
                out.setZero();
                for (int i = 0; i < _mat.rows(); ++i) {
                    const int k1 = _mat(i, i1);
                    const auto b = _n_levels_cont * k1;
                    const auto wi = sqrt_w[i] * sqrt_w[i];
                    const auto m = _mat(i, i0);
                    const auto mwi = m * wi;
                    out(b, b) += wi; 
                    out(b+1, b) += mwi;
                    out(b+1, b+1) += mwi * m;
                }
                for (int k = 0; k < l1; ++k) {
                    const int b = _n_levels_cont * k;
                    out(b, b+1) = out(b+1, b);
                }
                break;
            }
            case 3: {
                out.setZero();
                for (int i = 0; i < _mat.rows(); ++i) {
                    const int k0 = _mat(i, i0);
                    const int k1 = _mat(i, i1);
                    const int k = k1 * l0 + k0;
                    const auto wi = sqrt_w[i] * sqrt_w[i];
                    out(k, k) += wi;
                }
                break;
            }
        }

        const auto centers = _centers.segment(j, q);
        const auto scales = _scales.segment(j, q);

        if ((centers != 0).any()) {
            auto out_lower = out.template selfadjointView<Eigen::Lower>();
            vec_value_t x_mean(q);
            _bmul<false>(j, i0, i1, l0, l1, index, sqrt_w, sqrt_w, x_mean);
            out_lower.rankUpdate(centers.matrix().transpose(), x_mean.matrix().transpose(), -1);
            out_lower.rankUpdate(centers.matrix().transpose(), sqrt_w.square().sum());
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
        const auto routine = [&](auto k) {
            typename sp_mat_value_t::InnerIterator it(v, k);
            auto out_k = out.row(k);
            out_k.setZero();
            for (; it; ++it) {
                _ctmul(it.index(), it.value(), out_k);
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