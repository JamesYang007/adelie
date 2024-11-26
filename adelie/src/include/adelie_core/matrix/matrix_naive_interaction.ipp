#pragma once
#include <adelie_core/matrix/matrix_naive_interaction.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE_TP
auto
ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE::init_outer(
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

ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE_TP
auto
ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE::init_slice_map(
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

ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE_TP
auto
ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE::init_index_map(
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

ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE_TP
typename ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE::value_t
ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE::_cmul(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights,
    size_t n_threads,
    Eigen::Ref<vec_value_t> buff
) const
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
    switch (_case) {
        case 0: {
            switch (index) {
                case 0: {
                    return ddot((v * w).matrix(), _mat.col(i0).transpose(), n_threads, buff);
                    break;
                }
                case 1: {
                    return ddot((v * w).matrix(), _mat.col(i1).transpose(), n_threads, buff);
                    break;
                }
                case 2: {
                    return ddot((v * w).matrix(), _mat.col(i0).cwiseProduct(_mat.col(i1)).transpose(), n_threads, buff);
                    break;
                }
            }
            break;
        }
        case 1: {
            if (k1 == 0) {
                const auto mi0 = _mat.col(i0).transpose().array();
                return ddot(
                    (v * w).matrix(), 
                    (mi0 == k0).template cast<value_t>().matrix(), 
                    n_threads, 
                    buff
                );
            } else {
                const auto mi0 = _mat.col(i0).transpose().array();
                const auto mi1 = _mat.col(i1).transpose().array();
                return ddot(
                    (v * w).matrix(), 
                    (mi1 * (mi0 == k0).template cast<value_t>()).matrix(), 
                    n_threads, 
                    buff
                );
            }
            break;
        }
        case 2: {
            if (k0 == 0) {
                const auto mi1 = _mat.col(i1).transpose().array();
                return ddot(
                    (v * w).matrix(), 
                    (mi1 == k1).template cast<value_t>().matrix(), 
                    n_threads, 
                    buff
                );
            } else {
                const auto mi0 = _mat.col(i0).transpose().array();
                const auto mi1 = _mat.col(i1).transpose().array();
                return ddot(
                    (v * w).matrix(), 
                    (mi0 * (mi1 == k1).template cast<value_t>()).matrix(), 
                    n_threads, 
                    buff
                );
            }
            break;
        }
        case 3: {
            const auto mi0 = _mat.col(i0).transpose().array();
            const auto mi1 = _mat.col(i1).transpose().array();
            return ddot(
                (v * w).matrix(), 
                ((mi0 == k0) && (mi1 == k1)).template cast<value_t>().matrix(), 
                n_threads, 
                buff
            );
            break;
        }
    }
    return std::numeric_limits<value_t>::infinity() + 1; // lol
}

ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE::_ctmul(
    int j, 
    value_t v, 
    Eigen::Ref<vec_value_t> out,
    size_t n_threads
) const
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
    switch (_case) {
        case 0: {
            switch (index) {
                case 0: {
                    dvaddi(out, v * _mat.col(i0).transpose().array(), n_threads);
                    break;
                }
                case 1: {
                    dvaddi(out, v * _mat.col(i1).transpose().array(), n_threads);
                    break;
                }
                case 2: {
                    dvaddi(out, v * _mat.col(i0).transpose().array() * _mat.col(i1).transpose().array(), n_threads);
                    break;
                }
            }
            break;
        }
        case 1: {
            if (k1 == 0) {
                const auto mi0 = _mat.col(i0).transpose().array();
                dvaddi(
                    out,
                    v * (mi0 == k0).template cast<value_t>(),
                    n_threads
                );
            } else {
                const auto mi0 = _mat.col(i0).transpose().array();
                const auto mi1 = _mat.col(i1).transpose().array();
                dvaddi(
                    out,
                    v * mi1 * (mi0 == k0).template cast<value_t>(),
                    n_threads
                );
            }
            break;
        }
        case 2: {
            if (k0 == 0) {
                const auto mi1 = _mat.col(i1).transpose().array();
                dvaddi(
                    out,
                    v * (mi1 == k1).template cast<value_t>(),
                    n_threads
                );
            } else {
                const auto mi0 = _mat.col(i0).transpose().array();
                const auto mi1 = _mat.col(i1).transpose().array();
                dvaddi(
                    out,
                    v * mi0 * (mi1 == k1).template cast<value_t>(),
                    n_threads
                );
            }
            break;
        }
        case 3: {
            const auto mi0 = _mat.col(i0).transpose().array();
            const auto mi1 = _mat.col(i1).transpose().array();
            dvaddi(
                out,
                v * ((mi0 == k0) && (mi1 == k1)).template cast<value_t>(),
                n_threads
            );
            break;
        }
    }
}

ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE::_bmul(
    int begin,
    int i0, int i1,
    int l0, int l1,
    int index,
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out,
    Eigen::Ref<vec_value_t> buff,
    size_t n_threads
) const
{
    const size_t size = out.size();
    const auto both_cont = (l0 <= 0) & (l1 <= 0);
    const auto l0_exp = l0 <= 0 ? _n_levels_cont : l0;
    const auto l1_exp = l1 <= 0 ? _n_levels_cont : l1;
    const auto full_size = l0_exp * l1_exp - both_cont;
    // not a full-block
    if (index != 0 || size != full_size) {
        for (size_t l = 0; l < size; ++l) {
            out[l] = _cmul(begin+l, v, weights, n_threads, buff);
        }
        return;
    }
    const auto& w = weights;
    const auto _case = static_cast<int>(l0 > 0) | static_cast<int>(l1 > 0 ? _n_levels_cont : 0);
    switch (_case) {
        case 0: {
            out[0] = ddot(_mat.col(i0), (v * w).matrix(), n_threads, buff);
            out[1] = ddot(_mat.col(i1), (v * w).matrix(), n_threads, buff);
            out[2] = ddot(_mat.col(i0).cwiseProduct(_mat.col(i1)), (v * w).matrix(), n_threads, buff);
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
}

ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE::_sq_bmul(
    int i0, int i1,
    int l0, int l1,
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out,
    Eigen::Ref<vec_value_t> buff
) const
{
    constexpr size_t n_threads = 1;
    const auto& w = weights;
    const auto _case = static_cast<int>(l0 > 0) | static_cast<int>(l1 > 0 ? _n_levels_cont : 0);
    switch (_case) {
        case 0: {
            out[0] = ddot(_mat.col(i0).array().square().matrix(), w.matrix(), n_threads, buff);
            out[1] = ddot(_mat.col(i1).array().square().matrix(), w.matrix(), n_threads, buff);
            out[2] = ddot(_mat.col(i0).cwiseProduct(_mat.col(i1)).array().square().matrix(), w.matrix(), n_threads, buff);
            break;
        }
        case 1: {
            out.setZero();
            for (int i = 0; i < _mat.rows(); ++i) {
                const auto val = w[i];
                const int k0 = _mat(i, i0);
                const auto z_val = _mat(i, i1);
                out[k0] += val;
                out[l0 + k0] += val * z_val * z_val;
            }
            break;
        }
        case 2: {
            out.setZero();
            for (int i = 0; i < _mat.rows(); ++i) {
                const auto val = w[i];
                const int k1 = _mat(i, i1);
                const auto b = _n_levels_cont * k1;
                const auto z_val = _mat(i, i0);
                out[b] += val;
                out[b + 1] += val * z_val * z_val;
            }
            break;
        }
        case 3: {
            out.setZero();
            for (int i = 0; i < _mat.rows(); ++i) {
                const int k0 = _mat(i, i0);
                const int k1 = _mat(i, i1);
                out[k1 * l0 + k0] += w[i];
            }
            break;
        }
    }
}

ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE::_btmul(
    int begin,
    int i0, int i1,
    int l0, int l1,
    int index,
    size_t size,
    const Eigen::Ref<const vec_value_t>& v, 
    Eigen::Ref<vec_value_t> out,
    size_t n_threads
) const
{
    const auto both_cont = (l0 <= 0) & (l1 <= 0);
    const auto l0_exp = l0 <= 0 ? _n_levels_cont : l0;
    const auto l1_exp = l1 <= 0 ? _n_levels_cont : l1;
    const auto full_size = l0_exp * l1_exp - both_cont;
    // not a full-block
    if (index != 0 || size != full_size) {
        for (size_t l = 0; l < size; ++l) {
            _ctmul(begin+l, v[l], out, n_threads);
        }
        return;
    }
    const auto _case = static_cast<int>(l0 > 0) | static_cast<int>(l1 > 0 ? _n_levels_cont : 0);
    switch (_case) {
        case 0: {
            const auto mi0 = _mat.col(i0).transpose().array();
            const auto mi1 = _mat.col(i1).transpose().array();
            dvaddi(out, v[0] * mi0 + mi1 * (v[1] + v[2] * mi0), n_threads);
            break;
        }
        case 1: {
            dvaddi(
                out, 
                vec_value_t::NullaryExpr(_mat.rows(), 
                    [&](auto i) { 
                        const int k0 = _mat(i, i0);
                        return v[k0] + v[l0 + k0] * _mat(i, i1);
                    }
                ),
                n_threads
            );
            break;
        }
        case 2: {
            dvaddi(
                out,
                vec_value_t::NullaryExpr(_mat.rows(),
                    [&](auto i) {
                        const int k1 = _mat(i, i1);
                        const auto b = _n_levels_cont * k1;
                        return v[b] + v[b+1] * _mat(i, i0);
                    }
                ),
                n_threads
            );
            break;
        }
        case 3: {
            dvaddi(
                out,
                vec_value_t::NullaryExpr(_mat.rows(),
                    [&](auto i) {
                        const int k0 = _mat(i, i0);
                        const int k1 = _mat(i, i1);
                        return v[k1 * l0 + k0];
                    }
                ),
                n_threads
            );
            break;
        }
    }
}

ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE_TP
ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE::MatrixNaiveInteractionDense(
    const Eigen::Ref<const dense_t>& mat,
    const Eigen::Ref<const rowarr_index_t>& pairs,
    const Eigen::Ref<const vec_index_t>& levels,
    size_t n_threads
):
    _mat(mat.data(), mat.rows(), mat.cols()),
    _pairs(pairs.data(), pairs.rows(), pairs.cols()),
    _levels(levels.data(), levels.size()),
    _outer(init_outer(pairs, levels)),
    _cols(_outer[_outer.size()-1]),
    _slice_map(init_slice_map(pairs, levels, _cols)),
    _index_map(init_index_map(pairs, levels, _cols)),
    _n_threads(n_threads),
    _buff(_n_threads)
{
    const auto d = _mat.cols();
    if (pairs.cols() != 2) {
        throw util::adelie_core_error("pairs must be (G, 2).");
    }
    if (levels.size() != d) {
        throw util::adelie_core_error("levels must be (d,) where mat is (n, d).");
    }
    if (n_threads < 1) {
        throw util::adelie_core_error("n_threads must be >= 1.");
    }
}

ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE_TP
typename ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE::value_t
ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE::cmul(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights
) 
{
    base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
    return _cmul(j, v, weights, _n_threads, _buff);
}

ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE_TP
typename ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE::value_t
ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE::cmul_safe(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights
) const
{
    base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
    vec_value_t buff(_n_threads * (_n_threads > 1) * !util::omp_in_parallel());
    return _cmul(j, v, weights, _n_threads, buff);
}

ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE::ctmul(
    int j, 
    value_t v, 
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_ctmul(j, out.size(), rows(), cols());
    _ctmul(j, v, out, _n_threads);
}

ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE::bmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
)
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
        _bmul(jj, i0, i1, l0, l1, index, v, weights, out_curr, _buff, _n_threads);
        n_processed += size;
    }
}

ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE::bmul_safe(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
    vec_value_t buff(_n_threads * (_n_threads > 1) * !util::omp_in_parallel());
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
        _bmul(jj, i0, i1, l0, l1, index, v, weights, out_curr, buff, _n_threads);
        n_processed += size;
    }
}

ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE::btmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    Eigen::Ref<vec_value_t> out
)
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
        _btmul(jj, i0, i1, l0, l1, index, size, v_curr, out, _n_threads);
        n_processed += size;
    }
}

ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE::mul(
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
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
        _bmul(j, i0, i1, l0, l1, 0, v, weights, out_curr, out /* unused */, 1);
    };
    util::omp_parallel_for(routine, 0, _outer.size()-1, _n_threads);
}

ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE_TP
int
ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE::rows() const
{
    return _mat.rows();
}
    
ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE_TP
int
ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE::cols() const
{
    return _cols;
}

ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE::cov(
    int j, int q,
    const Eigen::Ref<const vec_value_t>& sqrt_weights,
    Eigen::Ref<colmat_value_t> out
) const
{
    base_t::check_cov(
        j, q, sqrt_weights.size(), 
        out.rows(), out.cols(),
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

    vec_value_t buff(_n_threads * (_n_threads > 1) * !util::omp_in_parallel());
    
    switch (_case) {
        case 0: {
            const auto mi0 = _mat.col(i0).array();
            const auto mi1 = _mat.col(i1).array();
            vec_value_t w(sqrt_w.size());
            dvveq(w, sqrt_w.square(), _n_threads);
            out(0, 0) = ddot(w.matrix(), mi0.square().matrix(), _n_threads, buff);
            out(1, 0) = ddot(w.matrix(), (mi0 * mi1).matrix(), _n_threads, buff);
            out(1, 1) = ddot(w.matrix(), mi1.square().matrix(), _n_threads, buff);
            out(2, 0) = ddot(w.matrix(), (mi0.square() * mi1).matrix(), _n_threads, buff);
            out(2, 1) = ddot(w.matrix(), (mi1.square() * mi0).matrix(), _n_threads, buff);
            out(2, 2) = ddot(w.matrix(), (mi0 * mi1).square().matrix(), _n_threads, buff);
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
}

ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE::sq_mul(
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
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
        _sq_bmul(i0, i1, l0, l1, weights, out_curr, out /* unused */);
    };
    util::omp_parallel_for(routine, 0, _outer.size()-1, _n_threads);
}

ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE::sp_tmul(
    const sp_mat_value_t& v, 
    Eigen::Ref<rowmat_value_t> out
) const
{
    base_t::check_sp_tmul(
        v.rows(), v.cols(), out.rows(), out.cols(), rows(), cols()
    );
    const auto routine = [&](auto k) {
        typename sp_mat_value_t::InnerIterator it(v, k);
        auto out_k = out.row(k);
        out_k.setZero();
        for (; it; ++it) {
            _ctmul(it.index(), it.value(), out_k, 1);
        }
    };
    util::omp_parallel_for(routine, 0, v.outerSize(), _n_threads);
}

ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE::mean(
    const Eigen::Ref<const vec_value_t>&,
    Eigen::Ref<vec_value_t> 
) const
{
    throw util::adelie_core_error(
        "MatrixNaiveInteractionDense: mean() not implemented! "
        "If this error occurred from standardizing the matrix, "
        "consider providing your own center vector. "
    );
}

ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE_TP
void
ADELIE_CORE_MATRIX_NAIVE_INTERACTION_DENSE::var(
    const Eigen::Ref<const vec_value_t>&,
    const Eigen::Ref<const vec_value_t>&,
    Eigen::Ref<vec_value_t> 
) const
{
    throw util::adelie_core_error(
        "MatrixNaiveInteractionDense: var() not implemented! "
        "If this error occurred from standardizing the matrix, "
        "consider providing your own scale vector. "
    );
}

} // namespace matrix 
} // namespace adelie_core