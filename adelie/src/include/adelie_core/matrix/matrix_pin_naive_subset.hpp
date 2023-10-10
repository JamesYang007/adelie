#pragma once
#include <unordered_map>
#include <adelie_core/matrix/matrix_pin_naive_base.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

template <class DenseType, class IndexType>
class MatrixPinNaiveSubset: public MatrixPinNaiveBase<typename DenseType::Scalar>
{
public:
    using base_t = MatrixPinNaiveBase<typename DenseType::Scalar>;
    using typename base_t::value_t;
    using typename base_t::rowvec_t;
    using dense_t = DenseType;
    using index_t = IndexType;

private:
    const int _rows; 
    const int _cols;
    const std::vector<dense_t>& _mats;
    const std::unordered_map<index_t, index_t>& _idx_map;
    const std::unordered_map<index_t, index_t>& _slice_map;
    const size_t _n_threads;
    util::rowmat_type<value_t> _buff;

public:
    MatrixPinNaiveSubset(
        int rows,
        int cols,
        const std::vector<dense_t>& mats,
        const std::unordered_map<index_t, index_t>& idx_map,
        const std::unordered_map<index_t, index_t>& slice_map,
        size_t n_threads
    ):
        _rows(rows),
        _cols(cols),
        _mats(mats),
        _idx_map(idx_map),
        _slice_map(slice_map),
        _n_threads(n_threads),
        _buff(_n_threads, std::min(rows, cols))
    {}

    value_t cmul(
        int j,
        const Eigen::Ref<const rowvec_t>& v
    ) const override
    {
        const auto& mat = _mats[_idx_map[j]];
        const auto k = _slice_map[j];
        return ddot(mat.col(k).matrix(), v.matrix(), _n_threads);
    }

    value_t ctmul(
        int j,
        value_t v,
        Eigen::Ref<rowvec_t> out
    ) const override
    {
        const auto& mat = _mats[_idx_map[j]];
        const auto k = _slice_map[j];
        return dax(v, mat.col(k), _n_threads, out);
    }

    void bmul(
        int j, int q,
        const Eigen::Ref<const rowvec_t>& v,
        Eigen::Ref<rowvec_t> out
    ) override
    {
        auto outm = out.matrix();
        dgemv(
            _mats[_idx_map[j]].middleCols(_slice_map[j], q),
            v.matrix(),
            _n_threads,
            _buff,
            outm
        );
    }

    void btmul(
        int j, int q,
        const Eigen::Ref<const rowvec_t>& v,
        Eigen::Ref<rowvec_t> out
    ) override
    {
        auto outm = out.matrix();
        dgemv(
            _mats[_idx_map[j]].middleCols(_slice_map[j], q).transpose(),
            v.matrix(),
            _n_threads,
            _buff,
            outm
        );
    }

    value_t cnormsq(int j) const override
    {
        return _mats[_idx_map[j]].col(_slice_map[j]).squaredNorm();
    }

    int rows() const override
    {
        return _rows;
    }

    int cols() const override
    {
        return _cols;
    }
};

} // namespace matrix
} // namespace adelie_core