#pragma once
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <vector>
#include <string>
#include <ghostbasil/util/macros.hpp>
#include <ghostbasil/util/types.hpp>
#include <ghostbasil/matrix/forward_decl.hpp>

namespace ghostbasil {

/*
 * This class represents a block-diagonal matrix
 * where each block has the same type MatrixType.
 * A block-diagonal matrix is of the form:
 *
 *      A_1 0   ... 0
 *      0   A_2 ... 0
 *      .   .   ... 0
 *      0   .   ... A_L
 *
 * where the above has L blocks of square matrices A_l.
 */
template <class MatrixType>
class BlockMatrix
{
    using mat_t = MatrixType;
    using value_t = typename mat_t::Scalar;

    const mat_t* mat_list_;
    size_t n_mats_;
    std::vector<uint32_t> n_cum_sum_; // [n_cum_sum_[i], n_cum_sum_[i+1])
                                      // is the range of features for block i.

    GHOSTBASIL_STRONG_INLINE
    auto n_features() const { return n_cum_sum_.back(); }

    GHOSTBASIL_STRONG_INLINE
    auto find_stride_at_index(
            Eigen::Index i,
            size_t begin,
            size_t end) const
    {
        auto stride_begin = std::next(n_cum_sum_.begin(), begin);
        auto stride_end = std::next(n_cum_sum_.begin(), end);
        auto it = std::upper_bound(stride_begin, stride_end, i);
        // guaranteed to not be the end
        assert(it != stride_end);
        return std::distance(n_cum_sum_.begin(), it)-1;
    }

public:
    using Scalar = value_t;
    using Index = typename Eigen::Index;

    class ConstBlockIterator;

    BlockMatrix(): mat_list_{nullptr}, n_mats_{0} {}

    template <class MatrixListType>
    BlockMatrix(const MatrixListType& mat_list)
        : mat_list_(mat_list.data()),
          n_mats_(mat_list.size())
    {
        for (size_t i = 0; i < n_mats_; ++i) {
            const auto& B = mat_list_[i];
            std::string error =
                "Matrix at index " + std::to_string(i) + " ";

            // Check that matrix is square.
            if (B.rows() != B.cols()) {
                error +=  "is not square. ";
                throw std::runtime_error(error);
            }

            // Check that matrix is not empty
            if (B.rows() == 0) {
                error +=  "is empty. ";
                throw std::runtime_error(error);
            }
        }

        // Compute the cumulative number of features.
        n_cum_sum_.resize(n_mats_+1);
        n_cum_sum_[0] = 0;
        for (size_t i = 0; i < n_mats_; ++i) {
            n_cum_sum_[i+1] = n_cum_sum_[i] + mat_list_[i].cols();
        }
    }

    GHOSTBASIL_STRONG_INLINE Index rows() const { return n_features(); }
    GHOSTBASIL_STRONG_INLINE Index cols() const { return n_features(); }
    GHOSTBASIL_STRONG_INLINE Index size() const { return rows() * cols(); }
    GHOSTBASIL_STRONG_INLINE Index n_blocks() const { return n_mats_; }
    GHOSTBASIL_STRONG_INLINE const mat_t* blocks() const { return mat_list_; }

    template <class VecType>
    GHOSTBASIL_STRONG_INLINE
    value_t col_dot(size_t k, const Eigen::DenseBase<VecType>& v) const
    {
        assert(k < cols());

        // Find the i(k) which is the closest index to k:
        // n_cum_sum_[i(k)] <= k < n_cum_sum_[i(k)+1]
        const auto ik_end = std::upper_bound(
                n_cum_sum_.begin(),
                n_cum_sum_.end(),
                k);
        const auto ik_begin = std::next(ik_end, -1);
        const auto ik = std::distance(n_cum_sum_.begin(), ik_begin);  
        assert((ik+1) < n_cum_sum_.size());

        // Find i(k)th block matrix.
        const auto& B = mat_list_[ik];

        // Find v_{i(k)}, i(k)th block of vector. 
        const auto vi = v.segment(n_cum_sum_[ik], n_cum_sum_[ik+1]-n_cum_sum_[ik]);

        // Find the shifted k relative to A_{i(k)}).
        const size_t k_shifted = k - n_cum_sum_[ik];

        return B.col_dot(k_shifted, vi);
    }

    template <class VecType>
    GHOSTBASIL_STRONG_INLINE
    value_t col_dot(size_t k, const Eigen::SparseCompressedBase<VecType>& v) const
    {
        assert(k < cols());

        // Find the i(k) which is the closest index to k:
        // n_cum_sum_[i(k)] <= k < n_cum_sum_[i(k)+1]
        const auto ik_end = std::upper_bound(
                n_cum_sum_.begin(),
                n_cum_sum_.end(),
                k);
        const auto ik_begin = std::next(ik_end, -1);
        const auto ik = std::distance(n_cum_sum_.begin(), ik_begin);  
        assert((ik+1) < n_cum_sum_.size());

        // Find i(k)th block matrix.
        const auto& B = mat_list_[ik];
        const auto stride = n_cum_sum_[ik];
        const size_t k_shifted = k - stride;

        // Find v_{i(k)}, i(k)th block of vector. 
        const auto v_inner = v.innerIndexPtr();
        const auto v_value = v.valuePtr();
        const auto v_nnz = v.nonZeros();
        const auto begin = std::lower_bound(v_inner, v_inner+v_nnz, stride)-v_inner;
        const auto end = std::lower_bound(v_inner+begin, v_inner+v_nnz, n_cum_sum_[ik+1])-v_inner;

        // Compute dot-product
        Scalar dp = 0; 
        for (auto j = begin; j != end; ++j) {
            dp += B.coeff(v_inner[j]-stride, k_shifted) * v_value[j]; 
        }

        return dp;
    }

    template <class VecType>
    GHOSTBASIL_STRONG_INLINE
    value_t quad_form(const VecType& v) const
    {
        value_t quadform = 0;
        for (size_t i = 0; i < n_mats_; ++i) {
            const auto& B = mat_list_[i];
            const auto vi = v.segment(n_cum_sum_[i], n_cum_sum_[i+1]-n_cum_sum_[i]);
            quadform += B.quad_form(vi);
        }
        return quadform;
    }

    template <class VecType>
    GHOSTBASIL_STRONG_INLINE
    value_t inv_quad_form(value_t s, const VecType& v) const
    {
        value_t inv_quadform = 0;
        for (size_t i = 0; i < n_mats_; ++i) {
            const auto& B = mat_list_[i];
            const auto vi = v.segment(n_cum_sum_[i], n_cum_sum_[i+1]-n_cum_sum_[i]);
            inv_quadform += B.inv_quad_form(s, vi);
        }
        return inv_quadform;
    }

    GHOSTBASIL_STRONG_INLINE
    ConstBlockIterator block_begin() const { 
        return ConstBlockIterator(*this);
    }
    GHOSTBASIL_STRONG_INLINE
    ConstBlockIterator block_end() const {
        return ConstBlockIterator(*this, n_cum_sum_.size()-1);
    }
    GHOSTBASIL_STRONG_INLINE
    const auto& strides() const { return n_cum_sum_; }

    GHOSTBASIL_STRONG_INLINE
    Scalar coeff(Index i, Index j) const 
    {
        auto stride_idx = find_stride_at_index(i, 0, n_cum_sum_.size());
        assert((stride_idx+1) < n_cum_sum_.size());
        auto begin = n_cum_sum_[stride_idx];
        auto end = n_cum_sum_[stride_idx+1];
        if (j < begin || j >= end) return 0;
        return mat_list_[stride_idx].coeff(i-begin, j-begin);
    }
    
    GHOSTBASIL_STRONG_INLINE
    auto to_dense() const
    {
        const auto p = n_features();
        const auto& strides_ = strides();

        util::mat_type<value_t> dm(p, p);
        dm.setZero();

        for (size_t i = 0; i < n_mats_; ++i) {
            const auto si = strides_[i];
            const auto li = strides_[i+1] - strides_[i];
            dm.block(si, si, li, li) = mat_list_[i].to_dense();
        }
        
        return dm;
    }
};

template <class MatrixType>
class BlockMatrix<MatrixType>::ConstBlockIterator
{
    const BlockMatrix& m_;
    size_t idx_;

public:
    ConstBlockIterator(const BlockMatrix& m, size_t idx=0)
        : m_(m), idx_{idx}
    {
        assert((idx+1) <= m_.n_cum_sum_.size());
    }

    const auto& block() const { return m_.mat_list_[idx_]; }
    auto shift(size_t k) const { return k - stride(); }
    auto stride() const { return m_.n_cum_sum_[idx_]; }
    ConstBlockIterator& operator++() { ++idx_; return *this; }
    ConstBlockIterator& advance_at(size_t k) {
        assert((m_.n_cum_sum_[idx_] <= k) && (k < m_.n_cum_sum_.back()));
        idx_ = m_.find_stride_at_index(k, idx_, m_.n_cum_sum_.size());
        return *this;
    }
    bool is_in_block(size_t k) const { 
        return (m_.n_cum_sum_[idx_] <= k) && (k < m_.n_cum_sum_[idx_+1]);
    }

    GHOSTBASIL_STRONG_INLINE
    constexpr bool operator==(const ConstBlockIterator& other) const
    {
        return (&m_ == &other.m_) && (idx_ == other.idx_);
    }
};

} // namespace ghostbasil
