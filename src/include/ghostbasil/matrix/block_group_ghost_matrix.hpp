#pragma once
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <ghostbasil/util/macros.hpp>
#include <ghostbasil/util/type_traits.hpp>
#include <ghostbasil/util/types.hpp>
#include <ghostbasil/matrix/forward_decl.hpp>
#include <ghostbasil/matrix/block_matrix.hpp>

namespace ghostbasil {

/**
 * This class represents a matrix of the form:
 *
 *      S+D S ... S   S
 *      S   S+D   .   .
 *      .   .    S+D  .
 *      S   .    ... S+D
 *
 * which is the typical form for the Gaussian covariance matrix
 * under multiple group-knockoffs framework where D is a block-diagonal matrix.
 * We say that a BlockGroupGhostMatrix has n groups if it is of size
 * (n*p) x (n*p) where S and D are p x p.
 *
 * @tparam  MatrixType  type of matrix that represents S.
 */
template <class MatrixType, class DType=BlockMatrix<MatrixType>>
class BlockGroupGhostMatrix
{
    using mat_t = std::decay_t<MatrixType>;
    using bmat_t = DType;
    using value_t = typename mat_t::Scalar;
    using sp_mat_t = Eigen::SparseMatrix<value_t>;

    Eigen::Map<const mat_t> S_;  // matrix S
    const bmat_t D_;  // matrix D 
    size_t n_groups_;       // number of groups
    Eigen::Index rows_; // number of total rows (same as total columns)

    // stores the shift information for each index k:
    //      shift_[k] == k % group_size
    std::vector<Eigen::Index> shift_;

    GHOSTBASIL_STRONG_INLINE
    auto compute_group_size() const { return S_.cols(); }

public:
    using Scalar = value_t;
    using Index = Eigen::Index;
    
    template <class MatType_>
    BlockGroupGhostMatrix(
        const MatType_& S,
        const bmat_t& D,
        size_t n_groups
    )
        : S_(S.data(), S.rows(), S.cols()),
          D_(D),
          n_groups_(n_groups),
          rows_(S.rows() * n_groups),
          shift_(rows_)
    {
        // Check that number of groups is at least 2.
        if (n_groups_ < 2) {
            throw std::runtime_error(
                "Number of groups must be at least 2. "
                "If number of groups <= 1, use Eigen::Matrix instead, "
                "since GhostMatrix degenerates to the top-left corner matrix. ");
        }

        // Check that the matrix is square and non-empty.
        if (S.rows() != S.cols()) {
            throw std::runtime_error("Matrix is not square.");
        }
        if (S.rows() <= 0) {
            throw std::runtime_error("Matrix and vector must have dimension/length > 0.");
        }

        // Check that the matrix and vector agree in size.
        if ((S.rows() != D.rows()) || (S.cols() != D.cols())) {
            std::string error = 
                "S and D do not have the same dimensions. "                
                "S has dimensions " + std::to_string(S.rows()) + " x " + std::to_string(S.cols()) + " and " +
                "D has dimensions " + std::to_string(D.rows()) + " x " + std::to_string(D.cols());
            throw std::runtime_error(error);
        }

        // populate stride_shift_
        const auto group_size = compute_group_size();
        for (size_t k = 0; k < shift_.size(); ++k) {
            shift_[k] = k % group_size;
        }
    }

    GHOSTBASIL_STRONG_INLINE Index rows() const { return rows_; }
    GHOSTBASIL_STRONG_INLINE Index cols() const { return rows(); }
    GHOSTBASIL_STRONG_INLINE Index size() const { return rows() * cols(); }
    GHOSTBASIL_STRONG_INLINE
    const auto& get_S() const { return S_; }
    GHOSTBASIL_STRONG_INLINE
    const auto& get_D() const { return D_; }
    GHOSTBASIL_STRONG_INLINE
    auto n_groups() const { return n_groups_; }
    GHOSTBASIL_STRONG_INLINE
    auto shift(Index i) const { return shift_[i]; }

    /*
     * Computes the dot product between kth column of the matrix with v: 
     *      A[:,k]^T * v
     */
    template <class VecType>
    value_t col_dot(size_t k, const Eigen::DenseBase<VecType>& v) const
    {
        assert(k < cols());
        assert(cols() == v.size());

        const size_t group_size = compute_group_size();
        const auto& S = get_S();
        const auto& D = get_D();

        // Find the index to block of K features containing k.
        const size_t k_block = shift_[k];

        // Get quantities for reuse.
        const auto S_k = S.col(k_block);

        // Compute the dot product.
        value_t dp = 0;
        size_t v_j_begin = 0;
        for (size_t j = 0; j < n_groups_; ++j, v_j_begin += group_size) {
            const auto v_j = v.segment(v_j_begin, group_size);
            dp += v_j.dot(S_k);
        }
        dp += D.col_dot(k_block, v.segment((k / group_size) * group_size, group_size));

        return dp;
    }

    template <class VecType>
    value_t col_dot(size_t k, const Eigen::SparseCompressedBase<VecType>& v) const
    {
        assert(k < cols());
        assert(cols() == v.size());

        if (v.nonZeros() == 0) return 0;

        const size_t group_size = compute_group_size();
        const auto& S = get_S();
        const auto& D = get_D();

        // Find the index to block of K features containing k.
        const size_t k_block = shift_[k];

        // Get quantities for reuse.
        const auto S_k = S.col(k_block);

        // Compute the dot product.
        value_t dp = 0;
        size_t j_group_size = 0;
        const auto v_inner = v.innerIndexPtr();
        const auto v_value = v.valuePtr();
        const auto v_nnz = v.nonZeros();
        size_t v_begin = 0;
        size_t v_end;
        for (size_t j = 0; j < n_groups_; ++j, j_group_size += group_size) {
            v_end = std::lower_bound(v_inner+v_begin, v_inner+v_nnz, j_group_size+group_size)-v_inner;
            for (size_t l = v_begin; l < v_end; ++l) {
                dp += v_value[l] * S_k[v_inner[l]-j_group_size];
            } 
            v_begin = v_end;
        }

        // Find the i(k) which is the closest index to k:
        // n_cum_sum_[i(k)] <= k < n_cum_sum_[i(k)+1]
        const auto& n_cum_sum = D.strides();
        const auto ik_end = std::upper_bound(
                n_cum_sum.begin(),
                n_cum_sum.end(),
                k_block);
        const auto ik_begin = std::next(ik_end, -1);
        const auto ik = std::distance(n_cum_sum.begin(), ik_begin);  
        assert((ik+1) < n_cum_sum.size());

        // Find i(k)th block matrix.
        const auto& B = D.blocks()[ik];
        const auto stride = n_cum_sum[ik];
        const size_t k_shifted = k_block - stride;

        // Find v_{i(k)}, i(k)th block of vector. 
        const auto k_block_begin = (k / group_size) * group_size;
        const auto B_begin = k_block_begin + stride;
        const auto B_end = k_block_begin+n_cum_sum[ik+1];
        const auto begin = std::lower_bound(v_inner, v_inner+v_nnz, B_begin)-v_inner;
        const auto end = std::lower_bound(v_inner+begin, v_inner+v_nnz, B_end)-v_inner;

        // Compute dot-product
        for (auto j = begin; j != end; ++j) {
            dp += B.coeff(v_inner[j]-B_begin, k_shifted) * v_value[j]; 
        }

        return dp;
    }

    /*
     * Computes the quadratic form of the matrix with v:
     *      v^T A v
     * We leave this unoptimized since it is not crucial to the library.
     */
    template <class VecType>
    value_t quad_form(const VecType& v) const
    {   
        value_t out = 0;
        for (size_t i = 0; i < cols(); ++i) {
            out += v.coeff(i) * col_dot(i, v);
        }
        return out;
    }

    /**
     * Computes an _estimate_ of inverse quadratic form:
     *      v^T [(1-s)A + sI]^{-1} v
     * where 0 <= s <= 1.
     * Note that it is undefined behavior if A
     * is not positive semi-definite.
     * If v == 0, then the result is 0.
     * If s == 0, A is not positive definite but semi-definite,
     * and v != 0 is in the kernel of A, then the result is Inf.
     * If s == 0, A is not positive definite, and not the previous cases,
     * it is undefined behavior.
     * In all other cases, the function will attempt to compute the desired quantity,
     * and is well-defined.
     * 
     * TODO: currently not written since it is not critical.
     */
    template <class VecType>
    value_t inv_quad_form(value_t s, const VecType& v) const
    { return std::numeric_limits<value_t>::quiet_NaN(); }

    inline Scalar coeff(Index i, Index j) const 
    {
        const auto group_size = compute_group_size();
        Index i_shift = shift_[i];
        Index j_shift = shift_[j];
        const auto& S = get_S();
        const auto& D = get_D();
        return S(i_shift, j_shift) + ((i/group_size == j/group_size) ? D.coeff(i_shift, j_shift) : 0);
    }
    
    inline auto to_dense() const
    {
        const auto p = rows_;
        const auto gsize = compute_group_size();
        const auto D_dense = D_.to_dense();
        util::mat_type<value_t> dm(p, p); 
        for (size_t i = 0; i < n_groups_; ++i) {
            const auto si = gsize * i;
            for (size_t j = 0; j < n_groups_; ++j) {
                const auto sj = gsize * j;
                auto curr_block = dm.block(si, sj, gsize, gsize);
                curr_block = S_.to_dense();
                if (i == j) {
                    curr_block += D_dense;
                }
            }
        }
        return dm;
    }
};

} // namespace ghostbasil
