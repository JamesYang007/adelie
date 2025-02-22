#pragma once
#include <adelie_core/matrix/matrix_naive_block_diag.hpp>
#include <adelie_core/util/omp.hpp>

namespace adelie_core {
namespace matrix {

ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG_TP
auto
ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG::init_rows(
    const std::vector<base_t*>& mat_list
) 
{
    size_t n = 0;
    for (auto mat : mat_list) n += mat->rows();
    return n;
}

ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG_TP
auto
ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG::init_cols(
    const std::vector<base_t*>& mat_list
) 
{
    size_t p = 0;
    for (auto mat : mat_list) p += mat->cols();
    return p;
}

ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG_TP
auto
ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG::init_max_cols(
    const std::vector<base_t*>& mat_list
) 
{
    return vec_index_t::NullaryExpr(
        mat_list.size(), 
        [&](auto i) { return mat_list[i]->cols(); }
    ).maxCoeff();
}

ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG_TP
auto 
ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG::init_col_slice_map(
    const std::vector<base_t*>& mat_list,
   size_t p
) 
{
    vec_index_t slice_map(p);
    size_t begin = 0;
    for (size_t i = 0; i < mat_list.size(); ++i) {
        const auto& mat = *mat_list[i];
        const auto pi = mat.cols();
        for (int j = 0; j < pi; ++j) {
            slice_map[begin + j] = i;
        }
        begin += pi;
    } 
    return slice_map;
}

ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG_TP
auto 
ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG::init_col_index_map(
    const std::vector<base_t*>& mat_list,
    size_t p
) 
{
    vec_index_t index_map(p);
    size_t begin = 0;
    for (size_t i = 0; i < mat_list.size(); ++i) {
        const auto& mat = *mat_list[i];
        const auto pi = mat.cols();
        for (int j = 0; j < pi; ++j) {
            index_map[begin + j] = j;
        }
        begin += pi;
    } 
    return index_map;
}

ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG_TP
auto 
ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG::init_row_outer(
    const std::vector<base_t*>& mat_list
) 
{
    vec_index_t outer(mat_list.size()+1);
    outer[0] = 0;
    for (size_t i = 0; i < mat_list.size(); ++i) {
        const auto& mat = *mat_list[i];
        const auto ni = mat.rows();
        outer[i+1] = ni + outer[i];
    } 
    return outer;
}

ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG_TP
auto 
ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG::init_col_outer(
    const std::vector<base_t*>& mat_list
) 
{
    vec_index_t outer(mat_list.size()+1);
    outer[0] = 0;
    for (size_t i = 0; i < mat_list.size(); ++i) {
        const auto& mat = *mat_list[i];
        const auto pi = mat.cols();
        outer[i+1] = pi + outer[i];
    } 
    return outer;
}

ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG_TP
ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG::MatrixNaiveBlockDiag(
    const std::vector<base_t*>& mat_list,
    size_t n_threads
):
    _mat_list(mat_list),
    _rows(init_rows(mat_list)),
    _cols(init_cols(mat_list)),
    _max_cols(init_max_cols(mat_list)),
    _col_slice_map(init_col_slice_map(mat_list, _cols)),
    _col_index_map(init_col_index_map(mat_list, _cols)),
    _row_outer(init_row_outer(mat_list)),
    _col_outer(init_col_outer(mat_list)),
    _n_threads(n_threads)
{
    if (mat_list.size() <= 0) {
        throw util::adelie_core_error("mat_list must be non-empty.");
    }
    if (n_threads < 1) {
        throw util::adelie_core_error("n_threads must be >= 1.");
    }
}

ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG_TP
typename ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG::value_t 
ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG::cmul(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights
)
{
    base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
    const auto slice = _col_slice_map[j];
    auto& mat = *_mat_list[slice];
    const auto index = _col_index_map[j];
    const auto r_begin = _row_outer[slice];
    const auto r_size = _row_outer[slice+1] - r_begin;
    const auto v_slice = v.segment(r_begin, r_size);
    const auto w_slice = weights.segment(r_begin, r_size);
    return mat.cmul(index, v_slice, w_slice);
}

ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG_TP
typename ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG::value_t 
ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG::cmul_safe(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights
) const
{
    base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
    const auto slice = _col_slice_map[j];
    const auto& mat = *_mat_list[slice];
    const auto index = _col_index_map[j];
    const auto r_begin = _row_outer[slice];
    const auto r_size = _row_outer[slice+1] - r_begin;
    const auto v_slice = v.segment(r_begin, r_size);
    const auto w_slice = weights.segment(r_begin, r_size);
    return mat.cmul_safe(index, v_slice, w_slice);
}

ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG_TP
void
ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG::ctmul(
    int j, 
    value_t v, 
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_ctmul(j, out.size(), rows(), cols());
    const auto slice = _col_slice_map[j];
    auto& mat = *_mat_list[slice];
    const auto index = _col_index_map[j];
    const auto r_begin = _row_outer[slice];
    const auto r_size = _row_outer[slice+1] - r_begin;
    auto out_slice = out.segment(r_begin, r_size);
    return mat.ctmul(index, v, out_slice);
}

ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG_TP
void
ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG::bmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
    int n_processed = 0;
    while (n_processed < q) {
        const auto k = j + n_processed;
        const auto slice = _col_slice_map[k];
        auto& mat = *_mat_list[slice];
        const auto index = _col_index_map[k];
        const auto size = std::min<size_t>(mat.cols()-index, q-n_processed);
        auto out_slice = out.segment(n_processed, size);
        const auto r_begin = _row_outer[slice];
        const auto r_size = _row_outer[slice+1] - r_begin;
        const auto v_slice = v.segment(r_begin, r_size);
        const auto w_slice = weights.segment(r_begin, r_size);
        mat.bmul(index, size, v_slice, w_slice, out_slice);
        n_processed += size;
    }
}

ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG_TP
void
ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG::bmul_safe(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
    int n_processed = 0;
    while (n_processed < q) {
        const auto k = j + n_processed;
        const auto slice = _col_slice_map[k];
        const auto& mat = *_mat_list[slice];
        const auto index = _col_index_map[k];
        const auto size = std::min<size_t>(mat.cols()-index, q-n_processed);
        auto out_slice = out.segment(n_processed, size);
        const auto r_begin = _row_outer[slice];
        const auto r_size = _row_outer[slice+1] - r_begin;
        const auto v_slice = v.segment(r_begin, r_size);
        const auto w_slice = weights.segment(r_begin, r_size);
        mat.bmul_safe(index, size, v_slice, w_slice, out_slice);
        n_processed += size;
    }
}

ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG_TP
void
ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG::btmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    Eigen::Ref<vec_value_t> out
)
{
    base_t::check_btmul(j, q, v.size(), out.size(), rows(), cols());
    int n_processed = 0;
    while (n_processed < q) {
        const auto k = j + n_processed;
        const auto slice = _col_slice_map[k];
        auto& mat = *_mat_list[slice];
        const auto index = _col_index_map[k];
        const auto size = std::min<size_t>(mat.cols()-index, q-n_processed);
        const auto v_slice = v.segment(n_processed, size);
        const auto r_begin = _row_outer[slice];
        const auto r_size = _row_outer[slice+1] - r_begin;
        auto out_slice = out.segment(r_begin, r_size);
        mat.btmul(index, size, v_slice, out_slice);
        n_processed += size;
    }
}

ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG_TP
void
ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG::mul(
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    const auto routine = [&](auto g) {
        const auto& mat = *_mat_list[g];
        const auto r_begin = _row_outer[g];
        const auto r_size = _row_outer[g+1] - r_begin;
        const auto v_slice = v.segment(r_begin, r_size);
        const auto w_slice = weights.segment(r_begin, r_size);
        const auto c_begin = _col_outer[g];
        const auto c_size = _col_outer[g+1] - c_begin;
        auto out_slice = out.segment(c_begin, c_size);
        mat.mul(v_slice, w_slice, out_slice);
    };
    util::omp_parallel_for(routine, 0, _mat_list.size(), _n_threads * (_n_threads <= _mat_list.size()));
}

ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG_TP
void
ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG::cov(
    int j, int q,
    const Eigen::Ref<const vec_value_t>& sqrt_weights,
    Eigen::Ref<colmat_value_t> out
) const
{
    base_t::check_cov(j, q, sqrt_weights.size(), out.rows(), out.cols(), rows(), cols());
    vec_value_t buff(_max_cols * _max_cols);
    out.setZero();
    int n_processed = 0;
    while (n_processed < q) {
        const auto k = j + n_processed;
        const auto slice = _col_slice_map[k];
        const auto& mat = *_mat_list[slice];
        const auto index = _col_index_map[k];
        const auto size = std::min<size_t>(mat.cols()-index, q-n_processed);
        Eigen::Map<colmat_value_t> out_curr(buff.data(), size, size);
        const auto r_begin = _row_outer[slice];
        const auto r_size = _row_outer[slice+1] - r_begin;
        const auto sqrt_w_slice = sqrt_weights.segment(r_begin, r_size);
        mat.cov(index, size, sqrt_w_slice, out_curr);
        out.block(n_processed, n_processed, size, size) = out_curr;
        n_processed += size;
    }
}

ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG_TP
int
ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG::rows() const
{
    return _rows;
}

ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG_TP
int
ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG::cols() const
{
    return _cols;
}

ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG_TP
void
ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG::sq_mul(
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    const auto routine = [&](auto g) {
        const auto& mat = *_mat_list[g];
        const auto r_begin = _row_outer[g];
        const auto r_size = _row_outer[g+1] - r_begin;
        const auto w_slice = weights.segment(r_begin, r_size);
        const auto c_begin = _col_outer[g];
        const auto c_size = _col_outer[g+1] - c_begin;
        auto out_slice = out.segment(c_begin, c_size);
        mat.sq_mul(w_slice, out_slice);
    };
    util::omp_parallel_for(routine, 0, _mat_list.size(), _n_threads * (_n_threads <= _mat_list.size()));
}

ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG_TP
void
ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG::sp_tmul(
    const sp_mat_value_t& v,
    Eigen::Ref<rowmat_value_t> out
) const
{
    base_t::check_sp_tmul(
        v.rows(), v.cols(), out.rows(), out.cols(), rows(), cols()
    );
    out.setZero();
    const auto routine = [&](auto g) {
        const auto& mat = *_mat_list[g];
        const auto c_begin = _col_outer[g];
        const auto c_size = _col_outer[g+1] - c_begin;
        rowmat_value_t out_curr(out.rows(), mat.rows());
        mat.sp_tmul(v.middleCols(c_begin, c_size), out_curr);
        const auto r_begin = _row_outer[g];
        const auto r_size = _row_outer[g+1] - r_begin;
        out.middleCols(r_begin, r_size) = out_curr;
    };
    util::omp_parallel_for(routine, 0, _mat_list.size(), _n_threads * (_n_threads <= _mat_list.size()));
}

ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG_TP
void
ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG::mean(
    const Eigen::Ref<const vec_value_t>&,
    Eigen::Ref<vec_value_t>
) const
{
    throw util::adelie_core_error(
        "MatrixNaiveBlockDiag: mean() not implemented! "
        "If this error occurred from standardizing the matrix, "
        "consider providing your own center vector. "
    );
}

ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG_TP
void
ADELIE_CORE_MATRIX_NAIVE_BLOCK_DIAG::var(
    const Eigen::Ref<const vec_value_t>&,
    const Eigen::Ref<const vec_value_t>&,
    Eigen::Ref<vec_value_t>
) const
{
    throw util::adelie_core_error(
        "MatrixNaiveBlockDiag: var() not implemented! "
        "If this error occurred from standardizing the matrix, "
        "consider providing your own scale vector. "
    );
}

} // namespace matrix 
} // namespace adelie_core