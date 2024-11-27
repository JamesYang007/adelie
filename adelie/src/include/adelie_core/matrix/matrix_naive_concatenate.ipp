#pragma once
#include <adelie_core/matrix/matrix_naive_concatenate.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE_TP
auto 
ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE::init_rows(
    const std::vector<base_t*>& mat_list
)
{
    if (mat_list.size() == 0) {
        throw util::adelie_core_error("List must be non-empty.");
    }

    const auto n = mat_list[0]->rows();
    for (auto mat : mat_list) {
        if (n != mat->rows()) {
            throw util::adelie_core_error("All matrices must have the same number of rows.");
        }
    }

    return n;
}

ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE_TP
auto 
ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE::init_cols(
    const std::vector<base_t*>& mat_list
)
{
    size_t p = 0;
    for (auto mat : mat_list) p += mat->cols();
    return p;
}

ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE_TP
auto
ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE::init_outer(
    const std::vector<base_t*>& mat_list
)
{
    vec_index_t outer(mat_list.size() + 1);
    outer[0] = 0;
    size_t begin = 0;
    for (size_t i = 0; i < mat_list.size(); ++i) {
        const auto& mat = *mat_list[i];
        const auto pi = mat.cols();
        outer[i+1] = outer[i] + pi;
        begin += pi;
    } 
    return outer;
}

ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE_TP
auto 
ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE::init_slice_map(
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

ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE_TP
auto 
ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE::init_index_map(
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

ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE_TP
ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE::MatrixNaiveCConcatenate(
    const std::vector<base_t*>& mat_list,
    size_t n_threads
): 
    _mat_list(mat_list),
    _rows(init_rows(mat_list)),
    _cols(init_cols(mat_list)),
    _outer(init_outer(mat_list)),
    _slice_map(init_slice_map(mat_list, _cols)),
    _index_map(init_index_map(mat_list, _cols)),
    _n_threads(n_threads)
{
    if (mat_list.size() <= 0) {
        throw util::adelie_core_error("mat_list must be non-empty.");
    }
    if (n_threads < 1) {
        throw util::adelie_core_error("n_threads must be >= 1.");
    }
}

ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE_TP
typename ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE::value_t
ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE::cmul(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights
) 
{
    base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
    const auto slice = _slice_map[j];
    auto& mat = *_mat_list[slice];
    const auto index = _index_map[j];
    return mat.cmul(index, v, weights);
}

ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE_TP
typename ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE::value_t
ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE::cmul_safe(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights
) const
{
    base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
    const auto slice = _slice_map[j];
    const auto& mat = *_mat_list[slice];
    const auto index = _index_map[j];
    return mat.cmul_safe(index, v, weights);
}

ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE_TP
void
ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE::ctmul(
    int j, 
    value_t v, 
    Eigen::Ref<vec_value_t> out
) 
{
    base_t::check_ctmul(j, out.size(), rows(), cols());
    const auto slice = _slice_map[j];
    auto& mat = *_mat_list[slice];
    const auto index = _index_map[j];
    mat.ctmul(index, v, out);
}

ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE_TP
void
ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE::bmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) 
{
    base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
    int n_processed = 0;
    while (n_processed < q) {
        const auto j_curr = j + n_processed;
        const auto slice = _slice_map[j_curr];
        auto& mat = *_mat_list[slice];
        const auto index = _index_map[j_curr];
        const int q_curr = std::min<int>(mat.cols()-index, q-n_processed);
        mat.bmul(index, q_curr, v, weights, out.segment(n_processed, q_curr));
        n_processed += q_curr;
    }
}

ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE_TP
void
ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE::bmul_safe(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const 
{
    base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
    int n_processed = 0;
    while (n_processed < q) {
        const auto j_curr = j + n_processed;
        const auto slice = _slice_map[j_curr];
        const auto& mat = *_mat_list[slice];
        const auto index = _index_map[j_curr];
        const int q_curr = std::min<int>(mat.cols()-index, q-n_processed);
        mat.bmul_safe(index, q_curr, v, weights, out.segment(n_processed, q_curr));
        n_processed += q_curr;
    }
}

ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE_TP
void
ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE::btmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    Eigen::Ref<vec_value_t> out
) 
{
    base_t::check_btmul(j, q, v.size(), out.size(), rows(), cols());
    int n_processed = 0;
    while (n_processed < q) {
        const auto j_curr = j + n_processed;
        const auto slice = _slice_map[j_curr];
        auto& mat = *_mat_list[slice];
        const auto index = _index_map[j_curr];
        const int q_curr = std::min<int>(mat.cols()-index, q-n_processed);
        mat.btmul(index, q_curr, v.segment(n_processed, q_curr), out);
        n_processed += q_curr;
    }
}

ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE_TP
void
ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE::mul(
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    const auto routine = [&](auto i) {
        const auto outer_i = _outer[i];
        const auto& mat = *_mat_list[i];
        const auto p = mat.cols();
        mat.mul(v, weights, out.segment(outer_i, p));
    };
    util::omp_parallel_for(routine, 0, _mat_list.size(), _n_threads * (_n_threads <= _mat_list.size()));
}

ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE_TP
int
ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE::rows() const
{
    return _rows;
}

ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE_TP
int
ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE::cols() const
{
    return _cols;
}

ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE_TP
void
ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE::cov(
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
    const auto& mat = *_mat_list[slice];
    const auto index = _index_map[j];

    // check that the block is fully contained in one matrix
    if (slice != _slice_map[j+q-1]) {
        throw util::adelie_core_error(
            "MatrixNaiveCConcatenate::cov() only allows the block to be fully contained in one of the matrices in the list."
        );
    }

    mat.cov(index, q, sqrt_weights, out);
}


ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE_TP
void
ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE::sq_mul(
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    const auto routine = [&](auto i) {
        const auto outer_i = _outer[i];
        const auto& mat = *_mat_list[i];
        const auto p = mat.cols();
        mat.sq_mul(weights, out.segment(outer_i, p));
    };
    util::omp_parallel_for(routine, 0, _mat_list.size(), _n_threads * (_n_threads <= _mat_list.size()));
}

ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE_TP
void
ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE::sp_tmul(
    const sp_mat_value_t& v, 
    Eigen::Ref<rowmat_value_t> out
) const
{
    base_t::check_sp_tmul(
        v.rows(), v.cols(), out.rows(), out.cols(), rows(), cols()
    );
    out.setZero();
    rowmat_value_t buff(out.rows(), out.cols());
    int n_processed = 0;
    for (size_t i = 0; i < _mat_list.size(); ++i) {
        const auto& mat = *_mat_list[i];
        const auto q_curr = mat.cols();
        mat.sp_tmul(v.middleCols(n_processed, q_curr), buff);
        out += buff;
        n_processed += q_curr;
    }
}

ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE_TP
void
ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE::mean(
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    const auto routine = [&](auto i) {
        const auto outer_i = _outer[i];
        const auto& mat = *_mat_list[i];
        const auto p = mat.cols();
        mat.mean(weights, out.segment(outer_i, p));
    };
    util::omp_parallel_for(routine, 0, _mat_list.size(), _n_threads * (_n_threads <= _mat_list.size()));
}

ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE_TP
void
ADELIE_CORE_MATRIX_NAIVE_CCONCATENATE::var(
    const Eigen::Ref<const vec_value_t>& centers,
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    const auto routine = [&](auto i) {
        const auto outer_i = _outer[i];
        const auto& mat = *_mat_list[i];
        const auto p = mat.cols();
        mat.var(centers.segment(outer_i, p), weights, out.segment(outer_i, p));
    };
    util::omp_parallel_for(routine, 0, _mat_list.size(), _n_threads * (_n_threads <= _mat_list.size()));
}

ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE_TP
auto 
ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE::init_rows(
    const std::vector<base_t*>& mat_list
)
{
    size_t n = 0;
    for (auto mat : mat_list) n += mat->rows();
    return n;
}

ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE_TP
auto 
ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE::init_cols(
    const std::vector<base_t*>& mat_list
)
{
    if (mat_list.size() == 0) {
        throw util::adelie_core_error("List must be non-empty.");
    }

    const auto p = mat_list[0]->cols();
    for (auto mat : mat_list) {
        if (p != mat->cols()) {
            throw util::adelie_core_error("All matrices must have the same number of columns.");
        }
    }

    return p;
}

ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE_TP
auto
ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE::init_outer(
    const std::vector<base_t*>& mat_list
)
{
    vec_index_t outer(mat_list.size() + 1);
    outer[0] = 0;
    size_t begin = 0;
    for (size_t i = 0; i < mat_list.size(); ++i) {
        const auto& mat = *mat_list[i];
        const auto pi = mat.rows();
        outer[i+1] = outer[i] + pi;
        begin += pi;
    } 
    return outer;
}

ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE_TP
ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE::MatrixNaiveRConcatenate(
    const std::vector<base_t*>& mat_list,
    size_t n_threads
): 
    _mat_list(mat_list),
    _rows(init_rows(mat_list)),
    _cols(init_cols(mat_list)),
    _outer(init_outer(mat_list)),
    _n_threads(n_threads),
    _buff(_cols)
{
    if (mat_list.size() <= 0) {
        throw util::adelie_core_error("mat_list must be non-empty.");
    }
    if (n_threads < 1) {
        throw util::adelie_core_error("n_threads must be >= 1.");
    }
}

ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE_TP
typename ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE::value_t
ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE::cmul(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights
) 
{
    base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
    value_t sum = 0;
    // NOTE: cannot parallelize since cmul may not be thread-safe!
    for (size_t i = 0; i < _mat_list.size(); ++i) {
        const auto outer_i = _outer[i];
        auto& mat = *_mat_list[i];
        const auto rows_curr = mat.rows();
        const Eigen::Map<const vec_value_t> v_curr(
            v.data() + outer_i, rows_curr
        );
        const Eigen::Map<const vec_value_t> weights_curr(
            weights.data() + outer_i, rows_curr
        );
        sum += mat.cmul(j, v_curr, weights_curr);
    };
    return sum;
}

ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE_TP
typename ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE::value_t
ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE::cmul_safe(
    int j, 
    const Eigen::Ref<const vec_value_t>& v,
    const Eigen::Ref<const vec_value_t>& weights
) const
{
    base_t::check_cmul(j, v.size(), weights.size(), rows(), cols());
    vec_value_t buff(_mat_list.size());
    const auto routine = [&](auto i) {
        const auto outer_i = _outer[i];
        const auto& mat = *_mat_list[i];
        const auto rows_curr = mat.rows();
        const Eigen::Map<const vec_value_t> v_curr(
            v.data() + outer_i, rows_curr
        );
        const Eigen::Map<const vec_value_t> weights_curr(
            weights.data() + outer_i, rows_curr
        );
        buff[i] = mat.cmul_safe(j, v_curr, weights_curr);
    };
    util::omp_parallel_for(routine, 0, _mat_list.size(), _n_threads * (_n_threads <= _mat_list.size()));
    return buff.sum();
}

ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE_TP
void
ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE::ctmul(
    int j, 
    value_t v, 
    Eigen::Ref<vec_value_t> out
) 
{
    base_t::check_ctmul(j, out.size(), rows(), cols());
    for (size_t i = 0; i < _mat_list.size(); ++i) {
        const auto outer_i = _outer[i];
        auto& mat = *_mat_list[i];
        const auto rows_curr = mat.rows();
        Eigen::Map<vec_value_t> out_curr(
            out.data() + outer_i, rows_curr
        );
        mat.ctmul(j, v, out_curr);
    }
}

ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE_TP
void
ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE::bmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) 
{
    base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
    out.setZero();
    Eigen::Map<vec_value_t> buff(_buff.data(), q);
    for (size_t i = 0; i < _mat_list.size(); ++i) {
        const auto outer_i = _outer[i];
        auto& mat = *_mat_list[i];
        const auto rows_curr = mat.rows();
        const Eigen::Map<const vec_value_t> v_curr(
            v.data() + outer_i, rows_curr
        );
        const Eigen::Map<const vec_value_t> weights_curr(
            weights.data() + outer_i, rows_curr
        );
        mat.bmul(j, q, v_curr, weights_curr, buff);
        out += buff;
    }
}

ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE_TP
void
ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE::bmul_safe(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const 
{
    base_t::check_bmul(j, q, v.size(), weights.size(), out.size(), rows(), cols());
    out.setZero();
    vec_value_t buff(q);
    for (size_t i = 0; i < _mat_list.size(); ++i) {
        const auto outer_i = _outer[i];
        const auto& mat = *_mat_list[i];
        const auto rows_curr = mat.rows();
        const Eigen::Map<const vec_value_t> v_curr(
            v.data() + outer_i, rows_curr
        );
        const Eigen::Map<const vec_value_t> weights_curr(
            weights.data() + outer_i, rows_curr
        );
        mat.bmul_safe(j, q, v_curr, weights_curr, buff);
        out += buff;
    }
}

ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE_TP
void
ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE::btmul(
    int j, int q, 
    const Eigen::Ref<const vec_value_t>& v, 
    Eigen::Ref<vec_value_t> out
) 
{
    base_t::check_btmul(j, q, v.size(), out.size(), rows(), cols());
    for (size_t i = 0; i < _mat_list.size(); ++i) {
        const auto outer_i = _outer[i];
        auto& mat = *_mat_list[i];
        const auto rows_curr = mat.rows();
        Eigen::Map<vec_value_t> out_curr(
            out.data() + outer_i, rows_curr
        );
        mat.btmul(j, q, v, out_curr);
    }
}

ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE_TP
void
ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE::mul(
    const Eigen::Ref<const vec_value_t>& v, 
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    out.setZero();
    vec_value_t buff(out.size());
    for (size_t i = 0; i < _mat_list.size(); ++i) {
        const auto outer_i = _outer[i];
        const auto& mat = *_mat_list[i];
        const auto rows_curr = mat.rows();
        const Eigen::Map<const vec_value_t> v_curr(
            v.data() + outer_i, rows_curr
        );
        const Eigen::Map<const vec_value_t> weights_curr(
            weights.data() + outer_i, rows_curr
        );
        mat.mul(v_curr, weights_curr, buff);
        out += buff;
    };
}

ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE_TP
int
ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE::rows() const
{
    return _rows;
}

ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE_TP
int
ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE::cols() const
{
    return _cols;
}

ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE_TP
void
ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE::cov(
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

    vec_value_t buff(q * q);

    out.setZero();
    for (size_t i = 0; i < _mat_list.size(); ++i) {
        const auto outer_i = _outer[i];
        const auto& mat = *_mat_list[i];
        const auto rows_curr = mat.rows();
        const Eigen::Map<const vec_value_t> sqrt_weights_curr(
            sqrt_weights.data() + outer_i, rows_curr
        );
        Eigen::Map<colmat_value_t> out_curr(
            buff.data(), q, q
        );
        mat.cov(j, q, sqrt_weights_curr, out_curr);
        out += out_curr;
    }
}

ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE_TP
void
ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE::sq_mul(
    const Eigen::Ref<const vec_value_t>& weights,
    Eigen::Ref<vec_value_t> out
) const
{
    out.setZero();
    vec_value_t buff(out.size());
    for (size_t i = 0; i < _mat_list.size(); ++i) {
        const auto outer_i = _outer[i];
        const auto& mat = *_mat_list[i];
        const auto rows_curr = mat.rows();
        const Eigen::Map<const vec_value_t> weights_curr(
            weights.data() + outer_i, rows_curr
        );
        mat.sq_mul(weights_curr, buff);
        out += buff;
    }
}

ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE_TP
void
ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE::sp_tmul(
    const sp_mat_value_t& v, 
    Eigen::Ref<rowmat_value_t> out
) const
{
    base_t::check_sp_tmul(
        v.rows(), v.cols(), out.rows(), out.cols(), rows(), cols()
    );
    vec_value_t buff;

    const auto L = v.rows();
    for (size_t i = 0; i < _mat_list.size(); ++i) {
        const auto outer_i = _outer[i];
        const auto& mat = *_mat_list[i];
        const auto rows_curr = mat.rows();
        if (buff.size() < L * rows_curr) buff.resize(L * rows_curr);
        Eigen::Map<rowmat_value_t> out_curr(
            buff.data(), L, rows_curr
        );
        mat.sp_tmul(v, out_curr);
        out.middleCols(outer_i, rows_curr) = out_curr;
    }
}

ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE_TP
void
ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE::mean(
    const Eigen::Ref<const vec_value_t>&,
    Eigen::Ref<vec_value_t> 
) const
{
    throw util::adelie_core_error(
        "MatrixNaiveRConcatenate: mean() not implemented! "
        "If this error occurred from standardizing the matrix, "
        "consider providing your own center vector. "
    );
}

ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE_TP
void
ADELIE_CORE_MATRIX_NAIVE_RCONCATENATE::var(
    const Eigen::Ref<const vec_value_t>&,
    const Eigen::Ref<const vec_value_t>&,
    Eigen::Ref<vec_value_t> 
) const
{
    throw util::adelie_core_error(
        "MatrixNaiveRConcatenate: var() not implemented! "
        "If this error occurred from standardizing the matrix, "
        "consider providing your own scale vector. "
    );
}

} // namespace matrix 
} // namespace adelie_core