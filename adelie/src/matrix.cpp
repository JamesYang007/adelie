#include "decl.hpp"
#include <adelie_core/matrix/matrix_cov_base.hpp>
#include <adelie_core/matrix/matrix_cov_block_diag.hpp>
#include <adelie_core/matrix/matrix_cov_dense.hpp>
#include <adelie_core/matrix/matrix_cov_lazy_cov.hpp>
#include <adelie_core/matrix/matrix_cov_sparse.hpp>
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/matrix/matrix_naive_concatenate.hpp>
#include <adelie_core/matrix/matrix_naive_dense.hpp>
#include <adelie_core/matrix/matrix_naive_interaction.hpp>
#include <adelie_core/matrix/matrix_naive_kronecker_eye.hpp>
#include <adelie_core/matrix/matrix_naive_one_hot.hpp>
#include <adelie_core/matrix/matrix_naive_snp_unphased.hpp>
#include <adelie_core/matrix/matrix_naive_snp_phased_ancestry.hpp>
#include <adelie_core/matrix/matrix_naive_sparse.hpp>
#include <adelie_core/matrix/matrix_naive_standardize.hpp>
#include <adelie_core/matrix/matrix_naive_subset.hpp>

namespace py = pybind11;
namespace ad = adelie_core;

template <class ValueType = double>
void utils(py::module_& m)
{
    using value_t = ValueType;
    using ref_rowarr_value_t = Eigen::Ref<ad::util::rowarr_type<value_t>>;
    using ref_rowmat_value_t = Eigen::Ref<ad::util::rowmat_type<value_t>>;
    using ref_vec_value_t = Eigen::Ref<ad::util::rowvec_type<value_t>>;
    using ref_mvec_value_t = Eigen::Ref<Eigen::Matrix<value_t, 1, Eigen::Dynamic, Eigen::RowMajor>>;
    using cref_vec_value_t = Eigen::Ref<const ad::util::rowvec_type<value_t>>;
    using cref_rowarr_value_t = Eigen::Ref<const ad::util::rowarr_type<value_t>>;
    using cref_colmat_value_t = Eigen::Ref<const ad::util::colmat_type<value_t>>;
    using cref_mvec_value_t = Eigen::Ref<const Eigen::Matrix<value_t, 1, Eigen::Dynamic, Eigen::RowMajor>>;

    m.def("dvaddi", ad::matrix::dvaddi<ref_vec_value_t, cref_vec_value_t>);
    m.def("dmmeq", ad::matrix::dmmeq<ref_rowarr_value_t, cref_rowarr_value_t>);
    m.def("dvzero", ad::matrix::dvzero<ref_vec_value_t>);
    m.def("ddot", ad::matrix::ddot<cref_mvec_value_t, cref_mvec_value_t, ref_vec_value_t>);
    m.def("dax", ad::matrix::dax<value_t, cref_vec_value_t, ref_vec_value_t>);
    m.def("dgemv", ad::matrix::dgemv<ad::util::operator_type::_eq, cref_colmat_value_t, cref_mvec_value_t, ref_rowmat_value_t, ref_mvec_value_t>);
}

template <class T>
class PyMatrixCovBase: public ad::matrix::MatrixCovBase<T>
{
    using base_t = ad::matrix::MatrixCovBase<T>;
public:
    /* Inherit the constructors */
    using base_t::base_t;
    using typename base_t::value_t;
    using typename base_t::vec_index_t;
    using typename base_t::vec_value_t;
    using typename base_t::colmat_value_t;

    void bmul(
        const Eigen::Ref<const vec_index_t>& subset,
        const Eigen::Ref<const vec_index_t>& indices,
        const Eigen::Ref<const vec_value_t>& values,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            bmul,
            subset, indices, values, out
        );
    }

    void mul(
        const Eigen::Ref<const vec_index_t>& indices,
        const Eigen::Ref<const vec_value_t>& values,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            mul,
            indices, values, out
        );
    }

    void to_dense(
        int i, int p,
        Eigen::Ref<colmat_value_t> out
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            to_dense,
            i, p, out
        );
    }

    int cols() const override
    {
        PYBIND11_OVERRIDE_PURE(
            int,
            base_t,
            cols,
        );
    }
};

template <class T>
class PyMatrixNaiveBase: public ad::matrix::MatrixNaiveBase<T>
{
    using base_t = ad::matrix::MatrixNaiveBase<T>;
public:
    /* Inherit the constructors */
    using base_t::base_t;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using typename base_t::rowmat_value_t;
    using typename base_t::sp_mat_value_t;

    /* Trampoline (need one for each virtual function) */
    value_t cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v,
        const Eigen::Ref<const vec_value_t>& weights
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            value_t,
            base_t,
            cmul,
            j, v, weights
        );
    }

    void ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            ctmul,
            j, v, out
        );
    }

    void bmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            bmul,
            j, q, v, weights, out
        );
    }

    void btmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            btmul,
            j, q, v, out
        );
    }

    void mul(
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            mul,
            v, weights, out
        );
    }

    void sp_btmul(
        const sp_mat_value_t& v,
        Eigen::Ref<rowmat_value_t> out
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            sp_btmul,
            v, out
        );
    }

    void cov(
        int j, int q,
        const Eigen::Ref<const vec_value_t>& sqrt_weights,
        Eigen::Ref<colmat_value_t> out,
        Eigen::Ref<colmat_value_t> buffer
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            cov,
            j, q, sqrt_weights, out, buffer
        );
    }

    int rows() const override
    {
        PYBIND11_OVERRIDE_PURE(
            int,
            base_t,
            rows,
        );
    }

    int cols() const override
    {
        PYBIND11_OVERRIDE_PURE(
            int,
            base_t,
            cols,
        );
    }
};

template <class T>
void matrix_naive_base(py::module_& m, const char* name)
{
    using trampoline_t = PyMatrixNaiveBase<T>;
    using internal_t = ad::matrix::MatrixNaiveBase<T>;
    py::class_<internal_t, trampoline_t>(m, name, R"delimiter(
        Base matrix class for naive method.
        )delimiter")
        .def(py::init<>())
        .def("cmul", &internal_t::cmul, R"delimiter(
        Column vector-vector multiplication.

        Computes the dot-product ``(v * w).T @ X[:,j]`` for a column ``j``.

        Parameters
        ----------
        j : int
            Column index.
        v : (n,) np.ndarray
            Vector to dot product with the ``j`` th column with.
        w : (n,) np.ndarray
            Vector of weights.
        )delimiter")
        .def("ctmul", &internal_t::ctmul, R"delimiter(
        Column vector-scalar multiplication.

        Computes the vector-scalar multiplication ``v * X[:,j]`` for a column ``j``.
        The result is *incremented* into the output vector.

        Parameters
        ----------
        j : int
            Column index.
        v : float
            Scalar to multiply with the ``j`` th column.
        out : (n,) np.ndarray
            Vector to increment in-place the result.
        )delimiter")
        .def("bmul", &internal_t::bmul, R"delimiter(
        Column block matrix-vector multiplication.

        Computes the matrix-vector multiplication ``(v * w).T @ X[:, j:j+q]``.

        Parameters
        ----------
        j : int
            Column index.
        q : int
            Number of columns.
        v : (n,) np.ndarray
            Vector to multiply with the block matrix.
        w : (n,) np.ndarray
            Vector of weights.
        out : (q,) np.ndarray
            Vector to store in-place the result.
        )delimiter")
        .def("btmul", &internal_t::btmul, R"delimiter(
        Column block matrix transpose-vector multiplication.

        Computes the matrix-vector multiplication
        ``v.T @ X[:, j:j+q].T``.
        The result is *incremented* into the output vector.

        Parameters
        ----------
        j : int
            Column index.
        q : int
            Number of columns.
        v : (q,) np.ndarray
            Vector to multiply with the block matrix.
        out : (n,) np.ndarray
            Vector to increment in-place the result.
        )delimiter")
        .def("mul", &internal_t::mul, R"delimiter(
        Matrix-vector multiplication.

        Computes the matrix-vector multiplication
        ``(v * w).T @ X``.

        Parameters
        ----------
        v : (n,) np.ndarray
            Vector to multiply with the block matrix.
        w : (n,) np.ndarray
            Vector of weights.
        out : (q,) np.ndarray
            Vector to store in-place the result.
        )delimiter")
        .def("sp_btmul", &internal_t::sp_btmul, R"delimiter(
        Matrix transpose-sparse matrix multiplication.

        Computes the matrix transpose-sparse matrix multiplication
        ``v @ X.T``.

        Parameters
        ----------
        v : (L, p) scipy.sparse.csr_matrix
            Sparse matrix to multiply with the matrix.
        out : (L, n) np.ndarray
            Matrix to store in-place the result.
        )delimiter")
        .def("cov", &internal_t::cov, R"delimiter(
        Weighted covariance matrix.

        Computes the weighted covariance matrix
        ``X[:, j:j+q].T @ W @ X[:, j:j+q]``.
        
        .. note::
            Although the name is "covariance", we do not center the columns of ``X``!

        Parameters
        ----------
        j : int
            Column index.
        q : int
            Number of columns.
        sqrt_weights : (n,) np.ndarray
            Square-root of the weights.
        out : (q, q) np.ndarray
            Matrix to store in-place the result.
        buffer : (n, q) np.ndarray
            Extra buffer space if needed.
        )delimiter")
        .def("rows", &internal_t::rows, R"delimiter(
        Number of rows.
        )delimiter")
        .def("cols", &internal_t::cols, R"delimiter(
        Number of columns.
        )delimiter")
        /* Augmented API for Python */
        .def_property_readonly("ndim", [](const internal_t&) { return 2; }, R"delimiter(
        Number of dimensions. It is always ``2``.
        )delimiter")
        .def_property_readonly("shape", [](const internal_t& m) {
            return std::make_tuple(m.rows(), m.cols());
        }, R"delimiter(
        Shape of the matrix.
        )delimiter")
        ;
}

template <class T>
void matrix_cov_base(py::module_& m, const char* name)
{
    using trampoline_t = PyMatrixCovBase<T>;
    using internal_t = ad::matrix::MatrixCovBase<T>;
    py::class_<internal_t, trampoline_t>(m, name, R"delimiter(
        Base matrix class for covariance method.
    )delimiter")
        .def(py::init<>())
        .def("bmul", &internal_t::bmul, R"delimiter(
        Block matrix-sparse vector multiplication.

        Computes the matrix-sparse vector multiplication
        ``v.T @ A[:, subset]`` where ``v`` is represented by the sparse-format
        ``indices`` and ``values``.

        Parameters
        ----------
        subset : (s,) np.ndarray
            Vector of column indices of ``A`` to subset in increasing order.
        indices : (nnz,) np.ndarray
            Vector of indices in increasing order.
        values : (nnz,) np.ndarray
            Vector of values associated with ``indices``.
        out : (s,) np.ndarray
            Vector to store the result.
        )delimiter")
        .def("mul", &internal_t::mul, R"delimiter(
        Matrix-sparse vector multiplication.

        Computes the matrix-sparse vector multiplication
        ``v.T @ A`` where ``v`` is represented by the sparse-format
        ``indices`` and ``values``.

        Parameters
        ----------
        indices : (nnz,) np.ndarray
            Vector of indices in increasing order.
        values : (nnz,) np.ndarray
            Vector of values associated with ``indices``.
        out : (n,) np.ndarray
            Vector to store in-place the result.
            The length is the number of columns of ``A``.
        )delimiter")
        .def("to_dense", &internal_t::to_dense, R"delimiter(
        Converts a block to a dense matrix.

        Converts the block ``A[i:i+p, i:i+p]`` into a dense matrix.

        Parameters
        ----------
        i : int
            Row index.
        p : int
            Number of rows.
        out : (p, p) np.ndarray
            Matrix to store the dense result.
        )delimiter")
        .def("rows", &internal_t::rows, R"delimiter(
        Number of rows.
        )delimiter")
        .def("cols", &internal_t::cols, R"delimiter(
        Number of columns.
        )delimiter")
        /* Augmented API for Python */
        .def_property_readonly("ndim", [](const internal_t&) { return 2; }, R"delimiter(
        Number of dimensions. It is always ``2``.
        )delimiter")
        .def_property_readonly("shape", [](const internal_t& m) {
            return std::make_tuple(m.rows(), m.cols());
        }, R"delimiter(
        Shape of the matrix.
        )delimiter")
        ;
}

template <class ValueType>
void matrix_cov_block_diag(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixCovBlockDiag<ValueType>;
    using base_t = typename internal_t::base_t;
    py::class_<internal_t, base_t>(m, name, 
        "Core matrix class for covariance block-diagonal matrix."
        )
        .def(
            py::init([](py::list mat_list_py, size_t n_threads) {
                std::vector<base_t*> mat_list;
                mat_list.reserve(mat_list_py.size());
                for (auto obj : mat_list_py) {
                    mat_list.push_back(py::cast<base_t*>(obj));
                }
                return new internal_t(mat_list, n_threads);
            }), 
            py::arg("mat_list").noconvert(),
            py::arg("n_threads")
        )
        ;
}

template <class DenseType>
void matrix_cov_dense(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixCovDense<DenseType>;
    using base_t = typename internal_t::base_t;
    using dense_t = typename internal_t::dense_t;
    py::class_<internal_t, base_t>(m, name,
        "Core matrix class for covariance dense matrix."
        )
        .def(
            py::init<const Eigen::Ref<const dense_t>&, size_t>(), 
            py::arg("mat").noconvert(),
            py::arg("n_threads")
        )
        ;
}

template <class DenseType>
void matrix_cov_lazy_cov(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixCovLazyCov<DenseType>;
    using base_t = typename internal_t::base_t;
    using dense_t = typename internal_t::dense_t;
    py::class_<internal_t, base_t>(m, name,
        "Core matrix class for covariance lazy-covariance matrix.")
        .def(
            py::init<const Eigen::Ref<const dense_t>&, size_t>(), 
            py::arg("mat").noconvert(),
            py::arg("n_threads")
        )
        ;
}

template <class SparseType>
void matrix_cov_sparse(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixCovSparse<SparseType>;
    using base_t = typename internal_t::base_t;
    using sparse_t = typename internal_t::sparse_t; 
    using vec_sp_value_t = typename internal_t::vec_sp_value_t;
    using vec_sp_index_t = typename internal_t::vec_sp_index_t;
    py::class_<internal_t, base_t>(m, name,
        "Core matrix class for covariance sparse matrix."
        )
        .def(
            py::init<
                size_t,
                size_t,
                size_t,
                const Eigen::Ref<const vec_sp_index_t>&,
                const Eigen::Ref<const vec_sp_index_t>&,
                const Eigen::Ref<const vec_sp_value_t>&,
                size_t
            >(), 
            py::arg("rows"),
            py::arg("cols"),
            py::arg("nnz"),
            py::arg("outer").noconvert(),
            py::arg("inner").noconvert(),
            py::arg("value").noconvert(),
            py::arg("n_threads")
        )
        ;
}

template <class ValueType>
void matrix_naive_cconcatenate(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixNaiveCConcatenate<ValueType>;
    using base_t = typename internal_t::base_t;
    py::class_<internal_t, base_t>(m, name,
        "Core matrix class for naive column-wise concatenated matrix."
        )
        .def(
            py::init([](py::list mat_list_py, size_t n_threads) {
                std::vector<base_t*> mat_list;
                mat_list.reserve(mat_list_py.size());
                for (auto obj : mat_list_py) {
                    mat_list.push_back(py::cast<base_t*>(obj));
                }
                return new internal_t(mat_list, n_threads);
            }), 
            py::arg("mat_list").noconvert(),
            py::arg("n_threads")
        )
        ;
}

template <class ValueType>
void matrix_naive_rconcatenate(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixNaiveRConcatenate<ValueType>;
    using base_t = typename internal_t::base_t;
    py::class_<internal_t, base_t>(m, name,
        "Core matrix class for naive row-wise concatenated matrix."
        )
        .def(
            py::init([](py::list mat_list_py, size_t n_threads) {
                std::vector<base_t*> mat_list;
                mat_list.reserve(mat_list_py.size());
                for (auto obj : mat_list_py) {
                    mat_list.push_back(py::cast<base_t*>(obj));
                }
                return new internal_t(mat_list, n_threads);
            }), 
            py::arg("mat_list").noconvert(),
            py::arg("n_threads")
        )
        ;
}

template <class DenseType>
void matrix_naive_dense(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixNaiveDense<DenseType>;
    using base_t = typename internal_t::base_t;
    using dense_t = typename internal_t::dense_t;
    py::class_<internal_t, base_t>(m, name,
        "Core matrix class for naive dense matrix."
        )
        .def(
            py::init<const Eigen::Ref<const dense_t>&, size_t>(), 
            py::arg("mat").noconvert(),
            py::arg("n_threads")
        )
        ;
}

template <class DenseType>
void matrix_naive_interaction_dense(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixNaiveInteractionDense<DenseType>;
    using base_t = typename internal_t::base_t;
    using dense_t = typename internal_t::dense_t;
    using rowarr_index_t = typename internal_t::rowarr_index_t;
    using vec_index_t = typename internal_t::vec_index_t;
    py::class_<internal_t, base_t>(m, name,
        "Core matrix class for naive interaction matrix."
        )
        .def(
            py::init<
                const Eigen::Ref<const dense_t>&,
                const Eigen::Ref<const rowarr_index_t>&,
                const Eigen::Ref<const vec_index_t>&,
                size_t 
            >(), 
            py::arg("mat").noconvert(),
            py::arg("pairs").noconvert(),
            py::arg("levels").noconvert(),
            py::arg("n_threads")
        )
        .def_property_readonly("groups", &internal_t::groups, R"delimiter(
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
        The groups are naturally defined by ``pairs``.
        In the order of the rows of ``pairs``,
        we group all columns of the current matrix
        corresponding to each row of ``pairs``.
        )delimiter")
        .def_property_readonly("group_sizes", &internal_t::group_sizes, R"delimiter(
        List of group sizes corresponding to each element in ``groups``.
        ``group_sizes[i]`` is the group size of the ``i`` th group. 
        )delimiter")
        ;
}

template <class ValueType>
void matrix_naive_kronecker_eye(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixNaiveKroneckerEye<ValueType>;
    using base_t = typename internal_t::base_t;
    py::class_<internal_t, base_t>(m, name,
        "Core matrix class for naive Kronecker product with identity matrix."
        )
        .def(
            py::init<base_t&, size_t, size_t>(), 
            py::arg("mat"),
            py::arg("K"),
            py::arg("n_threads")
        )
        ;
}

template <class DenseType>
void matrix_naive_kronecker_eye_dense(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixNaiveKroneckerEyeDense<DenseType>;
    using base_t = typename internal_t::base_t;
    using dense_t = typename internal_t::dense_t;
    py::class_<internal_t, base_t>(m, name,
        "Core matrix class for naive Kronecker product (dense) with identity matrix."
        )
        .def(
            py::init<const Eigen::Ref<const dense_t>&, size_t, size_t>(), 
            py::arg("mat").noconvert(),
            py::arg("K"),
            py::arg("n_threads")
        )
        ;
}

template <class DenseType>
void matrix_naive_one_hot_dense(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixNaiveOneHotDense<DenseType>;
    using base_t = typename internal_t::base_t;
    using dense_t = typename internal_t::dense_t;
    using vec_index_t = typename internal_t::vec_index_t;
    py::class_<internal_t, base_t>(m, name,
        "Core matrix class for naive (dense) one-hot encoded matrix."
        )
        .def(
            py::init<
                const Eigen::Ref<const dense_t>&,
                const Eigen::Ref<const vec_index_t>&,
                size_t 
            >(), 
            py::arg("mat").noconvert(),
            py::arg("levels").noconvert(),
            py::arg("n_threads")
        )
        .def_property_readonly("groups", &internal_t::groups, R"delimiter(
        List of starting indices to each group where `G` is the number of groups.
        ``groups[i]`` is the starting index of the ``i`` th group. 
        The groups are naturally defined by the columns of ``mat``.
        In the order of the columns of ``mat``,
        we group all columns of the current matrix 
        corresponding to each column of ``mat``.
        This way, the continuous features each form a group of size one
        and the discrete features form a group across their one-hot encodings.
        )delimiter")
        .def_property_readonly("group_sizes", &internal_t::group_sizes, R"delimiter(
        List of group sizes corresponding to each element in ``groups``.
        ``group_sizes[i]`` is the group size of the ``i`` th group. 
        )delimiter")
        ;
}

template <class ValueType>
void matrix_naive_snp_unphased(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixNaiveSNPUnphased<ValueType>;
    using base_t = typename internal_t::base_t;
    using string_t = typename internal_t::string_t;
    py::class_<internal_t, base_t>(m, name,
        "Core matrix class for naive SNP unphased matrix."
        )
        .def(
            py::init<
                const string_t&,
                const string_t&,
                size_t
            >(), 
            py::arg("filename"),
            py::arg("read_mode"),
            py::arg("n_threads")
        )
        ;
}

template <class ValueType>
void matrix_naive_snp_phased_ancestry(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixNaiveSNPPhasedAncestry<ValueType>;
    using base_t = typename internal_t::base_t;
    using string_t = typename internal_t::string_t;
    py::class_<internal_t, base_t>(m, name,
        "Core matrix class for naive SNP phased, ancestry matrix."
        )
        .def(
            py::init<
                const string_t&,
                const string_t&,
                size_t
            >(), 
            py::arg("filename"),
            py::arg("read_mode"),
            py::arg("n_threads")
        )
        ;
}

template <class SparseType>
void matrix_naive_sparse(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixNaiveSparse<SparseType>;
    using base_t = typename internal_t::base_t;
    using sparse_t = typename internal_t::sparse_t; 
    using vec_sp_value_t = typename internal_t::vec_sp_value_t;
    using vec_sp_index_t = typename internal_t::vec_sp_index_t;
    py::class_<internal_t, base_t>(m, name,
        "Core matrix class for naive sparse matrix."
        )
        .def(
            py::init<
                size_t,
                size_t,
                size_t,
                const Eigen::Ref<const vec_sp_index_t>&,
                const Eigen::Ref<const vec_sp_index_t>&,
                const Eigen::Ref<const vec_sp_value_t>&,
                size_t
            >(), 
            py::arg("rows"),
            py::arg("cols"),
            py::arg("nnz"),
            py::arg("outer").noconvert(),
            py::arg("inner").noconvert(),
            py::arg("value").noconvert(),
            py::arg("n_threads")
        )
        ;
}

template <class ValueType>
void matrix_naive_standardize(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixNaiveStandardize<ValueType>;
    using base_t = typename internal_t::base_t;
    using vec_value_t = typename internal_t::vec_value_t;
    py::class_<internal_t, base_t>(m, name,
        "Core matrix class for naive standardized matrix."
        )
        .def(py::init<
            base_t&,
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            size_t
        >(),
            py::arg("mat"),
            py::arg("centers").noconvert(),
            py::arg("scales").noconvert(),
            py::arg("n_threads")
        )
        ;
}

template <class ValueType>
void matrix_naive_csubset(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixNaiveCSubset<ValueType>;
    using base_t = typename internal_t::base_t;
    using vec_index_t = typename internal_t::vec_index_t;
    py::class_<internal_t, base_t>(m, name,
        "Core matrix class for naive column-subsetted matrix."
        )
        .def(
            py::init<
                base_t*,
                const Eigen::Ref<const vec_index_t>&,
                size_t
            >(),
            py::arg("mat"),
            py::arg("subset").noconvert(),
            py::arg("n_threads")
        )
        ;
}

template <class ValueType>
void matrix_naive_rsubset(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixNaiveRSubset<ValueType>;
    using base_t = typename internal_t::base_t;
    using vec_index_t = typename internal_t::vec_index_t;
    py::class_<internal_t, base_t>(m, name,
        "Core matrix class for naive row-subsetted matrix."
        )
        .def(
            py::init<
                base_t*,
                const Eigen::Ref<const vec_index_t>&,
                size_t
            >(),
            py::arg("mat"),
            py::arg("subset").noconvert(),
            py::arg("n_threads")
        )
        ;
}

template <class T, int Storage>
using dense_type = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Storage>;
template <class T, int Storage>
using sparse_type = Eigen::SparseMatrix<T, Storage>;

void register_matrix(py::module_& m)
{
    /* utils */
    utils(m);

    /* base matrices */
    matrix_naive_base<double>(m, "MatrixNaiveBase64");
    matrix_naive_base<float>(m, "MatrixNaiveBase32");
    matrix_cov_base<double>(m, "MatrixCovBase64");
    matrix_cov_base<float>(m, "MatrixCovBase32");

    /* cov matrices */
    matrix_cov_block_diag<double>(m, "MatrixCovBlockDiag64");
    matrix_cov_block_diag<float>(m, "MatrixCovBlockDiag32");

    matrix_cov_dense<dense_type<double, Eigen::RowMajor>>(m, "MatrixCovDense64C");
    matrix_cov_dense<dense_type<double, Eigen::ColMajor>>(m, "MatrixCovDense64F");
    matrix_cov_dense<dense_type<float, Eigen::RowMajor>>(m, "MatrixCovDense32C");
    matrix_cov_dense<dense_type<float, Eigen::ColMajor>>(m, "MatrixCovDense32F");

    matrix_cov_lazy_cov<dense_type<double, Eigen::RowMajor>>(m, "MatrixCovLazyCov64C");
    matrix_cov_lazy_cov<dense_type<double, Eigen::ColMajor>>(m, "MatrixCovLazyCov64F");
    matrix_cov_lazy_cov<dense_type<float, Eigen::RowMajor>>(m, "MatrixCovLazyCov32C");
    matrix_cov_lazy_cov<dense_type<float, Eigen::ColMajor>>(m, "MatrixCovLazyCov32F");

    matrix_cov_sparse<sparse_type<double, Eigen::ColMajor>>(m, "MatrixCovSparse64F");
    matrix_cov_sparse<sparse_type<float, Eigen::ColMajor>>(m, "MatrixCovSparse32F");

    /* naive matrices */
    matrix_naive_cconcatenate<double>(m, "MatrixNaiveCConcatenate64");
    matrix_naive_cconcatenate<float>(m, "MatrixNaiveCConcatenate32");
    matrix_naive_rconcatenate<double>(m, "MatrixNaiveRConcatenate64");
    matrix_naive_rconcatenate<float>(m, "MatrixNaiveRConcatenate32");

    matrix_naive_dense<dense_type<double, Eigen::RowMajor>>(m, "MatrixNaiveDense64C");
    matrix_naive_dense<dense_type<double, Eigen::ColMajor>>(m, "MatrixNaiveDense64F");
    matrix_naive_dense<dense_type<float, Eigen::RowMajor>>(m, "MatrixNaiveDense32C");
    matrix_naive_dense<dense_type<float, Eigen::ColMajor>>(m, "MatrixNaiveDense32F");

    matrix_naive_interaction_dense<dense_type<double, Eigen::RowMajor>>(m, "MatrixNaiveInteractionDense64C");
    matrix_naive_interaction_dense<dense_type<double, Eigen::ColMajor>>(m, "MatrixNaiveInteractionDense64F");
    matrix_naive_interaction_dense<dense_type<float, Eigen::RowMajor>>(m, "MatrixNaiveInteractionDense32C");
    matrix_naive_interaction_dense<dense_type<float, Eigen::ColMajor>>(m, "MatrixNaiveInteractionDense32F");

    matrix_naive_kronecker_eye<double>(m, "MatrixNaiveKroneckerEye64");
    matrix_naive_kronecker_eye<float>(m, "MatrixNaiveKroneckerEye32");
    matrix_naive_kronecker_eye_dense<dense_type<double, Eigen::RowMajor>>(m, "MatrixNaiveKroneckerEyeDense64C");
    matrix_naive_kronecker_eye_dense<dense_type<double, Eigen::ColMajor>>(m, "MatrixNaiveKroneckerEyeDense64F");
    matrix_naive_kronecker_eye_dense<dense_type<float, Eigen::RowMajor>>(m, "MatrixNaiveKroneckerEyeDense32C");
    matrix_naive_kronecker_eye_dense<dense_type<float, Eigen::ColMajor>>(m, "MatrixNaiveKroneckerEyeDense32F");

    matrix_naive_one_hot_dense<dense_type<double, Eigen::RowMajor>>(m, "MatrixNaiveOneHotDense64C");
    matrix_naive_one_hot_dense<dense_type<double, Eigen::ColMajor>>(m, "MatrixNaiveOneHotDense64F");
    matrix_naive_one_hot_dense<dense_type<float, Eigen::RowMajor>>(m, "MatrixNaiveOneHotDense32C");
    matrix_naive_one_hot_dense<dense_type<float, Eigen::ColMajor>>(m, "MatrixNaiveOneHotDense32F");

    matrix_naive_snp_unphased<double>(m, "MatrixNaiveSNPUnphased64");
    matrix_naive_snp_unphased<float>(m, "MatrixNaiveSNPUnphased32");
    matrix_naive_snp_phased_ancestry<double>(m, "MatrixNaiveSNPPhasedAncestry64");
    matrix_naive_snp_phased_ancestry<float>(m, "MatrixNaiveSNPPhasedAncestry32");

    matrix_naive_sparse<sparse_type<double, Eigen::ColMajor>>(m, "MatrixNaiveSparse64F");
    matrix_naive_sparse<sparse_type<float, Eigen::ColMajor>>(m, "MatrixNaiveSparse32F");

    matrix_naive_standardize<double>(m, "MatrixNaiveStandardize64");
    matrix_naive_standardize<float>(m, "MatrixNaiveStandardize32");

    matrix_naive_csubset<double>(m, "MatrixNaiveCSubset64");
    matrix_naive_csubset<float>(m, "MatrixNaiveCSubset32");
    matrix_naive_rsubset<double>(m, "MatrixNaiveRSubset64");
    matrix_naive_rsubset<float>(m, "MatrixNaiveRSubset32");
}