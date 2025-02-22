#include "py_decl.hpp"
#include <matrix/matrix.hpp>

namespace py = pybind11;
namespace ad = adelie_core;

template <class T>
class PyMatrixConstraintBase: public ad::matrix::MatrixConstraintBase<T>
{
    using base_t = ad::matrix::MatrixConstraintBase<T>;
public:
    /* Inherit the constructors */
    using base_t::base_t;
    using typename base_t::value_t;
    using typename base_t::vec_index_t;
    using typename base_t::vec_value_t;
    using typename base_t::colmat_value_t;

    void rmmul(
        int j, 
        const Eigen::Ref<const colmat_value_t>& Q,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            rmmul,
            j, Q, out
        );
    }

    void rmmul_safe(
        int j, 
        const Eigen::Ref<const colmat_value_t>& Q,
        Eigen::Ref<vec_value_t> out
    ) const override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            rmmul_safe,
            j, Q, out
        );
    }

    value_t rvmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            value_t,
            base_t,
            rvmul,
            j, v
        );
    }

    value_t rvmul_safe(
        int j, 
        const Eigen::Ref<const vec_value_t>& v
    ) const override
    {
        PYBIND11_OVERRIDE_PURE(
            value_t,
            base_t,
            rvmul_safe,
            j, v
        );
    }

    void rvtmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            rvtmul,
            j, v, out
        );
    }

    void mul(
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) const override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            mul,
            v, out
        );
    }

    void tmul(
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) const override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            tmul,
            v, out
        );
    }

    void cov(
        const Eigen::Ref<const colmat_value_t>& Q,
        Eigen::Ref<colmat_value_t> out
    ) const override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            cov,
            Q, out
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

    void sp_mul(
        const Eigen::Ref<const vec_index_t>& indices,
        const Eigen::Ref<const vec_value_t>& values,
        Eigen::Ref<vec_value_t> out
    ) const override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            sp_mul,
            indices, values, out
        );
    }
};

template <class T>
void matrix_constraint_base(py::module_& m, const char* name)
{
    using trampoline_t = PyMatrixConstraintBase<T>;
    using internal_t = ad::matrix::MatrixConstraintBase<T>;
    py::class_<internal_t, trampoline_t>(m, name, R"delimiter(
        Base matrix class for constraint matrices.
    )delimiter")
        .def(py::init<>())
        .def("rmmul", &internal_t::rmmul, R"delimiter(
        Computes a row vector-matrix multiplication.

        Computes the matrix-vector multiplication 
        ``A[j].T @ Q``.

        .. warning::
            This function is not thread-safe!

        Parameters
        ----------
        j : int
            Row index.
        Q : (d, d) ndarray
            Matrix to dot product with the ``j`` th row.
        out : (d,) ndarray
            Vector to store in-place the result.
        )delimiter")
        .def("rmmul_safe", &internal_t::rmmul_safe, R"delimiter(
        Computes a row vector-matrix multiplication.

        Thread-safe version of :func:`rmmul`.

        Parameters
        ----------
        j : int
            Row index.
        Q : (d, d) ndarray
            Matrix to dot product with the ``j`` th row.
        out : (d,) ndarray
            Vector to store in-place the result.
        )delimiter")
        .def("rvmul", &internal_t::rvmul, R"delimiter(
        Computes a row vector-vector multiplication.

        Computes the dot-product
        ``A[j].T @ v``.

        .. warning::
            This function is not thread-safe!

        Parameters
        ----------
        j : int
            Row index.
        v : (d,) ndarray
            Vector to dot product with the ``j`` th row.

        Returns
        -------
        dot : float
            Row vector-vector multiplication.
        )delimiter")
        .def("rvmul_safe", &internal_t::rvmul_safe, R"delimiter(
        Computes a row vector-vector multiplication.

        Thread-safe version of :func:`rvmul`.

        Parameters
        ----------
        j : int
            Row index.
        v : (d,) ndarray
            Vector to dot product with the ``j`` th row.

        Returns
        -------
        dot : float
            Row vector-vector multiplication.
        )delimiter")
        .def("rvtmul", &internal_t::rvtmul, R"delimiter(
        Computes a row vector-scalar multiplication increment.

        Computes the vector-scalar multiplication ``A[j] * v``.
        The result is *incremented* into the output vector.

        .. warning::
            This function is not thread-safe!

        Parameters
        ----------
        j : int
            Row index.
        v : float
            Scalar to multiply with the ``j`` th row.
        out : (d,) ndarray
            Vector to increment in-place the result.
        )delimiter")
        .def("mul", &internal_t::mul, R"delimiter(
        Computes a matrix-vector multiplication.

        Computes the matrix-vector multiplication
        ``v.T @ A``.

        Parameters
        ----------
        v : (m,) ndarray
            Vector to multiply with the matrix.
        out : (d,) ndarray
            Vector to store in-place the result.
        )delimiter")
        .def("tmul", &internal_t::tmul, R"delimiter(
        Computes a matrix transpose-vector multiplication.

        Computes the matrix transpose-vector multiplication
        ``v.T @ A.T``.

        Parameters
        ----------
        v : (d,) ndarray
            Vector to multiply with the matrix.
        out : (m,) ndarray
            Vector to store in-place the result.
        )delimiter")
        .def("cov", &internal_t::cov, R"delimiter(
        Computes the covariance matrix.

        Computes the covariance matrix ``A @ Q @ A.T``.

        Parameters
        ----------
        Q : (d, d) ndarray
            Matrix of weights.
        out : (m, m) ndarray
            Matrix to store in-place the result.
        )delimiter")
        .def("rows", &internal_t::rows, R"delimiter(
        Returns the number of rows.

        Returns
        -------
        rows : int
            Number of rows.
        )delimiter")
        .def("cols", &internal_t::cols, R"delimiter(
        Returns the number of columns.

        Returns
        -------
        cols : int
            Number of columns.
        )delimiter")
        .def("sp_mul", &internal_t::sp_mul, R"delimiter(
        Computes a matrix-sparse vector multiplication.

        Computes the matrix-sparse vector multiplication
        ``v.T @ A`` where ``v`` is represented by the sparse-format 
        ``indices`` and ``values``.

        Parameters
        ----------
        indices : (nnz,) ndarray
            Vector of indices with non-zero values of ``v``.
            It does not have to be sorted in increasing order.
        values : (nnz,) ndarray
            Vector of values corresponding to ``indices``.
        out : (d,) ndarray
            Vector to store in-place the result.
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

template <class DenseType>
void matrix_constraint_dense(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixConstraintDense<DenseType>;
    using base_t = typename internal_t::base_t;
    using dense_t = typename internal_t::dense_t;
    py::class_<internal_t, base_t>(m, name,
        "Core matrix class for dense constraint matrix."
        )
        .def(
            py::init<const Eigen::Ref<const dense_t>&, size_t>(), 
            py::arg("mat").noconvert(),
            py::arg("n_threads")
        )
        ;
}

template <class SparseType>
void matrix_constraint_sparse(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixConstraintSparse<SparseType>;
    using base_t = typename internal_t::base_t;
    using vec_sp_index_t = typename internal_t::vec_sp_index_t;
    using vec_sp_value_t = typename internal_t::vec_sp_value_t;
    py::class_<internal_t, base_t>(m, name,
        "Core matrix class for sparse constraint matrix."
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
    ) const override
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
    ) const override
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
void matrix_cov_base(py::module_& m, const char* name)
{
    using trampoline_t = PyMatrixCovBase<T>;
    using internal_t = ad::matrix::MatrixCovBase<T>;
    py::class_<internal_t, trampoline_t>(m, name, R"delimiter(
        Base matrix class for covariance matrices.
    )delimiter")
        .def(py::init<>())
        .def("bmul", &internal_t::bmul, R"delimiter(
        Computes a block matrix-sparse vector multiplication.

        Computes the matrix-sparse vector multiplication
        ``v.T @ A[:, subset]`` where ``v`` is represented by the sparse-format
        ``indices`` and ``values``.

        .. warning::
            This function is not thread-safe!

        Parameters
        ----------
        subset : (s,) ndarray
            Vector of column indices of ``A`` to subset in increasing order.
        indices : (nnz,) ndarray
            Vector of indices in increasing order.
        values : (nnz,) ndarray
            Vector of values associated with ``indices``.
        out : (s,) ndarray
            Vector to store the result.
        )delimiter")
        .def("mul", &internal_t::mul, R"delimiter(
        Computes a matrix-sparse vector multiplication.

        Computes the matrix-sparse vector multiplication
        ``v.T @ A`` where ``v`` is represented by the sparse-format
        ``indices`` and ``values``.

        Parameters
        ----------
        indices : (nnz,) ndarray
            Vector of indices in increasing order.
        values : (nnz,) ndarray
            Vector of values associated with ``indices``.
        out : (n,) ndarray
            Vector to store in-place the result.
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
        out : (p, p) ndarray
            Matrix to store the dense result.
        )delimiter")
        .def("rows", &internal_t::rows, R"delimiter(
        Returns the number of rows.

        Returns
        -------
        rows : int
            Number of rows.
        )delimiter")
        .def("cols", &internal_t::cols, R"delimiter(
        Returns the number of columns.

        Returns
        -------
        cols : int
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

    value_t cmul_safe(
        int j, 
        const Eigen::Ref<const vec_value_t>& v,
        const Eigen::Ref<const vec_value_t>& weights
    ) const override
    {
        PYBIND11_OVERRIDE_PURE(
            value_t,
            base_t,
            cmul_safe,
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

    void bmul_safe(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) const override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            bmul_safe,
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
    ) const override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            mul,
            v, weights, out
        );
    }

    void sq_mul(
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) const override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            sq_mul,
            weights, out
        );
    }

    void sp_tmul(
        const sp_mat_value_t& v,
        Eigen::Ref<rowmat_value_t> out
    ) const override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            sp_tmul,
            v, out
        );
    }

    void cov(
        int j, int q,
        const Eigen::Ref<const vec_value_t>& sqrt_weights,
        Eigen::Ref<colmat_value_t> out
    ) const override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            cov,
            j, q, sqrt_weights, out
        );
    }

    void mean(
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) const override
    {
        PYBIND11_OVERRIDE(
            void,
            base_t,
            mean,
            weights, out
        );
    }

    void var(
        const Eigen::Ref<const vec_value_t>& centers,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) const override
    {
        PYBIND11_OVERRIDE(
            void,
            base_t,
            var,
            centers, weights, out
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
        Base matrix class for naive matrices.
        )delimiter")
        .def(py::init<>())
        .def("cmul", &internal_t::cmul, R"delimiter(
        Computes a column vector-vector multiplication.

        Computes the dot-product 
        ``(v * w).T @ X[:,j]``.

        .. warning::
            This function is not thread-safe!

        Parameters
        ----------
        j : int
            Column index.
        v : (n,) ndarray
            Vector to dot product with the ``j`` th column.
        w : (n,) ndarray
            Vector of weights.

        Returns
        -------
        dot : float
            Column vector-vector multiplication.
        )delimiter")
        .def("cmul_safe", &internal_t::cmul_safe, R"delimiter(
        Computes a column vector-vector multiplication.

        Thread-safe version of :func:`cmul`.

        Parameters
        ----------
        j : int
            Column index.
        v : (n,) ndarray
            Vector to dot product with the ``j`` th column.
        w : (n,) ndarray
            Vector of weights.

        Returns
        -------
        dot : float
            Column vector-vector multiplication.
        )delimiter")
        .def("ctmul", &internal_t::ctmul, R"delimiter(
        Computes a column vector-scalar multiplication increment.

        Computes the vector-scalar multiplication ``v * X[:,j]``.
        The result is *incremented* into the output vector.

        .. warning::
            This function is not thread-safe!

        Parameters
        ----------
        j : int
            Column index.
        v : float
            Scalar to multiply with the ``j`` th column.
        out : (n,) ndarray
            Vector to increment in-place the result.
        )delimiter")
        .def("bmul", &internal_t::bmul, R"delimiter(
        Computes a column block matrix-vector multiplication.

        Computes the matrix-vector multiplication ``(v * w).T @ X[:, j:j+q]``.

        .. warning::
            This function is not thread-safe!

        Parameters
        ----------
        j : int
            Column index.
        q : int
            Number of columns.
        v : (n,) ndarray
            Vector to multiply with the block matrix.
        w : (n,) ndarray
            Vector of weights.
        out : (q,) ndarray
            Vector to store in-place the result.
        )delimiter")
        .def("bmul_safe", &internal_t::bmul_safe, R"delimiter(
        Computes a column block matrix-vector multiplication.

        Thread-safe version of :func:`bmul`.

        Parameters
        ----------
        j : int
            Column index.
        q : int
            Number of columns.
        v : (n,) ndarray
            Vector to multiply with the block matrix.
        w : (n,) ndarray
            Vector of weights.
        out : (q,) ndarray
            Vector to store in-place the result.
        )delimiter")
        .def("btmul", &internal_t::btmul, R"delimiter(
        Computes a column block matrix transpose-vector multiplication increment.

        Computes the matrix-vector multiplication
        ``v.T @ X[:, j:j+q].T``.
        The result is *incremented* into the output vector.

        .. warning::
            This function is not thread-safe!

        Parameters
        ----------
        j : int
            Column index.
        q : int
            Number of columns.
        v : (q,) ndarray
            Vector to multiply with the block matrix.
        out : (n,) ndarray
            Vector to increment in-place the result.
        )delimiter")
        .def("mul", &internal_t::mul, R"delimiter(
        Computes a matrix-vector multiplication.

        Computes the matrix-vector multiplication
        ``(v * w).T @ X``.

        Parameters
        ----------
        v : (n,) ndarray
            Vector to multiply with the matrix.
        w : (n,) ndarray
            Vector of weights.
        out : (p,) ndarray
            Vector to store in-place the result.
        )delimiter")
        .def("sq_mul", &internal_t::sq_mul, R"delimiter(
        Computes a squared matrix-vector multiplication.

        Computes the squared matrix-vector multiplication
        ``w.T @ X ** 2``.

        Parameters
        ----------
        weights : (n,) ndarray
            Vector of weights.
        out : (p,) ndarray
            Vector to store in-place the result.
        )delimiter")
        .def("sp_tmul", &internal_t::sp_tmul, R"delimiter(
        Computes a matrix transpose-sparse matrix multiplication.

        Computes the matrix transpose-sparse matrix multiplication
        ``v.T @ X.T``.

        Parameters
        ----------
        v : (L, p) csr_matrix
            Sparse matrix to multiply with the matrix.
        out : (L, n) ndarray
            Matrix to store in-place the result.
        )delimiter")
        .def("mean", &internal_t::mean, R"delimiter(
        Computes the implied column means.

        The default implied column means are given by ``w.T @ X``.
        Unless stated otherwise, this function will compute the default version.

        Parameters
        ----------
        weights : (n,) ndarray
            Vector of weights.
        out : (p,) ndarray
            Vector to store in-place the result.
        )delimiter")
        .def("var", &internal_t::var, R"delimiter(
        Computes the implied column variances.

        The default implied column variances are given by ``w.T @ (X-c[None])**2``.
        Unless stated otherwise, this function will compute the default version.

        Parameters
        ----------
        centers : (p,) ndarray
            Vector of centers.
        weights : (n,) ndarray
            Vector of weights.
        out : (p,) ndarray
            Vector to store in-place the result.
        )delimiter")
        .def("cov", &internal_t::cov, R"delimiter(
        Computes a weighted covariance matrix.

        Computes the weighted covariance matrix
        ``X[:, j:j+q].T @ W @ X[:, j:j+q]``.

        This function is thread-safe.
        
        .. note::
            Although the name is "covariance", we do not center the columns of ``X``!

        Parameters
        ----------
        j : int
            Column index.
        q : int
            Number of columns.
        sqrt_weights : (n,) ndarray
            Square-root of the weights.
        out : (q, q) ndarray
            Matrix to store in-place the result.
        )delimiter")
        .def("rows", &internal_t::rows, R"delimiter(
        Returns the number of rows.

        Returns
        -------
        rows : int
            Number of rows.
        )delimiter")
        .def("cols", &internal_t::cols, R"delimiter(
        Returns the number of columns.

        Returns
        -------
        cols : int
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
void matrix_naive_block_diag(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixNaiveBlockDiag<ValueType>;
    using base_t = typename internal_t::base_t;
    py::class_<internal_t, base_t>(m, name,
        "Core matrix class for naive block-diagonal matrix."
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
        .def("mean", &internal_t::mean, R"delimiter(
        Computes the implied column means.

        It is undefined for this matrix class and is only exposed for API consistency.

        Parameters
        ----------
        weights : (n,) ndarray
            Vector of weights.
        out : (p,) ndarray
            Vector to store in-place the result.
        )delimiter")
        .def("var", &internal_t::var, R"delimiter(
        Computes the implied column variances.

        It is undefined for this matrix class and is only exposed for API consistency.

        Parameters
        ----------
        centers : (p,) ndarray
            Vector of centers.
        weights : (n,) ndarray
            Vector of weights.
        out : (p,) ndarray
            Vector to store in-place the result.
        )delimiter")
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
        .def("mean", &internal_t::mean, R"delimiter(
        Computes the implied column means.

        The implied column means are a concatenation of the 
        implied column means of each sub-matrix.

        Parameters
        ----------
        weights : (n,) ndarray
            Vector of weights.
        out : (p,) ndarray
            Vector to store in-place the result.
        )delimiter")
        .def("var", &internal_t::var, R"delimiter(
        Computes the implied column variances.

        The implied column variances are a concatenation of the 
        implied column variances of each sub-matrix
        where the centers are subsetted to the corresponding entries.

        Parameters
        ----------
        centers : (p,) ndarray
            Vector of centers.
        weights : (n,) ndarray
            Vector of weights.
        out : (p,) ndarray
            Vector to store in-place the result.
        )delimiter")
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
        .def("mean", &internal_t::mean, R"delimiter(
        Computes the implied column means.

        It is undefined for this matrix class and is only exposed for API consistency.

        Parameters
        ----------
        weights : (n,) ndarray
            Vector of weights.
        out : (p,) ndarray
            Vector to store in-place the result.
        )delimiter")
        .def("var", &internal_t::var, R"delimiter(
        Computes the implied column variances.

        It is undefined for this matrix class and is only exposed for API consistency.

        Parameters
        ----------
        centers : (p,) ndarray
            Vector of centers.
        weights : (n,) ndarray
            Vector of weights.
        out : (p,) ndarray
            Vector to store in-place the result.
        )delimiter")
        ;
}

template <class DenseType, class MaskType>
void matrix_naive_convex_gated_relu_dense(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixNaiveConvexGatedReluDense<DenseType, MaskType>;
    using base_t = typename internal_t::base_t;
    using dense_t = typename internal_t::dense_t;
    using mask_t = typename internal_t::mask_t;
    py::class_<internal_t, base_t>(m, name,
        "Core matrix class for naive convex gated relu matrix with dense underlying."
        )
        .def(
            py::init<
                const Eigen::Ref<const dense_t>&,
                const Eigen::Ref<const mask_t>&,
                size_t
            >(), 
            py::arg("mat").noconvert(),
            py::arg("mask").noconvert(),
            py::arg("n_threads")
        )
        ;
}

template <class SparseType, class MaskType>
void matrix_naive_convex_gated_relu_sparse(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixNaiveConvexGatedReluSparse<SparseType, MaskType>;
    using base_t = typename internal_t::base_t;
    using vec_sp_index_t = typename internal_t::vec_sp_index_t;
    using vec_sp_value_t = typename internal_t::vec_sp_value_t;
    using mask_t = typename internal_t::mask_t;
    py::class_<internal_t, base_t>(m, name,
        "Core matrix class for naive convex gated relu matrix with sparse underlying."
        )
        .def(
            py::init<
                size_t,
                size_t,
                size_t,
                const Eigen::Ref<const vec_sp_index_t>&,
                const Eigen::Ref<const vec_sp_index_t>&,
                const Eigen::Ref<const vec_sp_value_t>&,
                const Eigen::Ref<const mask_t>&,
                size_t 
            >(), 
            py::arg("rows"),
            py::arg("cols"),
            py::arg("nnz"),
            py::arg("outer").noconvert(),
            py::arg("inner").noconvert(),
            py::arg("value").noconvert(),
            py::arg("mask").noconvert(),
            py::arg("n_threads")
        )
        ;
}

template <class DenseType, class MaskType>
void matrix_naive_convex_relu_dense(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixNaiveConvexReluDense<DenseType, MaskType>;
    using base_t = typename internal_t::base_t;
    using dense_t = typename internal_t::dense_t;
    using mask_t = typename internal_t::mask_t;
    py::class_<internal_t, base_t>(m, name,
        "Core matrix class for naive convex relu matrix with dense underlying."
        )
        .def(
            py::init<
                const Eigen::Ref<const dense_t>&,
                const Eigen::Ref<const mask_t>&,
                size_t
            >(), 
            py::arg("mat").noconvert(),
            py::arg("mask").noconvert(),
            py::arg("n_threads")
        )
        ;
}

template <class SparseType, class MaskType>
void matrix_naive_convex_relu_sparse(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixNaiveConvexReluSparse<SparseType, MaskType>;
    using base_t = typename internal_t::base_t;
    using vec_sp_index_t = typename internal_t::vec_sp_index_t;
    using vec_sp_value_t = typename internal_t::vec_sp_value_t;
    using mask_t = typename internal_t::mask_t;
    py::class_<internal_t, base_t>(m, name,
        "Core matrix class for naive convex relu matrix with sparse underlying."
        )
        .def(
            py::init<
                size_t,
                size_t,
                size_t,
                const Eigen::Ref<const vec_sp_index_t>&,
                const Eigen::Ref<const vec_sp_index_t>&,
                const Eigen::Ref<const vec_sp_value_t>&,
                const Eigen::Ref<const mask_t>&,
                size_t 
            >(), 
            py::arg("rows"),
            py::arg("cols"),
            py::arg("nnz"),
            py::arg("outer").noconvert(),
            py::arg("inner").noconvert(),
            py::arg("value").noconvert(),
            py::arg("mask").noconvert(),
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
        .def("mean", &internal_t::mean, R"delimiter(
        Computes the implied column means.

        It is undefined for this matrix class and is only exposed for API consistency.

        Parameters
        ----------
        weights : (n,) ndarray
            Vector of weights.
        out : (p,) ndarray
            Vector to store in-place the result.
        )delimiter")
        .def("var", &internal_t::var, R"delimiter(
        Computes the implied column variances.

        It is undefined for this matrix class and is only exposed for API consistency.

        Parameters
        ----------
        centers : (p,) ndarray
            Vector of centers.
        weights : (n,) ndarray
            Vector of weights.
        out : (p,) ndarray
            Vector to store in-place the result.
        )delimiter")
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
        .def("mean", &internal_t::mean, R"delimiter(
        Computes the implied column means.

        It is undefined for this matrix class and is only exposed for API consistency.

        Parameters
        ----------
        weights : (n,) ndarray
            Vector of weights.
        out : (p,) ndarray
            Vector to store in-place the result.
        )delimiter")
        .def("var", &internal_t::var, R"delimiter(
        Computes the implied column variances.

        It is undefined for this matrix class and is only exposed for API consistency.

        Parameters
        ----------
        centers : (p,) ndarray
            Vector of centers.
        weights : (n,) ndarray
            Vector of weights.
        out : (p,) ndarray
            Vector to store in-place the result.
        )delimiter")
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
        .def("mean", &internal_t::mean, R"delimiter(
        Computes the implied column means.

        It is undefined for this matrix class and is only exposed for API consistency.

        Parameters
        ----------
        weights : (n,) ndarray
            Vector of weights.
        out : (p,) ndarray
            Vector to store in-place the result.
        )delimiter")
        .def("var", &internal_t::var, R"delimiter(
        Computes the implied column variances.

        It is undefined for this matrix class and is only exposed for API consistency.

        Parameters
        ----------
        centers : (p,) ndarray
            Vector of centers.
        weights : (n,) ndarray
            Vector of weights.
        out : (p,) ndarray
            Vector to store in-place the result.
        )delimiter")
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
        .def("mean", &internal_t::mean, R"delimiter(
        Computes the implied column means.

        The default method is used for continuous features
        and the implied mean is zero for categorical features.

        Parameters
        ----------
        weights : (n,) ndarray
            Vector of weights.
        out : (p,) ndarray
            Vector to store in-place the result.
        )delimiter")
        .def("var", &internal_t::var, R"delimiter(
        Computes the implied column variances.

        The default method is used for continuous features
        and the implied variance is one for categorical features.

        Parameters
        ----------
        centers : (p,) ndarray
            Vector of centers.
        weights : (n,) ndarray
            Vector of weights.
        out : (p,) ndarray
            Vector to store in-place the result.
        )delimiter")
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
    using io_t = typename internal_t::io_t;
    py::class_<internal_t, base_t>(m, name,
        "Core matrix class for naive SNP unphased matrix."
        )
        .def(
            py::init<
                const io_t&,
                size_t
            >(), 
            py::arg("io"),
            py::arg("n_threads")
        )
        .def("mean", &internal_t::mean, R"delimiter(
        Computes the implied column means.

        The implied column means are zero.

        Parameters
        ----------
        weights : (n,) ndarray
            Vector of weights.
        out : (p,) ndarray
            Vector to store in-place the result.
        )delimiter")
        .def("var", &internal_t::var, R"delimiter(
        Computes the implied column variances.
        
        The implied column variances are one.

        Parameters
        ----------
        centers : (p,) ndarray
            Vector of centers.
        weights : (n,) ndarray
            Vector of weights.
        out : (p,) ndarray
            Vector to store in-place the result.
        )delimiter")
        ;
}

template <class ValueType>
void matrix_naive_snp_phased_ancestry(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixNaiveSNPPhasedAncestry<ValueType>;
    using base_t = typename internal_t::base_t;
    using io_t = typename internal_t::io_t;
    py::class_<internal_t, base_t>(m, name,
        "Core matrix class for naive SNP phased, ancestry matrix."
        )
        .def(
            py::init<
                const io_t&,
                size_t
            >(), 
            py::arg("io"),
            py::arg("n_threads")
        )
        .def("mean", &internal_t::mean, R"delimiter(
        Computes the implied column means.

        The implied column means are zero.

        Parameters
        ----------
        weights : (n,) ndarray
            Vector of weights.
        out : (p,) ndarray
            Vector to store in-place the result.
        )delimiter")
        .def("var", &internal_t::var, R"delimiter(
        Computes the implied column variances.
        
        The implied column variances are one.

        Parameters
        ----------
        centers : (p,) ndarray
            Vector of centers.
        weights : (n,) ndarray
            Vector of weights.
        out : (p,) ndarray
            Vector to store in-place the result.
        )delimiter")
        ;
}

template <class SparseType>
void matrix_naive_sparse(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixNaiveSparse<SparseType>;
    using base_t = typename internal_t::base_t;
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
        .def("mean", &internal_t::mean, R"delimiter(
        Computes the implied column means.

        The implied column means are zero.

        Parameters
        ----------
        weights : (n,) ndarray
            Vector of weights.
        out : (p,) ndarray
            Vector to store in-place the result.
        )delimiter")
        .def("var", &internal_t::var, R"delimiter(
        Computes the implied column variances.
        
        The implied column variances are one.

        Parameters
        ----------
        centers : (p,) ndarray
            Vector of centers.
        weights : (n,) ndarray
            Vector of weights.
        out : (p,) ndarray
            Vector to store in-place the result.
        )delimiter")
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
                base_t&,
                const Eigen::Ref<const vec_index_t>&,
                size_t
            >(),
            py::arg("mat"),
            py::arg("subset").noconvert(),
            py::arg("n_threads")
        )
        .def("mean", &internal_t::mean, R"delimiter(
        Computes the implied column means.

        The implied column means are the subset of the 
        implied column means of the underlying matrix.

        Parameters
        ----------
        weights : (n,) ndarray
            Vector of weights.
        out : (p,) ndarray
            Vector to store in-place the result.
        )delimiter")
        .def("var", &internal_t::var, R"delimiter(
        Computes the implied column variances.

        The implied column variances are the subset of the 
        implied column variances of the underlying matrix.

        Parameters
        ----------
        centers : (p,) ndarray
            Vector of centers.
        weights : (n,) ndarray
            Vector of weights.
        out : (p,) ndarray
            Vector to store in-place the result.
        )delimiter")
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
                base_t&,
                const Eigen::Ref<const vec_index_t>&,
                size_t
            >(),
            py::arg("mat"),
            py::arg("subset").noconvert(),
            py::arg("n_threads")
        )
        .def("mean", &internal_t::mean, R"delimiter(
        Computes the implied column means.

        The implied column means are the implied column means of the underlying matrix
        where the weights are zero outside of the subset.

        Parameters
        ----------
        weights : (n,) ndarray
            Vector of weights.
        out : (p,) ndarray
            Vector to store in-place the result.
        )delimiter")
        .def("var", &internal_t::var, R"delimiter(
        Computes the implied column variances.

        The implied column variances are the implied column variances of the underlying matrix
        where the weights are zero outside of the subset.

        Parameters
        ----------
        centers : (p,) ndarray
            Vector of centers.
        weights : (n,) ndarray
            Vector of weights.
        out : (p,) ndarray
            Vector to store in-place the result.
        )delimiter")
        ;
}

void register_matrix(py::module_& m)
{
    /* constraint matrices */
    matrix_constraint_base<double>(m, "MatrixConstraintBase64");
    matrix_constraint_base<float>(m, "MatrixConstraintBase32");

    matrix_constraint_dense<dense_type<double, Eigen::RowMajor>>(m, "MatrixConstraintDense64C");
    matrix_constraint_dense<dense_type<double, Eigen::ColMajor>>(m, "MatrixConstraintDense64F");
    matrix_constraint_dense<dense_type<float, Eigen::RowMajor>>(m, "MatrixConstraintDense32C");
    matrix_constraint_dense<dense_type<float, Eigen::ColMajor>>(m, "MatrixConstraintDense32F");

    matrix_constraint_sparse<sparse_type<double, Eigen::RowMajor>>(m, "MatrixConstraintSparse64C");
    matrix_constraint_sparse<sparse_type<float, Eigen::RowMajor>>(m, "MatrixConstraintSparse32C");

    /* cov matrices */
    matrix_cov_base<double>(m, "MatrixCovBase64");
    matrix_cov_base<float>(m, "MatrixCovBase32");

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
    matrix_naive_base<double>(m, "MatrixNaiveBase64");
    matrix_naive_base<float>(m, "MatrixNaiveBase32");

    matrix_naive_block_diag<double>(m, "MatrixNaiveBlockDiag64");
    matrix_naive_block_diag<float>(m, "MatrixNaiveBlockDiag32");

    matrix_naive_convex_gated_relu_dense<
        dense_type<double, Eigen::RowMajor>,
        dense_type<bool, Eigen::ColMajor>
    >(m, "MatrixNaiveConvexGatedReluDense64C");
    matrix_naive_convex_gated_relu_dense<
        dense_type<double, Eigen::ColMajor>,
        dense_type<bool, Eigen::ColMajor>
    >(m, "MatrixNaiveConvexGatedReluDense64F");
    matrix_naive_convex_gated_relu_dense<
        dense_type<float, Eigen::RowMajor>,
        dense_type<bool, Eigen::ColMajor>
    >(m, "MatrixNaiveConvexGatedReluDense32C");
    matrix_naive_convex_gated_relu_dense<
        dense_type<float, Eigen::ColMajor>,
        dense_type<bool, Eigen::ColMajor>
    >(m, "MatrixNaiveConvexGatedReluDense32F");
    matrix_naive_convex_gated_relu_sparse<
        sparse_type<double, Eigen::ColMajor>,
        dense_type<bool, Eigen::ColMajor>
    >(m, "MatrixNaiveConvexGatedReluSparse64F");
    matrix_naive_convex_gated_relu_sparse<
        sparse_type<float, Eigen::ColMajor>,
        dense_type<bool, Eigen::ColMajor>
    >(m, "MatrixNaiveConvexGatedReluSparse32F");

    matrix_naive_convex_relu_dense<
        dense_type<double, Eigen::RowMajor>,
        dense_type<bool, Eigen::ColMajor>
    >(m, "MatrixNaiveConvexReluDense64C");
    matrix_naive_convex_relu_dense<
        dense_type<double, Eigen::ColMajor>,
        dense_type<bool, Eigen::ColMajor>
    >(m, "MatrixNaiveConvexReluDense64F");
    matrix_naive_convex_relu_dense<
        dense_type<float, Eigen::RowMajor>,
        dense_type<bool, Eigen::ColMajor>
    >(m, "MatrixNaiveConvexReluDense32C");
    matrix_naive_convex_relu_dense<
        dense_type<float, Eigen::ColMajor>,
        dense_type<bool, Eigen::ColMajor>
    >(m, "MatrixNaiveConvexReluDense32F");
    matrix_naive_convex_relu_sparse<
        sparse_type<double, Eigen::ColMajor>,
        dense_type<bool, Eigen::ColMajor>
    >(m, "MatrixNaiveConvexReluSparse64F");
    matrix_naive_convex_relu_sparse<
        sparse_type<float, Eigen::ColMajor>,
        dense_type<bool, Eigen::ColMajor>
    >(m, "MatrixNaiveConvexReluSparse32F");

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