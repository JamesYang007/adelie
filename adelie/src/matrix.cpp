#include "decl.hpp"
#include <adelie_core/matrix/matrix_cov_base.hpp>
#include <adelie_core/matrix/matrix_cov_dense.hpp>
#include <adelie_core/matrix/matrix_cov_lazy.hpp>
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/matrix/matrix_naive_dense.hpp>

namespace py = pybind11;
namespace ad = adelie_core;

template <class T>
class PyMatrixNaiveBase : public ad::matrix::MatrixNaiveBase<T>
{
    using base_t = ad::matrix::MatrixNaiveBase<T>;
public:
    /* Inherit the constructors */
    using base_t::base_t;
    using typename base_t::value_t;
    using typename base_t::rowvec_t;
    using typename base_t::colmat_t;

    /* Trampoline (need one for each virtual function) */
    value_t cmul(
        int j, 
        const Eigen::Ref<const rowvec_t>& v
    ) const override
    {
        PYBIND11_OVERRIDE_PURE(
            value_t,
            base_t,
            cmul,
            j, v 
        );
    }

    void ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<rowvec_t> out
    ) const override
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
        const Eigen::Ref<const rowvec_t>& v, 
        Eigen::Ref<rowvec_t> out
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            bmul,
            j, q, v, out
        );
    }

    void btmul(
        int j, int q, 
        const Eigen::Ref<const rowvec_t>& v, 
        Eigen::Ref<rowvec_t> out
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            btmul,
            j, q, v, out
        );
    }

    void to_dense(
        int j, int q,
        Eigen::Ref<colmat_t> out
    ) const override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            to_dense,
            j, q, out
        );
    }

    int rows() const override
    {
        PYBIND11_OVERRIDE_PURE(
            int,
            base_t,
            rows
        );
    }

    int cols() const override
    {
        PYBIND11_OVERRIDE_PURE(
            int,
            base_t,
            cols
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
        Column vector multiplication.

        Computes the dot-product ``v.T @ X[:,j]`` for a column ``j``.

        Parameters
        ----------
        j : int
            Column index.
        v : (n,) np.ndarray
            Vector to multiply the ``j`` th column with.
        )delimiter")
        .def("ctmul", &internal_t::ctmul, R"delimiter(
        Column scalar multiplication.

        Computes the scalar-vector multiplication ``v * X[:,j]`` for a column ``j``.

        Parameters
        ----------
        j : int
            Column index.
        v : float
            Scalar to multiply the ``j`` th column with.
        out : (n,) np.ndarray
            Vector to store in-place the result.
        )delimiter")
        .def("bmul", &internal_t::bmul, R"delimiter(
        Block matrix-vector multiplication.

        Computes the matrix-vector multiplication
        ``v.T @ X[:, j:j+q]``.

        Parameters
        ----------
        j : int
            Column index.
        q : int
            Number of columns.
        v : (n,) np.ndarray
            Vector to multiply with the block matrix.
        out : (q,) np.ndarray
            Vector to store in-place the result.
        )delimiter")
        .def("btmul", &internal_t::btmul, R"delimiter(
        Block matrix transpose-vector multiplication.

        Computes the matrix-vector multiplication
        ``v.T @ X[:, j:j+q].T``.

        Parameters
        ----------
        j : int
            Column index.
        q : int
            Number of columns.
        v : (q,) np.ndarray
            Vector to multiply with the block matrix.
        out : (n,) np.ndarray
            Vector to store in-place the result.
        )delimiter")
        .def("to_dense", &internal_t::to_dense, R"delimiter(
        Converts block to a dense matrix.

        Converts the block ``X[:, j:j+q]`` into a dense matrix.

        Parameters
        ----------
        j : int
            Column index.
        q : int
            Number of columns.
        out : (n, q) np.ndarray
            Matrix to store the dense result.
        )delimiter")
        .def("rows", &internal_t::rows, R"delimiter(
        Number of rows.
        )delimiter")
        .def("cols", &internal_t::cols, R"delimiter(
        Number of columns.
        )delimiter")
        ;
}

template <class T>
class PyMatrixCovBase : public ad::matrix::MatrixCovBase<T>
{
    using base_t = ad::matrix::MatrixCovBase<T>;
public:
    /* Inherit the constructors */
    using base_t::base_t;
    using typename base_t::value_t;
    using typename base_t::rowvec_t;
    using typename base_t::colmat_t;

    void bmul(
        int i, int j, int p, int q, 
        const Eigen::Ref<const rowvec_t>& v, 
        Eigen::Ref<rowvec_t> out
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            bmul,
            i, j, p, q, v, out
        );
    }

    void to_dense(
        int i, int j, int p, int q,
        Eigen::Ref<colmat_t> out
    ) const override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            to_dense,
            i, j, p, q, out
        );
    }

    int rows() const override
    {
        PYBIND11_OVERRIDE(
            int,
            base_t,
            rows
        );
    }

    int cols() const override
    {
        PYBIND11_OVERRIDE_PURE(
            int,
            base_t,
            cols
        );
    }
};

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
        Block matrix-vector multiplication.

        Computes the matrix-vector multiplication
        ``v.T @ A[i:i+p, j:j+q]``.

        Parameters
        ----------
        i : int
            Row index.
        j : int
            Column index.
        p : int
            Number of rows.
        q : int
            Number of columns.
        v : (p,) np.ndarray
            Vector to multiply with the block matrix.
        out : (q,) np.ndarray
            Vector to store in-place the result.
        )delimiter")
        .def("to_dense", &internal_t::to_dense, R"delimiter(
        Converts block to a dense matrix.

        Converts the block ``X[i:i+p, j:j+q]`` into a dense matrix.

        Parameters
        ----------
        i : int
            Row index.
        j : int
            Column index.
        p : int
            Number of rows.
        q : int
            Number of columns.
        out : (p, q) np.ndarray
            Matrix to store the dense result.
        )delimiter")
        .def("rows", &internal_t::rows, R"delimiter(
        Number of rows.
        )delimiter")
        .def("cols", &internal_t::cols, R"delimiter(
        Number of columns.
        )delimiter")
        ;
}

template <class DenseType>
void matrix_naive_dense(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixNaiveDense<DenseType>;
    using base_t = typename internal_t::base_t;
    using dense_t = typename internal_t::dense_t;
    py::class_<internal_t, base_t>(m, name)
        .def(
            py::init<const Eigen::Ref<const dense_t>&, size_t>(), 
            py::arg("mat").noconvert(),
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
    py::class_<internal_t, base_t>(m, name)
        .def(
            py::init<const Eigen::Ref<const dense_t>&, size_t>(), 
            py::arg("mat").noconvert(),
            py::arg("n_threads")
        )
        ;
}

template <class DenseType>
void matrix_cov_lazy(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixCovLazy<DenseType>;
    using base_t = typename internal_t::base_t;
    using dense_t = typename internal_t::dense_t;
    py::class_<internal_t, base_t>(m, name)
        .def(
            py::init<const Eigen::Ref<const dense_t>&, size_t>(), 
            py::arg("mat").noconvert(),
            py::arg("n_threads")
        )
        ;
}

template <class T, int Storage>
using dense_type = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Storage>;

void register_matrix(py::module_& m)
{
    /* base matrices */
    matrix_naive_base<double>(m, "MatrixNaiveBase64");
    matrix_naive_base<float>(m, "MatrixNaiveBase32");
    matrix_cov_base<double>(m, "MatrixCovBase64");
    matrix_cov_base<float>(m, "MatrixCovBase32");

    /* naive matrices */
    matrix_naive_dense<dense_type<double, Eigen::RowMajor>>(m, "MatrixNaiveDense64C");
    matrix_naive_dense<dense_type<double, Eigen::ColMajor>>(m, "MatrixNaiveDense64F");
    matrix_naive_dense<dense_type<float, Eigen::RowMajor>>(m, "MatrixNaiveDense32C");
    matrix_naive_dense<dense_type<float, Eigen::ColMajor>>(m, "MatrixNaiveDense32F");

    /* cov matrices */
    matrix_cov_dense<dense_type<double, Eigen::RowMajor>>(m, "MatrixCovDense64C");
    matrix_cov_dense<dense_type<double, Eigen::ColMajor>>(m, "MatrixCovDense64F");
    matrix_cov_dense<dense_type<float, Eigen::RowMajor>>(m, "MatrixCovDense32C");
    matrix_cov_dense<dense_type<float, Eigen::ColMajor>>(m, "MatrixCovDense32F");
    matrix_cov_lazy<dense_type<double, Eigen::RowMajor>>(m, "MatrixCovLazy64C");
    matrix_cov_lazy<dense_type<double, Eigen::ColMajor>>(m, "MatrixCovLazy64F");
    matrix_cov_lazy<dense_type<float, Eigen::RowMajor>>(m, "MatrixCovLazy32C");
    matrix_cov_lazy<dense_type<float, Eigen::ColMajor>>(m, "MatrixCovLazy32F");
}