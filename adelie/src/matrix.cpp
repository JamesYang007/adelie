#include "decl.hpp"
#include <adelie_core/matrix/matrix_pin_cov_base.hpp>
#include <adelie_core/matrix/matrix_pin_cov_dense.hpp>
#include <adelie_core/matrix/matrix_pin_cov_lazy.hpp>
#include <adelie_core/matrix/matrix_pin_naive_base.hpp>
#include <adelie_core/matrix/matrix_pin_naive_dense.hpp>

namespace py = pybind11;
namespace ad = adelie_core;

template <class T>
class PyMatrixPinNaiveBase : public ad::matrix::MatrixPinNaiveBase<T>
{
    using base_t = ad::matrix::MatrixPinNaiveBase<T>;
public:
    /* Inherit the constructors */
    using base_t::base_t;
    using typename base_t::value_t;
    using typename base_t::rowvec_t;

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

    value_t cnormsq(int j) const override
    {
        PYBIND11_OVERRIDE_PURE(
            value_t,
            base_t,
            cnormsq,
            j
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
void matrix_pin_naive_base(py::module_& m, const char* name)
{
    using trampoline_t = PyMatrixPinNaiveBase<T>;
    using internal_t = ad::matrix::MatrixPinNaiveBase<T>;
    py::class_<internal_t, trampoline_t>(m, name, R"delimiter(
        Base matrix class for pin, naive method.
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
        .def("cnormsq", &internal_t::cnormsq, R"delimiter(
        Column norm-squared.

        Computes the :math:`\ell_2` norm square of a column.

        Parameters
        ----------
        j : int
            Column index.

        Returns
        -------
        normsq
            :math:`\ell_2` norm square of column ``j``.
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
class PyMatrixPinCovBase : public ad::matrix::MatrixPinCovBase<T>
{
    using base_t = ad::matrix::MatrixPinCovBase<T>;
public:
    /* Inherit the constructors */
    using base_t::base_t;
    using typename base_t::value_t;
    using typename base_t::rowvec_t;

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

    value_t diag(int i) const override
    {
        PYBIND11_OVERRIDE_PURE(
            value_t,
            base_t,
            diag,
            i
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
void matrix_pin_cov_base(py::module_& m, const char* name)
{
    using trampoline_t = PyMatrixPinCovBase<T>;
    using internal_t = ad::matrix::MatrixPinCovBase<T>;
    py::class_<internal_t, trampoline_t>(m, name, R"delimiter(
        Base matrix class for pin, covariance method.
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
        .def("diag", &internal_t::diag, R"delimiter(
        Diagonal entry.

        Returns the diagonal entry at index ``(i, i)``.

        Parameters
        ----------
        i : int
            Row index.

        Returns
        -------
        diag
            Diagonal entry at ``(i, i)``.
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
void matrix_pin_naive_dense(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixPinNaiveDense<DenseType>;
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
void matrix_pin_cov_dense(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixPinCovDense<DenseType>;
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
void matrix_pin_cov_lazy(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixPinCovLazy<DenseType>;
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
    /* pin base matrices */
    matrix_pin_naive_base<double>(m, "MatrixPinNaiveBase64");
    matrix_pin_naive_base<float>(m, "MatrixPinNaiveBase32");
    matrix_pin_cov_base<double>(m, "MatrixPinCovBase64");
    matrix_pin_cov_base<float>(m, "MatrixPinCovBase32");

    /* pin naive matrices */
    matrix_pin_naive_dense<dense_type<double, Eigen::RowMajor>>(m, "MatrixPinNaiveDense64C");
    matrix_pin_naive_dense<dense_type<double, Eigen::ColMajor>>(m, "MatrixPinNaiveDense64F");
    matrix_pin_naive_dense<dense_type<float, Eigen::RowMajor>>(m, "MatrixPinNaiveDense32C");
    matrix_pin_naive_dense<dense_type<float, Eigen::ColMajor>>(m, "MatrixPinNaiveDense32F");

    /* pin cov matrices */
    matrix_pin_cov_dense<dense_type<double, Eigen::RowMajor>>(m, "MatrixPinCovDense64C");
    matrix_pin_cov_dense<dense_type<double, Eigen::ColMajor>>(m, "MatrixPinCovDense64F");
    matrix_pin_cov_dense<dense_type<float, Eigen::RowMajor>>(m, "MatrixPinCovDense32C");
    matrix_pin_cov_dense<dense_type<float, Eigen::ColMajor>>(m, "MatrixPinCovDense32F");
    matrix_pin_cov_lazy<dense_type<double, Eigen::RowMajor>>(m, "MatrixPinCovLazy64C");
    matrix_pin_cov_lazy<dense_type<double, Eigen::ColMajor>>(m, "MatrixPinCovLazy64F");
    matrix_pin_cov_lazy<dense_type<float, Eigen::RowMajor>>(m, "MatrixPinCovLazy32C");
    matrix_pin_cov_lazy<dense_type<float, Eigen::ColMajor>>(m, "MatrixPinCovLazy32F");
}