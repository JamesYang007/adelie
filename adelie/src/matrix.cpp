#include "decl.hpp"
#include <adelie_core/matrix/matrix_cov_base.hpp>
#include <adelie_core/matrix/matrix_cov_dense.hpp>
#include <adelie_core/matrix/matrix_cov_lazy.hpp>
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/matrix/matrix_naive_concatenate.hpp>
#include <adelie_core/matrix/matrix_naive_dense.hpp>
#include <adelie_core/matrix/matrix_naive_kronecker_eye.hpp>
#include <adelie_core/matrix/matrix_naive_snp_unphased.hpp>
#include <adelie_core/matrix/matrix_naive_snp_phased_ancestry.hpp>

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
    m.def("dgemv", ad::matrix::dgemv<cref_colmat_value_t, cref_mvec_value_t, ref_rowmat_value_t, ref_mvec_value_t>);
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

        Parameters
        ----------
        j : int
            Column index.
        v : float
            Scalar to multiply with the ``j`` th column.
        out : (n,) np.ndarray
            Vector to store in-place the result.
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
    using typename base_t::vec_value_t;
    using typename base_t::colmat_value_t;

    void bmul(
        int i, int j, int p, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            bmul,
            i, j, p, q, v, out
        );
    }

    void mul(
        int i, int p,
        const Eigen::Ref<const vec_value_t>& v,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            mul,
            i, p, v, out
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
        .def("mul", &internal_t::mul, R"delimiter(
        Row matrix-vector multiplication.

        Computes the row matrix-vector multiplication
        ``v.T @ A[i:i+p, :]``.

        Parameters
        ----------
        i : int
            Row index.
        p : int
            Number of rows.
        v : (p,) np.ndarray
            Vector to multiply with the block matrix.
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
        ;
}

template <class ValueType>
void matrix_naive_concatenate(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixNaiveConcatenate<ValueType>;
    using base_t = typename internal_t::base_t;
    py::class_<internal_t, base_t>(m, name)
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
    py::class_<internal_t, base_t>(m, name)
        .def(
            py::init<const Eigen::Ref<const dense_t>&, size_t>(), 
            py::arg("mat").noconvert(),
            py::arg("n_threads")
        )
        ;
}

template <class ValueType>
void matrix_naive_kronecker_eye(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixNaiveKroneckerEye<ValueType>;
    using base_t = typename internal_t::base_t;
    py::class_<internal_t, base_t>(m, name)
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
    py::class_<internal_t, base_t>(m, name)
        .def(
            py::init<const Eigen::Ref<const dense_t>&, size_t, size_t>(), 
            py::arg("mat"),
            py::arg("K"),
            py::arg("n_threads")
        )
        ;
}

template <class ValueType>
void matrix_naive_snp_unphased(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixNaiveSNPUnphased<ValueType>;
    using base_t = typename internal_t::base_t;
    using string_t = typename internal_t::string_t;
    py::class_<internal_t, base_t>(m, name)
        .def(
            py::init<
                const string_t&,
                size_t
            >(), 
            py::arg("filename"),
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
    py::class_<internal_t, base_t>(m, name)
        .def(
            py::init<
                const string_t&,
                size_t
            >(), 
            py::arg("filename"),
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
    /* utils */
    utils(m);

    /* base matrices */
    matrix_naive_base<double>(m, "MatrixNaiveBase64");
    matrix_naive_base<float>(m, "MatrixNaiveBase32");
    matrix_cov_base<double>(m, "MatrixCovBase64");
    matrix_cov_base<float>(m, "MatrixCovBase32");

    /* naive matrices */
    matrix_naive_concatenate<double>(m, "MatrixNaiveConcatenate64");
    matrix_naive_concatenate<float>(m, "MatrixNaiveConcatenate32");

    matrix_naive_dense<dense_type<double, Eigen::RowMajor>>(m, "MatrixNaiveDense64C");
    matrix_naive_dense<dense_type<double, Eigen::ColMajor>>(m, "MatrixNaiveDense64F");
    matrix_naive_dense<dense_type<float, Eigen::RowMajor>>(m, "MatrixNaiveDense32C");
    matrix_naive_dense<dense_type<float, Eigen::ColMajor>>(m, "MatrixNaiveDense32F");

    matrix_naive_kronecker_eye<double>(m, "MatrixNaiveKroneckerEye64");
    matrix_naive_kronecker_eye<float>(m, "MatrixNaiveKroneckerEye32");
    matrix_naive_kronecker_eye_dense<dense_type<double, Eigen::RowMajor>>(m, "MatrixNaiveKroneckerEyeDense64C");
    matrix_naive_kronecker_eye_dense<dense_type<double, Eigen::ColMajor>>(m, "MatrixNaiveKroneckerEyeDense64F");
    matrix_naive_kronecker_eye_dense<dense_type<float, Eigen::RowMajor>>(m, "MatrixNaiveKroneckerEyeDense32C");
    matrix_naive_kronecker_eye_dense<dense_type<float, Eigen::ColMajor>>(m, "MatrixNaiveKroneckerEyeDense32F");

    matrix_naive_snp_unphased<double>(m, "MatrixNaiveSNPUnphased64");
    matrix_naive_snp_unphased<float>(m, "MatrixNaiveSNPUnphased32");
    matrix_naive_snp_phased_ancestry<double>(m, "MatrixNaiveSNPPhasedAncestry64");
    matrix_naive_snp_phased_ancestry<float>(m, "MatrixNaiveSNPPhasedAncestry32");

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