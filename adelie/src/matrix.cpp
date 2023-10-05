#include "decl.hpp"
#include <adelie_core/matrix/matrix_base.hpp>
#include <adelie_core/matrix/matrix_dense.hpp>

namespace py = pybind11;
namespace ad = adelie_core;

template <class T>
class PyMatrixBase : public ad::matrix::MatrixBase<T>
{
    using base_t = ad::matrix::MatrixBase<T>;
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

    void btmul(
        int i, int j, int p, int q, 
        const Eigen::Ref<const rowvec_t>& v, 
        Eigen::Ref<rowvec_t> out
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            btmul,
            i, j, p, q, v, out
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
void matrix_base(py::module_& m, const char* name)
{
    using trampoline_t = PyMatrixBase<T>;
    using internal_t = ad::matrix::MatrixBase<T>;
    py::class_<internal_t, trampoline_t>(m, name)
        .def(py::init<>())
        .def("cmul", &internal_t::cmul)
        .def("ctmul", &internal_t::ctmul)
        .def("bmul", &internal_t::bmul)
        .def("btmul", &internal_t::btmul)
        .def("cnormsq", &internal_t::cnormsq)
        .def("rows", &internal_t::rows)
        .def("cols", &internal_t::cols)
        ;
}

template <class DenseType>
void matrix_dense(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixDense<DenseType>;
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
    matrix_base<double>(m, "Base64");
    matrix_base<float>(m, "Base32");

    matrix_dense<dense_type<double, Eigen::RowMajor>>(m, "Dense64C");
    matrix_dense<dense_type<double, Eigen::ColMajor>>(m, "Dense64F");
    matrix_dense<dense_type<float, Eigen::RowMajor>>(m, "Dense32C");
    matrix_dense<dense_type<float, Eigen::ColMajor>>(m, "Dense32F");
}