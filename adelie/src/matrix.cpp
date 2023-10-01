#include "decl.hpp"
#include <adelie_core/matrix/matrix_base.hpp>
#include <adelie_core/matrix/matrix_dense.hpp>

namespace py = pybind11;
namespace ad = adelie_core;

template <class T, int S>
class PyMatrixBase : public ad::matrix::MatrixBase<T, S>
{
    using base_t = ad::matrix::MatrixBase<T, S>;
public:
    /* Inherit the constructors */
    using base_t::base_t;
    using typename base_t::vec_t;
    using typename base_t::mat_t;

    /* Trampoline (need one for each virtual function) */
    Eigen::Ref<const mat_t> block(int i, int j, int p, int q) const override
    {
        PYBIND11_OVERRIDE_PURE(
            Eigen::Ref<const mat_t>,
            base_t,
            block,
            i, j, p, q
        );
    }
    Eigen::Ref<const vec_t> col(int j) const override
    {
        PYBIND11_OVERRIDE_PURE(
            Eigen::Ref<const vec_t>,
            base_t,
            col,
            j
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

template <class T, int S>
void matrix_base(py::module_& m, const char* name)
{
    using trampoline_t = PyMatrixBase<T, S>;
    using internal_t = ad::matrix::MatrixBase<T, S>;
    py::class_<internal_t, trampoline_t>(m, name)
        .def(py::init<>())
        .def("block", &internal_t::block)
        .def("col", &internal_t::col)
        .def("cols", &internal_t::cols)
        ;
}

template <class T, int S>
void matrix_dense(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixDense<T, S>;
    using base_t = ad::matrix::MatrixBase<T, S>;
    using mat_t = typename internal_t::mat_t;
    py::class_<internal_t, base_t>(m, name)
        .def(
            py::init<const Eigen::Ref<const mat_t>&>(), 
            py::arg("mat").noconvert()
        )
        ;
}

void register_matrix(py::module_& m)
{
    matrix_base<double, Eigen::ColMajor>(m, "Base64F");
    matrix_base<double, Eigen::RowMajor>(m, "Base64C");
    matrix_base<float, Eigen::ColMajor>(m, "Base32F");
    matrix_base<float, Eigen::RowMajor>(m, "Base32C");

    matrix_dense<double, Eigen::ColMajor>(m, "Dense64F");
    matrix_dense<double, Eigen::RowMajor>(m, "Dense64C");
    matrix_dense<float, Eigen::ColMajor>(m, "Dense32F");
    matrix_dense<float, Eigen::RowMajor>(m, "Dense32C");
}