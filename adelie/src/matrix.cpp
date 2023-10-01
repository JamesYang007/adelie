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
        .def("block", &internal_t::block)
        .def("col", &internal_t::col)
        .def("rows", &internal_t::rows)
        .def("cols", &internal_t::cols)
        ;
}

template <class T>
void matrix_dense(py::module_& m, const char* name)
{
    using internal_t = ad::matrix::MatrixDense<T>;
    using base_t = ad::matrix::MatrixBase<T>;
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
    matrix_base<double>(m, "Base64");
    matrix_base<float>(m, "Base32");

    matrix_dense<double>(m, "Dense64");
    matrix_dense<float>(m, "Dense32");
}