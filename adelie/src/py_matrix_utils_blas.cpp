#if defined(__APPLE__)
#define EIGEN_USE_BLAS
#endif
#include "py_decl.hpp"
#include <adelie_core/matrix/utils.hpp>

namespace ad = adelie_core;

template <class ValueType=double>
void matrix_utils_blas(py::module_& m)
{
    using value_t = ValueType;
    using colmat_value_t = ad::util::colmat_type<value_t>;
    using ref_colmat_value_t = Eigen::Ref<colmat_value_t>;
    using cref_colmat_value_t = Eigen::Ref<const colmat_value_t>;
    m.def("dgemtm", ad::matrix::dgemtm<cref_colmat_value_t, ref_colmat_value_t>);
}

void register_matrix_utils_blas(py::module_& m)
{
    matrix_utils_blas(m);
}