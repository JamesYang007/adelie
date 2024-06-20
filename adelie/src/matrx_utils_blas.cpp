#if defined(__APPLE__)
#define EIGEN_USE_BLAS
#endif
#include "decl.hpp"
#include <adelie_core/configs.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

template <class MType, class OutType>
void dgemtm(
    const MType& m,
    OutType& out,
    size_t n_threads
)
{
    using value_t = typename std::decay_t<MType>::Scalar;
    using out_t = std::decay_t<OutType>;

    static_assert(!out_t::IsRowMajor, "out must be column-major!");

    const size_t n = m.rows();
    const size_t p = m.cols();
    const size_t n_bytes = sizeof(value_t) * n * p * p;
    if (n_threads <= 1 || n_bytes <= Configs::min_bytes) { 
        out.setZero();
        out.template selfadjointView<Eigen::Lower>().rankUpdate(m.transpose());
        out.template triangularView<Eigen::Upper>() = out.transpose();
        return; 
    }

    Eigen::setNbThreads(n_threads);
    out.noalias() = m.transpose() * m;
    Eigen::setNbThreads(1);
}

} // namespace matrix
} // namespace adelie_core

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