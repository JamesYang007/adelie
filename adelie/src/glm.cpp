#include "decl.hpp"
#include <adelie_core/glm/glm_base.hpp>
#include <adelie_core/glm/glm_gaussian.hpp>

namespace py = pybind11;
namespace ad = adelie_core;

template <class T>
class PyGlmBase : public ad::glm::GlmBase<T>
{
    using base_t = ad::glm::GlmBase<T>;
public:
    using base_t::base_t;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;

    void gradient(
        const Eigen::Ref<const vec_value_t>& eta,
        Eigen::Ref<vec_value_t> mu
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            gradient,
            eta, mu
        );
    }

    void hessian(
        const Eigen::Ref<const vec_value_t>& eta,
        Eigen::Ref<vec_value_t> var
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            hessian,
            eta, var
        );
    }
};

template <class T>
void glm_base(py::module_& m, const char* name)
{
    using trampoline_t = PyGlmBase<T>;
    using internal_t = ad::glm::GlmBase<T>;
    py::class_<internal_t, trampoline_t>(m, name, R"delimiter(
        Base GLM class.
        )delimiter")
        .def(py::init<>())
        .def("gradient", &internal_t::gradient, R"delimiter(
        Gradient of the log-partition function.

        Computes :math:`\nabla \underline{A}(\eta)`
        where :math:`\underline{A}(\eta)_k = A(\eta_k)`
        and :math:`A` is the log-partition function.

        Parameters
        ----------
        eta : (n,) np.ndarray
            Natural parameter.
        mu : (n,) np.ndarray
            The gradient, or mean parameter, to store.
        )delimiter")
        .def("hessian", &internal_t::hessian, R"delimiter(
        Hessian of the log-partition function.

        Computes :math:`\nabla^2 \underline{A}(\eta)`
        where :math:`\underline{A}(\eta)_k = A(\eta_k)`
        and :math:`A` is the log-partition function.
        Note that since the hessian is a diagonal matrix, 
        we only output the diagonal.

        Parameters
        ----------
        eta : (n,) np.ndarray
            Natural parameter.
        var : (n,) np.ndarray
            The hessian, or variance parameter, to store.
        )delimiter")
        ;
}

template <class T>
void glm_gaussian(py::module_& m, const char* name)
{
    using internal_t = ad::glm::GlmGaussian<T>;
    using base_t = typename internal_t::base_t;
    py::class_<internal_t, base_t>(m, name)
        .def(py::init<>())
        ;
}

void register_glm(py::module_& m)
{
    glm_base<double>(m, "GlmBase64");
    glm_base<float>(m, "GlmBase32");
    glm_gaussian<double>(m, "GlmGaussian64");
    glm_gaussian<float>(m, "GlmGaussian32");
}