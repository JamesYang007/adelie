#include "decl.hpp"
#include <adelie_core/glm/glm_base.hpp>
#include <adelie_core/glm/glm_gaussian.hpp>
#include <adelie_core/glm/glm_binomial.hpp>

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

    void gradient_inverse(
        const Eigen::Ref<const vec_value_t>& mu,
        Eigen::Ref<vec_value_t> eta
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            gradient_inverse,
            mu, eta
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

    void deviance(
        const Eigen::Ref<const vec_value_t>& y,
        const Eigen::Ref<const vec_value_t>& eta,
        Eigen::Ref<vec_value_t> dev
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            deviance,
            y, eta, dev
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
        .def("gradient_inverse", &internal_t::gradient_inverse, R"delimiter(
        Inverse gradient of the log-partition function.

        Computes :math:`(\nabla \underline{A})^{-1}(\mu)`
        where :math:`\underline{A}(\eta)_k = A(\eta_k)`
        and :math:`A` is the log-partition function.

        Parameters
        ----------
        mu : (n,) np.ndarray
            The mean parameter.
        eta : (n,) np.ndarray
            The natural parameter to store.
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
        .def("deviance", &internal_t::deviance, R"delimiter(
        Element-wise deviance function.

        Computes :math:`-y_i \eta_i + A(\eta_i)` for each entry :math:`i`
        where :math:`A` is the log-partition function.

        Parameters
        ----------
        y : (n,) np.ndarray
            Sufficient statistic.
        eta : (n,) np.ndarray
            Natural parameter.
        dev : (n,) np.ndarray
            The deviance to store.
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

template <class T>
void glm_binomial(py::module_& m, const char* name)
{
    using internal_t = ad::glm::GlmBinomial<T>;
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
    glm_binomial<double>(m, "GlmBinomial64");
    glm_binomial<float>(m, "GlmBinomial32");
}