#include "decl.hpp"
#include <adelie_core/glm/glm_base.hpp>
#include <adelie_core/glm/glm_gaussian.hpp>
#include <adelie_core/glm/glm_binomial.hpp>
#include <adelie_core/glm/glm_multinomial.hpp>
#include <adelie_core/glm/glm_poisson.hpp>

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
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> mu
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            gradient,
            eta, weights, mu
        );
    }

    void hessian(
        const Eigen::Ref<const vec_value_t>& mu,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> var
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            hessian,
            mu, weights, var
        );
    }

    value_t deviance(
        const Eigen::Ref<const vec_value_t>& y,
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& weights
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            value_t,
            base_t,
            deviance,
            y, eta, weights
        );
    }

    value_t deviance_full(
        const Eigen::Ref<const vec_value_t>& y,
        const Eigen::Ref<const vec_value_t>& weights
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            value_t,
            base_t,
            deviance_full,
            y, weights
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

        The generalized linear model is given by the (weighted) negative likelihood
        
        .. math::
            \begin{align*}
                D(\eta) = \sum\limits_{i=1}^n w_{i} \left(
                    -y_i \eta_i + A_i(\eta)
                \right)
            \end{align*}

        We define :math:`D(\eta)` as the *deviance* and :math:`A(\eta) := \sum_{i=1}^n w_{i} A_i(\eta)`
        as the *log-partition function*.
        Here, :math:`w \geq 0` and :math:`A_i` are any convex functions.

        The purpose of a GLM class is to define methods that evaluate key quantities regarding this model
        that are required for solving the group lasso problem.

        .. note::
            Our definition of deviance is non-standard.
            However, the differences are unimportant since deviance 
            is only relevant in terms of percent deviance explained.
            Both our definition and the standard one result in the same quantity
            for percent deviance explained.
            This was more of a design choice to be consistent with the group lasso problem.

        Every GLM-like class must inherit from this class and override the methods
        before passing into the solver.
        )delimiter")
        .def(py::init<>())
        .def("gradient", &internal_t::gradient, R"delimiter(
        Gradient of the log-partition function.

        Computes :math:`\nabla A(\eta)`.

        Parameters
        ----------
        eta : (n,) np.ndarray
            Natural parameter.
        weights : (n,) np.ndarray
            Observation weights.
        mu : (n,) np.ndarray
            The gradient, or mean parameter, to store.
        )delimiter")
        .def("hessian", &internal_t::hessian, R"delimiter(
        Hessian of the log-partition function.

        Computes :math:`\nabla^2 A(\eta)`.

        .. note::
            Although the hessian is in general a fully dense matrix,
            we only require the user to output a diagonal matrix.
            It is recommended that the diagonal matrix dominate the true hessian.
            However, in some cases, the diagonal of the hessian suffices.
            Interestingly, most hessian computations become greatly simplified
            when evaluated using the mean parameter instead of the natural parameter.
            Hence, the hessian computation assumes the mean parameter is provided.

        Parameters
        ----------
        mu : (n,) np.ndarray
            Mean parameter.
        weights : (n,) np.ndarray
            Observation weights.
        var : (n,) np.ndarray
            The hessian, or variance parameter, to store.
        )delimiter")
        .def("deviance", &internal_t::deviance, R"delimiter(
        Deviance function.

        Computes :math:`D(\eta)`.

        Parameters
        ----------
        y : (n,) np.ndarray
            Observations (sufficient statistics).
            It is assumed that ``y`` only takes on values assumed by the GLM.
        eta : (n,) np.ndarray
            Natural parameter.
        weights : (n,) np.ndarray
            Observation weights.

        Returns
        -------
        dev : float
            Deviance.
        )delimiter")
        .def("deviance_full", &internal_t::deviance_full, R"delimiter(
        Deviance function at the full-model.

        Computes :math:`D(\eta^\star)` where :math:`\eta^\star` is the minimizer.

        Parameters
        ----------
        y : (n,) np.ndarray
            Observations (sufficient statistics).
            It is assumed that ``y`` only takes on values assumed by the GLM.
        weights : (n,) np.ndarray
            Observation weights.

        Returns
        -------
        dev : float
            Deviance at the full model.
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

template <class T>
void glm_multinomial(py::module_& m, const char* name)
{
    using internal_t = ad::glm::GlmMultinomial<T>;
    using base_t = typename internal_t::base_t;
    py::class_<internal_t, base_t>(m, name)
        .def(py::init<size_t>(), 
            py::arg("K")
        )
        ;
}

template <class T>
void glm_poisson(py::module_& m, const char* name)
{
    using internal_t = ad::glm::GlmPoisson<T>;
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
    glm_multinomial<double>(m, "GlmMultinomial64");
    glm_multinomial<float>(m, "GlmMultinomial32");
    glm_poisson<double>(m, "GlmPoisson64");
    glm_poisson<float>(m, "GlmPoisson32");
}