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
        const Eigen::Ref<const vec_value_t>& mu,
        Eigen::Ref<vec_value_t> var
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            hessian,
            mu, var
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

        Suppose :math:`y \in \mathbb{R}` is a single observation
        that is statistically modeled with an exponential family.
        Recall that an exponential family is defined by the log-partition function :math:`A`
        so that the (negative) log-likelihood (up to a constant) is given by
        
        .. math::
            \begin{align*}
                D(\eta) = -y \eta + A(\eta)
            \end{align*}

        We define :math:`D(\eta)` as the *deviance*.
        For multidimensional data :math:`y \in \mathbb{R}^n`,
        we assume each :math:`y_i` is a sample from the same exponential family.
        It is useful to define :math:`\underline{A}(\eta) \in \mathbb{R}^n`
        where :math:`\eta \in \mathbb{R}^n` and :math:`\underline{A}(\eta)_i = A(\eta_i)`.

        The purpose of a GLM class is to define methods that evaluate key quantities regarding this model
        that are required for solving the group lasso problem.

        .. note::
            Our definition of deviance is the negative of the standard definition.
            Moreover, it is off by a factor of 2.
            This was more of a design choice to be consistent with the group lasso problem.

        Every GLM-like class must inherit from this class and override the methods
        before passing into the solver.
        )delimiter")
        .def(py::init<>())
        .def("gradient", &internal_t::gradient, R"delimiter(
        Element-wise gradient of the log-partition function.

        Computes :math:`A'(\eta_i)` for every element ``i``.

        Parameters
        ----------
        eta : (n,) np.ndarray
            Natural parameter.
        mu : (n,) np.ndarray
            The gradient, or mean parameter, to store.
        )delimiter")
        .def("gradient_inverse", &internal_t::gradient_inverse, R"delimiter(
        Element-wise inverse gradient of the log-partition function.

        Computes :math:`(A')^{-1}(\mu_i)` for every element ``i``.

        Parameters
        ----------
        mu : (n,) np.ndarray
            The mean parameter.
        eta : (n,) np.ndarray
            The natural parameter to store.
        )delimiter")
        .def("hessian", &internal_t::hessian, R"delimiter(
        Element-wise hessian of the log-partition function.

        Computes :math:`A''(\eta_i)` for every element ``i``.

        .. note::
            Since the hessian is a diagonal matrix, we only output the diagonal.
            Interestingly, most hessian computations become greatly simplified
            when evaluated using the mean parameter instead of the natural parameter.
            Hence, the hessian computation assumes the mean parameter is provided.

        Parameters
        ----------
        mu : (n,) np.ndarray
            The mean parameter.
        var : (n,) np.ndarray
            The hessian, or variance parameter, to store.
        )delimiter")
        .def("deviance", &internal_t::deviance, R"delimiter(
        Element-wise deviance function.

        Computes :math:`D(\eta_i)` for every element ``i``.

        Parameters
        ----------
        y : (n,) np.ndarray
            Observations (sufficient statistics).
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