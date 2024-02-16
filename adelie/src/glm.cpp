#include "decl.hpp"
#include <adelie_core/glm/glm_base.hpp>
#include <adelie_core/glm/glm_binomial.hpp>
#include <adelie_core/glm/glm_cox.hpp>
#include <adelie_core/glm/glm_gaussian.hpp>
#include <adelie_core/glm/glm_multibase.hpp>
#include <adelie_core/glm/glm_multigaussian.hpp>
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

    value_t loss(
        const Eigen::Ref<const vec_value_t>& y,
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& weights
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            value_t,
            base_t,
            loss,
            y, eta, weights
        );
    }

    value_t loss_full(
        const Eigen::Ref<const vec_value_t>& y,
        const Eigen::Ref<const vec_value_t>& weights
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            value_t,
            base_t,
            loss_full,
            y, weights
        );
    }
};

template <class T>
void glm_base(py::module_& m, const char* name)
{
    using trampoline_t = PyGlmBase<T>;
    using internal_t = ad::glm::GlmBase<T>;
    using string_t = typename internal_t::string_t;
    py::class_<internal_t, trampoline_t>(m, name, R"delimiter(
        Base GLM class.

        The generalized linear model is given by the (weighted) negative likelihood
        
        .. math::
            \begin{align*}
                \ell(\eta) = \sum\limits_{i=1}^n w_{i} \left(
                    -y_i \eta_i + A_i(\eta)
                \right)
            \end{align*}

        We define :math:`\ell(\eta)` as the *loss* and :math:`A(\eta) := \sum_{i=1}^n w_{i} A_i(\eta)`
        as the *log-partition function*.
        Here, :math:`w \geq 0` and :math:`A_i` are any convex functions.

        The purpose of a GLM class is to define methods that evaluate key quantities regarding this model
        that are required for solving the group lasso problem.

        Every GLM-like class must inherit from this class and override the methods
        before passing into the solver.
        )delimiter")
        .def(py::init<const string_t&>(),
            py::arg("name")
        )
        .def_readonly("name", &internal_t::name, R"delimiter(
            Name of the GLM family.
        )delimiter")
        .def_readonly("is_multi", &internal_t::is_multi, R"delimiter(
            ``True`` if it defines a multi-response GLM family.
            It is always ``False`` for this base class.
        )delimiter")
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
        .def("loss", &internal_t::loss, R"delimiter(
        Loss function.

        Computes :math:`\ell(\eta)`.

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
        loss : float
            Loss.
        )delimiter")
        .def("loss_full", &internal_t::loss_full, R"delimiter(
        Loss function at the full-model.

        Computes :math:`\ell(\eta^\star)` where :math:`\eta^\star` is the minimizer.

        Parameters
        ----------
        y : (n,) np.ndarray
            Observations (sufficient statistics).
            It is assumed that ``y`` only takes on values assumed by the GLM.
        weights : (n,) np.ndarray
            Observation weights.

        Returns
        -------
        loss : float
            Loss at the full model.
        )delimiter")
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
void glm_cox(py::module_& m, const char* name)
{
    using internal_t = ad::glm::GlmCox<T>;
    using base_t = typename internal_t::base_t;
    py::class_<internal_t, base_t>(m, name)
        .def(py::init<>())
        .def_static("_partial_sum", &ad::glm::cox::_partial_sum<
            Eigen::Ref<const ad::util::rowvec_type<T>>,
            Eigen::Ref<const ad::util::rowvec_type<T>>,
            Eigen::Ref<const ad::util::rowvec_type<T>>,
            Eigen::Ref<ad::util::rowvec_type<T>>
        >)
        .def_static("_average_ties", &ad::glm::cox::_average_ties<
            Eigen::Ref<const ad::util::rowvec_type<T>>,
            Eigen::Ref<const ad::util::rowvec_type<T>>,
            Eigen::Ref<ad::util::rowvec_type<T>>
        >)
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
void glm_poisson(py::module_& m, const char* name)
{
    using internal_t = ad::glm::GlmPoisson<T>;
    using base_t = typename internal_t::base_t;
    py::class_<internal_t, base_t>(m, name)
        .def(py::init<>())
        ;
}

template <class T>
class PyGlmMultiBase : public ad::glm::GlmMultiBase<T>
{
    using base_t = ad::glm::GlmMultiBase<T>;
public:
    using base_t::base_t;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::rowarr_value_t;

    void gradient(
        const Eigen::Ref<const rowarr_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<rowarr_value_t> mu
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
        const Eigen::Ref<const rowarr_value_t>& mu,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<rowarr_value_t> var
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            hessian,
            mu, weights, var
        );
    }

    value_t loss(
        const Eigen::Ref<const rowarr_value_t>& y,
        const Eigen::Ref<const rowarr_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& weights
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            value_t,
            base_t,
            loss,
            y, eta, weights
        );
    }

    value_t loss_full(
        const Eigen::Ref<const rowarr_value_t>& y,
        const Eigen::Ref<const vec_value_t>& weights
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            value_t,
            base_t,
            loss_full,
            y, weights
        );
    }
};

template <class T>
void glm_multibase(py::module_& m, const char* name)
{
    using trampoline_t = PyGlmMultiBase<T>;
    using internal_t = ad::glm::GlmMultiBase<T>;
    using string_t = typename internal_t::string_t;
    py::class_<internal_t, trampoline_t>(m, name, R"delimiter(
        Base Multi-response GLM class.

        The generalized multi-response linear model is given by the (weighted) negative likelihood
        
        .. math::
            \begin{align*}
                \ell(\eta) = \frac{1}{K} \sum\limits_{i=1}^n w_{i} \left(
                    -\sum\limits_{k=1}^K y_{ik} \eta_{ik} + A_i(\eta)
                \right)
            \end{align*}

        We define :math:`\ell(\eta)` as the *loss* and :math:`A(\eta) := K^{-1} \sum_{i=1}^n w_{i} A_i(\eta)`
        as the *log-partition function*.
        Here, :math:`w \geq 0` and :math:`A_i` are any convex functions.

        The purpose of a GLM class is to define methods that evaluate key quantities regarding this model
        that are required for solving the group lasso problem.

        Every multi-response GLM-like class must inherit from this class and override the methods
        before passing into the solver.
        )delimiter")
        .def(py::init<const string_t&, bool>(),
            py::arg("name"),
            py::arg("is_symmetric")
        )
        .def_readonly("name", &internal_t::name, R"delimiter(
            Name of the GLM family.
        )delimiter")
        .def_readonly("is_multi", &internal_t::is_multi, R"delimiter(
        ``True`` if it defines a multi-response GLM family.
        It is always ``True`` for this base class.
        )delimiter")
        .def_readonly("is_symmetric", &internal_t::is_symmetric, R"delimiter(
        ``True`` if the loss portion remains invariant under common scalar shift
        in the coefficients across the different responses.
        )delimiter")
        .def("gradient", &internal_t::gradient, R"delimiter(
        Gradient of the log-partition function.

        Computes :math:`\nabla A(\eta)`.

        Parameters
        ----------
        eta : (n, K) np.ndarray
            Natural parameter.
        weights : (n,) np.ndarray
            Observation weights.
        mu : (n, K) np.ndarray
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
        mu : (n, K) np.ndarray
            Mean parameter.
        weights : (n,) np.ndarray
            Observation weights.
        var : (n, K) np.ndarray
            The hessian, or variance parameter, to store.
        )delimiter")
        .def("loss", &internal_t::loss, R"delimiter(
        Loss function.

        Computes :math:`\ell(\eta)`.

        Parameters
        ----------
        y : (n, K) np.ndarray
            Observations (sufficient statistics).
            It is assumed that ``y`` only takes on values assumed by the GLM.
        eta : (n, K) np.ndarray
            Natural parameter.
        weights : (n,) np.ndarray
            Observation weights.

        Returns
        -------
        loss : float
            Loss.
        )delimiter")
        .def("loss_full", &internal_t::loss_full, R"delimiter(
        Loss function at the full-model.

        Computes :math:`\ell(\eta^\star)` where :math:`\eta^\star` is the minimizer.

        Parameters
        ----------
        y : (n, K) np.ndarray
            Observations (sufficient statistics).
            It is assumed that ``y`` only takes on values assumed by the GLM.
        weights : (n,) np.ndarray
            Observation weights.

        Returns
        -------
        loss : float
            Loss at the full model.
        )delimiter")
        ;
}

template <class T>
void glm_multigaussian(py::module_& m, const char* name)
{
    using internal_t = ad::glm::GlmMultiGaussian<T>;
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
        .def(py::init<>())
        ;
}

void register_glm(py::module_& m)
{
    glm_base<double>(m, "GlmBase64");
    glm_base<float>(m, "GlmBase32");
    glm_multibase<double>(m, "GlmMultiBase64");
    glm_multibase<float>(m, "GlmMultiBase32");
    glm_binomial<double>(m, "GlmBinomial64");
    glm_binomial<float>(m, "GlmBinomial32");
    glm_cox<double>(m, "GlmCox64");
    glm_cox<float>(m, "GlmCox32");
    glm_gaussian<double>(m, "GlmGaussian64");
    glm_gaussian<float>(m, "GlmGaussian32");
    glm_multigaussian<double>(m, "GlmMultiGaussian64");
    glm_multigaussian<float>(m, "GlmMultiGaussian32");
    glm_multinomial<double>(m, "GlmMultinomial64");
    glm_multinomial<float>(m, "GlmMultinomial32");
    glm_poisson<double>(m, "GlmPoisson64");
    glm_poisson<float>(m, "GlmPoisson32");
}