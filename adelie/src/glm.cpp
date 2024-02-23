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
        Eigen::Ref<vec_value_t> grad
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            gradient,
            eta, grad
        );
    }

    void hessian(
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& grad,
        Eigen::Ref<vec_value_t> hess
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            hessian,
            eta, grad, hess
        );
    }

    value_t loss(
        const Eigen::Ref<const vec_value_t>& eta
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            value_t,
            base_t,
            loss,
            eta 
        );
    }

    value_t loss_full() override
    {
        PYBIND11_OVERRIDE_PURE(
            value_t,
            base_t,
            loss_full,
        );
    }
};

template <class T>
void glm_base(py::module_& m, const char* name)
{
    using trampoline_t = PyGlmBase<T>;
    using internal_t = ad::glm::GlmBase<T>;
    using string_t = typename internal_t::string_t;
    using vec_value_t = typename internal_t::vec_value_t;
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
        .def(py::init<
            const string_t&,
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&
        >(),
            py::arg("name"),
            py::arg("y"),
            py::arg("weights")
        )
        .def_readonly("name", &internal_t::name, R"delimiter(
            Name of the GLM family.
        )delimiter")
        .def_readonly("is_multi", &internal_t::is_multi, R"delimiter(
            ``True`` if it defines a multi-response GLM family.
            It is always ``False`` for this base class.
        )delimiter")
        .def("gradient", &internal_t::gradient, R"delimiter(
        Gradient of the negative loss function.

        Computes :math:`-\nabla \ell(\eta)`.

        Parameters
        ----------
        eta : (n,) np.ndarray
            Natural parameter.
        grad : (n,) np.ndarray
            The gradient to store.
        )delimiter")
        .def("hessian", &internal_t::hessian, R"delimiter(
        Hessian of the loss function.

        Computes :math:`\nabla^2 \ell(\eta)`.

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
        eta : (n,) np.ndarray
            Natural parameter.
        grad : (n,) np.ndarray
            Gradient.
        hess : (n,) np.ndarray
            The hessian to store.
        )delimiter")
        .def("loss", &internal_t::loss, R"delimiter(
        Loss function.

        Computes :math:`\ell(\eta)`.

        Parameters
        ----------
        eta : (n,) np.ndarray
            Natural parameter.

        Returns
        -------
        loss : float
            Loss.
        )delimiter")
        .def("loss_full", &internal_t::loss_full, R"delimiter(
        Loss function at the saturated model.

        Computes :math:`\ell(\eta^\star)` where :math:`\eta^\star` is the minimizer.

        Returns
        -------
        loss : float
            Loss at the saturated model.
        )delimiter")
        ;
}

template <class T>
void glm_binomial(py::module_& m, const char* name)
{
    using internal_t = ad::glm::GlmBinomial<T>;
    using base_t = typename internal_t::base_t;
    using vec_value_t = typename internal_t::vec_value_t;
    py::class_<internal_t, base_t>(m, name)
        .def(py::init<
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&
        >())
        ;
}

template <class T>
void glm_cox(py::module_& m, const char* name)
{
    using internal_t = ad::glm::GlmCox<T>;
    using base_t = typename internal_t::base_t;
    using vec_value_t = typename internal_t::vec_value_t;
    py::class_<internal_t, base_t>(m, name)
        .def(py::init<
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>& ,
            const std::string&
        >(),
            py::arg("start"),
            py::arg("stop"),
            py::arg("status"),
            py::arg("weights"),
            py::arg("tie_method")
        )
        .def_readonly("start_order", &internal_t::start_order)
        .def_readonly("start_so", &internal_t::start_so)
        .def_readonly("stop_order", &internal_t::stop_order)
        .def_readonly("stop_to", &internal_t::stop_to)
        .def_readonly("status_to", &internal_t::status_to)
        .def_readonly("weights_to", &internal_t::weights_to)
        .def_readonly("weights_size_to", &internal_t::weights_size_to)
        .def_readonly("weights_mean_to", &internal_t::weights_mean_to)
        .def_readonly("scale_to", &internal_t::scale_to)
        .def_static("_partial_sum_fwd", &ad::glm::cox::_partial_sum_fwd<
            Eigen::Ref<const vec_value_t>,
            Eigen::Ref<const vec_value_t>,
            Eigen::Ref<const vec_value_t>,
            Eigen::Ref<vec_value_t>
        >)
        .def_static("_partial_sum_bwd", &ad::glm::cox::_partial_sum_bwd<
            Eigen::Ref<const vec_value_t>,
            Eigen::Ref<const vec_value_t>,
            Eigen::Ref<const vec_value_t>,
            Eigen::Ref<vec_value_t>
        >)
        .def_static("_at_risk_sum", &ad::glm::cox::_at_risk_sum<
            Eigen::Ref<const vec_value_t>,
            Eigen::Ref<const vec_value_t>,
            Eigen::Ref<const vec_value_t>,
            Eigen::Ref<const vec_value_t>,
            Eigen::Ref<const vec_value_t>,
            Eigen::Ref<vec_value_t>,
            Eigen::Ref<vec_value_t>,
            Eigen::Ref<vec_value_t>
        >)
        .def_static("_nnz_event_ties_sum", &ad::glm::cox::_nnz_event_ties_sum<
            Eigen::Ref<const vec_value_t>,
            Eigen::Ref<const vec_value_t>,
            Eigen::Ref<const vec_value_t>,
            Eigen::Ref<const vec_value_t>,
            Eigen::Ref<vec_value_t>
        >)
        .def_static("_scale", [](
            const Eigen::Ref<const vec_value_t>& t,
            const Eigen::Ref<const vec_value_t>& status,
            const Eigen::Ref<const vec_value_t>& w,
            const std::string& tie_method,
            Eigen::Ref<vec_value_t> out
        ){
            ad::glm::cox::_scale(t, status, w, ad::util::convert_tie_method(tie_method), out);
        })
        ;
}

template <class T>
void glm_gaussian(py::module_& m, const char* name)
{
    using internal_t = ad::glm::GlmGaussian<T>;
    using base_t = typename internal_t::base_t;
    using vec_value_t = typename internal_t::vec_value_t;
    py::class_<internal_t, base_t>(m, name)
        .def(py::init<
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&
        >())
        ;
}

template <class T>
void glm_poisson(py::module_& m, const char* name)
{
    using internal_t = ad::glm::GlmPoisson<T>;
    using base_t = typename internal_t::base_t;
    using vec_value_t = typename internal_t::vec_value_t;
    py::class_<internal_t, base_t>(m, name)
        .def(py::init<
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&
        >())
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
        Eigen::Ref<rowarr_value_t> grad
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            gradient,
            eta, grad
        );
    }

    void hessian(
        const Eigen::Ref<const rowarr_value_t>& eta,
        const Eigen::Ref<const rowarr_value_t>& grad,
        Eigen::Ref<rowarr_value_t> hess
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            hessian,
            eta, grad, hess
        );
    }

    value_t loss(
        const Eigen::Ref<const rowarr_value_t>& eta
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            value_t,
            base_t,
            loss,
            eta 
        );
    }

    value_t loss_full() override
    {
        PYBIND11_OVERRIDE_PURE(
            value_t,
            base_t,
            loss_full,
        );
    }
};

template <class T>
void glm_multibase(py::module_& m, const char* name)
{
    using trampoline_t = PyGlmMultiBase<T>;
    using internal_t = ad::glm::GlmMultiBase<T>;
    using string_t = typename internal_t::string_t;
    using rowarr_value_t = typename internal_t::rowarr_value_t;
    using vec_value_t = typename internal_t::vec_value_t;
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
        .def(py::init<
            const string_t&,
            const Eigen::Ref<const rowarr_value_t>&,
            const Eigen::Ref<const vec_value_t>& 
        >(),
            py::arg("name"),
            py::arg("y"),
            py::arg("weights")
        )
        .def_readonly("name", &internal_t::name, R"delimiter(
            Name of the GLM family.
        )delimiter")
        .def_readonly("is_multi", &internal_t::is_multi, R"delimiter(
        ``True`` if it defines a multi-response GLM family.
        It is always ``True`` for this base class.
        )delimiter")
        .def("gradient", &internal_t::gradient, R"delimiter(
        Gradient of the negative loss function.

        Computes :math:`-\nabla \ell(\eta)`.

        Parameters
        ----------
        eta : (n, K) np.ndarray
            Natural parameter.
        grad : (n, K) np.ndarray
            The gradient to store.
        )delimiter")
        .def("hessian", &internal_t::hessian, R"delimiter(
        Hessian of the loss function.

        Computes :math:`\nabla^2 \ell(\eta)`.

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
        eta : (n, K) np.ndarray
            Natural parameter.
        grad : (n, K) np.ndarray
            Gradient.
        hess : (n, K) np.ndarray
            The hessian to store.
        )delimiter")
        .def("loss", &internal_t::loss, R"delimiter(
        Loss function.

        Computes :math:`\ell(\eta)`.

        Parameters
        ----------
        eta : (n, K) np.ndarray
            Natural parameter.

        Returns
        -------
        loss : float
            Loss.
        )delimiter")
        .def("loss_full", &internal_t::loss_full, R"delimiter(
        Loss function at the saturated model.

        Computes :math:`\ell(\eta^\star)` where :math:`\eta^\star` is the minimizer.

        Returns
        -------
        loss : float
            Loss at the saturated model.
        )delimiter")
        ;
}

template <class T>
void glm_multigaussian(py::module_& m, const char* name)
{
    using internal_t = ad::glm::GlmMultiGaussian<T>;
    using base_t = typename internal_t::base_t;
    using rowarr_value_t = typename internal_t::rowarr_value_t;
    using vec_value_t = typename internal_t::vec_value_t;
    py::class_<internal_t, base_t>(m, name)
        .def(py::init<
            const Eigen::Ref<const rowarr_value_t>&,
            const Eigen::Ref<const vec_value_t>& 
        >())
        ;
}

template <class T>
void glm_multinomial(py::module_& m, const char* name)
{
    using internal_t = ad::glm::GlmMultinomial<T>;
    using base_t = typename internal_t::base_t;
    using rowarr_value_t = typename internal_t::rowarr_value_t;
    using vec_value_t = typename internal_t::vec_value_t;
    py::class_<internal_t, base_t>(m, name)
        .def(py::init<
            const Eigen::Ref<const rowarr_value_t>&,
            const Eigen::Ref<const vec_value_t>& 
        >())
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