#include "decl.hpp"
#include <adelie_core/constraint/constraint_base.hpp>

namespace py = pybind11;
namespace ad = adelie_core;

template <class T>
class PyConstraintBase : public ad::constraint::ConstraintBase<T>
{
    using base_t = ad::constraint::ConstraintBase<T>;
public:
    using base_t::base_t;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;

    void solve(
        Eigen::Ref<vec_value_t> x,
        Eigen::Ref<vec_value_t> mu,
        const Eigen::Ref<const vec_value_t>& quad,
        const Eigen::Ref<const vec_value_t>& linear,
        value_t l1,
        value_t l2
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            solve,
            x, mu, quad, linear, l1, l2
        );
    }

    void gradient(
        const Eigen::Ref<const vec_value_t>& x,
        const Eigen::Ref<const vec_value_t>& mu,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            gradient,
            x, mu, out
        );
    }

    int duals() override
    {
        PYBIND11_OVERRIDE_PURE(
            int,
            base_t,
            duals,
        );
    }
};

template <class T>
void constraint_base(py::module_& m, const char* name)
{
    using trampoline_t = PyConstraintBase<T>;
    using internal_t = ad::constraint::ConstraintBase<T>;
    using value_t = typename internal_t::value_t;
    using vec_value_t = typename internal_t::vec_value_t;
    py::class_<internal_t, trampoline_t>(m, name, R"delimiter(
        Base constraint class.
        
        The purpose of a constraint class is to define methods that 
        update certain quantities that are required for solving the constrained group lasso problem.

        Every constraint-like class must inherit from this class and override the methods
        before passing into the solver.
    )delimiter")
        .def(py::init<>())
        .def_property_readonly("dual_size", &internal_t::duals, R"delimiter(
        Number of duals.
        )delimiter")
        .def("solve", &internal_t::solve, R"delimiter(
        Computes the block-coordinate update.

        The block-coordinate update is given by solving

        .. math::
            \begin{align*}
                \mathrm{minimize}_x \quad&
                \frac{1}{2} x^\top \Sigma x - v^\top x + \lambda_1 \|x\|_2 + \frac{\lambda_2}{2} \|x\|_2^2
                \\
                \text{subject to} \quad&
                \phi(x) \leq 0
            \end{align*}

        where :math:`\phi` defines the current constraint.

        Parameters
        ----------
        x : (d,) np.ndarray 
            The primal :math:`x`.
            The passed-in values may be used as a warm-start for the internal solver.
            The output is stored back in this argument.
        mu : (m,) np.ndarray
            The dual :math:`\mu`.
            The passed-in values may be used as a warm-start for the internal solver.
            The output is stored back in this argument.
        quad : (d,) np.ndarray
            The quadratic component :math:`\Sigma`. 
        linear : (d,) np.ndarray
            The linear component :math:`v`.
        l1 : float
            The first regularization :math:`\lambda_1`.
        l2 : float
            The second regularization :math:`\lambda_2`.
        )delimiter")
        .def("gradient", &internal_t::gradient, R"delimiter(
        Computes the gradient of the Lagrangian.

        The gradient of the Lagrangian (with respect to the primal) is given by

        .. math::
            \begin{align*}
                \mu^\top \phi'(x)
            \end{align*}

        where :math:`\phi'(x)` is the Jacobian of :math:`\phi` at :math:`x`.

        Parameters
        ----------
        x : (d,) np.ndarray
            The primal :math:`x` at which to evaluate the gradient.
        mu : (m,) np.ndarray
            The dual :math:`\mu` at which to evaluate the gradient.
        out : (d,) np.ndarray
            The output vector to store the gradient.
        )delimiter")
        .def("duals", &internal_t::duals, R"delimiter(
        Number of dual variables.

        Returns
        -------
        size : int
            Number of dual variables.
        )delimiter")
        ;
}

void register_constraint(py::module_& m)
{
    py::bind_vector<std::vector<ad::constraint::ConstraintBase<double>*>>(m, "VectorConstraintBase64");
    py::bind_vector<std::vector<ad::constraint::ConstraintBase<float>*>>(m, "VectorConstraintBase32");

    constraint_base<double>(m, "ConstraintBase64");
    constraint_base<float>(m, "ConstraintBase32");
}