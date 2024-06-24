#include "decl.hpp"
#include <adelie_core/constraint/constraint_base.hpp>
#include <adelie_core/constraint/constraint_one_sided.hpp>
#include <adelie_core/constraint/constraint_box.hpp>

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
    using typename base_t::colmat_value_t;

    void solve(
        Eigen::Ref<vec_value_t> x,
        Eigen::Ref<vec_value_t> mu,
        const Eigen::Ref<const vec_value_t>& quad,
        const Eigen::Ref<const vec_value_t>& linear,
        value_t l1,
        value_t l2,
        const Eigen::Ref<const colmat_value_t>& Q
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            solve,
            x, mu, quad, linear, l1, l2, Q
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

    void project(
        Eigen::Ref<vec_value_t> x
    ) override
    {
        PYBIND11_OVERRIDE(
            void,
            base_t,
            project,
            x
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

    int primals() override
    {
        PYBIND11_OVERRIDE_PURE(
            int,
            base_t,
            primals,
        );
    }
};

template <class T>
void constraint_base(py::module_& m, const char* name)
{
    using trampoline_t = PyConstraintBase<T>;
    using internal_t = ad::constraint::ConstraintBase<T>;
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
        .def_property_readonly("primal_size", &internal_t::primals, R"delimiter(
        Number of primals.
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
                \phi(Q x) \leq 0
            \end{align*}

        where :math:`\phi` defines the current constraint.

        Parameters
        ----------
        x : (d,) ndarray 
            The primal :math:`x`.
            The passed-in values may be used as a warm-start for the internal solver.
            The output is stored back in this argument.
        mu : (m,) ndarray
            The dual :math:`\mu`.
            The passed-in values may be used as a warm-start for the internal solver.
            The output is stored back in this argument.
        quad : (d,) ndarray
            The quadratic component :math:`\Sigma`. 
        linear : (d,) ndarray
            The linear component :math:`v`.
        l1 : float
            The first regularization :math:`\lambda_1`.
        l2 : float
            The second regularization :math:`\lambda_2`.
        Q : (d, d) ndarray
            Orthogonal matrix :math:`Q`.
        )delimiter",
            py::arg("x").noconvert(),
            py::arg("mu").noconvert(),
            py::arg("quad").noconvert(),
            py::arg("linear").noconvert(),
            py::arg("l1"),
            py::arg("l2"),
            py::arg("Q").noconvert()
        )
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
        x : (d,) ndarray
            The primal :math:`x` at which to evaluate the gradient.
        mu : (m,) ndarray
            The dual :math:`\mu` at which to evaluate the gradient.
        out : (d,) ndarray
            The output vector to store the gradient.
        )delimiter",
            py::arg("x").noconvert(),
            py::arg("mu").noconvert(),
            py::arg("out").noconvert()
        )
        .def("project", &internal_t::project, R"delimiter(
        Computes a projection onto the feasible set.

        The feasible set is defined by :math:`\{x : \phi(x) \leq 0 \}`.
        A projection can be user-defined, that is, the user may define any
        norm :math:`\|\cdot\|` such that the function returns a solution to

        .. math::
            \begin{align*}
                \mathrm{minimize}_z \quad& \|x - z\| \\
                \text{subject to} \quad& \phi(z) \leq 0
            \end{align*}

        This function is only used by the solver after convergence
        to attempt to bring the coordinates into the feasible set.
        If not overriden, it will perform a no-op, assuming :math:`x` is already feasible.

        Parameters
        ----------
        x : (d,) ndarray
            The primal :math:`x` to project onto the feasible set.
            The output is stored back in this argument.
        )delimiter",
            py::arg("x").noconvert()
        )
        .def("duals", &internal_t::duals, R"delimiter(
        Returns the number of dual variables.

        Returns
        -------
        size : int
            Number of dual variables.
        )delimiter")
        .def("primals", &internal_t::primals, R"delimiter(
        Returns the number of primal variables.

        Returns
        -------
        size : int
            Number of primal variables.
        )delimiter")
        ;
}

template <class ValueType>
void constraint_one_sided_base(py::module_& m, const char* name)
{
    using internal_t = ad::constraint::ConstraintOneSidedBase<ValueType>;
    using base_t = typename internal_t::base_t;
    py::class_<internal_t, base_t>(m, name, 
        "Core constraint base class for one-sided bound constraint."
        )
        ;
}

template <class ValueType>
void constraint_one_sided_proximal_newton(py::module_& m, const char* name)
{
    using internal_t = ad::constraint::ConstraintOneSidedProximalNewton<ValueType>;
    using base_t = typename internal_t::base_t;
    using value_t = typename internal_t::value_t;
    using vec_value_t = typename internal_t::vec_value_t;
    py::class_<internal_t, base_t>(m, name, 
        "Core constraint class for one-sided bound constraint with proximal Newton solver."
        )
        .def(py::init<
            const Eigen::Ref<const vec_value_t>,
            const Eigen::Ref<const vec_value_t>,
            size_t,
            value_t,
            size_t,
            value_t,
            value_t,
            value_t
        >(), 
            py::arg("sgn"),
            py::arg("b"),
            py::arg("max_iters"),
            py::arg("tol"),
            py::arg("nnls_max_iters"),
            py::arg("nnls_tol"),
            py::arg("cs_tol"),
            py::arg("slack")
        )
        .def("debug_info", &internal_t::debug_info, R"delimiter(
        Returns debug information.

        This method is only intended for developers for debugging purposes.
        The package must be compiled with the compiler flag `-DADELIE_CORE_DEBUG`
        to see the debug information.
        )delimiter")
        ;
}

template <class ValueType>
void constraint_one_sided_admm(py::module_& m, const char* name)
{
    using internal_t = ad::constraint::ConstraintOneSidedADMM<ValueType>;
    using base_t = typename internal_t::base_t;
    using value_t = typename internal_t::value_t;
    using vec_value_t = typename internal_t::vec_value_t;
    py::class_<internal_t, base_t>(m, name, 
        "Core constraint class for one-sided bound constraint with ADMM solver."
        )
        .def(py::init<
            const Eigen::Ref<const vec_value_t>,
            const Eigen::Ref<const vec_value_t>,
            size_t,
            value_t,
            value_t,
            value_t
        >(), 
            py::arg("sgn"),
            py::arg("b"),
            py::arg("max_iters"),
            py::arg("tol_abs"),
            py::arg("tol_rel"),
            py::arg("rho")
        )
        .def("debug_info", &internal_t::debug_info, R"delimiter(
        Returns debug information.

        This method is only intended for developers for debugging purposes.
        The package must be compiled with the compiler flag `-DADELIE_CORE_DEBUG`
        to see the debug information.
        )delimiter")
        ;
}

template <class ValueType>
void constraint_box_base(py::module_& m, const char* name)
{
    using internal_t = ad::constraint::ConstraintBoxBase<ValueType>;
    using base_t = typename internal_t::base_t;
    py::class_<internal_t, base_t>(m, name, 
        "Core constraint base class for box constraint."
        )
        ;
}

template <class ValueType>
void constraint_box_proximal_newton(py::module_& m, const char* name)
{
    using internal_t = ad::constraint::ConstraintBoxProximalNewton<ValueType>;
    using base_t = typename internal_t::base_t;
    using value_t = typename internal_t::value_t;
    using vec_value_t = typename internal_t::vec_value_t;
    py::class_<internal_t, base_t>(m, name, 
        "Core constraint class for box constraint with proximal Newton solver."
        )
        .def(py::init<
            const Eigen::Ref<const vec_value_t>,
            const Eigen::Ref<const vec_value_t>,
            size_t,
            value_t,
            size_t,
            value_t,
            value_t,
            value_t
        >(), 
            py::arg("lower"),
            py::arg("upper"),
            py::arg("max_iters"),
            py::arg("tol"),
            py::arg("nnls_max_iters"),
            py::arg("nnls_tol"),
            py::arg("cs_tol"),
            py::arg("slack")
        )
        ;
}

void register_constraint(py::module_& m)
{
    py::bind_vector<std::vector<ad::constraint::ConstraintBase<double>*>>(m, "VectorConstraintBase64");
    py::bind_vector<std::vector<ad::constraint::ConstraintBase<float>*>>(m, "VectorConstraintBase32");

    constraint_base<double>(m, "ConstraintBase64");
    constraint_base<float>(m, "ConstraintBase32");

    constraint_one_sided_base<double>(m, "ConstraintOneSidedBase64");
    constraint_one_sided_base<float>(m, "ConstraintOneSidedBase32");
    constraint_one_sided_proximal_newton<double>(m, "ConstraintOneSidedProximalNewton64");
    constraint_one_sided_proximal_newton<float>(m, "ConstraintOneSidedProximalNewton32");
    constraint_one_sided_admm<double>(m, "ConstraintOneSidedADMM64");
    constraint_one_sided_admm<float>(m, "ConstraintOneSidedADMM32");

    constraint_box_base<double>(m, "ConstraintBoxBase64");
    constraint_box_base<float>(m, "ConstraintBoxBase32");
    constraint_box_proximal_newton<double>(m, "ConstraintBoxProximalNewton64");
    constraint_box_proximal_newton<float>(m, "ConstraintBoxProximalNewton32");
}