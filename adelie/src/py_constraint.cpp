#include "py_decl.hpp"
#include <constraint/constraint.hpp>

namespace py = pybind11;
namespace ad = adelie_core;

template <class T>
class PyConstraintBase : public ad::constraint::ConstraintBase<T>
{
    using base_t = ad::constraint::ConstraintBase<T>;
public:
    using base_t::base_t;
    using typename base_t::value_t;
    using typename base_t::vec_index_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_uint64_t;
    using typename base_t::colmat_value_t;

    void solve(
        Eigen::Ref<vec_value_t> x,
        const Eigen::Ref<const vec_value_t>& quad,
        const Eigen::Ref<const vec_value_t>& linear,
        value_t l1,
        value_t l2,
        const Eigen::Ref<const colmat_value_t>& Q,
        Eigen::Ref<vec_uint64_t> buffer
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            solve,
            x, quad, linear, l1, l2, Q, buffer
        );
    }

    void gradient(
        const Eigen::Ref<const vec_value_t>& x,
        Eigen::Ref<vec_value_t> out
    ) const override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            gradient,
            x, out
        );
    }

    void gradient(
        const Eigen::Ref<const vec_value_t>& x,
        const Eigen::Ref<const vec_value_t>& mu,
        Eigen::Ref<vec_value_t> out
    ) const override
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
    ) const override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            project,
            x
        );
    }

    value_t solve_zero(
        const Eigen::Ref<const vec_value_t>& v,
        Eigen::Ref<vec_uint64_t> buffer
    ) override
    {
        PYBIND11_OVERRIDE_PURE(
            value_t,
            base_t,
            solve_zero,
            v, buffer
        );
    }

    void clear() override 
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t,
            clear,
        );
    }

    void dual(
        Eigen::Ref<vec_index_t> indices,
        Eigen::Ref<vec_value_t> values
    ) const override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            base_t, 
            dual,
            indices, values
        );
    }

    int duals_nnz() const override
    {
        PYBIND11_OVERRIDE_PURE(
            int,
            base_t,
            duals_nnz,
        );
    }

    int duals() const override
    {
        PYBIND11_OVERRIDE_PURE(
            int,
            base_t,
            duals,
        );
    }

    int primals() const override
    {
        PYBIND11_OVERRIDE_PURE(
            int,
            base_t,
            primals,
        );
    }

    size_t buffer_size() const override
    {
        PYBIND11_OVERRIDE_PURE(
            size_t,
            base_t,
            buffer_size,
        );
    }
};

template <class T>
void constraint_base(py::module_& m, const char* name)
{
    using trampoline_t = PyConstraintBase<T>;
    using internal_t = ad::constraint::ConstraintBase<T>;
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

        .. warning::
            This function is not thread-safe!

        Parameters
        ----------
        x : (d,) ndarray 
            The primal :math:`x`.
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
        buffer : (b,) ndarray
            Buffer of type ``uint64_t`` aligned at 8 bytes.
            The size must be at least as large as :func:`buffer_size`.
        )delimiter",
            py::arg("x").noconvert(),
            py::arg("quad").noconvert(),
            py::arg("linear").noconvert(),
            py::arg("l1"),
            py::arg("l2"),
            py::arg("Q").noconvert(),
            py::arg("buffer").noconvert()
        )
        .def("gradient", py::overload_cast<
            const Eigen::Ref<const vec_value_t>&,
            Eigen::Ref<vec_value_t> 
        >(&internal_t::gradient, py::const_), R"delimiter(
        Computes the gradient of the Lagrangian.

        The gradient of the Lagrangian (with respect to the primal) is given by

        .. math::
            \begin{align*}
                \mu^\top \phi'(x)
            \end{align*}

        where :math:`\phi'(x)` is the Jacobian of :math:`\phi` at :math:`x`
        and :math:`\mu` is the dual solution from the last call to :func:`solve`. 

        Parameters
        ----------
        x : (d,) ndarray
            The primal :math:`x` at which to evaluate the gradient.
        out : (d,) ndarray
            The output vector to store the gradient.
        )delimiter",
            py::arg("x").noconvert(),
            py::arg("out").noconvert()
        )
        .def("gradient_static", py::overload_cast<
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            Eigen::Ref<vec_value_t> 
        >(&internal_t::gradient, py::const_), R"delimiter(
        Computes the gradient of the Lagrangian.

        The gradient of the Lagrangian (with respect to the primal) is given by

        .. math::
            \begin{align*}
                \mu^\top \phi'(x)
            \end{align*}

        where :math:`\phi'(x)` is the Jacobian of :math:`\phi` at :math:`x`
        and :math:`\mu` is the dual given by ``mu``. 

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
        .def("solve_zero", &internal_t::solve_zero, R"delimiter(
        Solves the zero primal KKT condition problem.

        The zero primal KKT condition problem is given by

        .. math::
            \begin{align*}
                \mathrm{minimize}_{\mu \geq 0}
                \|v - \phi'(0)^\top \mu\|_2
            \end{align*}

        where :math:`\phi` is the current constraint function
        and :math:`\mu` is the dual variable.
        It is advised, but not necessary, that the object stores the solution internally
        so that a subsequent call to :func:`dual` will return the solution.

        .. warning::
            This function is not thread-safe!

        Parameters
        ----------
        v : (d,) ndarray
            The vector :math:`v`.
        buffer : (b,) ndarray
            Buffer of type ``uint64_t`` aligned at 8 bytes.
            The size must be at least as large as :func:`buffer_size`.

        Returns
        -------
        norm : float
            The optimal objective for the zero primal KKT condition problem.
        )delimiter")
        .def("clear", &internal_t::clear, R"delimiter(
        Clears internal data.

        The state of the constraint object must return back to
        that of the initial construction.

        .. warning::
            This function is not thread-safe!

        )delimiter")
        .def("dual", &internal_t::dual, R"delimiter(
        Returns the current dual variable in sparse format.

        Parameters
        ----------
        indices : (nnz,) ndarray
            The indices with non-zero dual values.
            The size must be at least the value returned by :func:`duals_nnz`.
        values : (nnz,) ndarray
            The non-zero dual values corresponding to ``indices``.
            The size must be at least the value returned by :func:`duals_nnz`.
        )delimiter")
        .def("duals_nnz", &internal_t::duals_nnz, R"delimiter(
        Returns the number of non-zero dual values.

        Returns
        -------
        nnz : int
            Number of non-zero dual values.
        )delimiter")
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
        .def("buffer_size", &internal_t::buffer_size, R"delimiter(
        Returns the buffer size in unit of 8 bytes.

        Returns
        -------
        size : int
            Buffer size in unit of 8 bytes.
        )delimiter")
        ;
}

template <class ValueType>
void constraint_box(py::module_& m, const char* name)
{
    using internal_t = ad::constraint::ConstraintBox<ValueType>;
    using base_t = typename internal_t::base_t;
    using value_t = typename internal_t::value_t;
    using vec_value_t = typename internal_t::vec_value_t;
    py::class_<internal_t, base_t>(m, name, 
        "Core constraint class for box constraint."
        )
        .def(py::init<
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            size_t,
            value_t,
            size_t,
            value_t,
            value_t
        >(), 
            py::arg("lower").noconvert(),
            py::arg("upper").noconvert(),
            py::arg("max_iters"),
            py::arg("tol"),
            py::arg("pinball_max_iters"),
            py::arg("pinball_tol"),
            py::arg("slack")
        )
        ;
}

template <class AType>
void constraint_linear(py::module_& m, const char* name)
{
    using internal_t = ad::constraint::ConstraintLinear<AType>;
    using A_t = typename internal_t::A_t;
    using base_t = typename internal_t::base_t;
    using value_t = typename internal_t::value_t;
    using vec_value_t = typename internal_t::vec_value_t;
    py::class_<internal_t, base_t>(m, name, 
        "Core constraint class for linear constraint."
        )
        .def(py::init<
            A_t&,
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            size_t,
            value_t,
            size_t,
            value_t,
            size_t,
            value_t,
            value_t,
            size_t
        >(), 
            py::arg("A").noconvert(),
            py::arg("lower").noconvert(),
            py::arg("upper").noconvert(),
            py::arg("A_vars").noconvert(),
            py::arg("max_iters"),
            py::arg("tol"),
            py::arg("nnls_max_iters"),
            py::arg("nnls_tol"),
            py::arg("pinball_max_iters"),
            py::arg("pinball_tol"),
            py::arg("slack"),
            py::arg("n_threads")
        )
        ;
}


template <class ValueType>
void constraint_one_sided(py::module_& m, const char* name)
{
    using internal_t = ad::constraint::ConstraintOneSided<ValueType>;
    using base_t = typename internal_t::base_t;
    using value_t = typename internal_t::value_t;
    using vec_value_t = typename internal_t::vec_value_t;
    py::class_<internal_t, base_t>(m, name, 
        "Core constraint class for one-sided bound constraint."
        )
        .def(py::init<
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            size_t,
            value_t,
            size_t,
            value_t,
            value_t
        >(), 
            py::arg("sgn").noconvert(),
            py::arg("b").noconvert(),
            py::arg("max_iters"),
            py::arg("tol"),
            py::arg("pinball_max_iters"),
            py::arg("pinball_tol"),
            py::arg("slack")
        )
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
            const Eigen::Ref<const vec_value_t>&,
            const Eigen::Ref<const vec_value_t>&,
            size_t,
            value_t,
            value_t,
            value_t
        >(), 
            py::arg("sgn").noconvert(),
            py::arg("b").noconvert(),
            py::arg("max_iters"),
            py::arg("tol_abs"),
            py::arg("tol_rel"),
            py::arg("rho")
        )
        ;
}

void register_constraint(py::module_& m)
{
    py::bind_vector<std::vector<ad::constraint::ConstraintBase<double>*>>(m, "VectorConstraintBase64");
    py::bind_vector<std::vector<ad::constraint::ConstraintBase<float>*>>(m, "VectorConstraintBase32");

    constraint_base<double>(m, "ConstraintBase64");
    constraint_base<float>(m, "ConstraintBase32");

    constraint_box<double>(m, "ConstraintBox64");
    constraint_box<float>(m, "ConstraintBox32");

    constraint_linear<ad::matrix::MatrixConstraintBase<double>>(m, "ConstraintLinear64");
    constraint_linear<ad::matrix::MatrixConstraintBase<float>>(m, "ConstraintLinear32");

    constraint_one_sided<double>(m, "ConstraintOneSided64");
    constraint_one_sided<float>(m, "ConstraintOneSided32");
    constraint_one_sided_admm<double>(m, "ConstraintOneSidedADMM64");
    constraint_one_sided_admm<float>(m, "ConstraintOneSidedADMM32");
}