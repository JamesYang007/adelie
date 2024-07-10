#pragma once
#include <adelie_core/configs.hpp>
#include <adelie_core/constraint/constraint_base.hpp>
#include <adelie_core/optimization/hinge_full.hpp>
#include <adelie_core/optimization/hinge_low_rank.hpp>
#include <adelie_core/optimization/nnls.hpp>

namespace adelie_core {
namespace constraint { 

template <class ValueType>
class ConstraintLinearBase: public ConstraintBase<ValueType>
{
public:
    using base_t = ConstraintBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_uint64_t;
    using typename base_t::colmat_value_t;
    using rowmat_value_t = util::rowmat_type<value_t>;
    using map_crowmat_value_t = Eigen::Map<const rowmat_value_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;

protected:
    const map_crowmat_value_t _A;
    const map_cvec_value_t _l;
    const map_cvec_value_t _u;

public:
    explicit ConstraintLinearBase(
        const Eigen::Ref<const rowmat_value_t> A,
        const Eigen::Ref<const vec_value_t> l,
        const Eigen::Ref<const vec_value_t> u
    ):
        _A(A.data(), A.rows(), A.cols()),
        _l(l.data(), l.size()),
        _u(u.data(), u.size())
    {
        const auto m = A.rows();
        const auto d = A.cols();
        if (_u.size() != m) {
            throw util::adelie_core_error("upper must be (m,) where A is (m, d).");
        }
        if (_l.size() != m) {
            throw util::adelie_core_error("lower must be (m,) where A is (m, d).");
        }
        if ((_u < 0).any()) {
            throw util::adelie_core_error("upper must be >= 0.");
        }
        if ((_l < 0).any()) { 
            // NOTE: user passes in lower == -l
            throw util::adelie_core_error("lower must be <= 0.");
        }
    }

    using base_t::project;

    void gradient(
        const Eigen::Ref<const vec_value_t>&,
        const Eigen::Ref<const vec_value_t>& mu,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        out.matrix() = mu.matrix() * _A;
    }

    int duals() override { return _A.rows(); }
    int primals() override { return _A.cols(); }
};

template <class ValueType>
class ConstraintLinearProximalNewton: public ConstraintLinearBase<ValueType>
{
public:
    using base_t = ConstraintLinearBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_uint64_t;
    using typename base_t::colmat_value_t;
    using typename base_t::rowmat_value_t;
    using typename base_t::map_crowmat_value_t;
    using typename base_t::map_cvec_value_t;
    using map_ccolmat_value_t = Eigen::Map<const colmat_value_t>;
    using base_t::_A;
    using base_t::_l;
    using base_t::_u;

private:
    const map_ccolmat_value_t _A_u;
    const map_cvec_value_t _A_d;
    const map_crowmat_value_t _A_vh;
    const map_cvec_value_t _A_vars;
    const size_t _A_rank;
    const size_t _max_iters;
    const value_t _tol;
    const size_t _nnls_batch_size;
    const size_t _nnls_max_iters;
    const value_t _nnls_tol;
    const value_t _cs_tol;
    const value_t _slack;
    const size_t _n_threads;

    static size_t init_A_rank(
        const Eigen::Ref<const vec_value_t>& A_d
    )
    {
        const value_t A_d_sum = A_d.sum();
        value_t cumsum = 0; 
        for (Eigen::Index i = 0; i < A_d.size(); ++i) {
            if (cumsum > 0.99 * A_d_sum) return i;
            cumsum += A_d[i];
        }
        return A_d.size();
    }

public:
    explicit ConstraintLinearProximalNewton(
        const Eigen::Ref<const rowmat_value_t>& A,
        const Eigen::Ref<const vec_value_t>& l,
        const Eigen::Ref<const vec_value_t>& u,
        const Eigen::Ref<const colmat_value_t>& A_u,
        const Eigen::Ref<const vec_value_t>& A_d,
        const Eigen::Ref<const rowmat_value_t>& A_vh,
        const Eigen::Ref<const vec_value_t>& A_vars,
        size_t max_iters,
        value_t tol,
        size_t nnls_batch_size,
        size_t nnls_max_iters,
        value_t nnls_tol,
        value_t cs_tol,
        value_t slack,
        size_t n_threads
    ):
        base_t(A, l, u),
        _A_u(A_u.data(), A_u.rows(), A_u.cols()),
        _A_d(A_d.data(), A_d.size()),
        _A_vh(A_vh.data(), A_vh.rows(), A_vh.cols()),
        _A_vars(A_vars.data(), A_vars.size()),
        _A_rank(init_A_rank(A_d)),
        _max_iters(max_iters),
        _tol(tol),
        _nnls_batch_size(nnls_batch_size),
        _nnls_max_iters(nnls_max_iters),
        _nnls_tol(nnls_tol),
        _cs_tol(cs_tol),
        _slack(slack),
        _n_threads(n_threads)
    {
        const auto m = A.rows();
        const auto d = A.cols();

        if (A_u.rows() != m) {
            throw util::adelie_core_error("A_u must be (m, r) where A is (m, d).");
        }
        if (A_d.size() > std::min(A_u.cols(), A_vh.rows())) {
            throw util::adelie_core_error("A_d must be (b,) where b <= min(r, s), A_u is (m, r) and A_vh is (s, d).");
        }
        if (A_vh.cols() != d) {
            throw util::adelie_core_error("A_vh must be (s, d) where A is (m, d).");
        }
        if (A_vars.size() != m) {
            throw util::adelie_core_error("A_vars must be (m,) where A is (m, d).");
        }
        if (tol < 0) {
            throw util::adelie_core_error("tol must be >= 0.");
        }
        if (nnls_tol < 0) {
            throw util::adelie_core_error("nnls_tol must be >= 0.");
        }
        if (cs_tol < 0) {
            throw util::adelie_core_error("cs_tol must be >= 0.");
        }
        if (slack <= 0 || slack >= 1) {
            throw util::adelie_core_error("slack must be in (0,1).");
        }
    }

    size_t buffer_size() override 
    {
        const auto m = _A.rows();
        const auto d = _A.cols();
        return d * (10 + 2 * d) + 2 * m + ((m < d) ? (m * m) : ((2 + d) * m));
    }

    void solve(
        Eigen::Ref<vec_value_t> x,
        Eigen::Ref<vec_value_t> mu,
        const Eigen::Ref<const vec_value_t>& quad,
        const Eigen::Ref<const vec_value_t>& linear,
        value_t l1,
        value_t l2,
        const Eigen::Ref<const colmat_value_t>& Q,
        Eigen::Ref<vec_uint64_t> buffer
    ) override
    {
        // This ensures that buffer is aligned by the same byte size.
        using index_t = std::conditional_t<
            std::is_same_v<value_t, float>,
            int32_t,
            int64_t
        >;
        using vec_index_t = util::rowvec_type<index_t>;

        const auto m = _A.rows();
        const auto d = _A.cols();

        auto buff_ptr = reinterpret_cast<value_t*>(buffer.data());
        const auto buff_begin = buff_ptr;
        Eigen::Map<vec_value_t> grad_prev(buff_ptr, d); buff_ptr += d;
        Eigen::Map<vec_value_t> grad(buff_ptr, d); buff_ptr += d;
        Eigen::Map<vec_value_t> ATmu_prev(buff_ptr, d); buff_ptr += d;
        Eigen::Map<vec_value_t> ATmu(buff_ptr, d); buff_ptr += d;
        Eigen::Map<vec_value_t> hinge_grad(buff_ptr, m); buff_ptr += m;
        const auto m_large = (m < d) ? 0 : m;
        Eigen::Map<vec_index_t> active_set(reinterpret_cast<index_t*>(buff_ptr), m_large); buff_ptr += m_large;
        Eigen::Map<vec_value_t> active_vars(buff_ptr, m_large); buff_ptr += m_large;
        Eigen::Map<rowmat_value_t> active_AQ(buff_ptr, m_large, d); buff_ptr += m_large * d;
        const auto m_small = (m < d) ? m : 0;
        Eigen::Map<colmat_value_t> hess_small(buff_ptr, m_small, m_small); buff_ptr += m_small * m_small;
        const auto n_read = std::distance(buff_begin, buff_ptr);
        Eigen::Map<vec_uint64_t> next_buff(buffer.data() + n_read, buffer.size() - n_read);

        const auto compute_ATmu = [&](
            auto& out
        ) {
            if (m < d * d) {
                out.matrix() = mu.matrix() * _A;
            } else {
                // TODO: would be nice to keep track of mu active set to reduce loop size
                out.setZero();
                for (Eigen::Index i = 0; i < mu.size(); ++i) {
                    if (mu[i] == 0) continue;
                    out += mu[i] * _A.row(i).array();
                }
            } 
        };

        // must be initialized prior to calling solver
        compute_ATmu(ATmu);

        const auto compute_mu_resid = [&](
            const auto&,
            const auto& linear,
            const auto& Q,
            auto& mu_resid
        ) {
            mu_resid.matrix() = linear.matrix() - ATmu.matrix() * Q;
        };
        const auto compute_min_mu_resid = [&](
            auto& mu,
            const auto& Qv,
            bool is_prev_valid_old,
            auto cs_tol
        ) {
            auto Qmu_resid = grad.matrix();
            Qmu_resid = Qv - ATmu;
            const value_t loss = 0.5 * Qmu_resid.squaredNorm();
            const Eigen::Map<const colmat_value_t> AT(
                _A.data(), _A.cols(), _A.rows()
            );
            optimization::StateNNLS<colmat_value_t> state_nnls(
                AT, _A_vars, _nnls_max_iters, _nnls_tol,
                mu, Qmu_resid, loss
            );
            const auto is_u_zero = (_u <= 0).template cast<value_t>();
            const auto is_l_zero = (_l <= 0).template cast<value_t>();
            const auto lower_constraint = (
                (-cs_tol) * (1 - is_l_zero) / (_l + is_l_zero) +
                (-Configs::max_solver_value) * is_l_zero
            );
            const auto upper_constraint = (
                cs_tol * (1 - is_u_zero) / (_u + is_u_zero) +
                Configs::max_solver_value * is_u_zero
            );
            state_nnls.solve(
                [&]() { return state_nnls.loss <= 0.5 * l1 * l1; },
                lower_constraint,
                upper_constraint
            );
            if (!is_prev_valid_old) {
                ATmu = Qv - Qmu_resid.array();
            }
            return 2 * state_nnls.loss;
        };
        const auto compute_backtrack_a = [&](
            const auto&,
            const auto& 
        ) {
            const auto ATdmu = ATmu - ATmu_prev;
            return ATdmu.square().sum();
        };
        const auto compute_backtrack_b = [&](
            const auto&,
            const auto&,
            const auto&,
            const auto& mu_resid
        ) {
            const auto ATdmu = ATmu - ATmu_prev;
            return (mu_resid.matrix() * Q.transpose()).dot(ATdmu.matrix());
        };
        const auto compute_gradient = [&](
            const auto& x,
            const auto& Q
        ) {
            grad.matrix() = x.matrix() * Q.transpose();
        };
        const auto compute_hard_optimality = [&](
            const auto& 
        ) {
            // This is an optional check for optimization purposes.
            // The cost of computing this check may be too much if m >> d
            // so we omit this check.
            return false;
        };
        const auto compute_convergence_measure = [&](
            const auto&,
            const auto&,
            bool is_in_ellipse
        ) {
            const auto ATdmu = (ATmu - ATmu_prev);
            return is_in_ellipse ? (
                std::abs((ATdmu * grad_prev).mean())
            ) : (
                std::abs((ATdmu * (grad_prev-grad)).mean())
            );
        };
        const auto compute_proximal_newton_step = [&](
            const auto& hess,
            auto& mu
        ) {
            if (m < d) {
                hess_small = _A * hess * _A.transpose();
                hinge_grad = grad.matrix() * _A.transpose();
                optimization::StateHingeFull<colmat_value_t> state_hinge(
                    hess_small, _l, _u, _nnls_max_iters, _nnls_tol,
                    mu, hinge_grad
                );
                state_hinge.solve();
            } else {
                optimization::StateHingeLowRank<value_t, index_t> state_hinge(
                    hess, _A, _l, _u, _nnls_batch_size, _nnls_max_iters, _nnls_tol, _n_threads,
                    mu, grad, active_set, active_vars, active_AQ, hinge_grad
                );
                //try {
                state_hinge.solve();
                //} catch (...) {
                //    PRINT(this);
                //    PRINT(hess);
                //    PRINT(state_hinge.iters);
                //    PRINT(mu);
                //    PRINT(hinge_grad);
                //    throw;
                //}
            }
            compute_ATmu(ATmu);
        };
        const auto save_additional_prev = [&](bool is_in_ellipse) {
            ATmu_prev = ATmu;
            if (is_in_ellipse) grad_prev.setZero();
            else grad_prev = grad;
        };
        base_t::_solve_proximal_newton(
            x, mu, quad, linear, l1, l2, Q, _max_iters, _tol, _cs_tol, _slack, next_buff,
            compute_mu_resid,
            compute_min_mu_resid,
            compute_backtrack_a,
            compute_backtrack_b,
            compute_gradient,
            compute_hard_optimality,
            compute_convergence_measure,
            compute_proximal_newton_step,
            save_additional_prev
        );
    }
};

} // namespace constraint
} // namespace adelie_core