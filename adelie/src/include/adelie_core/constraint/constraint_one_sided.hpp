#pragma once
#include <adelie_core/configs.hpp>
#include <adelie_core/constraint/constraint_base.hpp>
#include <adelie_core/optimization/nnqp_full.hpp>

namespace adelie_core {
namespace constraint {

template <class ValueType, class IndexType=Eigen::Index>
class ConstraintOneSidedBase: public ConstraintBase<ValueType, IndexType>
{
public:
    using base_t = ConstraintBase<ValueType, IndexType>;
    using typename base_t::value_t;
    using typename base_t::vec_index_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_uint64_t;
    using typename base_t::colmat_value_t;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;

protected:
    const map_cvec_value_t _sgn;
    const map_cvec_value_t _b;

public:
    explicit ConstraintOneSidedBase(
        const Eigen::Ref<const vec_value_t>& sgn,
        const Eigen::Ref<const vec_value_t>& b
    ):
        _sgn(sgn.data(), sgn.size()),
        _b(b.data(), b.size())
    {
        if ((_sgn.abs() != 1).any()) {
            throw util::adelie_core_error("sgn must be a vector of +/-1.");
        }
        if ((_b < 0).any()) {
            throw util::adelie_core_error("b must be >= 0.");
        }
        if (_sgn.size() != _b.size()) {
            throw util::adelie_core_error("sgn be (d,) where b is (d,).");
        }
    }

    ADELIE_CORE_STRONG_INLINE
    void solve_1d(
        Eigen::Ref<vec_value_t> x,
        Eigen::Ref<vec_value_t> mu,
        const Eigen::Ref<const vec_value_t>& quad,
        const Eigen::Ref<const vec_value_t>& linear,
        value_t l1,
        value_t l2,
        const Eigen::Ref<const colmat_value_t>& Q
    ) const
    {
        const auto A = _sgn[0] * Q(0,0);
        const auto b = _b[0];
        const auto q = quad[0];
        const auto v = linear[0];

        // check if x == 0 is optimal
        auto mu0 = (b > 0) ? 0 : std::max<value_t>(A * v, 0.0);
        const auto is_zero_opt = std::abs(v - A * mu0) <= l1;

        // if optimal, take previous solution, else compute general solution
        const auto x0 = is_zero_opt ? 0 : (
            A * std::min(
                A * std::copysign(std::abs(v) - l1, v) / (q + l2), 
                b
            )
        );
        mu0 = is_zero_opt ? mu0 : (
            (A * x0 < b) ? 0 : (A * (v - ((q + l2) * x0 + std::copysign(l1, x0))))
        );

        // store output
        x[0] = x0;
        mu[0] = mu0;
    }

    void project(
        Eigen::Ref<vec_value_t> x
    ) override
    {
        x = _sgn * (_sgn * x).min(_b);
    }
    int duals() override { return _b.size(); }
    int primals() override { return _b.size(); }
};

template <class ValueType, class IndexType=Eigen::Index>
class ConstraintOneSidedProximalNewton: public ConstraintOneSidedBase<ValueType, IndexType>
{
public:
    using base_t = ConstraintOneSidedBase<ValueType, IndexType>;
    using typename base_t::value_t;
    using typename base_t::vec_index_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_uint64_t;
    using typename base_t::colmat_value_t;
    using typename base_t::map_cvec_value_t;
    using base_t::_sgn;
    using base_t::_b;

private:
    const size_t _max_iters;
    const value_t _tol;
    const size_t _nnls_max_iters;
    const value_t _nnls_tol;
    const value_t _cs_tol;
    const value_t _slack;

    vec_value_t _mu;

public:
    explicit ConstraintOneSidedProximalNewton(
        const Eigen::Ref<const vec_value_t>& sgn,
        const Eigen::Ref<const vec_value_t>& b,
        size_t max_iters,
        value_t tol,
        size_t nnls_max_iters,
        value_t nnls_tol,
        value_t cs_tol,
        value_t slack
    ):
        base_t(sgn, b),
        _max_iters(max_iters),
        _tol(tol),
        _nnls_max_iters(nnls_max_iters),
        _nnls_tol(nnls_tol),
        _cs_tol(cs_tol),
        _slack(slack),
        _mu(vec_value_t::Zero(sgn.size()))
    {
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
        const auto d = _b.size();
        return (d <= 1) ? 0 : (d * (9 + 2 * d));
    }

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
        const auto m = _b.size();
        const auto d = m;

        if (d == 1) {
            base_t::solve_1d(x, _mu, quad, linear, l1, l2, Q);
            return;
        }

        auto buff_ptr = reinterpret_cast<value_t*>(buffer.data());
        const auto buff_begin = buff_ptr;
        Eigen::Map<vec_value_t> grad_prev(buff_ptr, m); buff_ptr += m;
        Eigen::Map<vec_value_t> grad(buff_ptr, m); buff_ptr += m;
        Eigen::Map<vec_value_t> mu_prev(buff_ptr, m); buff_ptr += m;
        const auto n_read = std::distance(buff_begin, buff_ptr);
        Eigen::Map<vec_uint64_t> next_buff(buffer.data() + n_read, buffer.size() - n_read);

        const auto compute_mu_resid = [&](
            auto& mu_resid
        ) {
            mu_resid.matrix() = linear.matrix() - (_sgn * _mu).matrix() * Q;
        };
        const auto compute_min_mu_resid = [&](
            const auto& Qv,
            bool is_prev_valid_old
        ) {
            auto& mu_curr = grad;
            if (is_prev_valid_old) {
                mu_curr = _mu;
            }
            const auto is_b_zero = (_b <= 0).template cast<value_t>();
            _mu = (_sgn * Qv).max(0).min(
                _cs_tol * (1 - is_b_zero) / (_b + is_b_zero) +
                Configs::max_solver_value * is_b_zero
            );
            const auto mu_resid_norm_sq = (Qv - _sgn * _mu).square().sum();
            if (is_prev_valid_old && mu_resid_norm_sq > l1 * l1) {
                _mu = mu_curr;
            } 
            return mu_resid_norm_sq;
        };
        const auto compute_backtrack_a = [&]() {
            return (_mu - mu_prev).square().sum();
        };
        const auto compute_backtrack_b = [&](
            const auto& Qv,
            const auto& 
        ) {
            return ((_sgn * Qv - _mu) * (_mu - mu_prev)).sum();
        };
        const auto compute_backtrack = [&](auto step_size) {
            _mu = mu_prev + step_size * (_mu - mu_prev);
        };
        const auto compute_gradient = [&]() {
            grad.matrix() = (x.matrix() * Q.transpose()).cwiseProduct(_sgn.matrix()) - _b.matrix();
        };
        const auto compute_hard_optimality = [&]() {
            return (grad <= 0).all() && (_mu * grad == 0).all();
        };
        const auto compute_convergence_measure = [&](
            bool is_in_ellipse
        ) {
            return is_in_ellipse ? (
                std::abs(((_mu-mu_prev) * (grad_prev+_b)).mean())
            ) : (
                std::abs(((_mu-mu_prev) * (grad_prev-grad)).mean())
            );
        };
        const auto compute_proximal_newton_step = [&](
            const auto& hess
        ) {
            // reparametrize
            grad *= _sgn;
            _mu *= _sgn;

            // solve NNQP for new mu
            optimization::StateNNQPFull<colmat_value_t, true> state_nnqp(
                _sgn, hess, _nnls_max_iters, _nnls_tol, _mu, grad
            );
            state_nnqp.solve();

            // reparametrize
            _mu *= _sgn;
        };
        const auto save_additional_prev = [&](bool is_in_ellipse) {
            mu_prev = _mu;
            if (is_in_ellipse) grad_prev = -_b;
            else grad_prev = grad;
        };
        base_t::_solve_proximal_newton(
            x, quad, linear, l1, l2, Q, _max_iters, _tol, _slack, next_buff,
            compute_mu_resid,
            compute_min_mu_resid,
            compute_backtrack_a,
            compute_backtrack_b,
            compute_backtrack,
            compute_gradient,
            compute_hard_optimality,
            compute_convergence_measure,
            compute_proximal_newton_step,
            save_additional_prev
        );
    }

    void gradient(
        const Eigen::Ref<const vec_value_t>&,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        out = _sgn * _mu;
    }

    void clear() override 
    {
        _mu.setZero();
    }

    void dual(
        Eigen::Ref<vec_index_t> indices,
        Eigen::Ref<vec_value_t> values
    ) override
    {
        size_t nnz = 0;
        for (Eigen::Index i = 0; i < _mu.size(); ++i) {
            const auto mi = _mu[i];
            if (mi == 0) continue;
            indices[nnz] = i;
            values[nnz] = mi;
            ++nnz;
        }
    }

    int duals_nnz() override 
    {
        return (_mu != 0).count();
    }
};

template <class ValueType>
class ConstraintOneSidedADMM: public ConstraintOneSidedBase<ValueType>
{
public:
    using base_t = ConstraintOneSidedBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_index_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_uint64_t;
    using typename base_t::colmat_value_t;
    using typename base_t::map_cvec_value_t;
    using base_t::_sgn;
    using base_t::_b;

private:
    const size_t _max_iters;
    const value_t _tol_abs;
    const value_t _tol_rel;
    const value_t _rho;

    vec_value_t _mu;

public:
    explicit ConstraintOneSidedADMM(
        const Eigen::Ref<const vec_value_t>& sgn,
        const Eigen::Ref<const vec_value_t>& b,
        size_t max_iters,
        value_t tol_abs,
        value_t tol_rel,
        value_t rho
    ):
        base_t(sgn, b),
        _max_iters(max_iters),
        _tol_abs(tol_abs),
        _tol_rel(tol_rel),
        _rho(rho),
        _mu(vec_value_t::Zero(sgn.size()))
    {
        if (tol_abs < 0) {
            throw util::adelie_core_error("tol_abs must be >= 0.");
        }
        if (tol_rel < 0) {
            throw util::adelie_core_error("tol_rel must be >= 0.");
        }
        if (rho <= 0) {
            throw util::adelie_core_error("rho must be > 0.");
        }
    }

    size_t buffer_size() override 
    {
        const auto d = _b.size();
        return (d <= 1) ? 0 : (7 * d);
    }

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
        const auto m = _b.size();
        const auto d = m;

        if (d == 1) {
            base_t::solve_1d(x, _mu, quad, linear, l1, l2, Q);
            return;
        } 

        const auto& v = linear;

        size_t iters = 0;

        auto buff_ptr = reinterpret_cast<value_t*>(buffer.data());
        Eigen::Map<vec_value_t> x_buffer1(buff_ptr, d); buff_ptr += d;
        Eigen::Map<vec_value_t> x_buffer2(buff_ptr, d); buff_ptr += d;
        Eigen::Map<vec_value_t> z(buff_ptr, m); buff_ptr += m;
        Eigen::Map<vec_value_t> z_prev(buff_ptr, m); buff_ptr += m;
        Eigen::Map<vec_value_t> u(buff_ptr, m); buff_ptr += m;
        Eigen::Map<vec_value_t> linear_shifted(buff_ptr, d); buff_ptr += d;
        Eigen::Map<vec_value_t> s(buff_ptr, m); buff_ptr += m;

        // TODO: these warm-starts may not be optimal
        z = ((x.matrix() * Q.transpose()).array() * _sgn).min(_b);
        u.setZero();

        const auto compute_primal_x = [&]() {
            constexpr size_t _newton_max_iters = 100000;
            constexpr value_t _newton_tol = 1e-12;
            size_t x_iters;
            bcd::unconstrained::newton_abs_solver(
                quad, linear_shifted, l1, l2+_rho, _newton_tol, _newton_max_iters, 
                x, x_iters, x_buffer1, x_buffer2
            );
        };

        const auto compute_primal_z = [&]() {
            z = linear_shifted.min(_b);
        };

        const auto compute_dual = [&]() {
            const auto x_norm = x.matrix().norm();
            if (x_norm <= 0) {
                _mu = (
                    ((v.matrix() * Q.transpose()).array() * _sgn).max(0) 
                    * (_b <= 0).template cast<value_t>()
                );
                return;
            }
            _mu = (((v - (quad + l2 + l1 / x_norm) * x).matrix() * Q.transpose()).array() * _sgn).max(0);
        };

        while (iters < _max_iters) {
            ++iters;
            linear_shifted.matrix() = v.matrix() + _rho * (_sgn * (z - u)).matrix() * Q;
            compute_primal_x();
            linear_shifted = (x.matrix() * Q.transpose()).array() * _sgn + u;
            z_prev = z;
            compute_primal_z();
            auto& r = linear_shifted;
            r = linear_shifted - z - u;
            u += r;
            s.matrix() = -_rho * (_sgn * (z-z_prev)).matrix() * Q;
            const value_t eps_pri = (
                std::sqrt(m) * _tol_abs
                + _tol_rel * std::max(x.matrix().norm(), z.matrix().norm())
            );
            const value_t eps_dual = (
                std::sqrt(d) * _tol_abs 
                + _tol_rel * _rho * u.matrix().norm()
            );
            if (
                (r.matrix().norm() <= eps_pri) &&
                (s.matrix().norm() <= eps_dual)
            ) {
                compute_dual();
                return;
            }
        }

        throw util::adelie_core_solver_error("ConstraintOneSidedADMM: max iterations reached!");
    }

    void gradient(
        const Eigen::Ref<const vec_value_t>&,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        out = _sgn * _mu;
    }

    void clear() override 
    {
        _mu.setZero();
    }

    void dual(
        Eigen::Ref<vec_index_t> indices,
        Eigen::Ref<vec_value_t> values
    ) override
    {
        size_t nnz = 0;
        for (Eigen::Index i = 0; i < _mu.size(); ++i) {
            const auto mi = _mu[i];
            if (mi == 0) continue;
            indices[nnz] = i;
            values[nnz] = mi;
            ++nnz;
        }
    }

    int duals_nnz() override 
    {
        return (_mu != 0).count();
    }
};

} // namespace constraint
} // namespace adelie_core