#pragma once
#include <unordered_set>
#include <vector>
#include <adelie_core/configs.hpp>
#include <adelie_core/constraint/constraint_base.hpp>
#include <adelie_core/optimization/hinge_full.hpp>
#include <adelie_core/optimization/hinge_low_rank.hpp>
#include <adelie_core/optimization/nnls.hpp>

namespace adelie_core {
namespace constraint { 

template <class ValueType, class IndexType=Eigen::Index>
class ConstraintLinearBase: public ConstraintBase<ValueType, IndexType>
{
public:
    using base_t = ConstraintBase<ValueType, IndexType>;
    using typename base_t::index_t;
    using typename base_t::value_t;
    using typename base_t::vec_index_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_uint64_t;
    using typename base_t::colmat_value_t;
    using rowmat_value_t = util::rowmat_type<value_t>;
    using map_crowmat_value_t = Eigen::Map<const rowmat_value_t>;
    using map_cvec_value_t = Eigen::Map<const vec_value_t>;

protected:
    using base_t::check_solve;

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

    int duals() override { return _A.rows(); }
    int primals() override { return _A.cols(); }
};

template <class ValueType, class IndexType=Eigen::Index>
class ConstraintLinearProximalNewton: public ConstraintLinearBase<ValueType, IndexType>
{
public:
    using base_t = ConstraintLinearBase<ValueType, IndexType>;
    using typename base_t::index_t;
    using typename base_t::value_t;
    using typename base_t::vec_index_t;
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

    std::unordered_set<index_t> _mu_active_set;
    std::unordered_set<index_t> _mu_active_set_prev;
    std::vector<index_t> _mu_active;
    std::vector<index_t> _mu_active_prev;
    std::vector<value_t> _mu_value;
    std::vector<value_t> _mu_value_prev;
    vec_value_t _ATmu;

    ADELIE_CORE_STRONG_INLINE
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

    ADELIE_CORE_STRONG_INLINE
    void compute_ATmu(
        Eigen::Ref<vec_value_t> out
    ) 
    {
        out.setZero();
        for (size_t i = 0; i < _mu_active.size(); ++i) {
            out += _mu_value[i] * _A.row(_mu_active[i]).array();
        }
    };

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
        _n_threads(n_threads),
        _ATmu(vec_value_t::Zero(A.cols()))
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
        return d * (10 + 2 * d) + 2 * m + ((m < d) ? (m * m) : ((1 + d) * m));
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
        const auto m = _A.rows();
        const auto d = _A.cols();

        base_t::check_solve(x.size(), quad.cols(), linear.size(), m, d);

        // check if x = 0, mu = 0 is optimal
        if (linear.matrix().norm() <= l1) {
            x.setZero();
            _mu_active_set.clear();
            _mu_active.clear();
            _mu_value.clear();
            _ATmu.setZero();
            return;
        }

        auto buff_ptr = reinterpret_cast<value_t*>(buffer.data());
        const auto buff_begin = buff_ptr;
        Eigen::Map<vec_value_t> grad_prev(buff_ptr, d); buff_ptr += d;
        Eigen::Map<vec_value_t> grad(buff_ptr, d); buff_ptr += d;
        Eigen::Map<vec_value_t> ATmu_prev(buff_ptr, d); buff_ptr += d;
        Eigen::Map<vec_value_t> ATmu(buff_ptr, d); buff_ptr += d;
        Eigen::Map<vec_value_t> mu(buff_ptr, m); buff_ptr += m;
        Eigen::Map<vec_value_t> hinge_grad(buff_ptr, m); buff_ptr += m;
        const auto m_large = (m < d) ? 0 : m;
        Eigen::Map<vec_value_t> active_vars(buff_ptr, m_large); buff_ptr += m_large;
        Eigen::Map<rowmat_value_t> active_AQ(buff_ptr, m_large, d); buff_ptr += m_large * d;
        const auto m_small = (m < d) ? m : 0;
        Eigen::Map<colmat_value_t> hess_small(buff_ptr, m_small, m_small); buff_ptr += m_small * m_small;
        const auto n_read = std::distance(buff_begin, buff_ptr);
        Eigen::Map<vec_uint64_t> next_buff(buffer.data() + n_read, buffer.size() - n_read);

        const auto mu_prune = [&]() {
            size_t n_active = 0;
            for (size_t i = 0; i < _mu_active.size(); ++i) {
                const auto idx = _mu_active[i];
                const auto mi = _mu_value[i];
                if (mi == 0) {
                    _mu_active_set.erase(idx);
                    continue;
                }
                _mu_active[n_active] = idx;
                _mu_value[n_active] = mi;
                ++n_active;
            }
            _mu_active.erase(
                std::next(_mu_active.begin(), n_active),
                _mu_active.end()
            );
            _mu_value.erase(
                std::next(_mu_value.begin(), n_active),
                _mu_value.end()
            );
        };
        const auto mu_to_dense = [&]() {
            mu.setZero();
            for (size_t i = 0; i < _mu_active.size(); ++i) {
                mu[_mu_active[i]] = _mu_value[i]; 
            }
        };
        const auto mu_to_sparse = [&]() {
            for (size_t i = 0; i < _mu_active.size(); ++i) {
                _mu_value[i] = mu[_mu_active[i]];
            }
            for (Eigen::Index i = 0; i < mu.size(); ++i) {
                const auto mi = mu[i];
                if (mi == 0 || _mu_active_set.find(i) != _mu_active_set.end()) continue;
                _mu_active_set.insert(i);
                _mu_active.push_back(i);
                _mu_value.push_back(mi);
            }
            mu_prune();
        };

        const auto compute_mu_resid = [&](
            auto& mu_resid
        ) {
            mu_resid.matrix() = linear.matrix() - _ATmu.matrix() * Q;
        };
        const auto compute_min_mu_resid = [&](
            const auto& Qv,
            bool is_prev_valid_old,
            bool is_init
        ) {
            // check if current mu_resid norm is small enough
            if ((Qv - _ATmu).square().sum() <= l1 * l1) return value_t(0);

            // check SVD-based warm-start
            const auto u0 = _A_u.leftCols(_A_rank);
            const auto d0 = _A_d.head(_A_rank);
            const auto vh0 = _A_vh.topRows(_A_rank);
            auto DinvVTQv = grad.head(_A_rank).matrix();
            DinvVTQv = (Qv.matrix() * vh0.transpose()).cwiseQuotient(d0.matrix()); 
            auto mu_m = mu.matrix();
            if (u0.rows() >= u0.cols()) {
                matrix::dgemv(
                    u0.transpose(),
                    DinvVTQv,
                    _n_threads,
                    grad /* unused dummy input */,
                    mu_m
                );
            } else {
                mu_m = DinvVTQv * u0.transpose();
            }
            ATmu.matrix() = DinvVTQv.cwiseProduct(d0.matrix()) * vh0;

            // refine check with NNLS
            auto& Qmu_resid = grad;
            Qmu_resid = Qv - ATmu;
            const value_t loss = 0.5 * Qmu_resid.square().sum();
            const Eigen::Map<const colmat_value_t> AT(
                _A.data(), _A.cols(), _A.rows()
            );
            optimization::StateNNLS<colmat_value_t> state_nnls(
                AT, _A_vars, _nnls_max_iters, _nnls_tol,
                mu, Qmu_resid, loss
            );
            const auto lower_constraint = vec_value_t::NullaryExpr(_l.size(), [&](auto i) {
                const auto li = _l[i];
                return (li <= 0) ? (-Configs::max_solver_value) : (-_cs_tol / li);
            });
            const auto upper_constraint = vec_value_t::NullaryExpr(_u.size(), [&](auto i) {
                const auto ui = _u[i];
                return (ui <= 0) ? Configs::max_solver_value : (_cs_tol / ui);
            });
            state_nnls.solve(
                [&]() { return (state_nnls.iters > 1) && (state_nnls.loss <= 0.5 * l1 * l1); },
                lower_constraint,
                upper_constraint
            );

            const auto mu_resid_norm_sq = 2 * state_nnls.loss;

            // Extra invariance required under these conditions
            if ((!is_init && !is_prev_valid_old) || (mu_resid_norm_sq <= l1 * l1)) {
                // The relaxed constraint version above was just to check if a zero solution is 
                // a good approximation to the true solution.
                // For sparsity, we omit values that are hitting the boundaries (these are the ones that are "relaxed").
                mu = (
                    mu.min(0) * (mu.min(0) * _l > -_cs_tol).template cast<value_t>()
                    + mu.max(0) * (mu.max(0) * _u < _cs_tol).template cast<value_t>()
                );
                mu_to_sparse();
                compute_ATmu(_ATmu);
            } 

            return mu_resid_norm_sq;
        };
        const auto compute_backtrack_a = [&]() {
            const auto ATdmu = _ATmu - ATmu_prev;
            return ATdmu.square().sum();
        };
        const auto compute_backtrack_b = [&](
            const auto&,
            const auto& mu_resid
        ) {
            const auto ATdmu = _ATmu - ATmu_prev;
            return (mu_resid.matrix() * Q.transpose()).dot(ATdmu.matrix());
        };
        const auto compute_backtrack = [&](auto step_size) {
            for (size_t i = 0; i < _mu_active_prev.size(); ++i) {
                mu[_mu_active_prev[i]] = (1-step_size) * _mu_value_prev[i];
            }
            for (size_t i = 0; i < _mu_active.size(); ++i) {
                const auto idx = _mu_active[i];
                auto& mi = _mu_value[i];
                mi = step_size * mi + (
                    (_mu_active_set_prev.find(idx) != _mu_active_set_prev.end()) ?
                    mu[idx] : 0
                );
            }
            for (size_t i = 0; i < _mu_active_prev.size(); ++i) {
                const auto idx = _mu_active_prev[i];
                if (_mu_active_set.find(idx) != _mu_active_set.end()) continue;
                _mu_active_set.insert(idx);
                _mu_active.push_back(idx);
                _mu_value.push_back(mu[idx]);
            }
            compute_ATmu(_ATmu);
        };
        const auto compute_gradient = [&]() {
            grad.matrix() = x.matrix() * Q.transpose();
        };
        const auto compute_hard_optimality = [&]() {
            // This is an optional check for optimization purposes.
            // The cost of computing this check may be too much if m >> d
            // so we omit this check.
            return false;
        };
        const auto compute_convergence_measure = [&](
            bool is_in_ellipse
        ) {
            const auto ATdmu = (_ATmu - ATmu_prev);
            return is_in_ellipse ? (
                std::abs((ATdmu * grad_prev).mean())
            ) : (
                std::abs((ATdmu * (grad_prev-grad)).mean())
            );
        };
        const auto compute_proximal_newton_step = [&](
            const auto& hess
        ) {
            if (m < d) {
                mu_to_dense();
                hess_small = _A * hess * _A.transpose();
                hinge_grad = grad.matrix() * _A.transpose();
                optimization::StateHingeFull<colmat_value_t> state_hinge(
                    hess_small, _l, _u, _nnls_max_iters, _nnls_tol,
                    mu, hinge_grad
                );
                state_hinge.solve();
                mu_to_sparse();
            } else {
                const auto active_invariance = [&](auto ii) {
                    const auto i = _mu_active[ii];
                    const auto Ai = _A.row(i);
                    active_AQ.row(ii) = Ai * hess;
                    active_vars[ii] = std::max<value_t>(
                        active_AQ.row(ii).dot(Ai),
                        1e-14
                    );
                };
                const size_t active_size = _mu_active.size();
                const size_t n_bytes = sizeof(value_t) * d * (d + 1) * active_size;
                if (_n_threads <= 1 || n_bytes <= Configs::min_bytes) {
                    for (Eigen::Index ii = 0; ii < active_size; ++ii) active_invariance(ii);
                } else {
                    #pragma omp parallel for schedule(static) num_threads(_n_threads)
                    for (Eigen::Index ii = 0; ii < active_size; ++ii) active_invariance(ii);
                }
                optimization::StateHingeLowRank<value_t, index_t> state_hinge(
                    hess, _A, _l, _u, _nnls_batch_size, _nnls_max_iters, _nnls_tol, _n_threads,
                    _mu_active, _mu_value, active_vars, active_AQ, grad, hinge_grad
                );
                state_hinge.solve();
                _mu_active_set.insert(
                    _mu_active.data() + active_size,
                    _mu_active.data() + _mu_active.size()
                );
                mu_prune();
            }
            compute_ATmu(_ATmu);
        };
        const auto save_additional_prev = [&](bool is_in_ellipse) {
            _mu_active_set_prev = _mu_active_set;
            _mu_active_prev = _mu_active;
            _mu_value_prev = _mu_value;
            ATmu_prev = _ATmu;
            if (is_in_ellipse) grad_prev.setZero();
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
        out = _ATmu;
    }

    void gradient(
        const Eigen::Ref<const vec_value_t>&,
        const Eigen::Ref<const vec_value_t>& mu,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        out.matrix() = mu.matrix() * _A;
    }

    void clear() override 
    {
        _mu_active_set.clear();
        _mu_active.clear();
        _mu_value.clear();
        _ATmu.setZero();
    }

    void dual(
        Eigen::Ref<vec_index_t> indices,
        Eigen::Ref<vec_value_t> values
    ) override
    {
        const size_t nnz = _mu_active.size();
        indices.head(nnz) = Eigen::Map<vec_index_t>(
            _mu_active.data(),
            nnz
        );
        values.head(nnz) = Eigen::Map<vec_value_t>(
            _mu_value.data(),
            nnz
        );
    }

    int duals_nnz() override 
    {
        return _mu_active.size();
    }
};

} // namespace constraint
} // namespace adelie_core