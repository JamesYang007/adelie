#pragma once
#include <adelie_core/constraint/constraint_base.hpp>
#include <adelie_core/optimization/hinge_low_rank.hpp>

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
        if (_A.rows() != _u.size()) {
            throw util::adelie_core_error("A must have same number of rows as the length of lower and upper.");
        }
        if (_u.size() != _l.size()) {
            throw util::adelie_core_error("upper and lower must have the same length.");
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
    using base_t::_A;
    using base_t::_l;
    using base_t::_u;

private:
    const size_t _max_iters;
    const value_t _tol;
    const size_t _nnls_max_iters;
    const value_t _nnls_tol;
    const value_t _cs_tol;
    const value_t _slack;

public:
    explicit ConstraintLinearProximalNewton(
        const Eigen::Ref<const rowmat_value_t>& A,
        const Eigen::Ref<const vec_value_t>& l,
        const Eigen::Ref<const vec_value_t>& u,
        size_t max_iters,
        value_t tol,
        size_t nnls_max_iters,
        value_t nnls_tol,
        value_t cs_tol,
        value_t slack
    ):
        base_t(A, l, u),
        _max_iters(max_iters),
        _tol(tol),
        _nnls_max_iters(nnls_max_iters),
        _nnls_tol(nnls_tol),
        _cs_tol(cs_tol),
        _slack(slack)
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
        const auto m = _A.rows();
        const auto d = _A.cols();
        return d * (6 + 2 * d) + m; // TODO
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
        const auto m = _A.rows();
        const auto d = _A.cols();

        auto buff_ptr = reinterpret_cast<value_t*>(buffer.data());
        const auto buff_begin = buff_ptr;
        Eigen::Map<vec_value_t> grad_prev(buff_ptr, m); buff_ptr += m;
        Eigen::Map<vec_value_t> grad(buff_ptr, m); buff_ptr += m;
        const auto n_read = std::distance(buff_begin, buff_ptr);
        Eigen::Map<vec_uint64_t> next_buff(buffer.data() + n_read, buffer.size() - n_read);

        const auto compute_mu_resid = [&](
            const auto& mu,
            const auto& linear,
            const auto& Q,
            auto& mu_resid
        ) {
            mu_resid.matrix() = linear.matrix() - (mu.matrix() * _A) * Q;
        };
        const auto compute_hard_min_mu_resid = [&](
            auto& mu,
            const auto& Qv
        ) {
            mu = Qv;
            return 0;
        };
    }
};

} // namespace constraint
} // namespace adelie_core