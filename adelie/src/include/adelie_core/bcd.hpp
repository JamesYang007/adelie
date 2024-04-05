#pragma once
#include <Eigen/Core>
#include <adelie_core/optimization/bisect.hpp>
#include <adelie_core/optimization/newton.hpp>
#include <adelie_core/util/types.hpp>

namespace adelie_core {
namespace bcd {

// =======================================================================
// BCD unconstrained
// =======================================================================

/*
 * ISTA solver.
 */
template <class LType, class VType, class ValueType, class XType>
inline
void ista_solver(
    const LType& L,
    const VType& v,
    ValueType l1,
    ValueType l2,
    ValueType tol,
    size_t max_iters,
    XType& x_sol,
    size_t& iters
)
{
    using value_t = ValueType;
    const auto lip = L.maxCoeff();
    const auto nu = 1. / lip;
    const auto p = L.size();
    util::rowvec_type<value_t> x_diff(p);
    util::rowvec_type<value_t> x_old(p);
    util::rowvec_type<value_t> x(p); x.setZero();
    util::rowvec_type<value_t> v_tilde(p);

    iters = 0;
    for (; iters < max_iters; ++iters) {
        x_old.swap(x);
        v_tilde = x_old - nu * (L * x_old - v);
        const auto v_tilde_l2 = v_tilde.matrix().norm();
        if (v_tilde_l2 <= l1 * nu) {
            x.setZero();
        } else {
            x = ((lip /(lip + l2)) * (1 - (l1 * nu) / v_tilde_l2)) * v_tilde;
        }
        x_diff = x - x_old;
        if ((x_diff.abs() <= (tol * x.abs())).all()) break;
    }
    
    x_sol = x;
}
    
/*
 * FISTA solver.
 */
template <class LType, class VType, class ValueType, class XType>
inline
void fista_solver(
    const LType& L,
    const VType& v,
    ValueType l1,
    ValueType l2,
    ValueType tol,
    size_t max_iters,
    XType& x_sol,
    size_t& iters
)
{
    using value_t = ValueType;
    const auto lip = L.maxCoeff();
    const auto nu = 1. / lip;
    const auto p = L.size();
    util::rowvec_type<value_t> x_diff(p);
    util::rowvec_type<value_t> x_old(p);
    util::rowvec_type<value_t> x(p); x.setZero();
    util::rowvec_type<value_t> y(p); y = x;
    util::rowvec_type<value_t> v_tilde(p);
    value_t t = 1;

    iters = 0;
    for (; iters < max_iters; ++iters) {
        x_old.swap(x);
        v_tilde = y - nu * (L * y - v);
        const auto v_tilde_l2 = v_tilde.matrix().norm();
        if (v_tilde_l2 <= l1 * nu) {
            x.setZero();
        } else {
            x = (lip /(lip + l2)) * (1 - (l1 * nu) / v_tilde_l2) * v_tilde;
        }
        const auto t_old = t;
        t = (1 + std::sqrt(1 + 4 * t * t)) * 0.5;
        x_diff = x - x_old;
        y = x + (t_old - 1) / t * x_diff;
        
        if ((x_diff.abs() <= (tol * x.abs())).all()) break;
    }
    
    x_sol = x;
}

/*
 * FISTA with adaptive restart.
 */
template <class LType, class VType, class ValueType, class XType>
inline
void fista_adares_solver(
    const LType& L,
    const VType& v,
    ValueType l1,
    ValueType l2,
    ValueType tol,
    size_t max_iters,
    XType& x_sol,
    size_t& iters
)
{
    using value_t = ValueType;
    const auto lip = L.maxCoeff();
    const auto nu = 1. / lip;
    const auto p = L.size();
    util::rowvec_type<value_t> x_diff(p);
    util::rowvec_type<value_t> x_old(p);
    util::rowvec_type<value_t> x(p); x.setZero();
    util::rowvec_type<value_t> y(p); y = x;
    util::rowvec_type<value_t> v_tilde(p);
    util::rowvec_type<value_t> df(p);
    value_t t = 1;

    iters = 0;
    for (; iters < max_iters; ++iters) {
        x_old.swap(x);
        df = (L * y - v);
        v_tilde = y - nu * df;
        const auto v_tilde_l2 = v_tilde.matrix().norm();
        if (v_tilde_l2 <= l1 * nu) {
            x.setZero();
        } else {
            x = (lip /(lip + l2)) * (1 - (l1 * nu) / v_tilde_l2) * v_tilde;
        }
        x_diff = x - x_old;
        
        // accelerated
        if (v_tilde.matrix().dot(x_diff.matrix()) > 0) {
            y = x;
            t = 1;
        } else {
            const auto t_old = t;
            t = (1 + std::sqrt(1 + 4 * t * t)) * 0.5;
            y = x + (t_old - 1) / t * x_diff;
        }

        if ((x_diff.abs() <= (tol * x.abs())).all()) break;
    }
    
    x_sol = x;
}

/**
 * @brief Compute lower bound h_min >= 0 such that 
 * root_function(h_min) >= 0.
 * 
 * @tparam DiagType     vector type.
 * @tparam VType        vector type.
 * @tparam ValueType    float type.
 * @param vbuffer1      vector containing L + l2.
 * @param v             any vector of same length as vbuffer1.
 * @param l1            l1 regularization.
 * @return h_min 
 */
template <class DiagType, class VType, class ValueType>
inline
auto root_lower_bound(
    const DiagType& vbuffer1,
    const VType& v,
    ValueType l1
)
{
    using value_t = ValueType;
    const value_t b = l1 * vbuffer1.sum();
    const value_t a = vbuffer1.matrix().squaredNorm();
    const value_t v_l1 = v.matrix().template lpNorm<1>();
    const value_t c = l1 * l1 * vbuffer1.size() - v_l1 * v_l1;
    const value_t discr = b*b - a*c;
    value_t h_min = (discr > -1e-12) ? 
        (-b + std::sqrt(std::max<value_t>(discr, 0.0))) / a : 0.0;
    
    // Otherwise, if h <= 0, we know at least 0 is a reasonable solution.
    // The only case h <= 0 is when 0 is already close to the solution.
    h_min = std::max<value_t>(h_min, 0.0);
    return h_min;
}

/**
 * @brief 
 * Compute upper bound h_max >= 0 such that root_function(h_max) <= 0.
 * NOTE: if zero_tol > 0,
 * the result may NOT be a true upper bound in the sense that group_elnet_objective(result) <= 0.
 * 
 * @tparam DiagType     vector type.
 * @tparam VType        vector type.
 * @tparam ValueType    float type.
 * @param vbuffer1      vector containing L + l2.
 * @param v             any vector of same length as vbuffer1.
 * @param zero_tol      if a float is <= zero_tol, it is considered to be 0.
 * @return (h_max, vbuffer1_min_nnz)
 *  h_max: the upper bound
 *  vbuffer1_min_nnz:   smallest value in vbuffer1 among non-zero values based on zero_tol.
 */
template <class DiagType, class VType, class ValueType>
inline
auto root_upper_bound(
    const DiagType& vbuffer1,
    const VType& v,
    ValueType zero_tol=1e-10
)
{
    using value_t = ValueType;

    const value_t vbuffer1_min = vbuffer1.minCoeff();

    value_t vbuffer1_min_nnz = std::numeric_limits<value_t>::infinity();
    value_t h_max = 0;

    // If L+l2 have entries <= threshold,
    // find h_max with more numerically-stable routine.
    // If threshold > 0, there is NO guarantee that f(h_max) <= 0, 
    // but we will use this to bisect and find an h where f(h) >= 0,
    // so we don't necessarily need h_max to be f(h_max) <= 0.
    if (vbuffer1_min <= zero_tol) {
        for (int i = 0; i < vbuffer1.size(); ++i) {
            const bool is_nonzero = vbuffer1[i] > zero_tol;
            const auto vi2 = v[i] * v[i];
            h_max += is_nonzero ? vi2 / (vbuffer1[i] * vbuffer1[i]) : 0;
            vbuffer1_min_nnz = is_nonzero ? std::min(vbuffer1_min_nnz, vbuffer1[i]) : vbuffer1_min_nnz;
        }
        h_max = std::sqrt(h_max);
    } else {
        vbuffer1_min_nnz = vbuffer1_min;
        h_max = (v / vbuffer1).matrix().norm();
    }

    return std::make_pair(h_max, vbuffer1_min_nnz);
}

template <class ValueType, class DiagType, class VType>
inline
auto root_function(
    ValueType h,
    const DiagType& D,
    const VType& v,
    ValueType l1
)
{
    return (v.array() / (D.array() * h + l1)).matrix().squaredNorm() - 1;
}

/**
 * @brief Base Newton solver for the block update norm objective.
 * IMPORTANT: This solver assumes that v is of the form sqrt(L) * u.
 * 
 * @tparam LType        vector type.
 * @tparam VType        vector type.
 * @tparam ValueType    float type.
 * @tparam XType        vector type.
 * @tparam BufferType   vector type.
 * @tparam InitialType  see newton_root_find.
 * @param L             vector with non-negative entries.
 * @param v             any vector same length as L.
 * @param l1            l1 regularization.
 * @param l2            l2 regularization.
 * @param tol           see newton_root_find.
 * @param max_iters     see newton_root_find.
 * @param x             solution vector.
 *                      Also used as buffer to store (v / buffer2).square() 
 *                      in each Newton step.
 * @param iters         number of iterations taken. Set to 0 before anything.
 * @param buffer1       buffer to store L + l2.
 * @param buffer2       buffer to store buffer1 * h + l1.
 * @param initial_f     see newton_root_find.
 */
template <class LType, class VType, class ValueType, 
          class XType, class BufferType, class InitialType>
inline
void newton_solver_base(
    const LType& L,
    const VType& v,
    ValueType l1,
    ValueType l2,
    ValueType tol,
    size_t max_iters,
    InitialType initial_f,
    XType& x,
    size_t& iters,
    BufferType& buffer1,
    BufferType& buffer2
)
{
    using value_t = ValueType;
    iters = 0;

    // Easy case: ||v||_2 <= l1 -> x = 0
    const auto v_l2 = v.matrix().norm();
    if (v_l2 <= l1) {
        x.setZero();
        return;
    }
    
    // Difficult case: ||v||_2 > l1
    
    // Still easy if l1 == 0.0
    // IMPORTANT: user's responsibility that L + l2 does not have any zeros.
    if (l1 <= 0.0) {
        x = v / (L + l2);
        return;
    }
    
    // First solve for h := ||x||_2
    auto vbuffer1 = buffer1.head(L.size());
    auto vbuffer2 = buffer2.head(L.size());

    vbuffer1 = (L + l2);

    const auto step_f = [&](auto h) {
        vbuffer2 = vbuffer1 * h + l1;
        x = (v / vbuffer2).square();
        auto fh = x.sum() - 1.0;
        auto dfh = -2.0 * (
            x * (vbuffer1 / vbuffer2)
        ).sum();
        return std::make_pair(fh, dfh);
    };

    const auto project_f = [&](auto h) {
        return std::max<value_t>(h, 0.0);
    };

    const auto root_find_state = optimization::newton_root_find(
        initial_f,
        step_f,
        project_f,
        tol, 
        max_iters
    );

    const auto h = std::get<0>(root_find_state);

    x = h * v / vbuffer2;
    iters = std::get<3>(root_find_state);
}

/*
 * Vanilla Newton solver.
 */
template <class LType, class VType, class ValueType, 
          class XType, class BufferType>
inline
void newton_solver(
    const LType& L,
    const VType& v,
    ValueType l1,
    ValueType l2,
    ValueType tol,
    size_t max_iters,
    XType& x,
    size_t& iters,
    BufferType& buffer1,
    BufferType& buffer2
)
{
    const auto initial_f = [](){ return std::make_pair(0.0, 0); };
    newton_solver_base(
        L, v, l1, l2, tol, max_iters, initial_f,
        x, iters, buffer1, buffer2
    );
}

/*
 * Newton + Brent solver.
 */
template <class LType, class VType, class ValueType, 
          class XType, class BufferType>
inline
void newton_brent_solver(
    const LType& L,
    const VType& v,
    ValueType l1,
    ValueType l2,
    ValueType tol,
    ValueType brent_tol,
    size_t max_iters,
    XType& x,
    size_t& iters,
    BufferType& buffer1,
    BufferType& buffer2
)
{
    using value_t = ValueType;
    auto vbuffer1 = buffer1.head(L.size());

    const auto initial_f = [&]() {
        const value_t h_min = root_lower_bound(vbuffer1, v, l1);
        const auto h_max_out = root_upper_bound(vbuffer1, v, 0.0); // IMPORTANT: NEEDS GUARANTEE
        const value_t h_max = std::get<0>(h_max_out);

        value_t h = 0;
        size_t iters_brent;
        optimization::brent(
            [&](auto x) { return root_function(x, vbuffer1, v, l1); },
            tol,
            brent_tol, 
            max_iters,
            h_min,
            h_min,
            h_max,
            [](auto, auto, auto, auto) { 
                return std::make_tuple(false, 0.0);
            },
            h,
            iters_brent
        );
        return std::make_pair(h, 0); 
    };

    newton_solver_base(
        L, v, l1, l2, tol, max_iters, initial_f,
        x, iters, buffer1, buffer2
    );
}

/*
 * Newton-ABS solver
 */
template <class LType, class VType, class ValueType, 
          class XType, class BufferType>
inline
void newton_abs_solver(
    const LType& L,
    const VType& v,
    ValueType l1,
    ValueType l2,
    ValueType tol,
    size_t max_iters,
    XType& x,
    size_t& iters,
    BufferType& buffer1,
    BufferType& buffer2
)
{
    using value_t = ValueType;
    auto vbuffer1 = buffer1.head(L.size());

    const auto initial_f = [&]() {
        const value_t h_min = root_lower_bound(vbuffer1, v, l1);
        const auto h_max_out = root_upper_bound(vbuffer1, v, l1);
        const value_t h_max = std::get<0>(h_max_out);
        const value_t vbuffer1_min_nnz = std::get<1>(h_max_out);

        value_t h;

        // If range is very small, just set h = h_min and Newton.
        // NOTE: this accounts for when h_max <= h_min as well on purpose!
        // The numerically stable way of computing h_max may lead to h_max <= h_min.
        // This is an indication that the correct range was small to begin with.
        if (h_max - h_min <= 1e-1) {
            h = h_min;
        } else {
            // NEW METHOD: adaptive bisection
            value_t h_cand = h_max;
            value_t w;  // prior
            value_t fh; // current function value

            const auto ada_bisect = [&]() {
                // enforce some movement towards h_min for safety.
                w = std::max<value_t>(l1 / (vbuffer1_min_nnz * h_cand + l1), 0.05);
                h_cand = w * h_min + (1-w) * h_cand;
                fh = root_function(h_cand, vbuffer1, v, l1);
            };
            
            ada_bisect();
            while ((fh < 0) && (std::abs(fh) > tol)) { 
                ada_bisect();
            }
            
            // save candidate solution
            h = h_cand;
        }

        return std::make_pair(h, 0);
    };

    newton_solver_base(
        L, v, l1, l2, tol, max_iters, initial_f,
        x, iters, buffer1, buffer2
    );
}

/**
 * Same as newton_solver above but returns way more information.
 *      
 * @param   init        initial point if used.
 * @param   x           matrix of iterate vectors (column-wise).
 * @param   buffer1     any vector with L.size() <= buffer1.size().
 * @param   buffer2     any vector with L.size() <= buffer2.size().
 */
template <class LType, class VType, class ValueType, 
          class XType, class ItersType, class BufferType>
inline
void newton_abs_debug_solver(
    const LType& L,
    const VType& v,
    ValueType l1,
    ValueType l2,
    ValueType tol,
    size_t max_iters,
    bool smart_init,
    ValueType& h_min,
    ValueType& h_max,
    XType& x,
    ItersType& iters,
    ItersType& smart_iters,
    BufferType& buffer1,
    BufferType& buffer2
)
{
    using value_t = ValueType;
    
    const value_t zero = 0.0;
    value_t h = 0;

    // Easy case: ||v||_2 <= l1 -> x = 0
    const auto v_l2 = v.matrix().norm();
    if (v_l2 <= l1) {
        x.setZero();
        return;
    }
    
    // Difficult case: ||v||_2 > l1
    if (l1 <= 0.0) {
        x = v / (L + l2);
        return;
    }
    
    // First solve for h := ||x||_2
    auto vbuffer1 = buffer1.head(L.size());
    auto vbuffer2 = buffer2.head(L.size());

    // pre-compute L + l2
    vbuffer1 = (L + l2);
    
    // Find good initialization
    // The following helps tremendously if the resulting h > 0.
    if (smart_init) {
        // compute h_min
        {
            const value_t b = l1 * vbuffer1.sum();
            const value_t a = vbuffer1.matrix().squaredNorm();
            const value_t v_l1 = v.matrix().template lpNorm<1>();
            const value_t c = l1 * l1 * L.size() - v_l1 * v_l1;
            const value_t discr = b*b - a*c;
            h_min = (discr > -1e-12) ? 
                (-b + std::sqrt(std::max<value_t>(discr, zero))) / a : 0.0;
            
            // Otherwise, if h <= 0, we know at least 0 is a reasonable solution.
            // The only case h <= 0 is when 0 is already close to the solution.
            h_min = std::max<value_t>(h_min, zero);
        }
        
        // compute h_max
        const value_t vbuffer1_min = vbuffer1.minCoeff();
        value_t vbuffer1_min_nnz = vbuffer1_min;

        // If L+l2 have entries <= threshold,
        // find h_max with more numerically-stable routine.
        // There is NO guarantee that f(h_max) <= 0, 
        // but we will use this to bisect and find an h where f(h) >= 0,
        // so we don't necessarily need h_max to be f(h_max) <= 0.
        if (vbuffer1_min <= 1e-10) {
            vbuffer1_min_nnz = std::numeric_limits<value_t>::infinity();
            h_max = 0;
            for (int i = 0; i < vbuffer1.size(); ++i) {
                const bool is_nonzero = vbuffer1[i] > 1e-10;
                const auto vi2 = v[i] * v[i];
                h_max += is_nonzero ? vi2 / (vbuffer1[i] * vbuffer1[i]) : 0;
                vbuffer1_min_nnz = is_nonzero ? std::min<value_t>(vbuffer1_min_nnz, vbuffer1[i]) : vbuffer1_min_nnz;
            }
            h_max = std::sqrt(h_max);
        } else {
            h_max = (v / vbuffer1).matrix().norm();
        }
        
        // If range is very small, just set h = h_min and Newton.
        // NOTE: this accounts for when h_max <= h_min as well on purpose!
        // The numerically stable way of computing h_max may lead to h_max <= h_min.
        // This is an indication that the correct range was small to begin with.
        if (h_max - h_min <= 1e-1) {
            h = h_min;
        } else {
            //// NEW METHOD: bisection
            // Adaptive method enforces some movement towards h_min for safety.

            value_t w = std::max<value_t>(l1 / (vbuffer1_min_nnz * h_max + l1), 0.05);
            h = w * h_min + (1-w) * h_max;
            value_t fh = (v / (vbuffer1 * h + l1)).matrix().squaredNorm() - 1;
            
            smart_iters.push_back(h);

            while ((fh < 0) && std::abs(fh) > tol) {
                w = std::max<value_t>(l1 / (vbuffer1_min_nnz * h + l1), 0.05);
                h = w * h_min + (1-w) * h;
                fh = (v / (vbuffer1 * h + l1)).matrix().squaredNorm() - 1;
                smart_iters.push_back(h);
            }

            if (std::abs(fh) <= tol) {
                h = std::max<value_t>(h, zero);
                x = h * v / vbuffer2;
                return;
            }
        }
    }

    // Newton method
    value_t fh;
    value_t dfh;

    const auto newton_update = [&]() {
        vbuffer2 = vbuffer1 * h + l1;
        x = (v / vbuffer2).square();
        fh = x.sum() - 1;
        dfh = -2 * (x * (vbuffer1 / vbuffer2)).sum();
    };
    
    newton_update();

    size_t i = 0;
    while (std::abs(fh) > tol && i < max_iters) {
        // Newton update 
        h -= fh / dfh;
        iters.push_back(h);
        newton_update();
        ++i;
    }
    
    // numerical stability
    h = std::max<value_t>(h, zero);

    // final solution
    x = h * v / vbuffer2;
}

template <class LType, class VType, class ValueType, class XType>
void brent_solver(
    const LType& L,
    const VType& v,
    ValueType l1,
    ValueType l2,
    ValueType tol,
    size_t max_iters,
    XType& x,
    size_t& iters
)
{
    using value_t = ValueType;
    value_t h = 0;
    const util::rowvec_type<value_t> buffer1 = L + l2;
    const auto a = root_lower_bound(buffer1, v, l1);
    const auto b = std::get<0>(root_upper_bound(buffer1, v, 0.0));
    iters = 0;
    const auto phi = [&](auto h) {
        return root_function(h, buffer1, v, l1);
    };
    optimization::brent(
        phi, tol, tol, max_iters, a, a, b, 
        [](auto, auto, auto, auto) { return std::make_pair(false, 0.0); },
        h, iters
    );
    x = h * v / (buffer1 * h + l1);
}

// =======================================================================
// BCD constrained
// =======================================================================

template <class QuadCType, class QCType,
          class AQCType, class QTVCType, class AType, class BType,
          class ValueType, class OutType, class BuffType>
void admm_cnstr_solver(
    const QuadCType& quad_c,
    ValueType l1,
    ValueType l2,
    const QCType& Q_c,
    const AQCType& AQ_c,
    const QTVCType& QTv_c,
    const AType& A,
    const BType& b,
    ValueType rho,
    size_t max_iters,
    ValueType tol_abs,
    ValueType tol_rel,
    OutType& x,
    OutType& z,
    OutType& u,
    size_t& iters,
    BuffType& buff
)
{
    using value_t = ValueType;
    using vec_value_t = util::rowvec_type<value_t>;

    const auto m = A.rows();
    const auto d = A.cols();

    iters = 0;
    while (iters < max_iters) {
        // increment counter no matter what
        ++iters;

        // x-update
        Eigen::Map<vec_value_t> zmu(buff.data(), m);
        Eigen::Map<vec_value_t> linear_curr(buff.data() + m, d);
        Eigen::Map<vec_value_t> QTx(buff.data() + m+d, d);
        Eigen::Map<vec_value_t> x_buffer1(buff.data() + m+2*d, d);
        Eigen::Map<vec_value_t> x_buffer2(buff.data() + m+3*d, d);
        zmu = z - u;
        linear_curr.matrix().noalias() = zmu.matrix() * AQ_c;
        linear_curr = QTv_c + rho * linear_curr;
        size_t x_iters;
        newton_abs_solver(
            quad_c,
            linear_curr,
            l1,
            l2,
            1e-12,
            1000,
            QTx,
            x_iters,
            x_buffer1,
            x_buffer2
        );
        x.matrix().noalias() = QTx.matrix() * Q_c.transpose();

        // z-update
        Eigen::Map<vec_value_t> Ax(buff.data(), m);
        Eigen::Map<vec_value_t> z_prev(buff.data()+m, m);
        Ax.matrix().noalias() = x.matrix() * A.transpose();
        z_prev = z;
        z = (u + Ax).min(b);

        // u-update
        Eigen::Map<vec_value_t> r(buff.data()+2*m, m);
        r = Ax - z;
        u += r;

        // check convergence
        Eigen::Map<vec_value_t> s(buff.data()+3*m, d);
        s.matrix().noalias() = (z - z_prev).matrix() * A;
        const auto eps_pri = std::sqrt(m) * tol_abs + tol_rel * std::max<value_t>(
            Ax.matrix().norm(), z.matrix().norm()
        );
        Eigen::Map<vec_value_t> ATu(buff.data()+3*m+d, d);
        ATu.matrix().noalias() = u.matrix() * A;
        const auto eps_dual = std::sqrt(d) * tol_abs + tol_rel * rho * ATu.matrix().norm();
        if (
            (r.square().sum() <= eps_pri * eps_pri) &&
            (rho * rho * s.square().sum() <= eps_dual * eps_dual)
        ) break;
    }
}

template <class QuadType, class LinearType, class ValueType,
          class AType, class BType, class AVarType,
          class OutType, class BuffType>
void newton_cnstr_solver(
    const QuadType& quad,
    const LinearType& linear,
    ValueType l1,
    ValueType l2,
    const AType& A,
    const BType& b,
    const AVarType& A_vars,
    size_t max_iters,
    ValueType tol,
    size_t pnewton_max_iters,
    ValueType pnewton_tol,
    size_t newton_max_iters,
    ValueType newton_tol,
    size_t& iters,
    OutType& x,
    OutType& mu,
    OutType& mu_resid,
    ValueType& mu_rsq,
    BuffType& buff
)
{
    using value_t = ValueType;
    using vec_value_t = util::rowvec_type<value_t>;

    if (l1 <= 0) {
        // TODO: this is just QP with linear inequality constraint.
        throw util::adelie_core_error(
            "newton_cnstr_solver: case l1 <= 0 not implemented yet!"
        );
    }

    const auto m = A.rows();
    const auto d = A.cols();
    const auto& S = quad;
    const auto& v = linear;

    iters = 0;

    Eigen::Map<vec_value_t> x_buffer1(buff.data(), d);
    Eigen::Map<vec_value_t> x_buffer2(buff.data()+d, d);

    // optimization: check if unconstrained solution is feasible.
    {
        Eigen::Map<vec_value_t> x_uncnstr(buff.data()+2*d, d);
        size_t x_iters;
        newton_abs_solver(
            quad, linear, l1, l2, newton_tol, newton_max_iters, 
            x_uncnstr, x_iters, x_buffer1, x_buffer2
        );
        x_buffer1.matrix().noalias() = x_uncnstr.matrix() * A.transpose();
        // if unconstrained solution is feasible
        if ((x_buffer1 <= b).all()) {
            x = x_uncnstr;
            mu.setZero();
            mu_resid = v;
            mu_rsq = v.square().sum();
            return;
        }
    }

    // since unconstrained solution is wrong, 

    /* invariance quantities */
    // mu_resid = v - A.T @ mu
    // mu_rsq = ||mu_resid||_2^2

    value_t convg_measure = 0;

    const auto invariance_f = [&](
        size_t k, value_t mu_k, value_t Akr
    ) {
        const auto del = mu_k - mu[k];
        mu[k] = mu_k;
        mu_resid -= del * A.row(k).array();
        mu_rsq -= del * (2 * Akr - A_vars[k] * del);
    };

    while (iters < max_iters) {
        bool compute_x = false;
        convg_measure = 0;
        ++iters;

        // coordinate descent
        for (int k = 0; k < m; ++k) {
            const auto A_vars_k = A_vars[k];

            // if kth constraint can be removed.
            if (A_vars_k <= 0) {
                mu[k] = 0;
                continue;
            }

            // compute l_star
            auto Akr = A.row(k).dot(mu_resid.matrix());
            const auto discr = Akr * Akr - A_vars_k * (mu_rsq - l1 * l1);
            const bool l_star_finite = discr >= 0;
            value_t l_star = l_star_finite ? 
                (mu[k] + (Akr - std::sqrt(discr)) / A_vars_k) :
                std::numeric_limits<value_t>::max()
            ;

            /* case 1: b_k = 0 and l_star < infty */
            if (b[k] == 0 && l_star_finite) {
                const auto mu_k_new = std::max<value_t>(l_star, 0);
                if (mu_k_new == mu[k]) continue; 
                invariance_f(k, mu_k_new, Akr);
                const auto del = mu_k_new - mu[k];
                convg_measure = std::max(convg_measure, A_vars[k] * del * del);
                compute_x = true;
                continue;
            }

            // optimization: l_star <= 0 (and b_k > 0)
            if (l_star <= 0) {
                if (mu[k] == 0) continue;
                invariance_f(k, 0, Akr);
                const auto del = -mu[k];
                convg_measure = std::max(convg_measure, A_vars[k] * del * del);
                compute_x = true;
                continue;
            }

            /* case 2: b_k > 0 and 0 < l_star < infty */
            /* case 3: l_star = infty */

            // compute x
            size_t x_iters;
            newton_abs_solver(
                quad, mu_resid, l1, l2, newton_tol, newton_max_iters, 
                x, x_iters, x_buffer1, x_buffer2
            );
            compute_x = false;

            // optimization: k is inactive and will stay inactive
            if (mu[k] == 0) {
                const auto h_k = A.row(k).dot(x.matrix()) - b[k];
                if (h_k <= 0) continue;
            }

            /* projected Newton method */

            // NOTE: the following are additional invariance quantities from now:
            // - x
            // - Akr
            const auto mu_k_old = mu[k];
            const auto initial_f = [&]() {
                return std::make_tuple(mu[k], 0);
            };
            bool is_first_call = true;
            const auto step_f = [&](auto muk) {
                if (is_first_call) {
                    is_first_call = false;
                } else {
                    invariance_f(k, muk, Akr);
                    newton_abs_solver(
                        quad, mu_resid, l1, l2, newton_tol, newton_max_iters, 
                        x, x_iters, x_buffer1, x_buffer2
                    );
                    Akr = A.row(k).dot(mu_resid.matrix());
                }
                const auto x_norm = x.matrix().norm();

                // NOTE: this really should not happen, but just in case..
                if (x_norm <= 0) {
                    const value_t fh = -b[k];
                    const value_t dfh = -Akr * Akr / ((S + l2) * mu_resid.square()).sum();
                    return std::make_tuple(fh, dfh);
                }

                // Since x_norm > 0 and l1 > 0,
                //x_buffer1 = S + l2;
                //x_buffer2 = x_buffer1 * x_norm + l1;

                // compute intermediate values
                Eigen::Map<vec_value_t> t1(buff.data()+2*d, d);
                Eigen::Map<vec_value_t> t2(buff.data()+3*d, d);
                t1 = A.row(k).array() / x_buffer2;
                t2 = mu_resid / x_buffer2;
                const auto t3 = (t1 * t2).sum();

                // compute output
                const value_t fh = x_norm * (t1 * mu_resid).sum() - b[k];
                const value_t dfh = -(
                    x_norm * (t1 * A.row(k).array()).sum()
                    + l1 * t3 * t3 / (x_buffer1 * t2.square() / x_buffer2).sum()
                );
                return std::make_tuple(fh, dfh);
            };
            const auto project_f = [&](auto h) { 
                return std::max<value_t>(std::min<value_t>(h, l_star), 0); 
            };
            const auto tup = optimization::newton_root_find(
                initial_f,
                step_f,
                project_f,
                pnewton_tol,
                pnewton_max_iters
            );

            const auto del = mu[k] - mu_k_old;
            convg_measure = std::max(convg_measure, A_vars[k] * del * del);
        }

        // check convergence
        if (convg_measure <= tol) {
            if (compute_x) {
                size_t x_iters;
                newton_abs_solver(
                    quad, mu_resid, l1, l2, newton_tol, newton_max_iters, 
                    x, x_iters, x_buffer1, x_buffer2
                );
            }
            break;
        }
    }    
}

} // namespace bcd
} // namespace adelie_core