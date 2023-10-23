#pragma once

namespace adelie_core {
namespace optimization {

template <class XType, class YType, class MSESType>
int search_pivot(
    const XType& x,
    const YType& y,
    MSESType& mses
)
{
    using value_t = typename std::decay_t<XType>::Scalar;

    assert(x.size() == y.size());
    assert(x.size() == mses.size());

    const auto n = x.size();

    if (n <= 0) return -1;

    mses[0] = std::numeric_limits<value_t>::infinity();

    if (n == 1) return 0;

    // assume x is sorted and y follows the same ordering
    // starting with the second point x[1] as the pivot candidate,
    // run linear regression of y = b0 + b1 * (pivot - x) 1(x <= pivot) + eps.
    // Find the pivot choice that yields lowest MSE.
    const value_t y_mean = y.mean();
    value_t x_sum = x[0];
    value_t xsq_sum = x[0] * x[0];
    value_t y_sum = y[0];
    value_t yx_sum = y[0] * x[0];
    value_t min_mse = mses[0];
    int argmin_mse = 0;

    for (int i = 1; i < n; ++i) {
        x_sum += x[i];
        xsq_sum += x[i] * x[i];
        y_sum += y[i];
        yx_sum += y[i] * x[i];
        const value_t x_bar = ((i+1) * x[i] - x_sum) / n;
        const value_t var_x = (
            (i+1) * x[i] * x[i] 
            - 2 * x[i] * x_sum
            + xsq_sum
            - n * x_bar * x_bar
        );
        const value_t cov_xy = (
            x[i] * (y_sum - i * y_mean) 
            - (yx_sum - y_mean * x_sum)
        );
        const value_t beta1_hat = cov_xy / var_x;
        mses[i] = -beta1_hat * beta1_hat * var_x;
        argmin_mse = (mses[i] < min_mse) ? i : argmin_mse;
        min_mse = std::min(min_mse, mses[i]);
    }

    return argmin_mse;
}

} // namespace optimization
} // namespace adelie_core