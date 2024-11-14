#include <tools/eigen_wrap.hpp>
#include <tools/types.hpp>
#include <adelie_core/state/state_css_cov.ipp>

template class adelie_core::state::StateCSSCov<dense_type<float, Eigen::ColMajor>>;
template class adelie_core::state::StateCSSCov<dense_type<double, Eigen::ColMajor>>;