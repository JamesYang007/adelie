#pragma once
#include <Eigen/Core>

namespace adelie_core {
namespace util {

template <class T>
struct is_dense 
{
    using po_t = typename std::decay_t<T>::PlainObject;
    static constexpr bool value =
        std::is_base_of<Eigen::DenseBase<po_t>, po_t>::value;
};

} // namespace util
} // namespace adelie_core
