#pragma once
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <grpglmnet_core/util/macros.hpp>

namespace grpglmnet_core {
namespace util {

// forward declaration
template <class IntType, class F>
class functor_iterator;

template <class IntType, class F>
inline constexpr bool 
operator==(const functor_iterator<IntType, F>& it1,
           const functor_iterator<IntType, F>& it2)
{ 
    assert(it1.f_ == it2.f_);
    return (it1.curr_ == it2.curr_); 
}

template <class IntType, class F>
inline constexpr bool 
operator!=(const functor_iterator<IntType, F>& it1,
           const functor_iterator<IntType, F>& it2)
{ 
    assert(it1.f_ == it2.f_);
    return (it1.curr_ != it2.curr_); 
}

template <class IntType, class F>
inline constexpr functor_iterator<IntType, F> 
operator+(const functor_iterator<IntType, F>& it1,
          typename functor_iterator<IntType, F>::difference_type n)
{ 
    return functor_iterator<IntType, F>(it1.curr_ + n, it1.f_);
}

template <class IntType, class F>
inline constexpr functor_iterator<IntType, F> 
operator+(typename functor_iterator<IntType, F>::difference_type n,
          const functor_iterator<IntType, F>& it1)
{ 
    return functor_iterator<IntType, F>(it1.curr_ + n, it1.f_);
}

template <class IntType, class F>
inline constexpr functor_iterator<IntType, F> 
operator-(const functor_iterator<IntType, F>& it,
          typename functor_iterator<IntType, F>::difference_type n)
{
    return functor_iterator<IntType, F>(it.curr_-n, it.f_);
}

template <class IntType, class F>
inline constexpr typename functor_iterator<IntType, F>::difference_type
operator-(const functor_iterator<IntType, F>& it1,
          const functor_iterator<IntType, F>& it2)
{
    return it1.curr_ - it2.curr_;
}

template <class IntType, class F>
class functor_iterator
{
    using int_t = IntType;
    using f_t = F;

public:
    using difference_type = int32_t;
#if __cplusplus >= 201703L
    using value_type = std::invoke_result_t<F, int_t>;
#elif __cplusplus >= 201103L 
    using value_type = typename std::result_of<F(int_t)>::type;
#endif
    using pointer = value_type*;
    using reference = value_type&;
    using iterator_category = std::random_access_iterator_tag;

private:
    f_t* f_;
    int_t curr_;

public:

    functor_iterator(int_t begin, f_t& f)
        : f_(&f), curr_(begin)
    {}

    GRPGLMNET_CORE_STRONG_INLINE
    functor_iterator& operator+=(difference_type n) { curr_ += n; return *this; }
    GRPGLMNET_CORE_STRONG_INLINE
    functor_iterator& operator-=(difference_type n) { curr_ -= n; return *this; }
    GRPGLMNET_CORE_STRONG_INLINE functor_iterator& operator++() { ++curr_; return *this; }
    GRPGLMNET_CORE_STRONG_INLINE functor_iterator& operator--() { --curr_; return *this; }

    // Weirdly, returning reference destroys speed in lasso coordinate descent.
    // Make sure to return by value!
    GRPGLMNET_CORE_STRONG_INLINE auto operator*() { return (*f_)(curr_); }

    GRPGLMNET_CORE_STRONG_INLINE auto operator[](difference_type n) { return (*f_)(n); }

    friend constexpr functor_iterator operator+<>(const functor_iterator&,
                                      difference_type);
    friend constexpr functor_iterator operator+<>(difference_type,
                                      const functor_iterator&);
    friend constexpr functor_iterator operator-<>(const functor_iterator&,
                                      difference_type);
    friend constexpr difference_type operator-<>(const functor_iterator&,
                                      const functor_iterator&);
    friend constexpr bool operator==<>(const functor_iterator&,
                                       const functor_iterator&);
    friend constexpr bool operator!=<>(const functor_iterator&,
                                       const functor_iterator&);
};

template <class IntType, class F>
auto make_functor_iterator(IntType i, F& f)
{
    return functor_iterator<IntType, F>(i, f);
}

} // namespace util
} // namespace ghosbasil
