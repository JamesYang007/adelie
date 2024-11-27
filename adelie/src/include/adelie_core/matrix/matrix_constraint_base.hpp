#pragma once
#include <adelie_core/util/types.hpp>

#ifndef ADELIE_CORE_MATRIX_CONSTRAINT_BASE_TP
#define ADELIE_CORE_MATRIX_CONSTRAINT_BASE_TP \
    template <class ValueType, class IndexType>
#endif
#ifndef ADELIE_CORE_MATRIX_CONSTRAINT_BASE
#define ADELIE_CORE_MATRIX_CONSTRAINT_BASE \
    MatrixConstraintBase<ValueType, IndexType>
#endif

namespace adelie_core {
namespace matrix {

template <class ValueType, class IndexType=Eigen::Index> 
class MatrixConstraintBase
{
public:
    using value_t = ValueType;
    using index_t = IndexType;
    using vec_value_t = util::rowvec_type<value_t>;
    using vec_index_t = util::rowvec_type<index_t>;
    using colmat_value_t = util::colmat_type<value_t>;
    
    virtual ~MatrixConstraintBase() {}

    virtual void rmmul(
        int j, 
        const Eigen::Ref<const colmat_value_t>& Q,
        Eigen::Ref<vec_value_t> out
    ) =0;

    virtual void rmmul_safe(
        int j, 
        const Eigen::Ref<const colmat_value_t>& Q,
        Eigen::Ref<vec_value_t> out
    ) const =0;

    virtual value_t rvmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v
    ) =0;

    virtual value_t rvmul_safe(
        int j, 
        const Eigen::Ref<const vec_value_t>& v
    ) const =0;

    virtual void rvtmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out
    ) =0;

    virtual void mul(
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) const =0;

    virtual void tmul(
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) const =0;

    virtual void cov(
        const Eigen::Ref<const colmat_value_t>& Q,
        Eigen::Ref<colmat_value_t> out
    ) const =0;

    virtual int rows() const =0;
    
    virtual int cols() const =0;

    virtual void sp_mul(
        const Eigen::Ref<const vec_index_t>& indices,
        const Eigen::Ref<const vec_value_t>& values,
        Eigen::Ref<vec_value_t> out
    ) const =0;
};

} // namespace matrix
} // namespace adelie_core

#ifndef ADELIE_CORE_MATRIX_CONSTRAINT_PURE_OVERRIDE_DECL
#define ADELIE_CORE_MATRIX_CONSTRAINT_PURE_OVERRIDE_DECL \
    void rmmul(\
        int j,\
        const Eigen::Ref<const colmat_value_t>& Q,\
        Eigen::Ref<vec_value_t> out\
    ) override;\
    void rmmul_safe(\
        int j,\
        const Eigen::Ref<const colmat_value_t>& Q,\
        Eigen::Ref<vec_value_t> out\
    ) const override;\
    value_t rvmul(\
        int j,\
        const Eigen::Ref<const vec_value_t>& v\
    ) override;\
    value_t rvmul_safe(\
        int j,\
        const Eigen::Ref<const vec_value_t>& v\
    ) const override;\
    void rvtmul(\
        int j,\
        value_t v,\
        Eigen::Ref<vec_value_t> out\
    ) override;\
    void mul(\
        const Eigen::Ref<const vec_value_t>& v,\
        Eigen::Ref<vec_value_t> out\
    ) const override;\
    void tmul(\
        const Eigen::Ref<const vec_value_t>& v,\
        Eigen::Ref<vec_value_t> out\
    ) const override;\
    void cov(\
        const Eigen::Ref<const colmat_value_t>& Q,\
        Eigen::Ref<colmat_value_t> out\
    ) const override;\
    int rows() const override;\
    int cols() const override;\
    void sp_mul(\
        const Eigen::Ref<const vec_index_t>& indices,\
        const Eigen::Ref<const vec_value_t>& values,\
        Eigen::Ref<vec_value_t> out\
    ) const override;
#endif