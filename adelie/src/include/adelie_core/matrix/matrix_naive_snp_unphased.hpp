#pragma once
#include <functional>
#include <memory>
#include <string>
#include <adelie_core/io/io_snp_unphased.hpp>
#include <adelie_core/matrix/matrix_naive_base.hpp>

#ifndef ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED_TP
#define ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED_TP \
    template <class ValueType, class MmapPtrType, class IndexType>
#endif
#ifndef ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED
#define ADELIE_CORE_MATRIX_NAIVE_SNP_UNPHASED \
    MatrixNaiveSNPUnphased<ValueType, MmapPtrType, IndexType>
#endif

namespace adelie_core {
namespace matrix {

template <
    class ValueType,
    class MmapPtrType=std::unique_ptr<char, std::function<void(char*)>>,
    class IndexType=Eigen::Index
>
class MatrixNaiveSNPUnphased: public MatrixNaiveBase<ValueType, IndexType>
{
public:
    using base_t = MatrixNaiveBase<ValueType, IndexType>;
    using typename base_t::value_t;
    using typename base_t::index_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using typename base_t::rowmat_value_t;
    using typename base_t::sp_mat_value_t;
    using string_t = std::string;
    using io_t = io::IOSNPUnphased<MmapPtrType>;

private:
    static constexpr value_t _max = std::numeric_limits<value_t>::max();
    const io_t& _io;             // IO handler
    const size_t _n_threads;    // number of threads
    vec_value_t _buff;

    inline value_t _cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v,
        const Eigen::Ref<const vec_value_t>& weights,
        size_t n_threads,
        Eigen::Ref<vec_value_t> buff
    ) const;

    inline value_t _sq_cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> buff
    ) const;

    inline void _ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out,
        size_t n_threads
    ) const;

public:
    explicit MatrixNaiveSNPUnphased(
        const io_t& io,
        size_t n_threads
    );

    ADELIE_CORE_MATRIX_NAIVE_PURE_OVERRIDE_DECL
    ADELIE_CORE_MATRIX_NAIVE_OVERRIDE_DECL
};

} // namespace matrix
} // namespace adelie_core