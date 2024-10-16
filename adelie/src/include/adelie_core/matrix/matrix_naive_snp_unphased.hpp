#pragma once
#include <functional>
#include <memory>
#include <string>
#include <adelie_core/io/io_snp_unphased.hpp>
#include <adelie_core/matrix/matrix_naive_base.hpp>

namespace adelie_core {
namespace matrix {

template <class ValueType,
          class MmapPtrType=std::unique_ptr<char, std::function<void(char*)>>,
          class IndexType=Eigen::Index>
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
    
protected:
    static constexpr value_t _max = std::numeric_limits<value_t>::max();
    const io_t& _io;             // IO handler
    const size_t _n_threads;    // number of threads
    vec_index_t _ibuff;
    vec_value_t _vbuff;
    vec_value_t _buff;

    inline value_t _cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v,
        const Eigen::Ref<const vec_value_t>& weights,
        size_t n_threads
    );

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

    value_t cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v,
        const Eigen::Ref<const vec_value_t>& weights
    ) override;

    void ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out
    ) override;

    void bmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override;

    void btmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override;

    void mul(
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override;

    void cov(
        int j, int q,
        const Eigen::Ref<const vec_value_t>& sqrt_weights,
        Eigen::Ref<colmat_value_t> out,
        Eigen::Ref<colmat_value_t> buffer
    ) override;

    int rows() const override;
    int cols() const override;

    void sp_tmul(
        const sp_mat_value_t& v,
        Eigen::Ref<rowmat_value_t> out
    ) override;
};

} // namespace matrix
} // namespace adelie_core