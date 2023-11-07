#pragma once
#include <string>
#include <vector>
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/matrix/utils.hpp>

namespace adelie_core {
namespace matrix {

template <class ValueType>
class MatrixNaiveSNPUnphased : public MatrixNaiveBase<ValueType>
{
public:
    using base_t = MatrixNaiveBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using typename base_t::rowmat_value_t;
    using typename base_t::sp_mat_value_t;
    
private:
    const size_t _n_threads;                // number of threads

public:
    MatrixNaiveSNPUnphased(
        std::vector<std::string> filenames,
        size_t n_threads
    ): 
        _n_threads(n_threads)
    {}

    value_t cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v
    ) const override
    {

    }
};

} // namespace matrix
} // namespace adelie_core