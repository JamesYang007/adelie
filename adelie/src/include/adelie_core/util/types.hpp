#pragma once
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <string>

namespace adelie_core {
namespace util {
    
template <class Scalar_, int Rows_=Eigen::Dynamic, int Cols_=Eigen::Dynamic>
using colmat_type = Eigen::Matrix<Scalar_, Rows_, Cols_, Eigen::ColMajor>;

template <class Scalar_, int Rows_=Eigen::Dynamic, int Cols_=Eigen::Dynamic>
using rowmat_type = Eigen::Matrix<Scalar_, Rows_, Cols_, Eigen::RowMajor>;

template <class Scalar_, int Rows_=Eigen::Dynamic, int Cols_=Eigen::Dynamic>
using colarr_type = Eigen::Array<Scalar_, Rows_, Cols_, Eigen::ColMajor>;

template <class Scalar_, int Rows_=Eigen::Dynamic, int Cols_=Eigen::Dynamic>
using rowarr_type = Eigen::Array<Scalar_, Rows_, Cols_, Eigen::RowMajor>;

template <class Scalar_, int Rows_=Eigen::Dynamic>
using colvec_type = colarr_type<Scalar_, Rows_, 1>;

template <class Scalar_, int Cols_=Eigen::Dynamic>
using rowvec_type = rowarr_type<Scalar_, 1, Cols_>;

template <class Scalar_, int Options_=Eigen::ColMajor, class StorageIndex_=int>
using sp_vec_type = Eigen::SparseVector<Scalar_, Options_, StorageIndex_>;

enum class screen_rule_type
{
    _strong,
    _pivot
};

enum class multi_group_type
{
    _grouped,
    _ungrouped
};

enum class tie_method_type
{
    _efron,
    _breslow
};

enum class hessian_type
{
    _diagonal,
    _full
};

inline screen_rule_type convert_screen_rule(
    const std::string& rule
)
{
    if (rule == "strong") return screen_rule_type::_strong;
    if (rule == "pivot") return screen_rule_type::_pivot;
    throw std::runtime_error("Invalid screen rule type: " + rule);
}

inline multi_group_type convert_multi_group(
    const std::string& group
) 
{
    if (group == "grouped") return multi_group_type::_grouped;
    if (group == "ungrouped") return multi_group_type::_ungrouped;
    throw std::runtime_error("Invalid multi-response grouping type: " + group);
}

inline tie_method_type convert_tie_method(
    const std::string& method
) 
{
    if (method == "breslow") return tie_method_type::_breslow;
    if (method == "efron") return tie_method_type::_efron;
    throw std::runtime_error("Invalid tie method: " + method);
}

inline hessian_type convert_hessian(
    const std::string& hessian
) 
{
    if (hessian == "diagonal") return hessian_type::_diagonal;
    if (hessian == "full") return hessian_type::_full;
    throw std::runtime_error("Invalid hessian type: " + hessian);
}

} // namespace util
} // namespace adelie_core
