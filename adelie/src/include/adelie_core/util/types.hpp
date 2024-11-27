#pragma once
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <adelie_core/util/eigen/map_sparsevector.hpp>
#include <string>
#include <adelie_core/util/exceptions.hpp>

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

enum class nnqp_screen_rule_type
{
    _greedy
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

enum class read_mode_type
{
    _file,
    _mmap,
    _auto
};

enum class impute_method_type
{
    _mean,
    _user
};

enum class operator_type
{
    _eq,
    _add
};

enum class css_method_type
{
    _greedy,
    _swapping
};

enum class css_loss_type
{
    _least_squares,
    _subset_factor,
    _min_det
};

enum class omp_schedule_type
{
    _static,
    _dynamic,
    _guided,
    _runtime
};

inline screen_rule_type convert_screen_rule(
    const std::string& rule
)
{
    if (rule == "strong") return screen_rule_type::_strong;
    if (rule == "pivot") return screen_rule_type::_pivot;
    throw util::adelie_core_error("Invalid screen rule type: " + rule);
}

inline nnqp_screen_rule_type convert_nnqp_screen_rule(
    const std::string& rule
)
{
    if (rule == "greedy") return nnqp_screen_rule_type::_greedy;
    throw util::adelie_core_error("Invalid NNQP screen rule type: " + rule);
}

inline tie_method_type convert_tie_method(
    const std::string& method
) 
{
    if (method == "breslow") return tie_method_type::_breslow;
    if (method == "efron") return tie_method_type::_efron;
    throw util::adelie_core_error("Invalid tie method: " + method);
}

inline hessian_type convert_hessian(
    const std::string& hessian
) 
{
    if (hessian == "diagonal") return hessian_type::_diagonal;
    if (hessian == "full") return hessian_type::_full;
    throw util::adelie_core_error("Invalid hessian type: " + hessian);
}

inline read_mode_type convert_read_mode(
    const std::string& read_mode
)
{
    if (read_mode == "file") return read_mode_type::_file;
    if (read_mode == "mmap") return read_mode_type::_mmap;
    throw util::adelie_core_error("Invalid read mode type: " + read_mode);
}

inline impute_method_type convert_impute_method(
    const std::string& impute_method
)
{
    if (impute_method == "mean") return impute_method_type::_mean;
    if (impute_method == "user") return impute_method_type::_user;
    throw util::adelie_core_error("Invalid impute mode type: " + impute_method);
}

inline css_method_type convert_css_method(
    const std::string& css_method
)
{
    if (css_method == "greedy") return css_method_type::_greedy;
    if (css_method == "swapping") return css_method_type::_swapping;
    throw util::adelie_core_error("Invalid CSS method type: " + css_method);
}

inline css_loss_type convert_css_loss(
    const std::string& css_loss
)
{
    if (css_loss == "least_squares") return css_loss_type::_least_squares;
    if (css_loss == "subset_factor") return css_loss_type::_subset_factor;
    if (css_loss == "min_det") return css_loss_type::_min_det;
    throw util::adelie_core_error("Invalid CSS loss type: " + css_loss);
}

} // namespace util
} // namespace adelie_core
