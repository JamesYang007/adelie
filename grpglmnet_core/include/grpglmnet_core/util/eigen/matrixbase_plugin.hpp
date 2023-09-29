using value_type = Scalar;
using value_t = Scalar;

template <class VecType>
inline Scalar col_dot(size_t k, const VecType& v) const
{
    return v.dot(this->col(k));
}

template <class VecType>
inline Scalar quad_form(const VecType& v) const
{
    return v.dot((*this) * v);
}

template <class VecType>
inline Scalar inv_quad_form(Scalar s, const Eigen::MatrixBase<VecType>& v) const
{
    using matrix_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    matrix_t m = (1-s) * (*this);
    m.diagonal().array() += s;
    Eigen::FullPivLU<matrix_t> lu(m); 
    return v.dot(lu.solve(v));
}

template <class VecType>
inline Scalar inv_quad_form(Scalar s, const Eigen::SparseMatrixBase<VecType>& v) const
{
    using colvec_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    colvec_t vd = v;
    return inv_quad_form(s, vd);
}

inline const auto& to_dense() const { return *this; }

inline void cache(Eigen::Index, Eigen::Index) const {}