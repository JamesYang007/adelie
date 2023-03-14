#pragma once

namespace ghostbasil {

template <class MatType>
class BlockMatrix;

template <class MatrixType, class VectorType>
class GhostMatrix;

template <class MatrixType>
class GroupGhostMatrix;

template <class MatrixType, class DType>
class BlockGroupGhostMatrix;
    
template <class XType, class ValueType>
class CovCache;

} // namespace ghostbasil
