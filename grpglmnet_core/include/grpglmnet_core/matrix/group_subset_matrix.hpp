#pragma once

namespace grpglmnet_core {

class GroupSubsetMatrix
{
public:
    // col(i): return view of column of matrix
    // only needs to be well-defined for i == beginning index of a group

    // block(i,j,p,q): return block view of matrix
    // only needs to be well-defined for j == beginning index of a group

    // rows, cols: for the original matrix
};

} // namespace grpglmnet_core