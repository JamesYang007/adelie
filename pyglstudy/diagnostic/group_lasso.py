from pyglstudy.group_lasso import GroupLassoPack
import numpy as np


def check_data(
    A, r, groups, group_sizes, 
):
    assert A.shape[0] == r.shape[0]
    assert A.shape[0] == A.shape[1]
    
    p = A.shape[-1]

    assert groups.shape[0] == group_sizes.shape[0]
    assert groups[0] == 0
    for i in range(len(groups)-1):
        assert groups[i+1] - groups[i] == group_sizes[i]
    assert p - groups[-1] == group_sizes[-1]
    assert np.sum(group_sizes) == p
    
    #assert 0 <= alpha and alpha <= 1
    #assert np.all(penalty >= 0)
