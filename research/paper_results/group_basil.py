import pyglstudy as gl
from pyglstudy.group_lasso import *
from pyglstudy.group_basil import group_basil
import numpy as np
import matplotlib.pyplot as plt

n = 1000
p = 100000
n_groups = int(p/1)
seed = 0

np.random.seed(seed)
X, beta, y, groups, group_sizes = generate_group_lasso_data(
    n, p, n_groups, rho=0.5, svd_transform=False, group_split_type="random",
).values()

out_naive = group_basil(X, y, groups, group_sizes, method='naive', max_n_cds=int(1))
diag_naive = out_naive['diagnostic']