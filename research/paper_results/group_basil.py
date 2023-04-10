import pyglstudy as gl
from pyglstudy.group_lasso import *
from pyglstudy.group_basil import group_basil
import numpy as np
import matplotlib.pyplot as plt

n = 100
p = 1000000
n_groups = int(p/500)
seed = 0

np.random.seed(seed)
X, beta, y, groups, group_sizes = generate_group_lasso_data(
    n, p, n_groups, rho=0, svd_transform=False
).values()

out = group_basil(X, y, groups, group_sizes, method='naive')

