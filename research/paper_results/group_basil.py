import pyglstudy as gl
from pyglstudy.group_lasso import *
from pyglstudy.group_basil import group_basil
import numpy as np
import matplotlib.pyplot as plt

n = 100
p = 100000
n_groups = int(p/100)
seed = 0

np.random.seed(seed)
X, beta, y, groups, group_sizes = generate_group_lasso_data(
    n, p, n_groups, rho=0.1, svd_transform=False, group_split_type="random",
).values()

alpha = 1
penalty = np.sqrt(group_sizes)

# naive
out_naive = group_basil(
    X, y, groups, group_sizes, 
    alpha=alpha,
    penalty=penalty,
    method='naive', 
    max_n_cds=int(1e5), 
    n_lambdas_iter=29,#max(int(n_groups * 0.1), 5),
    use_strong_rule=False,
    max_strong_size=p-1,
    verbose_diagnostic=False,
)
print(out_naive)