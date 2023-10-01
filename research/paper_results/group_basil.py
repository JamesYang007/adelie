import adelie as gl
from adelie.solver import *
from adelie.group_basil import group_basil
import numpy as np
import matplotlib.pyplot as plt

n = 10
p = 100
n_groups = int(p/10)
seed = 0

np.random.seed(seed)
X, beta, y, groups, group_sizes = generate_group_elnet_data(
    n, p, n_groups, rho=0.4, svd_transform=False, group_split_type="random",
).values()

alpha = 0.999
penalty = np.sqrt(group_sizes)

# naive
out_naive = group_basil(
    X, y, groups, group_sizes, 
    alpha=alpha,
    penalty=penalty,
    method='naive', 
    max_n_cds=int(1e5), 
    n_lambdas_iter=5,
    #use_strong_rule=False,
    use_strong_rule=True,
    max_strong_size=p-1,
    verbose_diagnostic=False,
)
diag_naive = out_naive['diagnostic']
print(out_naive)