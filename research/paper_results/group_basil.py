# %%
import pyglstudy as gl
from pyglstudy.group_lasso import *
import numpy as np
import matplotlib.pyplot as plt

# %%
n = 1000
p = 100
n_groups = int(p/10)
seed = 0

np.random.seed(seed)
X, beta, y, groups, group_sizes = generate_group_lasso_data(
    n, p, n_groups, rho=0
).values()

# %%
groups, group_sizes

# %%
alpha = 1.0
penalty = np.ones(n_groups)
user_lmdas = [] #[1e-4, 1e-6]
max_n_lambdas = 100
n_lambdas_iter = 5
use_strong_rule = True
do_early_exit = True
verbose_diagnostic = False
delta_strong_size = 1
max_strong_size = p
max_n_cds = 100000
thr = 1e-8
newton_tol = 1e-8
newton_max_iters = 100000
min_ratio = 1e-2
n_threads = 1

# %%
out = gl.group_basil(
    X, y, groups, group_sizes, alpha, penalty, user_lmdas,
    max_n_lambdas,
    n_lambdas_iter,
    use_strong_rule,
    do_early_exit,
    verbose_diagnostic,
    delta_strong_size,
    max_strong_size,
    max_n_cds,
    thr,
    newton_tol,
    newton_max_iters,
    min_ratio,
    n_threads,
)

