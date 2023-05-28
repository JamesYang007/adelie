from pyglstudy.pyglstudy_ext import group_basil_cov__, group_basil_naive__
import numpy as np
import os

def group_basil(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    group_sizes: np.ndarray,
    method: str = "auto",
    alpha: float = 1,
    penalty: np.ndarray = None,
    user_lmdas: np.ndarray = None,
    max_n_lambdas: int = 100,
    n_lambdas_iter: int = 5,
    use_strong_rule: bool = True,
    do_early_exit: bool = True,
    verbose_diagnostic: bool = False,
    delta_strong_size: int = 5,
    max_strong_size: int = None,
    max_n_cds: int = int(1e5),
    thr: float = 1e-7,
    cond_0_thresh: float = 1e-3,
    cond_1_thresh: float = 1e-3,
    newton_tol: float = 1e-8,
    newton_max_iters: int = 100,
    min_ratio: float = 1e-2,
    n_threads: int = os.cpu_count(),
):
    method_dict = {
        "cov" : group_basil_cov__,
        "naive" : group_basil_naive__,
    }

    n, p = X.shape
    
    if method == "auto":
        method = "cov" if n > p or p < 1e3 else "naive"
    
    if penalty is None:
        penalty = np.sqrt(group_sizes)

    if user_lmdas is None:
        user_lmdas = []
        
    if max_strong_size is None:
        max_strong_size = p

    return method_dict[method](
        X, y, groups, group_sizes, alpha, penalty,
        user_lmdas, max_n_lambdas, n_lambdas_iter,
        use_strong_rule, do_early_exit, verbose_diagnostic,
        delta_strong_size, max_strong_size,
        max_n_cds, thr, cond_0_thresh, cond_1_thresh,
        newton_tol, newton_max_iters, min_ratio, n_threads,
    )
