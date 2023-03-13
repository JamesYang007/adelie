import numpy as np


def generate_block_norm_data(p, L_zero_prop, L_small_prop, small_min=1e-14, small_max=1e-8, large_min=0, large_max=1):
    L = np.concatenate(
        [np.zeros(int(p * L_zero_prop)),
         np.random.uniform(small_min, small_max, int(p * L_small_prop))]
    )
    L = np.concatenate(
        [L, np.random.uniform(large_min, large_max, p - len(L))]
    )
    v = np.random.normal(0, 1, size=(p,)) * np.sqrt(L)
    return L, v