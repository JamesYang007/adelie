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


def generate_group_elnet_data(
    n,
    p,
    n_groups,
    rho=0.1,
    svd_transform=True,
    group_split_type="random",
):
    X = np.random.normal(size=(n, p))
    X = rho * np.sum(X, axis=-1)[:,None] + (1-rho) * X
    beta = np.random.normal(size=(p,)) / np.sqrt(p * 0.99)
    beta[:int(p * 0.99)] = 0
    y = X @ beta + np.random.normal(size=(n,))
    
    X /= np.sqrt(n)
    y /= np.sqrt(n)

    if group_split_type == "random":
        order = np.arange(1, p) 
        groups = np.sort(np.random.choice(
            order, (n_groups-1,), replace=False,
        ))
        groups = np.concatenate([[0], groups, [p]], dtype=np.int32)
        group_sizes = groups[1:(n_groups+1)] - groups[:n_groups]
        groups = groups[:n_groups]

    elif group_split_type == "even":
        full_group_size = p // n_groups
        groups = full_group_size * np.arange(0, n_groups)
        groups = np.concatenate(
            [groups, [p]],
            dtype=np.int32,
        )
        assert len(groups) == n_groups + 1
        group_sizes = groups[1:(n_groups+1)] - groups[:n_groups]
        groups = groups[:n_groups]
    
    else:
        raise RuntimeError(f"Not a valid group_split_type: {group_split_type}")

    if svd_transform:
        for i in range(len(groups)):
            begin = groups[i]
            end = begin + group_sizes[i]
            _, _, vh = np.linalg.svd(X[:, begin:end])
            X[:, begin:end] = X[:, begin:end] @ vh.T

    X = np.asfortranarray(X)
    
    return {
        "X": X,
        "beta": beta,
        "y": y,
        "groups": groups,
        "group_sizes": group_sizes,
    }


def generate_group_elnet_state(
    X,
    y,
    groups,
    group_sizes,
    alpha,
    penalty=None, 
    strong_set=None,
    strong_g1=None,
    strong_g2=None,
    strong_begins=None,
    strong_A_diag=None,
    lmda_max=None,
    lmdas=None,
    log10_min_ratio=-2,
    n_lmdas=100,
    max_cds=int(1e5),
    thr=1e-7,
    newton_tol=1e-8,
    newton_max_iters=100,
    rsq=0.0, 
    strong_beta=None,
    strong_grad=None,
    active_set=None,
    active_g1=None,
    active_g2=None,
    active_begins=None,
    active_order=None,
    is_active=None,
):
    A_diag = np.sum(X ** 2, axis=0)
    r = X.T @ y
    n_groups = len(groups)

    if penalty is None:
        penalty = np.ones(n_groups)
    if strong_set is None:
        strong_set = np.arange(0, n_groups, dtype=np.int32)
    if strong_g1 is None:
        strong_g1 = np.array([i for i in range(len(strong_set)) if group_sizes[strong_set[i]] == 1], dtype=np.int32)
    if strong_g2 is None:
        strong_g2 = np.array([i for i in range(len(strong_set)) if group_sizes[strong_set[i]] > 1], dtype=np.int32)

    assert((len(strong_g1) + len(strong_g2)) == n_groups)

    if strong_begins is None:
        strong_begins = np.cumsum(np.concatenate(
            [[0], np.array([group_sizes[i] for i in range(len(strong_set))], dtype=np.int32)],
        ), dtype=np.int32)[:-1]

    if strong_A_diag is None:
        strong_A_diag = np.concatenate(
            [
                A_diag[groups[i] : (groups[i] + group_sizes[i])]
                for i in strong_set
            ]
        )

    if lmda_max is None:
        lmda_max = np.max([
            np.linalg.norm(r[groups[i] : (groups[i]+group_sizes[i])]) / np.maximum((alpha * penalty[i]), 1e-3)
            for i in strong_set if penalty[i] > 0
        ])

    if lmdas is None:
        lmdas = lmda_max * np.logspace(0, log10_min_ratio, n_lmdas)
        
    n_total_group_size = np.sum(group_sizes)
    assert len(strong_A_diag) == n_total_group_size

    if strong_beta is None: 
        strong_beta = np.zeros((n_total_group_size,))
    if strong_grad is None:
        indices = np.concatenate([
            np.arange(groups[i], groups[i] + group_sizes[i])
            for i in strong_set
        ])
        assert len(indices) == n_total_group_size
        
        correction = np.array([
            X[:, groups[i]:(groups[i]+group_sizes[i])] @ strong_beta[strong_begins[i]:(strong_begins[i]+group_sizes[i])]
            for i in strong_set
        ])
        strong_grad = X.T @ (y - np.sum(correction, axis=0))
        strong_grad -= np.concatenate([
            X[:, groups[i]:(groups[i]+group_sizes[i])].T @ correction[i]
            for i in strong_set
        ])

    if active_set is None:
        active_set = np.empty((0,), dtype=np.int32)
    if active_g1 is None:
        active_g1 = np.empty((0,), dtype=np.int32)
    if active_g2 is None:
        active_g2 = np.empty((0,), dtype=np.int32)
    if active_begins is None:
        active_begins = np.empty((0,), dtype=np.int32)
    if active_order is None:
        active_order = np.empty((0,), dtype=np.int32)
    if is_active is None:
        is_active = np.zeros((n_groups,), dtype=bool)
        
    return CommonPack(
        groups=groups,
        group_sizes=group_sizes,
        alpha=alpha,
        penalty=penalty, 
        strong_set=strong_set,
        strong_g1=strong_g1,
        strong_g2=strong_g2,
        strong_begins=strong_begins,
        strong_A_diag=strong_A_diag,
        lmda_max=lmda_max,
        lmdas=lmdas,
        max_cds=max_cds,
        thr=thr,
        newton_tol=newton_tol,
        newton_max_iters=newton_max_iters,
        rsq=rsq, 
        strong_beta=strong_beta,
        strong_grad=strong_grad,
        active_set=active_set,
        active_g1=active_g1,
        active_g2=active_g2,
        active_begins=active_begins,
        active_order=active_order,
        is_active=is_active,
    )
