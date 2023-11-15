import numpy as np


def create_dense(
    n: int, 
    p: int, 
    G: int,
    *,
    equal_groups=False,
    rho: float =0,
    sparsity: float =0.95,
    zero_penalty: float =0,
    snr: float = 1,
    seed: int =0,
):
    """Creates a dense dataset.

    The groups and group sizes are generated randomly
    such that ``G`` groups are created and the sum of the group sizes is ``p``.
    The data matrix ``X`` is generated from a normal distribution
    where each feature is equicorrelated with the other features by ``rho``.
    The response ``y`` is generated from a linear model :math:`y = X\\beta + \\epsilon`
    with :math:`\\epsilon \\sim N(0, \\sigma^2 I_n)` and :math:`\\beta` such that ``sparsity`` proportion
    of the entries are set to :math:`0`.
    We compute :math:`\\sigma^2` such that the signal-to-noise ratio is given by ``snr``.
    The penalty factors are by default set to ``np.sqrt(group_sizes)``,
    however if ``zero_penalty > 0``, a random set of penalties will be set to zero,
    in which case, ``penalty`` is rescaled such that the :math:`\\ell_2` norm squared is ``p``.

    Parameters
    ----------
    n : int
        Number of data points.
    p : int
        Number of features.
    G : int
        Number of groups.
    equal_groups : bool, optional
        If ``True``, group sizes are made as equal as possible.
        Default is ``False``.
    rho : float, optional
        Feature (equi)-correlation.
        Default is ``0`` so that the features are independent.
    sparsity : float, optional
        Proportion of :math:`\\beta` entries to be zeroed out.
        Default is ``0.95``.
    zero_penalty : float, optional
        Proportion of ``penalty`` entries to be zeroed out.
        Default is ``0``.
    snr : float, optional
        Signal-to-noise ratio.
        Default is ``1``.
    seed : int, optional
        Random seed.
        Default is ``0``.

    Returns
    -------
    data : dict
        A dictionary containing the generated data:
        
            - ``"X"``: feature matrix.
            - ``"y"``: response vector.
            - ``"groups"``: mapping of group index to the starting column index of ``X``.
            - ``"group_sizes"``: mapping of group index to the group size.
            - ``"penalty"``: penalty factor for each group index.
    """
    assert n >= 1
    assert p >= 1
    assert G >= 1
    assert rho >= -1 / (p-1)
    assert (0 <= sparsity) & (sparsity <= 1)
    assert (0 <= zero_penalty) & (zero_penalty <= 1)
    assert snr > 0
    assert seed >= 0

    np.random.seed(seed)

    # define groups
    if equal_groups:
        equal_group_size = p // G
        groups = equal_group_size * np.arange(G)
    else:
        groups = np.concatenate([
            [0],
            np.random.choice(np.arange(1, p), size=G-1, replace=False)
        ])
        groups = np.sort(groups).astype(int)
    group_sizes = np.concatenate([groups, [p]], dtype=int)
    group_sizes = group_sizes[1:] - group_sizes[:-1]

    penalty = np.sqrt(group_sizes)
    penalty[np.random.choice(G, int(zero_penalty * G), replace=False)] = 0
    penalty /= np.linalg.norm(penalty) / np.sqrt(p)

    # generate raw data
    X = np.random.normal(0, 1, (n, p))
    Z = np.random.normal(0, 1, n)
    X = np.sqrt(rho) * Z[:, None] + np.sqrt(1-rho) * X
    X = np.asfortranarray(X)

    beta = np.random.normal(0, 1, p)
    beta_zero_indices = np.random.choice(p, int(sparsity * p), replace=False)
    beta_nnz_indices = np.array(list(set(np.arange(p)) - set(beta_zero_indices)))
    X_sub = X[:, beta_nnz_indices]
    beta_sub = beta[beta_nnz_indices]

    noise_scale = np.sqrt(
        (rho * np.sum(beta_sub) ** 2 + (1-rho) * np.sum(beta_sub ** 2))
        / snr
    )
    y = X_sub @ beta_sub + noise_scale * np.random.normal(0, 1, n)

    return {
        "X": X, 
        "y": y,
        "groups": groups,
        "group_sizes": group_sizes,
        "penalty": penalty,
    }


def create_snp_unphased(
    n: int, 
    p: int, 
    *,
    sparsity: float =0.95,
    one_ratio: float =0.25,
    two_ratio: float =0.05,
    zero_penalty: float =0,
    snr: float =1,
    seed: int =0,
):
    """Creates a SNP Unphased dataset.

    This dataset is only used for lasso, so ``groups`` is simply each individual feature
    and ``group_sizes`` is a vector of ones.
    The data matrix ``X`` has sparsity ratio ``1 - one_ratio - two_ratio``
    where ``one_ratio`` of the entries are randomly set to ``1``
    and ``two_ratio`` are randomly set to ``2``.
    The response ``y`` is generated from a linear model :math:`y = X\\beta + \\epsilon`
    with :math:`\\epsilon \\sim N(0, \\sigma^2 I_n)` 
    and :math:`\\beta` such that ``sparsity`` proportion of the entries are set to :math:`0`.
    We compute :math:`\\sigma^2` such that the signal-to-noise ratio is given by ``snr``.
    The penalty factors are by default set to ``np.sqrt(group_sizes)``,
    however if ``zero_penalty > 0``, a random set of penalties will be set to zero,
    in which case, ``penalty`` is rescaled such that the :math:`\\ell_2` norm squared is ``p``.

    Parameters
    ----------
    n : int
        Number of data points.
    p : int
        Number of SNPs.
    sparsity : float, optional
        Proportion of :math:`\\beta` entries to be zeroed out.
        Default is ``0.95``.
    one_ratio : float, optional
        Proportion of the entries of ``X`` that is set to ``1``.
        Default is ``0.25``.
    two_ratio : float, optional
        Proportion of the entries of ``X`` that is set to ``2``.
        Default is ``0.05``.
    zero_penalty : float, optional
        Proportion of ``penalty`` entries to be zeroed out.
        Default is ``0``.
    snr : float, optional
        Signal-to-noise ratio.
        Default is ``1``.
    seed : int, optional
        Random seed.
        Default is ``0``.

    Returns
    -------
    data : dict
        A dictionary containing the generated data:
        
            - ``"X"``: feature matrix.
            - ``"y"``: response vector.
            - ``"groups"``: mapping of group index to the starting column index of ``X``.
            - ``"group_sizes"``: mapping of group index to the group size.
            - ``"penalty"``: penalty factor for each group index.
    """
    assert n >= 1
    assert p >= 1
    assert sparsity >= 0 and sparsity <= 1
    assert one_ratio >= 0 and one_ratio <= 1
    assert two_ratio >= 0 and two_ratio <= 1
    assert zero_penalty >= 0 and zero_penalty <= 1
    assert snr > 0
    assert seed >= 0

    np.random.seed(seed)

    nnz_ratio = one_ratio + two_ratio
    one_ratio = one_ratio / nnz_ratio
    two_ratio = two_ratio / nnz_ratio
    nnz = int(nnz_ratio * n * p)
    nnz_indices = np.random.choice(n * p, nnz, replace=False)
    nnz_indices = np.random.permutation(nnz_indices)
    one_indices = nnz_indices[:int(one_ratio * nnz)]
    two_indices = nnz_indices[int(one_ratio * nnz):]
    X = np.zeros((n, p), dtype=np.int8)
    X.ravel()[one_indices] = 1
    X.ravel()[two_indices] = 2

    groups = np.arange(p)
    group_sizes = np.ones(p)

    penalty = np.sqrt(group_sizes)
    penalty[np.random.choice(p, int(zero_penalty * p), replace=False)] = 0
    penalty /= np.linalg.norm(penalty) / np.sqrt(p)

    beta = np.random.normal(0, 1, p)
    beta_nnz_indices = np.random.choice(p, int((1-sparsity) * p), replace=False)
    X_sub = X[:, beta_nnz_indices]
    beta_sub = beta[beta_nnz_indices]

    signal_var = np.dot(beta_sub, np.dot(np.cov(X_sub.T), beta_sub))
    noise_scale = np.sqrt(signal_var / snr)
    y = X_sub @ beta_sub + noise_scale * np.random.normal(0, 1, n)

    return {
        "X": X, 
        "y": y,
        "groups": groups,
        "group_sizes": group_sizes,
        "penalty": penalty,
    }


def create_snp_phased_ancestry(
    n: int, 
    s: int, 
    A: int,
    *,
    sparsity: float =0.95,
    one_ratio: float =0.25,
    two_ratio: float =0.05,
    zero_penalty: float =0,
    snr: float =1,
    seed: int =0,
):
    """Creates a SNP Unphased dataset.

    The data matrix ``X`` is a phased version of a matrix with sparsity ratio ``1 - one_ratio - two_ratio``
    where ``one_ratio`` of the entries are randomly set to ``1``
    and ``two_ratio`` are randomly set to ``2``.
    The response ``y`` is generated from a linear model :math:`y = X\\beta + \\epsilon`
    with :math:`\\epsilon \\sim N(0, \\sigma^2 I_n)` 
    and :math:`\\beta` such that ``sparsity`` proportion of the entries are set to :math:`0`.
    We compute :math:`\\sigma^2` such that the signal-to-noise ratio is given by ``snr``.
    The penalty factors are by default set to ``np.sqrt(group_sizes)``,
    however if ``zero_penalty > 0``, a random set of penalties will be set to zero,
    in which case, ``penalty`` is rescaled such that the :math:`\\ell_2` norm squared is ``p``.

    Parameters
    ----------
    n : int
        Number of data points.
    s : int
        Number of SNPs.
    A : int
        Number of ancestries.
    sparsity : float, optional
        Proportion of :math:`\\beta` entries to be zeroed out.
        Default is ``0.95``.
    one_ratio : float, optional
        Proportion of the entries of ``X`` that is set to ``1``.
        Default is ``0.25``.
    two_ratio : float, optional
        Proportion of the entries of ``X`` that is set to ``2``.
        Default is ``0.05``.
    zero_penalty : float, optional
        Proportion of ``penalty`` entries to be zeroed out.
        Default is ``0``.
    snr : float, optional
        Signal-to-noise ratio.
        Default is ``1``.
    seed : int, optional
        Random seed.
        Default is ``0``.

    Returns
    -------
    data : dict
        A dictionary containing the generated data:
        
            - ``"X"``: feature matrix.
            - ``"ancestries"``: ancestry label of the same shape as ``X``.
            - ``"y"``: response vector.
            - ``"groups"``: mapping of group index to the starting column index of ``X``.
            - ``"group_sizes"``: mapping of group index to the group size.
            - ``"penalty"``: penalty factor for each group index.
    """
    assert n >= 1
    assert s >= 1
    assert A >= 1
    assert sparsity >= 0 and sparsity <= 1
    assert one_ratio >= 0 and one_ratio <= 1
    assert two_ratio >= 0 and two_ratio <= 1
    assert zero_penalty >= 0 and zero_penalty <= 1
    assert snr > 0
    assert seed >= 0

    np.random.seed(seed)

    nnz_ratio = one_ratio + two_ratio
    one_ratio = one_ratio / nnz_ratio
    two_ratio = two_ratio / nnz_ratio
    nnz = int(nnz_ratio * n * s)
    nnz_indices = np.random.choice(n * s, nnz, replace=False)
    nnz_indices = np.random.permutation(nnz_indices)
    one_indices = nnz_indices[:int(one_ratio * nnz)]
    two_indices = nnz_indices[int(one_ratio * nnz):]
    one_indices = 2 * one_indices + np.random.binomial(1, 0.5, len(one_indices))
    two_indices *= 2
    X = np.zeros((n, 2 * s), dtype=np.int8)
    X.ravel()[one_indices] = 1
    X.ravel()[two_indices] = 1
    X.ravel()[two_indices + 1] = 1
    ancestries = np.zeros((n, 2 * s), dtype=np.int8)
    ancestries.ravel()[one_indices] = np.random.choice(A, len(one_indices), replace=True)
    ancestries.ravel()[two_indices] = np.random.choice(A, len(two_indices), replace=True)
    ancestries.ravel()[two_indices + 1] = np.random.choice(A, len(two_indices), replace=True)

    groups = np.arange(s)
    group_sizes = np.full(s, A)

    penalty = np.sqrt(group_sizes)
    penalty[np.random.choice(s, int(zero_penalty * s), replace=False)] = 0
    penalty /= np.linalg.norm(penalty) / np.sqrt(s * A)

    beta = np.random.normal(0, 1, 2 * s)
    beta_nnz_indices = np.random.choice(2 * s, int((1-sparsity) * s * 2), replace=False)
    X_sub = X[:, beta_nnz_indices]
    beta_sub = beta[beta_nnz_indices]

    signal_var = np.dot(beta_sub, np.dot(np.cov(X_sub.T), beta_sub))
    noise_scale = np.sqrt(signal_var / snr)
    y = X_sub @ beta_sub + noise_scale * np.random.normal(0, 1, n)

    return {
        "X": X, 
        "ancestries": ancestries,
        "y": y,
        "groups": groups,
        "group_sizes": group_sizes,
        "penalty": penalty,
    }