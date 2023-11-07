import numpy as np


def create_test_data_basil(
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
    """Creates a test dataset for BASIL method.

    The groups and group sizes are generated randomly
    such that ``G`` groups are created and the sum of the group sizes is ``p``.
    The data matrix ``X`` is generated from a normal distribution
    where each feature is equicorrelated with the other features by ``rho``.
    The response ``y`` is generated from a linear model :math:`y = X\\beta + \\epsilon`
    with :math:`\\epsilon \\sim N(0, \\sigma^2 I_n)` and :math:`\\beta` such that ``sparsity`` proportion
    of the entries are set to :math:`0`.
    We compute :math:`\\sigma^2`` such that the signal-to-noise ratio is given by ``snr``.
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
    beta[np.random.choice(p, int(sparsity * p), replace=False)] = 0

    noise_scale = np.sqrt(
        (rho * np.sum(beta) ** 2 + (1-rho) * np.sum(beta ** 2))
        / snr
    )
    y = X @ beta + noise_scale * np.random.normal(0, 1, n)

    return {
        "X": X, 
        "y": y,
        "groups": groups,
        "group_sizes": group_sizes,
        "penalty": penalty,
    }