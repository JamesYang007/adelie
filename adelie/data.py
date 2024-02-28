import adelie as ad
import numpy as np
import warnings 


def _sample_y(
    glm: str,
    eta: np.ndarray,
    beta: np.ndarray,
    rho: float =0,
    snr: float =1,
):
    n, K = eta.shape
    is_multi = "multi" in glm
    if not is_multi and K > 1:
        warnings.warn("Ignoring K and taking only first class response.")
        eta = eta[:, 0][:, None]
        K = 1

    if "gaussian" in glm:
        signal_scale = np.sqrt(rho * np.sum(beta) ** 2 + (1-rho) * np.sum(beta ** 2))
        noise_scale = signal_scale / np.sqrt(snr)
        y = eta + noise_scale * np.random.normal(0, 1, eta.shape)
        if not is_multi:
            y = y.ravel()
            return ad.glm.gaussian(y=y)
        return ad.glm.multigaussian(y=y)
    elif glm == "multinomial":
        mu = np.empty((n, K), dtype=eta.dtype)
        glm = ad.glm.multinomial(y=np.zeros(eta.shape)) 
        glm.gradient(
            snr * (eta / np.sqrt(np.sum(beta**2, axis=0))[None]), 
            mu,
        )
        mu *= -K * n
        y = np.array([
            np.random.multinomial(1, m)
            for m in mu
        ], dtype=eta.dtype)
        return ad.glm.multinomial(y=y)
    elif glm == "cox":
        signal_scale = np.sqrt(rho * np.sum(beta) ** 2 + (1-rho) * np.sum(beta ** 2))
        noise_scale = signal_scale / np.sqrt(snr)
        eta = eta.ravel()
        s = np.round(np.random.exponential(1, n))
        t = 1 + s + np.round(np.exp(eta / noise_scale + np.random.normal(0, 1, n)))
        C = 1 + s + np.round(np.exp(np.random.normal(0, 1, n)))
        d = t < C
        t = np.minimum(t, C)
        return ad.glm.cox(
            start=s,
            stop=t,
            status=d,
        )
    else:
        func_map = {
            "binomial": ad.glm.binomial,
            "poisson": ad.glm.poisson,
        }
        sample_map = {
            "binomial": lambda mu: np.random.binomial(1, mu),
            "poisson": lambda mu: np.random.poisson(mu),
        }
        glm_o = func_map[glm](y=np.zeros(n))
        eta = eta.ravel()
        mu = np.empty(eta.shape[0], dtype=eta.dtype)
        glm_o.gradient(snr * eta / np.sqrt(np.sum(beta**2)), mu)
        mu *= -n
        y = sample_map[glm](mu).astype(eta.dtype)
        return func_map[glm](y=y)


def dense(
    n: int, 
    p: int, 
    G: int,
    *,
    K: int=1,
    glm: str ="gaussian",
    equal_groups=False,
    rho: float =0,
    sparsity: float =0.95,
    zero_penalty: float =0,
    snr: float = 1,
    seed: int =0,
):
    """Creates a dense dataset.

    - The groups and group sizes are generated randomly
      such that ``G`` groups are created and the sum of the group sizes is ``p``.
    - The data matrix ``X`` is generated from a normal distribution
      where each feature is equicorrelated with the other features by ``rho``.
    - The true coefficients :math:`\\beta` are such that ``sparsity`` proportion
      of the entries are set to :math:`0`.
    - The response ``y`` is generated from the GLM specified by ``glm``.
    - The penalty factors are by default set to ``np.sqrt(group_sizes)``,
      however if ``zero_penalty > 0``, a random set of penalties will be set to zero,
      in which case, ``penalty`` is rescaled such that the :math:`\\ell_2` norm squared equals ``p``.

    Parameters
    ----------
    n : int
        Number of data points.
    p : int
        Number of features.
    G : int
        Number of groups.
    K : int, optional
        Number of classes for multi-response GLMs.
        Default is ``1``.
    glm : str, optional
        GLM name.
        It must be one of the following:

            - ``"binomial"``
            - ``"cox"``
            - ``"gaussian"``
            - ``"multigaussian"``
            - ``"multinomial"``
            - ``"poisson"``

        Default is ``"gaussian"``.
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

    beta = np.random.normal(0, 1, (p, K))
    beta_zero_indices = np.random.choice(p, int(sparsity * p), replace=False)
    beta_nnz_indices = np.array(list(set(np.arange(p)) - set(beta_zero_indices)))
    X_sub = X[:, beta_nnz_indices]
    beta_sub = beta[beta_nnz_indices]

    eta = X_sub @ beta_sub
    glm = _sample_y(
        glm=glm,
        eta=eta,
        beta=beta_sub,
        rho=rho,
        snr=snr,
    )

    return {
        "X": X, 
        "glm": glm,
        "groups": groups,
        "group_sizes": group_sizes,
        "penalty": penalty,
    }


def snp_unphased(
    n: int, 
    p: int, 
    *,
    K: int =1,
    glm: str ="gaussian",
    sparsity: float =0.95,
    one_ratio: float =0.25,
    two_ratio: float =0.05,
    zero_penalty: float =0,
    snr: float =1,
    seed: int =0,
):
    """Creates a SNP unphased dataset.

    - This dataset is only used for lasso, so ``groups`` is simply each individual feature
      and ``group_sizes`` is a vector of ones.
    - The calldata matrix ``X`` has sparsity ratio ``1 - one_ratio - two_ratio``
      where ``one_ratio`` of the entries are randomly set to ``1``
      and ``two_ratio`` are randomly set to ``2``.
    - The true coefficients :math:`\\beta` are such that ``sparsity`` proportion
      of the entries are set to :math:`0`.
    - The response ``y`` is generated from the GLM specified by ``glm``.
    - The penalty factors are by default set to ``np.sqrt(group_sizes)``,
      however if ``zero_penalty > 0``, a random set of penalties will be set to zero,
      in which case, ``penalty`` is rescaled such that the :math:`\\ell_2` norm squared is ``p``.

    Parameters
    ----------
    n : int
        Number of data points.
    p : int
        Number of SNPs.
    K : int, optional
        Number of classes for multi-response GLMs.
        Default is ``1``.
    glm : str, optional
        GLM name.
        It must be one of the following:

            - ``"binomial"``
            - ``"cox"``
            - ``"gaussian"``
            - ``"multigaussian"``
            - ``"multinomial"``
            - ``"poisson"``

        Default is ``"gaussian"``.
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
    group_sizes = np.ones(p, dtype=int)

    penalty = np.sqrt(group_sizes)
    penalty[np.random.choice(p, int(zero_penalty * p), replace=False)] = 0
    penalty /= np.linalg.norm(penalty) / np.sqrt(p)

    beta = np.random.normal(0, 1, (p, K))
    beta_nnz_indices = np.random.choice(p, int((1-sparsity) * p), replace=False)
    X_sub = X[:, beta_nnz_indices]
    beta_sub = beta[beta_nnz_indices]

    eta = X_sub @ beta_sub 
    glm = _sample_y(
        glm=glm,
        eta=eta,
        beta=beta_sub,
        snr=snr,
    )

    return {
        "X": np.asfortranarray(X), 
        "glm": glm,
        "groups": groups,
        "group_sizes": group_sizes,
        "penalty": penalty,
    }


def snp_phased_ancestry(
    n: int, 
    s: int, 
    A: int,
    *,
    K: int =1,
    glm: str ="gaussian",
    sparsity: float =0.95,
    one_ratio: float =0.25,
    two_ratio: float =0.05,
    zero_penalty: float =0,
    snr: float =1,
    seed: int =0,
):
    """Creates a SNP phased, ancestry dataset.

    - The groups and group sizes are generated randomly
      such that ``G`` groups are created and the sum of the group sizes is ``p``.
    - The calldata matrix ``X`` is a phased version of a matrix with sparsity ratio 
      ``1 - one_ratio - two_ratio``
      where ``one_ratio`` of the entries are randomly set to ``1``
      and ``two_ratio`` are randomly set to ``2``.
    - The ancestry matrix randomly generates integers in the range ``[0, A)``.
    - The true coefficients :math:`\\beta` is such that ``sparsity`` proportion of the entries are set to :math:`0`.
    - The response ``y`` is generated from the GLM specified by ``glm``.
    - The penalty factors are by default set to ``np.sqrt(group_sizes)``,
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
    K : int, optional
        Number of classes for multi-response GLMs.
        Default is ``1``.
    glm : str, optional
        GLM name.
        It must be one of the following:

            - ``"binomial"``
            - ``"cox"``
            - ``"gaussian"``
            - ``"multigaussian"``
            - ``"multinomial"``
            - ``"poisson"``

        Default is ``"gaussian"``.
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

    groups = A * np.arange(s)
    group_sizes = np.full(s, A, dtype=int)

    penalty = np.sqrt(group_sizes)
    penalty[np.random.choice(s, int(zero_penalty * s), replace=False)] = 0
    penalty /= np.linalg.norm(penalty) / np.sqrt(s * A)

    beta = np.random.normal(0, 1, (2 * s, K))
    beta_nnz_indices = np.random.choice(2 * s, int((1-sparsity) * s * 2), replace=False)
    X_sub = X[:, beta_nnz_indices]
    beta_sub = beta[beta_nnz_indices]

    eta = X_sub @ beta_sub 
    glm = _sample_y(
        glm=glm,
        eta=eta,
        beta=beta_sub,
        snr=snr,
    )

    return {
        "X": np.asfortranarray(X), 
        "ancestries": np.asfortranarray(ancestries),
        "glm": glm,
        "groups": groups,
        "group_sizes": group_sizes,
        "penalty": penalty,
    }
