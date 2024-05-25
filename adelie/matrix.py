from . import adelie_core as core
from . import io
from .adelie_core.matrix import (
    MatrixNaiveBase64,
    MatrixNaiveBase32,
    MatrixCovBase64,
    MatrixCovBase32,
)
from scipy.sparse import (
    csc_matrix,
    csr_matrix,
)
from typing import Union
import numpy as np
import warnings


# ------------------------------------------------------------------------
# Extra Python API
# ------------------------------------------------------------------------


def _to_dtype(mat):
    if (
        isinstance(mat, MatrixNaiveBase32) or
        isinstance(mat, MatrixCovBase32)
    ): 
        return np.float32
    elif (
        isinstance(mat, MatrixNaiveBase64) or
        isinstance(mat, MatrixCovBase64)
    ): 
        return np.float64
    return None


class PyMatrixCovBase:
    # TODO?
    def __init__(self, n_threads=1):
        self._n_threads = n_threads


class PyMatrixNaiveTranspose:
    def __init__(self, mat):
        self._mat = mat
        self.T = mat

    def __matmul__(self, v):
        dtype = _to_dtype(self._mat)
        v = np.asarray(v, dtype=dtype)

        if (len(v.shape) <= 0) or (len(v.shape) > 2):
            raise ValueError("Right argument must be either 1 or 2-dimensional.")

        n, p = self._mat.shape
        ones = np.ones(n, dtype=dtype)

        if len(v.shape) == 1:
            out = np.empty(p, dtype=dtype)
            self._mat.mul(v, ones, out)
            return out

        v = np.asfortranarray(v)
        out = np.empty((v.shape[1], p), dtype=dtype)
        for i in range(out.shape[0]):
            self._mat.mul(v[:, i], ones, out[i])
        return out.T


class PyMatrixNaiveBase:
    def __init__(self, n_threads=1):
        self._n_threads = n_threads
        self.T = PyMatrixNaiveTranspose(self)

    def __getitem__(self, key):
        valid_one_type = (int, np.integer, slice)
        valid_some_type = valid_one_type + (list, np.ndarray)
        if isinstance(key, tuple):
            if len(key) == 0:
                return self

            if len(key) > 2:
                raise ValueError(
                    "Key must be of length 1 or 2 if it is a tuple."
                )
            
            if (
                (len(key) == 2) and
                (not isinstance(key[0], valid_one_type)) and 
                (not isinstance(key[1], valid_one_type))
            ):
                raise ValueError(
                    "If row and column subsets are provided, "
                    "at least one must not be a list-like object. "
                )
        elif isinstance(key, valid_some_type):
            key = (key,)
        else:
            raise ValueError(
                "Subsets must be integer, slice, list, or np.ndarray objects."
            )

        def _convert_subset(s, size):
            if isinstance(s, (int, np.integer)):
                return np.array([s])
            elif isinstance(s, (list, np.ndarray)):
                if s.dtype == np.dtype("bool"):
                    s = np.where(s)[0]
                else:
                    s = np.array(s, dtype=int)
                    if np.unique(s).shape[0] != s.shape[0]:
                        raise ValueError(
                            "Subset does not contain unique elements."
                        )
                return s
            elif isinstance(s, slice):
                start = 0 if s.start is None else s.start
                stop = size if s.stop is None else s.stop
                step = 1 if s.step is None else s.step
                if (
                    (start != 0) or 
                    (stop != size) or
                    (step != 1)
                ):
                    return np.arange(start, stop, step=step)
                return None
            else:
                raise ValueError(
                    "Subsets must be integer, slice, list, or np.ndarray objects."
                )

        n, p = self.shape

        # process row subset
        row_subset = _convert_subset(key[0], n)
        this = self
        if len(key) == 2:
            # process column subset
            column_subset = _convert_subset(key[1], p)
            if not (column_subset is None):
                this = subset(
                    this, 
                    column_subset, 
                    axis=1, 
                    n_threads=self._n_threads,
                )

        if row_subset is None:
            return this
        return subset(
            this, 
            row_subset, 
            axis=0, 
            n_threads=self._n_threads,
        )

    def __matmul__(self, v):
        dtype = _to_dtype(self)
        n, p = self.shape

        if isinstance(v, (csr_matrix, csc_matrix)):
            v = v.tocsr().transpose()
            out = np.empty((v.shape[0], n), dtype=dtype)
            self.sp_btmul(v, out)
            return out.T

        v = np.asarray(v, dtype=dtype)

        if (len(v.shape) <= 0) or (len(v.shape) > 2):
            raise ValueError("Right argument must be either 1 or 2-dimensional.")

        if len(v.shape) == 1:
            out = np.zeros(n, dtype=dtype)
            self.btmul(0, p, v, out)
            return out

        v = np.asfortranarray(v)
        out = np.zeros((v.shape[1], n), dtype=dtype)
        for i in range(out.shape[0]):
            self.btmul(0, p, v[:, i], out[i])
        return out.T


# ------------------------------------------------------------------------
# Matrix Classes
# ------------------------------------------------------------------------


def block_diag(
    mats: list,
    *,
    n_threads: int =1,
):
    """Creates a block-diagonal matrix given by the list of matrices.

    If ``mats`` represents a list of matrices :math:`A_1,\\ldots, A_L`,
    then the resulting matrix represents the block-diagonal matrix given by

    .. math::
        \\begin{align*}
            \\begin{bmatrix}
                A_1 & 0 & \\cdots & 0 \\\\
                0 & A_2 & \\cdots & 0 \\\\
                \\vdots & \\vdots & \\ddots & \\vdots \\\\
                0 & 0 & \\cdots & A_L
            \\end{bmatrix}
        \\end{align*}

    .. note::
        This matrix only works for covariance method!

    Parameters
    ----------
    mats : list
        List of matrices to represent the block diagonal matrix.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.

    Returns
    -------
    wrap
        Wrapper matrix object.

    See Also
    --------
    adelie.adelie_core.matrix.MatrixCovBlockDiag64
    """
    mats = [
        dense(mat, method="cov", n_threads=1)
        if isinstance(mat, np.ndarray) else
        mat
        for mat in mats
    ]
    dtype = _to_dtype(mats[0])

    for mat in mats:
        if dtype == _to_dtype(mat): continue
        raise ValueError("All matrices must have the same underlying data type.")

    dispatcher = {
        np.float64: core.matrix.MatrixCovBlockDiag64,
        np.float32: core.matrix.MatrixCovBlockDiag32,
    }

    core_base = dispatcher[dtype]
    py_base = PyMatrixCovBase

    class _block_diag(core_base, py_base):
        def __init__(self):
            self._mats = mats
            core_base.__init__(self, self._mats, n_threads)
            py_base.__init__(self, n_threads=n_threads)

    return _block_diag()


def concatenate(
    mats: list,
    *,
    axis: int=0,
    n_threads: int =1,
):
    """Creates a concatenation of the matrices.

    If ``mats`` represents a list of matrices :math:`X_1,\\ldots, X_L`,
    then the resulting matrix represents the concatenated matrix along 
    the given axis ``axis``.
    
    If ``axis=0``, the matrix is concatenated row-wise:

    .. math::
        \\begin{align*}
            \\begin{bmatrix}
                \\unicode{x2014} & X_1 & \\unicode{x2014} \\\\
                \\unicode{x2014} & X_2 & \\unicode{x2014} \\\\
                \\vdots & \\vdots & \\vdots \\\\
                \\unicode{x2014} & X_L & \\unicode{x2014}
            \\end{bmatrix}
        \\end{align*}

    If ``axis=1``, the matrix is concatenated column-wise:

    .. math::
        \\begin{align*}
            \\begin{bmatrix}
                \\vert & \\vert & \\cdots & \\vert \\\\
                X_1 & X_2 & \\cdots & X_L \\\\
                \\vert & \\vert & \\cdots & \\vert
            \\end{bmatrix}
        \\end{align*}

    .. note::
        This matrix only works for naive method!

    Parameters
    ----------
    mats : list
        List of matrices to concatenate.
    axis : int, optional
        The axis along which the matrices will be joined.
        Default is ``0``.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.

    Returns
    -------
    wrap
        Wrapper matrix object.

    See Also
    --------
    adelie.adelie_core.matrix.MatrixNaiveCConcatenate64
    adelie.adelie_core.matrix.MatrixNaiveRConcatenate64
    """
    mats = [
        dense(mat, method="naive", n_threads=n_threads)
        if isinstance(mat, np.ndarray) else
        mat
        for mat in mats
    ]
    dtype = _to_dtype(mats[0])

    for mat in mats:
        if dtype == _to_dtype(mat): continue
        raise ValueError("All matrices must have the same underlying data type.")

    cdispatcher = {
        np.float64: core.matrix.MatrixNaiveCConcatenate64,
        np.float32: core.matrix.MatrixNaiveCConcatenate32,
    }
    rdispatcher = {
        np.float64: core.matrix.MatrixNaiveRConcatenate64,
        np.float32: core.matrix.MatrixNaiveRConcatenate32,
    }
    dispatcher = {
        0: rdispatcher,
        1: cdispatcher,
    }

    core_base = dispatcher[axis][dtype]
    py_base = PyMatrixNaiveBase

    class _concatenate(core_base, py_base):
        def __init__(self):
            self._mats = mats
            core_base.__init__(self, self._mats)
            py_base.__init__(self, n_threads=n_threads)

    return _concatenate()


def dense(
    mat: np.ndarray,
    *,
    method: str ="naive",
    copy: bool =False,
    n_threads: int =1,
):
    """Creates a viewer of a dense matrix.
    
    Parameters
    ----------
    mat : np.ndarray
        The dense matrix to view.
    method : str, optional
        Method type. It must be one of the following:

            - ``"naive"``: naive method.
            - ``"cov"``: covariance method.

        Default is ``"naive"``.
    copy : bool, optional
        If ``True``, a copy of ``mat`` is stored internally.
        Otherwise, a reference is stored instead.
        Default is ``False``.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.

    Returns
    -------
    wrap
        Wrapper matrix object.

    See Also
    --------
    adelie.adelie_core.matrix.MatrixCovDense64F
    adelie.adelie_core.matrix.MatrixNaiveDense64F
    """
    naive_dispatcher = {
        np.dtype("float64"): {
            "C": core.matrix.MatrixNaiveDense64C,
            "F": core.matrix.MatrixNaiveDense64F,
        },
        np.dtype("float32"): {
            "C": core.matrix.MatrixNaiveDense32C,
            "F": core.matrix.MatrixNaiveDense32F,
        },
    }

    cov_dispatcher = {
        np.dtype("float64"): {
            "C": core.matrix.MatrixCovDense64C,
            "F": core.matrix.MatrixCovDense64F,
        },
        np.dtype("float32"): {
            "C": core.matrix.MatrixCovDense32C,
            "F": core.matrix.MatrixCovDense32F,
        },
    }

    dispatcher = {
        "naive" : naive_dispatcher,
        "cov" : cov_dispatcher,
    }

    dtype = mat.dtype
    order = (
        "F"
        # prioritize choosing Fortran contiguity
        if mat.flags.f_contiguous else
        "C"
    )
    if order == "C":
        warnings.warn(
            "Detected matrix to be C-contiguous. "
            "Performance may improve with F-contiguous matrix."
        )
    core_base = dispatcher[method][dtype][order]

    py_base = {
        "naive" : PyMatrixNaiveBase,
        "cov" : PyMatrixCovBase,
    }[method]

    class _dense(core_base, py_base):
        def __init__(self):
            self._mat = np.array(mat, copy=copy)
            core_base.__init__(self, self._mat, n_threads)
            py_base.__init__(self, n_threads=n_threads)

    return _dense()


def interaction(
    mat: np.ndarray,
    intr_map: dict,
    levels: np.ndarray =None,
    *,
    copy: bool =False,
    n_threads: int =1,
):
    """Creates a matrix with pairwise interactions.

    This matrix :math:`X \\in \\mathbb{R}^{n\\times p}` represents pairwise interaction terms
    within a given base matrix :math:`Z \\in \\mathbb{R}^{n\\times d}` 
    where the interaction structure is defined as follows.
    We assume :math:`Z` contains, in general, a combination of
    continuous and discrete features (as columns).
    Denote :math:`L : \\{1,\\ldots, d\\} \\to \\mathbb{N}` as
    the mapping that maps each feature index of :math:`Z` to the number of levels of that feature
    where a value of :math:`0` means the feature is continuous
    and otherwise means it is discrete with that many levels (or categories).
    Let :math:`S \\subseteq \\{1,\\ldots, d\\}^2` denote the set of
    valid and unique pairs of feature indices of :math:`Z`.
    A pair is *valid* if the two values are not equal.
    We define uniqueness up to ordering so that :math:`(x,y)` and :math:`(y,x)` are considered the same pairs.
    Finally, for each pair :math:`(i, j) \\in S`,
    define the interaction term :math:`Z_{i:j}` as

    .. math::
        \\begin{align*}
            Z_{i:j}
            &:=
            \\begin{cases}
                \\begin{bmatrix}
                    Z_{i} & Z_{j} & Z_i \\odot Z_j
                \\end{bmatrix}
                ,& L(i) = 0, L(j) = 0 \\\\
                \\begin{bmatrix}
                    \\mathbf{1} & Z_{i}
                \\end{bmatrix}
                \\star
                I_{Z_{j}}
                ,& L(i) = 0, L(j) > 0 \\\\
                I_{Z_{i}}
                \\star
                \\begin{bmatrix}
                    \\mathbf{1} & Z_{j}
                \\end{bmatrix}
                ,& L(i) > 0, L(j) = 0 \\\\
                I_{Z_{i}}
                \\star
                I_{Z_{j}}
                ,& L(i) > 0, L(j) > 0
            \\end{cases}
        \\end{align*}

    Here, :math:`Z_i` is the :math:`i` th column of :math:`Z`,
    :math:`I_{v}` is the indicator matrix, or one-hot encoding, of :math:`v`,
    and for any two matrices :math:`A \\in \\mathbb{R}^{n\\times d_A}`, :math:`B \\in \\mathbb{R}^{n\\times d_B}`,

    .. math::
        \\begin{align*}
            A \\star B
            &=
            \\begin{bmatrix}
                A_{1} \\odot B_{1} &
                \\cdots &
                A_{d_A} \\odot B_{1} & 
                A_{1} \\odot B_{2} &
                \\cdots &
                A_{d_A} \\odot B_{2} &
                \\cdots
            \\end{bmatrix}
        \\end{align*}

    Then, :math:`X` is defined as the column-wise concatenation of :math:`Z_{i:j}`
    in lexicographical order of :math:`(i,j) \\in S`.

    .. note::
        Every discrete feature of `Z` *must* take on values in the set 
        :math:`\\{0, \\ldots, \\ell-1\\}`
        where :math:`\\ell` is the number of levels for that feature.

    .. note::
        This matrix only works for naive method!

    Parameters
    ----------
    mat : (n, d) np.ndarray
        The dense matrix :math:`Z` from which to construct interaction terms.
    intr_map : dict
        Dictionary mapping a column index of ``mat``
        to a list of (column) indices to pair with.
        If the value of a key-value pair is ``None``,
        then every column is paired with the key.
        Internally, only valid and unique (as defined above) pairs are registered
        to construct :math:`S`.
        Moreover, the pairs are stored in lexicographical order of ``(key, val)``
        for each ``val`` in ``intr_map[key]`` and for each ``key``.
    levels : (d,) np.ndarray, optional
        Number of levels for each column in ``mat``.
        A non-positive value indicates that the column is a continuous variable
        whereas a positive value indicates that it is a discrete variable with
        that many levels (or categories).
        If ``None``, it is initialized to be ``np.zeros(d)``
        so that every column is a continuous variable.
        Default is ``None``.
    copy : bool, optional
        If ``True``, a copy of ``mat`` is stored internally.
        Otherwise, a reference is stored instead.
        Default is ``False``.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.

    Returns
    -------
    wrap
        Wrapper matrix object.

    See Also
    --------
    adelie.adelie_core.matrix.MatrixNaiveInteractionDense64F
    """
    dispatcher = {
        np.dtype("float64"): {
            "C": core.matrix.MatrixNaiveInteractionDense64C,
            "F": core.matrix.MatrixNaiveInteractionDense64F,
        },
        np.dtype("float32"): {
            "C": core.matrix.MatrixNaiveInteractionDense32C,
            "F": core.matrix.MatrixNaiveInteractionDense32F,
        },
    }
    dtype = mat.dtype
    order = (
        "F"
        # prioritize choosing Fortran contiguity
        if mat.flags.f_contiguous else
        "C"
    )
    if order == "C":
        warnings.warn(
            "Detected matrix to be C-contiguous. "
            "Performance may improve with F-contiguous matrix."
        )
    core_base = dispatcher[dtype][order]
    py_base = PyMatrixNaiveBase

    _, d = mat.shape

    if levels is None:
        levels = np.zeros(d, dtype=int)

    if len(intr_map) <= 0:
        raise ValueError("intr_map must be non-empty.")
    arange_d = np.arange(d)
    keys = np.sort(list(intr_map.keys()))
    pairs_seen = set()
    pairs = []
    for key in keys:
        if (key < 0) or (key >= d):
            warnings.warn(f"key not in range [0,{d}): {key}.")
        value_lst = intr_map[key]
        if value_lst is None: 
            value_lst = arange_d
        else:
            value_lst = np.sort(np.unique(value_lst))

        for val in value_lst:
            if (
                ((key, val) in pairs_seen) or
                ((val, key) in pairs_seen) or
                (key == val)
            ): 
                continue
            if (val < 0) or (val >= d):
                warnings.warn(f"value not in range [0,{d}): {val}.")
            pairs.append((key, val))
            pairs_seen.add((key, val))
    if len(pairs) <= 0:
        raise ValueError("No valid pairs exist. There must be at least one valid pair.")
    pairs = np.array(pairs, dtype=np.int32)

    class _interaction(core_base, py_base):
        def __init__(self):
            self._mat = np.array(mat, copy=copy)
            self._pairs = pairs
            self._levels = np.array(levels, copy=True, dtype=np.int32)
            core_base.__init__(self, self._mat, self._pairs, self._levels, n_threads)
            py_base.__init__(self, n_threads=n_threads)
        
    return _interaction()


def kronecker_eye(
    mat: Union[np.ndarray, MatrixNaiveBase32, MatrixNaiveBase64],
    K: int,
    *,
    copy: bool =False,
    n_threads: int =1,
):
    """Creates a Kronecker product with identity matrix.

    The matrix is represented as :math:`X \\otimes I_K`
    where :math:`X` is the underlying dense matrix and 
    :math:`I_K` is the identity matrix of dimension :math:`K`.

    .. note::
        This matrix only works for naive method!
    
    Parameters
    ----------
    mat : Union[np.ndarray, MatrixNaiveBase32, MatrixNaiveBase64]
        The matrix to view as a Kronecker product with identity matrix.
        If ``np.ndarray``, a specialized class is created with more optimized routines.
    K : int
        Dimension of the identity matrix.
    copy : bool, optional
        This argument is only used if ``mat`` is a ``np.ndarray``.
        If ``True``, a copy of ``mat`` is stored internally.
        Otherwise, a reference is stored instead.
        Default is ``False``.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.

    Returns
    -------
    wrap
        Wrapper matrix object.

    See Also
    --------
    adelie.adelie_core.matrix.MatrixNaiveKroneckerEye64
    adelie.adelie_core.matrix.MatrixNaiveKroneckerEyeDense64F
    """
    if isinstance(mat, np.ndarray):
        dispatcher = {
            np.dtype("float64"): {
                "C": core.matrix.MatrixNaiveKroneckerEyeDense64C,
                "F": core.matrix.MatrixNaiveKroneckerEyeDense64F,
            },
            np.dtype("float32"): {
                "C": core.matrix.MatrixNaiveKroneckerEyeDense32C,
                "F": core.matrix.MatrixNaiveKroneckerEyeDense32F,
            },
        }
        dtype = mat.dtype
        order = (
            "C"
            if mat.flags.c_contiguous else
            "F"
        )
        core_base = dispatcher[dtype][order]
        mat = np.array(mat, copy=copy)
    else:
        dispatcher = {
            np.float64: core.matrix.MatrixNaiveKroneckerEye64,
            np.float32: core.matrix.MatrixNaiveKroneckerEye32,
        }
        dtype = _to_dtype(mat)
        core_base = dispatcher[dtype]

    py_base = PyMatrixNaiveBase
    class _kronecker_eye(core_base, py_base):
        def __init__(self):
            self._mat = mat
            core_base.__init__(self, self._mat, K, n_threads)
            py_base.__init__(self, n_threads=n_threads)

    return _kronecker_eye()


def one_hot(
    mat: np.ndarray,
    levels: np.ndarray =None,
    *,
    copy: bool =False,
    n_threads: int =1,
):
    """Creates a one-hot encoded matrix.

    This matrix :math:`X \\in \\mathbb{R}^{n \\times p}`
    represents a one-hot encoding of a given base matrix
    :math:`Z \\in \\mathbb{R}^{n \\times d}`.
    We assume :math:`Z` contains, in general, a combination of
    continuous and discrete features (as columns).
    Denote :math:`L : \\{1, \\ldots, d\\} \\to \\mathbb{N}` as
    the mapping that maps each feature index of :math:`Z` to the number of levels of that feature
    where a value of :math:`0` means the feature is continuous
    and otherwise means it is discrete with that many levels (or categories).
    For every :math:`j` th column of :math:`Z`,
    define the possibly one-hot encoded version :math:`\\tilde{Z}_j` as

    .. math::
        \\begin{align*}
            \\tilde{Z}_j
            &:=
            \\begin{cases}
                Z_j ,& L(j) = 0 \\\\
                I_{Z_j} ,& L(j) > 0
            \\end{cases}
        \\end{align*}

    Here, :math:`I_{v}` is the indicator matrix, or one-hot encoding, of :math:`v`.

    Then, :math:`X` is defined as the column-wise concatenation of :math:`\\tilde{Z}_j`
    in order of :math:`j`.

    .. note::
        Every discrete feature of `Z` *must* take on values in the set 
        :math:`\\{0, \\ldots, \\ell-1\\}`
        where :math:`\\ell` is the number of levels for that feature.

    .. note::
        This matrix only works for naive method!
    
    Parameters
    ----------
    mat : (n, d) np.ndarray
        The dense matrix :math:`Z` from which to construct one-hot encodings.
    levels : (d,) np.ndarray, optional
        Number of levels for each column in ``mat``.
        A non-positive value indicates that the column is a continuous variable
        whereas a positive value indicates that it is a discrete variable with
        that many levels (or categories).
        If ``None``, it is initialized to be ``np.zeros(d)``
        so that every column is a continuous variable.
        Default is ``None``.
    copy : bool, optional
        If ``True``, a copy of ``mat`` is stored internally.
        Otherwise, a reference is stored instead.
        Default is ``False``.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.

    Returns
    -------
    wrap
        Wrapper matrix object.

    See Also
    --------
    adelie.adelie_core.matrix.MatrixNaiveOneHotDense64F
    """
    dispatcher = {
        np.dtype("float64"): {
            "C": core.matrix.MatrixNaiveOneHotDense64C,
            "F": core.matrix.MatrixNaiveOneHotDense64F,
        },
        np.dtype("float32"): {
            "C": core.matrix.MatrixNaiveOneHotDense32C,
            "F": core.matrix.MatrixNaiveOneHotDense32F,
        },
    }
    dtype = mat.dtype
    order = (
        "F"
        # prioritize choosing Fortran contiguity
        if mat.flags.f_contiguous else
        "C"
    )
    if order == "C":
        warnings.warn(
            "Detected matrix to be C-contiguous. "
            "Performance may improve with F-contiguous matrix."
        )
    core_base = dispatcher[dtype][order]
    py_base = PyMatrixNaiveBase

    _, d = mat.shape

    if levels is None:
        levels = np.zeros(d, dtype=int)

    class _one_hot(core_base, py_base):
        def __init__(self):
            self._mat = np.array(mat, copy=copy)
            self._levels = np.array(levels, copy=True, dtype=np.int32)
            core_base.__init__(self, self._mat, self._levels, n_threads)
            py_base.__init__(self, n_threads=n_threads)
        
    return _one_hot()


def lazy_cov(
    mat: np.ndarray,
    *,
    copy: bool =False,
    n_threads: int =1,
):
    """Creates a lazy covariance matrix.

    The lazy covariance matrix :math:`A` uses 
    the underlying matrix :math:`X` given by ``mat``
    to compute the values of :math:`A` dynamically.
    It only computes rows of :math:`A`
    on-the-fly that are needed when calling its member functions.
    This is useful in ``adelie.solver.gaussian_cov`` where
    the covariance method must be used but the dimensions of :math:`A`
    is too large to construct the entire matrix as a dense matrix.
    
    .. note::
        This matrix only works for covariance method!

    Parameters
    ----------
    mat : (n, p) np.ndarray
        The data matrix from which to lazily compute the covariance.
    copy : bool, optional
        If ``True``, a copy of ``mat`` is stored internally.
        Otherwise, a reference is stored instead.
        Default is ``False``.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.

    Returns
    -------
    wrap
        Wrapper matrix object.

    See Also
    --------
    adelie.adelie_core.matrix.MatrixCovLazyCov64F
    """
    dispatcher = {
        np.dtype("float64"): {
            "C": core.matrix.MatrixCovLazyCov64C,
            "F": core.matrix.MatrixCovLazyCov64F,
        },
        np.dtype("float32"): {
            "C": core.matrix.MatrixCovLazyCov32C,
            "F": core.matrix.MatrixCovLazyCov32F,
        },
    }

    dtype = mat.dtype
    order = (
        "C"
        if mat.flags.c_contiguous else
        "F"
    )
    core_base = dispatcher[dtype][order]
    py_base = PyMatrixCovBase

    class _lazy_cov(core_base, py_base):
        def __init__(self):
            self._mat = np.array(mat, copy=copy)
            core_base.__init__(self, self._mat, n_threads)
            py_base.__init__(self, n_threads=n_threads)

    return _lazy_cov()


def snp_phased_ancestry(
    io: io.snp_phased_ancestry,
    *,
    n_threads: int =1,
    dtype: Union[np.float32, np.float64] =np.float64,
):
    """Creates a SNP phased, ancestry matrix.

    The SNP phased, ancestry matrix is a wrapper around 
    the corresponding IO handler ``adelie.io.snp_phased_ancestry``
    exposing some matrix operations.

    .. note::
        This matrix only works for naive method!
    
    Parameters
    ----------
    io : adelie.io.snp_phased_ancestry
        IO handler for SNP phased, ancestry data.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.
    dtype : Union[np.float32, np.float64], optional
        Underlying value type.
        Default is ``np.float64``.

    Returns
    -------
    wrap
        Wrapper matrix object.

    See Also
    --------
    adelie.io.snp_phased_ancestry
    adelie.adelie_core.matrix.MatrixNaiveSNPPhasedAncestry64
    """
    dispatcher = {
        np.float64: core.matrix.MatrixNaiveSNPPhasedAncestry64,
        np.float32: core.matrix.MatrixNaiveSNPPhasedAncestry32,
    }
    core_base = dispatcher[dtype]
    py_base = PyMatrixNaiveBase

    if not io.is_read:
        io.read() 

    class _snp_phased_ancestry(core_base, py_base):
        def __init__(self):
            self._io = io
            core_base.__init__(self, self._io, n_threads)
            py_base.__init__(self, n_threads=n_threads)

    return _snp_phased_ancestry()


def snp_unphased(
    io: io.snp_unphased,
    *,
    n_threads: int =1,
    dtype: Union[np.float32, np.float64] =np.float64,
):
    """Creates a SNP unphased matrix.

    The SNP unphased matrix is a wrapper around    
    the corresponding IO handler ``adelie.io.snp_unphased``
    exposing some matrix operations.

    .. note::
        This matrix only works for naive method!
    
    Parameters
    ----------
    io : adelie.io.snp_unphased
        IO handler for SNP unphased data.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.
    dtype : Union[np.float32, np.float64], optional
        Underlying value type.
        Default is ``np.float64``.

    Returns
    -------
    wrap
        Wrapper matrix object.

    See Also
    --------
    adelie.io.snp_unphased
    adelie.adelie_core.matrix.MatrixNaiveSNPUnphased64
    """
    dispatcher = {
        np.float64: core.matrix.MatrixNaiveSNPUnphased64,
        np.float32: core.matrix.MatrixNaiveSNPUnphased32,
    }
    core_base = dispatcher[dtype]
    py_base = PyMatrixNaiveBase

    if not io.is_read:
        io.read()

    class _snp_unphased(core_base, py_base):
        def __init__(self):
            self._io = io
            core_base.__init__(self, self._io, n_threads)
            py_base.__init__(self, n_threads=n_threads)

    return _snp_unphased()


def sparse(
    mat: Union[csc_matrix, csr_matrix],
    *,
    method: str ="naive",
    copy: bool =False,
    n_threads: int =1,
):
    """Creates a viewer of a sparse matrix.

    .. note::
        Regardless of the storage order of the input matrix,
        we internally convert the matrix to CSC order for performance reasons.
    
    Parameters
    ----------
    mat : Union[scipy.sparse.csc_matrix, scipy.sparse.csr_matrix]
        The sparse matrix to view.
    method : str, optional
        Method type. It must be one of the following:

            - ``"naive"``: naive method.
            - ``"cov"``: covariance method.

        Default is ``"naive"``.
    copy : bool, optional
        If ``True``, a copy of ``mat`` is stored internally.
        Otherwise, a reference is stored instead.
        Default is ``False``.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.

    Returns
    -------
    wrap
        Wrapper matrix object.

    See Also
    --------
    adelie.adelie_core.matrix.MatrixCovSparse64F
    adelie.adelie_core.matrix.MatrixNaiveSparse64F
    """
    if not (isinstance(mat, csr_matrix) or isinstance(mat, csc_matrix)):
        raise TypeError("mat must be scipy.sparse.csr_matrix or scipy.sparse.csc_matrix.")

    if isinstance(mat, csr_matrix):
        warnings.warn("Converting to CSC format.")
        mat = mat.tocsc(copy=True)
    elif copy:
        mat = mat.copy()

    mat.prune()
    mat.sort_indices()

    naive_dispatcher = {
        np.dtype("float64"): core.matrix.MatrixNaiveSparse64F,
        np.dtype("float32"): core.matrix.MatrixNaiveSparse32F,
    }
    
    cov_dispatcher = {
        np.dtype("float64"): core.matrix.MatrixCovSparse64F,
        np.dtype("float32"): core.matrix.MatrixCovSparse32F,
    }

    dispatcher = {
        "naive" : naive_dispatcher,
        "cov" : cov_dispatcher,
    }

    dtype = mat.dtype
    core_base = dispatcher[method][dtype]
    py_base = {
        "naive" : PyMatrixNaiveBase,
        "cov" : PyMatrixCovBase,
    }[method]

    class _sparse(core_base, py_base):
        def __init__(self):
            self._mat = mat
            core_base.__init__(
                self, 
                self._mat.shape[0], 
                self._mat.shape[1], 
                self._mat.nnz,
                self._mat.indptr,
                self._mat.indices,
                self._mat.data,
                n_threads,
            )
            py_base.__init__(self, n_threads=n_threads)

    return _sparse()


def standardize(
    mat: Union[np.ndarray, MatrixNaiveBase32, MatrixNaiveBase64],
    centers: np.ndarray =None,
    scales: np.ndarray =None,
    ddof: int =0,
    *,
    n_threads: int =1,
):
    """Creates a standardized matrix.

    Given a matrix :math:`Z \\in \\mathbb{R}^{n \\times p}`,
    the standardized matrix :math:`X \\in \\mathbb{R}^{n \\times p}`
    of :math:`Z` centered by :math:`c \\in \\mathbb{R}^{p}` and scaled by :math:`s \\in \\mathbb{R}^p`
    is defined by 
    
    .. math::
        \\begin{align*}
            X = (Z - \\mathbf{1} c^\\top) \\mathrm{diag}(s)^{-1}
        \\end{align*}

    We define the column means :math:`\\overline{Z} \\in \\mathbb{R}^p` to be

    .. math::
        \\begin{align*}
            \\overline{Z}
            =
            \\frac{1}{n} Z^\\top \\mathbf{1}
        \\end{align*}

    Lastly, we define the column standard deviations :math:`\\hat{\\sigma} \\in \\mathbb{R}^p` to be

    .. math::
        \\begin{align*}
            \\hat{\\sigma}_j
            =
            \\frac{1}{\\sqrt{n - \\mathrm{df}}} \\|Z_{\\cdot j} - c_j \\mathbf{1} \\|_2
        \\end{align*}

    where :math:`\\mathrm{df}` is the degrees of freedom given by ``ddof``.

    .. note::
        This matrix only works for naive method!
    
    Parameters
    ----------
    mat : Union[np.ndarray, MatrixNaiveBase32, MatrixNaiveBase64]
        The underlying matrix :math:`Z` to standardize.
    centers : np.ndarray, optional
        The center values :math:`c` for each column of ``mat``.
        If ``None``, the column means :math:`\\overline{Z}` are used as centers.
        Default is ``None``.
    scales : np.ndarray, optional
        The scale values :math:`s` for each column of ``mat``.
        If ``None``, the column standard deviations :math:`\\hat{\\sigma}` are used as scales.
        Default is ``None``.
    ddof : int, optional
        The degrees of freedom used to compute ``scales``.
        This is only used if ``scales`` is ``None``.
        Default is ``0``.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.

    Returns
    -------
    wrap
        Wrapper matrix object.

    See Also
    --------
    adelie.adelie_core.matrix.MatrixNaiveStandardize64
    """
    if isinstance(mat, (list, np.ndarray)):
        mat = np.ndarray(mat, order="F", copy=True)
        if centers is None:
            centers = np.mean(mat, axis=0)
        mat -= centers[None]
        if scales is None:
            n = mat.shape[0]
            scales = np.sqrt(
                np.sum(mat ** 2, axis=0) / (n - ddof)
            )
        mat /= scales[None]
        return mat

    dtype = _to_dtype(mat)
    dispatcher = {
        np.float32: core.matrix.MatrixNaiveStandardize32,
        np.float64: core.matrix.MatrixNaiveStandardize64,
    }
    core_base = dispatcher[dtype]
    py_base = PyMatrixNaiveBase

    n, p = mat.shape
    sqrt_weights = np.full(n, 1/np.sqrt(n), dtype=dtype) 
    is_centers_none = centers is None

    if is_centers_none:
        centers = np.empty(p, dtype=dtype) 
        mat.mul(sqrt_weights, sqrt_weights, centers)

    if scales is None:
        if is_centers_none:
            means = centers
        else:
            means = np.empty(p, dtype=dtype) 
            mat.mul(sqrt_weights, sqrt_weights, means)

        vars = np.empty((p, 1, 1), dtype=dtype, order="F")
        buffer = np.empty((n, 1), dtype=dtype, order="F")
        for j in range(p):
            mat.cov(j, 1, sqrt_weights, vars[j], buffer) 
        vars = vars.reshape(p)
        vars += centers * (centers - 2 * means)
        scales = np.sqrt((n / (n - ddof)) * vars)

    class _standardize(core_base, py_base):
        def __init__(self):
            self._mat = mat
            self._centers = np.array(centers, copy=True, dtype=dtype)
            self._scales = np.array(scales, copy=True, dtype=dtype)
            core_base.__init__(self, self._mat, self._centers, self._scales, n_threads)
            py_base.__init__(self, n_threads=n_threads)
        
    return _standardize()


def subset(
    mat: Union[np.ndarray, MatrixNaiveBase32, MatrixNaiveBase64],
    indices: np.ndarray,
    *,
    axis: int=0,
    n_threads: int =1,
):
    """Creates a subset of the matrix along an axis.

    If ``axis=0``, then ``mat`` is subsetted 
    along the rows as if we had done ``X[indices]`` for numpy arrays.
    If ``axis=1``, then it is subsetted
    along the columns as if we had done ``X[:, indices]`` for numpy arrays.

    For syntactic sugar, the above numpy syntax works for all naive matrix classes.
    That is, if ``mat`` is a ``MatrixNaiveBaseXX``,
    then ``mat[indices]`` and ``mat[:, indices]`` yield the same return values as 
    ``subset(mat, indices, axis=0, n_threads=n_threads)`` and
    ``subset(mat, indices, axis=1, n_threads=n_threads)``, respectively,
    where ``n_threads`` is deduced from ``mat``.
    This function allows the user to further specify the number of threads.

    .. note::
        This matrix only works for naive method!

    .. warning::
        For users intending to subset rows of ``mat``
        and pass the subsetted matrix to our group elastic net solver,
        it is much more efficient to rather set observation weights 
        along ``indices`` to ``0`` when supplying the GLM object.
        For example, suppose the user wishes to run 
        ``adelie.solver.grpnet`` with ``mat`` and ``ad.glm.gaussian(y)``
        but subsetting the samples along ``indices``.
        Then, instead of supplying ``mat[indices]`` and ``ad.glm.gaussian(y[indices])``,
        we recommend creating a weight vector ``w`` where it is ``0`` outside ``indices``
        and supply ``mat`` and ``ad.glm.gaussian(y, weights=w)``.

    Parameters
    ----------
    mat : Union[np.ndarray, MatrixNaiveBase32, MatrixNaiveBase64]
        The matrix to subset.
    indices : np.ndarray
        Array of indices to subset the matrix.
    axis : int, optional
        The axis along which to subset.
        Default is ``0``.
    n_threads : int, optional
        Number of threads.
        Default is ``1``.

    Returns
    -------
    wrap
        Wrapper matrix object.
        If ``mat`` is ``np.ndarray`` then the usual numpy subsetted matrix is returned.

    See Also
    --------
    adelie.adelie_core.matrix.MatrixNaiveCSubset64
    adelie.adelie_core.matrix.MatrixNaiveRSubset64
    """
    if isinstance(mat, np.ndarray):
        if axis == 0:
            return mat[indices]
        else:
            return mat[:, indices]

    dtype = _to_dtype(mat)

    cdispatcher = {
        np.float64: core.matrix.MatrixNaiveCSubset64,
        np.float32: core.matrix.MatrixNaiveCSubset32,
    }
    rdispatcher = {
        np.float64: core.matrix.MatrixNaiveRSubset64,
        np.float32: core.matrix.MatrixNaiveRSubset32,
    }
    dispatcher = {
        0: rdispatcher,
        1: cdispatcher,
    }

    core_base = dispatcher[axis][dtype]
    py_base = PyMatrixNaiveBase

    class _subset(core_base, py_base):
        def __init__(self):
            self._mat = mat
            self._indices = np.array(indices, copy=True, dtype=np.int32)
            core_base.__init__(self, self._mat, self._indices, n_threads)
            py_base.__init__(self, n_threads=n_threads)
        
    return _subset()