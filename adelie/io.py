from .adelie_core import io as core_io
from typing import Union
import numpy as np


class snp_phased_ancestry(core_io.IOSNPPhasedAncestry):
    """IO handler for SNP phased, ancestry matrix.

    A SNP phased, ancestry matrix is a matrix 
    that combines a (phased) calldata and local ancestry information.

    Let :math:`X \\in \\mathbb{R}^{n \\times sA}` denote such a matrix
    where
    :math:`n` is the number of samples,
    :math:`s` is the number of SNPs,
    and :math:`A` is the number of ancestries.
    Every :math:`A` (contiguous) columns is called an
    *ancestry block* corresponding to a single SNP
    with the same structure described next.
    Let :math:`H \\in \\mathbb{R}^{n \\times A}` denote any such ancestry block for some SNP.
    Then, :math:`H = H_0 + H_1` where :math:`H_k \\in \\mathbb{R}^{n \\times A}`
    represents the phased calldata marked by the ancestry indicator
    for each of the two haplotypes of a SNP, that is,

    .. math::
        \\begin{align*}
            H_k
            =
            \\begin{bmatrix}
                \\unicode{x2014} & \\delta^k_1 \\cdot e_{a^k_1}^\\top & \\unicode{x2014} \\\\
                \\unicode{x2014} & \\delta^k_2 \\cdot e_{a^k_2}^\\top & \\unicode{x2014} \\\\
                \\vdots & \\vdots & \\vdots \\\\
                \\unicode{x2014} & \\delta^k_n \\cdot e_{a^k_n}^\\top & \\unicode{x2014} \\\\
            \\end{bmatrix}
        \\end{align*}

    where for each individual :math:`i` and haplotype :math:`k`,
    :math:`\\delta^k_i \\in \\{0,1\\}` is :math:`1` 
    if and only if there is a mutation 
    and :math:`a^k_i \\in \\{1,\\ldots,A\\}` is 
    the ancestry labeling.
    Here, :math:`e_j \\in \\mathbb{R}^A` is the :math:`j` th standard basis vector.
         
    Parameters
    ----------
    filename : str
        File name containing the SNP data in ``.snpdat`` format.
    read_mode : str, optional
        Reading mode of the SNP data. 
        It must be one of the following:

            - ``"file"``: reads the file using standard file IO.
              This method is the most general and portable,
              however, with large files, it is the slowest option.
            - ``"mmap"``: reads the file using mmap.
              This method is only supported on Linux and MacOS.
              It is the most efficient way to read large files.

        Default is ``"file"``.
    """
    def __init__(
        self,
        filename: str,
        read_mode: str ="file",
    ):
        core_io.IOSNPPhasedAncestry.__init__(self, filename, read_mode)

    def write(
        self, 
        calldata: np.ndarray,
        ancestries: np.ndarray,
        A: int,
        n_threads: int =1,
    ):
        """Writes a dense SNP phased, ancestry matrix to the file in ``.snpdat`` format.

        .. note::
            The calldata and ancestries matrices must not contain
            any missing values.

        Parameters
        ----------
        calldata : (n, 2*s) ndarray
            SNP phased calldata in dense format.
            ``calldata[i, 2*j+k]`` is the data for individual ``i``, SNP ``j``, and haplotype ``k``.
            It must only contain values in :math:`\\{0,1\\}`.
        ancestries : (n, 2*s) ndarray
            Local ancestry information in dense format.
            ``ancestries[i, 2*j+k]`` is the ancestry for individual ``i``, SNP ``j``, and haplotype ``k``.
            It must only contain values in :math:`\\{0,\\ldots, A-1\\}`.
        A : int
            Number of ancestries.
        n_threads : int, optional
            Number of threads.
            Default is ``1``.

        Returns
        -------
        total_bytes : int
            Number of bytes written.
        benchmark : dict
            Dictionary of benchmark timings for each step of the serializer.
        """
        (
            total_bytes, 
            benchmark, 
            error,
        ) = core_io.IOSNPPhasedAncestry.write(self, calldata, ancestries, A, n_threads)
        if error != "":
            raise RuntimeError(error)
        return total_bytes, benchmark


class snp_unphased(core_io.IOSNPUnphased):
    """IO handler for SNP unphased matrix.

    A SNP unphased matrix is a matrix that contains values in the set ``{0, 1, 2, NA}``
    where ``NA`` indicates a missing value.
    Typically, ``NA`` is encoded as ``-9``, but for more generality
    we assume *any* negative value is equivalent to ``NA``.

    Parameters
    ----------
    filename : str
        File name to either read or write the SNP unphased matrix in ``.snpdat`` format.
    read_mode : str, optional
        Reading mode of the file ``filename``. 
        It must be one of the following:

            - ``"file"``: reads the file using standard file IO.
              This method is the most general and portable method,
              however, with large files, it is the slowest one.
            - ``"mmap"``: reads the file using mmap.
              This method is only supported on Linux and MacOS.
              It is the most efficient way to read large files.

        Default is ``"file"``.
    """
    def __init__(
        self,
        filename: str,
        read_mode: str ="file",
    ):
        core_io.IOSNPUnphased.__init__(self, filename, read_mode)

    def write(
        self, 
        calldata: np.ndarray,
        impute_method: Union[str, np.ndarray] ="mean",
        n_threads: int =1,
    ):
        """Writes a dense SNP unphased matrix to the file in ``.snpdat`` format.

        Parameters
        ----------
        calldata : (n, p) ndarray
            SNP unphased matrix in dense format.
        impute_method : Union[str, ndarray], optional
            Impute method for missing values. 
            It must be one of the following:

                - ``"mean"``: mean-imputation. Missing values in column ``j`` of ``calldata`` are replaced with
                  the mean of column ``j`` where the mean is computed using the non-missing values.
                  If every value is missing, we impute with ``0``.
                - :class:`numpy.ndarray`: user-specified vector of imputed values for each column of ``calldata``.
                
            Default is ``"mean"``.
        n_threads : int, optional
            Number of threads.
            Default is ``1``.

        Returns
        -------
        total_bytes : int
            Number of bytes written.
        benchmark : dict
            Dictionary of benchmark timings for each step of the serializer.
        """
        if isinstance(impute_method, str):
            p = calldata.shape[1]
            impute = np.empty(p)
        elif isinstance(impute_method, np.ndarray):
            impute = impute_method
            impute_method = "user"
        else:
            raise ValueError("impute_method must be a valid option.")
        (
            total_bytes, 
            benchmark, 
            error,
        ) = core_io.IOSNPUnphased.write(self, calldata, impute_method, impute, n_threads)
        if error != "":
            raise RuntimeError(error)
        return total_bytes, benchmark


class snp_combine_r(core_io.IOSNPCombineR):
    """IO handler for SNP unphased, ancestry matrix.

    A SNP unphased, ancestry matrix is a matrix 
    that combines unphased calldata and phased local ancestry information.

    Let :math:`X \\in \\mathbb{R}^{n \\times s(1+A)}` denote such a matrix
    where
    :math:`n` is the number of samples,
    :math:`s` is the number of SNPs,
    and :math:`A` is the number of ancestries.
    Every :math:`1+A` (contiguous) columns is called a
    *SNP block* corresponding to a single SNP
    with the following structure:
    - The first column contains the SNP data (0, 1, or 2)
    - The next :math:`A` columns contain the unphased ancestry dosages for each ancestry (0, 1, or 2)

    For example, for SNP :math:`j`, the columns are:
    - Column :math:`j(1+A)`: SNP data
    - Columns :math:`j(1+A)+1` to :math:`j(1+A)+A`: Ancestry dosages
         
    Parameters
    ----------
    filename : str
        File name containing the SNP data in ``.snpdat`` format.
    read_mode : str, optional
        Reading mode of the SNP data. 
        It must be one of the following:

            - ``"file"``: reads the file using standard file IO.
              This method is the most general and portable,
              however, with large files, it is the slowest option.
            - ``"mmap"``: reads the file using mmap.
              This method is only supported on Linux and MacOS.
              It is the most efficient way to read large files.

        Default is ``"file"``.
    """
    def __init__(
        self,
        filename: str,
        read_mode: str ="file",
    ):
        core_io.IOSNPCombineR.__init__(self, filename, read_mode)

    def write(
        self, 
        calldata: np.ndarray,
        ancestries: np.ndarray,
        A: int,
        n_threads: int =1,
        *,
        selected_ancestries: np.ndarray = None,
    ):
        """Writes a dense SNP unphased, ancestry matrix to the file in ``.snpdat`` format.

        .. note::
            The calldata and ancestries matrices must not contain
            any missing values.

        Parameters
        ----------
        calldata : (n, s) ndarray
            SNP unphased calldata in dense format.
            ``calldata[i, j]`` is the data for individual ``i`` and SNP ``j``.
            It must only contain values in :math:`\\{0,1,2\\}`.
        ancestries : (n, 2*s) ndarray
            Local ancestry information in dense format.
            ``ancestries[i, 2*j]`` and ``ancestries[i, 2*j+1]`` are the ancestries 
            for individual ``i`` and SNP ``j`` for each of the two haplotypes.
            It must only contain values in :math:`\\{0,\\ldots, A-1\\}`.
        A : int
            Total number of ancestries present in ``ancestries`` (upper bound of labels).
        n_threads : int, optional
            Number of threads.
            Default is ``1``.
        selected_ancestries : ndarray of uint32, optional
            If provided, restrict the ancestry dosage columns per SNP to these
            ancestry labels (in the given order). The output block width per SNP
            becomes ``1 + len(selected_ancestries)`` instead of ``1 + A``.

        Returns
        -------
        total_bytes : int
            Number of bytes written.
        benchmark : dict
            Dictionary of benchmark timings for each step of the serializer.
        """
        if selected_ancestries is None:
            (
                total_bytes, 
                benchmark, 
                error,
            ) = core_io.IOSNPCombineR.write(self, calldata, ancestries, A, n_threads)
        else:
            selected_ancestries = np.asarray(selected_ancestries, dtype=np.uint32)
            (
                total_bytes,
                benchmark,
                error,
            ) = core_io.IOSNPCombineR.write(self, calldata, ancestries, A, selected_ancestries, n_threads)
        if error != "":
            raise RuntimeError(error)
        return total_bytes, benchmark


class snp_combine_s(core_io.IOSNPCombineS):
    """IO handler for SNP both-ancestry matrix.

    Combines:
    - Mutated haplotype counts per ancestry (like phased ancestry summed across haps)
    - Ancestry dosage counts per ancestry (like unphased ancestry dosages)

    Shape: (n, s * (2*A)). For each SNP j:
    - Columns j*(2*A) .. j*(2*A)+(A-1): number of mutated haplotypes labeled with ancestry a
    - Columns j*(2*A)+A .. j*(2*A)+(2*A-1): ancestry dosages per ancestry a
    """
    def __init__(
        self,
        filename: str,
        read_mode: str ="file",
    ):
        core_io.IOSNPCombineS.__init__(self, filename, read_mode)

    def write(
        self,
        calldata: np.ndarray,
        ancestries: np.ndarray,
        A: int,
        n_threads: int =1,
        *,
        selected_ancestries: np.ndarray = None,
    ):
        if selected_ancestries is None:
            (
                total_bytes,
                benchmark,
                error,
            ) = core_io.IOSNPCombineS.write(self, calldata, ancestries, A, n_threads)
        else:
            selected_ancestries = np.asarray(selected_ancestries, dtype=np.uint32)
            (
                total_bytes,
                benchmark,
                error,
            ) = core_io.IOSNPCombineS.write(self, calldata, ancestries, A, selected_ancestries, n_threads)
        if error != "":
            raise RuntimeError(error)
        return total_bytes, benchmark
