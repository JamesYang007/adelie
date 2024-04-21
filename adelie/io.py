from .adelie_core import io as core_io
from typing import Union
import numpy as np


class snp_base:
    def __init__(self, core):
        self._core = core


class snp_phased_ancestry(snp_base):
    """IO handler for SNP phased, ancestry data.

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
            - ``"auto"``: automatic way of choosing one of the options above.
              The mode is set to ``"mmap"`` whenever the option is allowed.
              Otherwise, the mode is set to ``"file"``.

        Default is ``"auto"``.
    """
    def __init__(
        self,
        filename: str,
        read_mode: str ="auto",
    ):
        super().__init__(core_io.IOSNPPhasedAncestry(filename, read_mode))

    def outer(self):
        """Outer indexing vector.

        Returns
        -------
        outer : (2*s+1,) np.ndarray
            Outer indexing vector.
        """
        return self._core.outer()

    def nnz(self, j: int, hap: int):
        """Number of non-zero entries at a SNP/haplotype.

        Parameters
        ----------
        j : int
            SNP index.
        hap : int
            Haplotype for SNP ``j``.

        Returns
        -------
        nnz : int
            Number of non-zero entries at SNP ``j`` and haplotype ``hap``.
        """
        return self._core.nnz(j, hap)

    def inner(self, j: int, hap: int):
        """Inner indexing vector at a SNP/haplotype.

        Parameters
        ----------
        j : int
            SNP index.
        hap : int
            Haplotype for SNP ``j``.

        Returns
        -------
        inner : np.ndarray
            Inner indexing vector at SNP ``j`` and haplotype ``hap``.
        """
        return self._core.inner(j, hap)

    def ancestry(self, j: int, hap: int):
        """Ancestry vector at a SNP/haplotype.

        Parameters
        ----------
        j : int
            SNP index.
        hap : int
            Haplotype for SNP ``j``.

        Returns
        -------
        v : np.ndarray
            Ancestry vector at SNP ``j`` and haplotype ``hap``.
        """
        return self._core.ancestry(j, hap)

    def write(
        self, 
        calldata: np.ndarray,
        ancestries: np.ndarray,
        A: int,
        n_threads: int =1,
    ):
        """Write a dense array of calldata and ancestry information to the file in ``.snpdat`` format.

        Parameters
        ----------
        calldata : (n, 2*s) np.ndarray
            SNP phased calldata in dense format.
            ``calldata[i, 2*j+k]`` is the data for individual ``i``, SNP ``j``, and haplotype ``k``.
            It must only contain values in :math:`\\{0,1\\}`.
        ancestries : (n, 2*s) np.ndarray
            Ancestry information in dense format.
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
        """
        return self._core.write(calldata, ancestries, A, n_threads)


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
            - ``"auto"``: automatically choose one of the options above.
              The mode is set to ``"mmap"`` whenever the option is allowed.
              Otherwise, the mode is set to ``"file"``.

        Default is ``"auto"``.
    """
    def __init__(
        self,
        filename: str,
        read_mode: str ="auto",
    ):
        core_io.IOSNPUnphased.__init__(self, filename, read_mode)

    def write(
        self, 
        calldata: np.ndarray,
        impute_method: Union[str, np.ndarray] ="mean",
        n_threads: int =1,
    ):
        """Write a dense SNP unphased matrix to the file in ``.snpdat`` format.

        Parameters
        ----------
        calldata : (n, p) np.ndarray
            SNP unphased matrix in dense format.
        impute_method : Union[str, np.ndarray], optional
            Impute method for missing values. 
            It must be one of the following:

                - ``"mean"``: mean-imputation. Missing values in column ``j`` of ``calldata`` are replaced with
                  the mean of column ``j`` where the mean is computed using the non-missing values.
                - ``np.ndarray``: user-specified vector of imputed values for each column of ``calldata``.
                
            Default is ``"mean"``.
        n_threads : int, optional
            Number of threads.
            Default is ``1``.

        Returns
        -------
        total_bytes : int
            Number of bytes written.
        """
        if isinstance(impute_method, str):
            p = calldata.shape[1]
            impute = np.empty(p)
        elif isinstance(impute_method, np.ndarray):
            impute = impute_method
            impute_method = "user"
        else:
            raise ValueError("impute_method must be a valid option.")
        return core_io.IOSNPUnphased.write(self, calldata, impute_method, impute, n_threads)
