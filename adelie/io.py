import numpy as np
from .adelie_core import io as core_io


class snp_base:
    def __init__(self, core):
        self._core = core
    
    def endian(self):
        """Endianness used in the file.

        .. note::
            We recommend that users read/write from the file on the *same* machine.
            The ``.snpdat`` format may depend on the endianness of the machine.
            So, unless the endianness is the same across two different machines,
            it is undefined behavior reading a file that was generated on a different machine.

        Returns
        -------
        endian : str
            ``"big-endian"`` if big-endian and ``"little-endian"`` if little-endian.
        """
        return (
            "big-endian" 
            if self._core.endian() else
            "little-endian"
        )

    def rows(self):
        """Number of rows of the matrix.

        Returns
        -------
        rows : int
            Number of rows.
        """
        return self._core.rows()

    def cols(self):
        """Number of columns of the matrix.

        Returns
        -------
        cols : int
            Number of columns.
        """
        return self._core.cols()

    def read(self):
        """Read and load the matrix from file.

        Returns
        -------
        total_bytes : int
            Number of bytes read.
        """
        return self._core.read()

    def to_dense(self, n_threads: int =1):
        """Creates a dense matrix.

        Parameters
        ----------
        n_threads : int, optional
            Number of threads.
            Default is ``1``.

        Returns
        -------
        dense : (n, p) np.ndarray
            Dense matrix.
        """
        return self._core.to_dense(n_threads)


class snp_unphased(snp_base):
    """IO handler for SNP unphased data.

    Parameters
    ----------
    filename : str
        File name containing the SNP data in ``.snpdat`` format.
    """
    def __init__(
        self,
        filename,
    ):
        super().__init__(core_io.IOSNPUnphased(filename))

    def outer(self):
        """Outer indexing vector.

        Returns
        -------
        outer : (p+1,) np.ndarray
            Outer indexing vector.
        """
        return self._core.outer()

    def nnz(self, j: int):
        """Number of non-zero entries at a column.

        Parameters
        ----------
        j : int
            Column index.

        Returns
        -------
        nnz : int
            Number of non-zero entries in column ``j``.
        """
        return self._core.nnz(j)

    def inner(self, j: int):
        """Inner indexing vector at a column.

        Parameters
        ----------
        j : int
            Column index.

        Returns
        -------
        inner : np.ndarray
            Inner indexing vector at column ``j``.
        """
        return self._core.inner(j)

    def value(self, j: int):
        """Value vector at a column.

        Parameters
        ----------
        j : int
            Column index.

        Returns
        -------
        v : np.ndarray
            Value vector at column ``j``.
        """
        return self._core.value(j)

    def write(
        self, 
        calldata: np.ndarray,
        n_threads: int =1,
    ):
        """Write a dense array to the file in ``.snpdat`` format.

        Parameters
        ----------
        calldata : (n, p) np.ndarray
            SNP unphased calldata in dense format.
        n_threads : int, optional
            Number of threads.
            Default is ``1``.

        Returns
        -------
        total_bytes : int
            Number of bytes written.
        """
        return self._core.write(calldata, n_threads)


class snp_phased_ancestry(snp_base):
    """IO handler for SNP phased, ancestry data.

    Parameters
    ----------
    filename : str
        File name containing the SNP data in ``.snpdat`` format.
    """
    def __init__(
        self,
        filename,
    ):
        super().__init__(core_io.IOSNPPhasedAncestry(filename))

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
        ancestries : (n, 2*s) np.ndarray
            Ancestry information in dense format.
            ``ancestries[i, 2*j+k]`` is the ancestry for individual ``i``, SNP ``j``, and haplotype ``k``.
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

