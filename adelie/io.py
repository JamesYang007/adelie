import numpy as np
from .adelie_core import io as core_io


class snp_unphased:
    """IO handler for SNP Unphased data.

    Parameters
    ----------
    filename : str
        File name to either read from or write to related to the SNP data.
    """
    def __init__(
        self,
        filename,
    ):
        self._core = core_io.IOSNPUnphased(filename)

    def endian(self):
        """Gets the endianness used in the file.

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
        """Gets the number of rows of the matrix.

        Returns
        -------
        rows : int
            Number of rows.
        """
        return self._core.rows()

    def cols(self):
        """Gets the number of columns of the matrix.

        Returns
        -------
        cols : int
            Number of columns.
        """
        return self._core.cols()

    def outer(self):
        """Gets the outer indexing vector.

        Returns
        -------
        outer : (p+1,) np.ndarray
            Outer indexing vector.
        """
        return self._core.outer()

    def nnz(self, j: int):
        """Gets the number of non-zero entries at a column.

        Parameters
        ----------
        j : int
            Column index.

        Returns
        -------
        nnz : int
            Number of non-zero entries column ``j``.
        """
        return self._core.nnz(j)

    def inner(self, j: int):
        """Gets the inner indexing vector at a column.

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
        """Gets the value vector at a column.

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

    def read(self):
        """Read and load the matrix from file.

        Returns
        -------
        total_bytes : int
            Number of bytes read.
        """
        return self._core.read()

    def write(
        self, 
        calldata: np.ndarray,
        n_threads: int =1,
    ):
        """Write dense array to the file in special format.

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
