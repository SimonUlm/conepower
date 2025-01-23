import warnings

import numpy as np
from scipy import sparse


class HermitianSubmatrix:
    """
    Represents a Hermitian submatrix, i.e., a partially defined Hermitian matrix.

    A Hermitian submatrix is a square matrix where:
    1. Only some of the elements are explicitly defined, while others are left undefined.
    2. The explicitly defined elements satisfy the Hermitian property,
    i.e., the element at (i, j) is the complex conjugate of the element at (j, i).

    Example:
        The matrix
        [1, -, 5;
        -, 2, 6;
        5, 6, 3]
        is a Hermitian submatrix, where '-' represents an undefined element.

    To leverage the Hermitian property, a Hermitian submatrix is stored by only including the variables
    in the upper triangle of the matrix, in the following order:
    1. The diagonal elements.
    2. The real parts of the off-diagonal elements in the strict upper triangle, stored row-wise.
    3. The imaginary parts of the off-diagonal elements in the strict upper triangle, stored row-wise.

    Example:
        The submatrix above would be stored as [1, 2, 3, 5, 6, 0, 0].

    This class also provides methods for working with Hermitian submatrices,
    such as checking consistency or extracting properties of the submatrix.

    This class provides efficient storage and manipulation of Hermitian submatrices,
    along with methods for tasks such as checking consistency or extracting properties of the submatrix.

    Attributes
    ----------
    dim : int
        The dimension of the submatrix (i.e., the number of rows or columns, as it is square).
    _data: sparse.coo_array
        The stored data of the submatrix in a sparse format.
    _nof_unique_edges : int
        The number of (complex-valued) elements in the strict upper triangle.
    _complex_size: int
        The number of diagonal elements plus the number of (complex-valued) elements in the strict upper triangle.
    _real_size: int
        The total number of real-valued components stored:
        the number of diagonal elements plus twice the number of complex-valued elements in the strict upper triangle.
        This value matches the size of `_data`.
    _offset_complex_to_real : int
        The offset between the real parts of the elements in the strict upper triangle and their imaginary parts.
        This value equals `_nof_unique_edges`.
    _half_index_matrix : sparse.csr_matrix
        A sparse matrix where each element indicates the position in `_data`
        where the real part of the corresponding matrix element is stored.
        A value of 0 means the element is either not in the strict upper triangle or is undefined in the submatrix.
    _full_off_diag_to_sym_upper_tri : sparse.csc_matrix
        Assume `d` is a vector storing the real parts of **all** off-diagonal elements of the submatrix
        in row-wise order (not leveraging Hermiticity).
        Let `v` be a vector multiplied with `d` such that `v^T * d = s`.
        This attribute helps to transform the problem allowing the same result `s` to be computed
        using the real parts of the off-diagonal elements stored in `_data` (denoted as `r`):
        `v^T * full_off_diag_to_sym_upper_tri * r = s`.
    _full_off_diag_to_antisym_upper_tri : sparse.csc_matrix
        Assume `d` is a vector storing the imaginary parts of **all** off-diagonal elements of the submatrix
        in row-wise order (not leveraging Hermiticity).
        Let `v` be a vector multiplied with `d` such that `v^T * d = s`.
        This attribute helps to transform the problem allowing the same result `s` to be computed
        using the imaginary parts of the off-diagonal elements stored in `_data` (denoted as `c`):
        `v^T * full_off_diag_to_antisym_upper_tri * c = s`.
    """
    dim: int
    _data: sparse.coo_array
    _nof_unique_edges: int
    _complex_size: int
    _real_size: int
    _offset_complex_to_real: int
    _half_index_matrix: sparse.csr_matrix
    _full_off_diag_to_sym_upper_tri: sparse.csc_matrix
    _full_off_diag_to_antisym_upper_tri: sparse.csc_matrix

    @staticmethod
    def _validate_matrix(matrix: sparse.csr_matrix):
        """
        Validates whether the given matrix is a valid adjacency matrix for an undirected graph.

        Raises
        ------
        TypeError
            If the matrix is not stored in the csr format.
        ValueError
            If it is not a square matrix that implicitly defines an undirected graph.
        """
        if type(matrix) is not sparse.csr_matrix:
            raise TypeError("The matrix needs to be stored in the csr format.")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("The matrix needs to be a square matrix.")
        assert matrix.shape[0] > 0
        if matrix.shape[0] == 0:
            raise ValueError("The dimension of the matrix needs to be greater than zero.")
        transposed: sparse.csr_matrix = matrix.transpose().tocsr()
        if not (np.all(np.equal(matrix.indices, transposed.indices)) and
                np.all(np.equal(matrix.indptr, transposed.indptr))):
            raise ValueError("The graph defined by the matrix needs to be undirected.")

    @staticmethod
    def _remove_lower_triangle(full_matrix: sparse.csr_matrix) -> sparse.csr_matrix:
        """
        Sets all elements below the main diagonal (the strict lower triangle) to zero
        and removes them from the sparse matrix.
        """
        half_matrix = full_matrix.copy()
        for row_idx in range(half_matrix.shape[0]):
            row_start = half_matrix.indptr[row_idx]
            row_end = half_matrix.indptr[row_idx + 1]
            col_indices = half_matrix.indices[row_start:row_end]
            mask = col_indices < row_idx
            half_matrix.data[row_start:row_end][mask] = 0
            half_matrix.eliminate_zeros()
        return half_matrix

    def __init__(self,
                 adjacency_matrix: sparse.csr_matrix,
                 allocated_data: np.ndarray = None):
        # validate
        self._validate_matrix(adjacency_matrix)
        self.dim = adjacency_matrix.shape[0]

        # prepare index matrices (see below)
        full_index_matrix: sparse.csr_matrix = adjacency_matrix.copy()
        full_index_matrix.setdiag(0)
        full_index_matrix.eliminate_zeros()
        half_index_matrix = self._remove_lower_triangle(full_index_matrix)

        # initialize
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            half_index_matrix = half_index_matrix.astype(dtype=int, casting='unsafe')
            full_index_matrix = full_index_matrix.astype(dtype=int, casting='unsafe')
        self._nof_unique_edges = half_index_matrix.size
        self._offset_complex_to_real = self._nof_unique_edges
        self._complex_size = self.dim + self._nof_unique_edges
        self._real_size = self.dim + self._nof_unique_edges * 2

        # create half index matrix that indicates where (in the coo array) the respective matrix entries are stored
        starting_index = self.dim
        ending_index = self._complex_size
        half_index_matrix.data = np.arange(starting_index, ending_index)
        self._half_index_matrix = half_index_matrix

        # create data as coo matrix
        if allocated_data is not None:
            assert allocated_data.size == self._real_size
            data = allocated_data
        else:
            # noinspection PyTypeChecker
            data = np.zeros(self._real_size, dtype=float)
        complex_off_diag_coords = half_index_matrix.tocoo()
        row = np.concatenate((np.arange(self.dim), complex_off_diag_coords.row, complex_off_diag_coords.row))
        col = np.concatenate((np.arange(self.dim), complex_off_diag_coords.col, complex_off_diag_coords.col))
        self._data = sparse.coo_array((data, (row, col)), shape=(self.dim, self.dim))

        # create full index matrix that indicates where the respective matrix entries would have been stored
        # if the full matrix were saved instead of only the upper triangle
        starting_index = self.dim
        ending_index = self.dim + self._nof_unique_edges * 2
        full_index_matrix.data = np.arange(starting_index, ending_index)

        # map half index matrix to full index matrix while disregarding the diagonal
        mapping_matrix = sparse.lil_matrix((self._nof_unique_edges, self._nof_unique_edges * 2), dtype=int)
        indices = np.arange(self._nof_unique_edges)
        pos_array = self._remove_lower_triangle(full_index_matrix).data
        pos_array[:] -= self.dim  # disregard diagonal
        neg_array = self._remove_lower_triangle(full_index_matrix.transpose().tocsr()).data
        neg_array[:] -= self.dim  # disregard diagonal
        mapping_matrix[indices, pos_array] = 1
        mapping_matrix[indices, neg_array] = -1
        self._full_off_diag_to_antisym_upper_tri = mapping_matrix.tocsr(copy=True).transpose()
        self._full_off_diag_to_sym_upper_tri = mapping_matrix.tocsr(copy=True).transpose()
        self._full_off_diag_to_sym_upper_tri.data[:] = 1

    def _get_idx(self, i: int, j: int) -> (int, bool):
        """
        Returns the index in the `_data` storage that corresponds to the element at (i, j) in the Hermitian submatrix,
        along with a boolean indicating whether the element is in the strict lower triangle.

        Parameters
        ----------
        i : int
            The row index of the element.
        j : int
            The column index of the element.

        Returns
        -------
        index, conj : (int, bool)
            The index in the `_data` array corresponding to the element at (i, j),
            and boolean value that indicates whether the element is in the strict lower triangle.

        Raises
        ------
        IndexError
            If the submatrix does not contain an element with row index i and column index j.
        """
        # validate
        if i >= self.dim:
            raise IndexError(f"Row index {i} exceeds the dimensions of the submatrix.")
        if j >= self.dim:
            raise IndexError(f"Row index {i} exceeds the dimensions of the submatrix.")

        # diagonal
        if i == j:
            return i, False

        # upper triangle
        if i < j:
            idx = self._half_index_matrix[i, j]
            if idx == 0:
                raise IndexError(f"No element exists at ({i}, {j}).")
            return idx, False

        # lower triangle
        if i > j:
            idx = self._half_index_matrix[j, i]
            if idx == 0:
                raise IndexError(f"No element exists at ({i}, {j}).")
            return idx, True

    def __getitem__(self, key: (int, int)) -> complex:
        i, j = key
        idx, conj = self._get_idx(i, j)
        if idx < self.dim:
            return complex(self._data.data[idx], 0)
        elif conj:
            return complex(self._data.data[idx], -self._data.data[idx+self._offset_complex_to_real])
        else:
            return complex(self._data.data[idx], self._data.data[idx+self._offset_complex_to_real])

    def __setitem__(self, key: (int, int), value: complex):
        i, j = key
        idx, conj = self._get_idx(i, j)
        if idx < self.dim:
            if value.imag != 0:
                raise ValueError(f"The element at ({i}, {j}) must be real-valued.")
            self._data.data[idx] = value.real
        elif conj:
            self._data.data[idx] = value.real
            self._data.data[idx+self._offset_complex_to_real] = -value.imag
        else:
            self._data.data[idx] = value.real
            self._data.data[idx+self._offset_complex_to_real] = value.imag

    def get_connectivity_matrix(self) -> sparse.csc_matrix:
        """
        Returns the adjacency matrix of the undirected graph implicitly defined by the Hermitian submatrix.

        Returns
        -------
        adj_matrix : sparse.csc_matrix
            The corresponding adjacency matrix.
        """
        matrix = self._half_index_matrix.tocsc() + self._half_index_matrix.transpose()
        matrix.data[:] = 1
        return matrix

    def get_diagonal(self) -> np.ndarray:
        """
        Returns the diagonal of the submatrix as a dense array.

        Returns
        -------
        diag : np.ndarray
            The diagonal.
        """
        return self._data.data[:self.dim]
