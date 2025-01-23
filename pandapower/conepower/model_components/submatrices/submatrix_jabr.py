import math
from typing import List, Tuple

import numpy as np
from scipy import sparse

from pandapower.conepower.model_components.submatrices.submatrix_base import HermitianSubmatrix


class JabrSubmatrix(HermitianSubmatrix):
    """
    Represents a Hermitian submatrix, i.e., a partially defined Hermitian matrix, that is used for Jabr's Relaxation.

    A Hermitian submatrix is a square matrix where:
    1. Only some of the elements are explicitly defined, while others are left undefined.
    2. The explicitly defined elements satisfy the Hermitian property, i.e.,
    the element at (i, j) is the complex conjugate of the element at (j, i).

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
    along with methods for tasks such as checking consistency or extracting properties of the submatrix,
    as well as necessary methods deploying Jabr's Relaxation.

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
    _nof_lines : int
        The number of electrical lines. This value can be higher than `_nof_unique_edges`,
        as edges with more than one line can exist.
    _lines_to_diag_ff : sparse.csc_matrix
        A transformation matrix where each row represents a line.
        For each row representing the line (`i`, `j`), it contains a `1` at the position that corresponds to the index
        in `_data` where (`i`, `i`) is stored.
    _lines_to_diag_tt : sparse.csc_matrix
        A transformation matrix where each row represents a line.
        For each row representing the line (`i`, `j`), it contains a `1` at the position that corresponds to the index
        in `_data` where (`j`, `j`) is stored.
    _lines_to_real_off_diag_ft : sparse.csc_matrix
        A transformation matrix where each row represents a line.
        For each row representing the line (`i`, `j`), it contains a `1` at the position that corresponds to the index
        in `_data` where the real part of (`i`, `j`) is stored.
    _lines_to_real_off_diag_tf : sparse.csc_matrix
        A transformation matrix where each row represents a line.
        For each row representing the line (`i`, `j`), it contains a `1` at the position that corresponds to the index
        in `_data` where the real part of (`j`, `i`) is stored.
    _lines_to_imag_off_diag_ft : sparse.csc_matrix
        A transformation matrix where each row represents a line.
        For each row representing the line (`i`, `j`), it contains a `1` at the position that corresponds to the index
        in `_data` where the imaginary part of (`i`, `j`) is stored.
        If (`i`, `j`) is not explicitly stored (but instead (`j`, `i`)), it is indicated by `-1`instead of `1`.
    _lines_to_imag_off_diag_ft : sparse.csc_matrix
        A transformation matrix where each row represents a line.
        For each row representing the line (`i`, `j`), it contains a `1` at the position that corresponds to the index
        in `_data` where the imaginary part of (`j`, `i`) is stored.
        If (`j`, `i`) is not explicitly stored (but instead (`i`, `j`)), it is indicated by `-1`instead of `1`.
    """
    _nof_lines: int
    _lines_to_diag_ff: sparse.csc_matrix
    _lines_to_diag_tt: sparse.csc_matrix
    _lines_to_real_off_diag_ft: sparse.csc_matrix
    _lines_to_real_off_diag_tf: sparse.csc_matrix
    _lines_to_imag_off_diag_ft: sparse.csc_matrix
    _lines_to_imag_off_diag_tf: sparse.csc_matrix

    def _create_transformation_matrices(self, nodes_from: np.ndarray, nodes_to: np.ndarray):
        """
        Creates the transformation matrices needed for assembling the line constraints.

        Parameters
        ----------
        nodes_from : np.ndarray
            The list of starting nodes of the lines.
        nodes_to : np.ndarray
            The list of ending nodes of the lines.
        """
        # initialize
        self._nof_lines = nodes_from.size
        assert nodes_to.size == self._nof_lines
        assert self._nof_unique_edges <= self._nof_lines
        dummy_diag = sparse.csc_matrix((self._nof_lines, self.dim), dtype=int)
        dummy_off_diag = sparse.csc_matrix((self._nof_lines, self._nof_unique_edges), dtype=int)

        # get how the order of edges relates to the order of variables in submatrix
        mask = nodes_from > nodes_to
        temp_nodes_from = np.where(np.invert(mask), nodes_from, nodes_to)
        temp_nodes_to = np.where(mask, nodes_from, nodes_to)
        lines = np.column_stack((temp_nodes_from, temp_nodes_to))
        line_indices = np.arange(self._nof_lines)
        sorted_line_indices = np.lexsort((temp_nodes_to, temp_nodes_from))
        sorted_lines = lines[sorted_line_indices]
        _, counts = np.unique(sorted_lines, axis=0, return_counts=True)
        data_index_per_line = np.empty(self._nof_lines, dtype=int)
        data_index_per_line[sorted_line_indices] = np.repeat(np.arange(self._nof_unique_edges, dtype=int), counts)

        # diagonal ff
        diag_ff = sparse.coo_matrix((np.ones(self._nof_lines, dtype=int),
                                     (line_indices, nodes_from)),
                                    shape=(self._nof_lines, self.dim))
        self._lines_to_diag_ff = sparse.hstack((diag_ff, dummy_off_diag, dummy_off_diag), format='csc')

        # diagonal tt
        diag_tt = sparse.coo_matrix((np.ones(self._nof_lines, dtype=int),
                                     (line_indices, nodes_to)),
                                    shape=(self._nof_lines, self.dim))
        self._lines_to_diag_tt = sparse.hstack((diag_tt, dummy_off_diag, dummy_off_diag), format='csc')

        # real off-diagonal ft and tf
        real_off_diag_ft = sparse.coo_matrix((np.ones(self._nof_lines, dtype=int),
                                              (line_indices, data_index_per_line)),
                                             shape=(self._nof_lines, self._nof_unique_edges))
        real_off_diag_ft = sparse.hstack((dummy_diag, real_off_diag_ft, dummy_off_diag), format='csc')
        self._lines_to_real_off_diag_ft = real_off_diag_ft
        self._lines_to_real_off_diag_tf = real_off_diag_ft

        # imaginary off-diagonal ft and tf
        off_diag_data = np.ones(self._nof_lines)
        off_diag_data[mask] = -1
        imag_off_diag_ft = sparse.coo_matrix((off_diag_data,
                                              (line_indices, data_index_per_line)),
                                             shape=(self._nof_lines, self._nof_unique_edges))
        imag_off_diag_ft = sparse.hstack((dummy_diag, dummy_off_diag, imag_off_diag_ft), format='csc')
        self._lines_to_imag_off_diag_ft = imag_off_diag_ft
        self._lines_to_imag_off_diag_tf = -imag_off_diag_ft

    def __init__(self,
                 adjacency_matrix: sparse.csr_matrix,
                 nodes_from: np.ndarray,
                 nodes_to: np.ndarray,
                 allocated_data: np.ndarray = None):
        HermitianSubmatrix.__init__(self, adjacency_matrix, allocated_data)
        self._create_transformation_matrices(nodes_from, nodes_to)

    @staticmethod
    def _transform_to_diagonal_block_matrices_without_diagonal(matrix: sparse.csr_matrix)\
            -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """
        Eliminates all diagonal elements as well as all zero elements from the matrix.
        Each row is than considered to be a block and a corresponding diagonal block matrix is assembled.

        Example:
            Consider the matrix
            [1, 5, 0, 6;
            5, 2, 7, 0;
            0, 7, 3, 0;
            6, 0, 0, 4].
            Then, after eliminating the diagonal elements and all zeroes, and regarding each row as block, the matrix
            [5, 6, 0, 0, 0, 0;
            0, 0, 5, 7, 0, 0;
            0, 0, 0, 0, 7, 0;
            0, 0, 0, 0, 0, 6]
            is obtained.

        Parameters
        ----------
        matrix : sparse.csr_matrix
            A matrix.

        Returns
        -------
        matrices : Tuple[sparse.csr_matrix, sparse.csr_matrix]
            The resulting diagonal block matrix,
            separated into two matrices representing real and imaginary part, respectively.
        """
        matr_without_diag = matrix.copy()
        matr_without_diag.setdiag(0)
        filtered_rows = [sparse.csr_matrix(matr_without_diag[i][matr_without_diag[i] != 0])
                         for i in range(matr_without_diag.shape[0])]
        diag_block = sparse.block_diag(filtered_rows, 'csr')
        return np.real(diag_block), np.imag(diag_block)

    def create_pg_linear_system_matrix(self, adm_matr: sparse.csr_matrix) -> sparse.csr_matrix:
        """
        Creates the real part of the power flow equations (excluding generators) with respect to Jabr's Relaxation.
        The equations are a linear system of the form A * x = b,
        where x is the variable vector which is equivalent to `_data`.

        Parameters
        ----------
        adm_matr : sparse.csr_matrix
            Admittance matrix representing the electrical network.

        Returns
        -------
        A : sparse.csr_matrix
            The matrix A of the linear system A * x = b.
        """
        adm_real_diag = sparse.diags(np.real(adm_matr.diagonal()))
        real_diag_block, imag_diag_block = self._transform_to_diagonal_block_matrices_without_diagonal(adm_matr)
        real_diag_block = real_diag_block @ self._full_off_diag_to_sym_upper_tri
        imag_diag_block = imag_diag_block @ self._full_off_diag_to_antisym_upper_tri
        return sparse.hstack((adm_real_diag, real_diag_block, imag_diag_block), 'csr')

    def create_qg_linear_system_matrix(self, adm_matr: sparse.csr_matrix) -> sparse.csr_matrix:
        """
        Creates the imaginary part of the power flow equations (excluding generators) with respect to Jabr's Relaxation.
        The equations are a linear system of the form A * x = b,
        where x is the variable vector representing `_data`.

        Parameters
        ----------
        adm_matr : sparse.csr_matrix
            Admittance matrix representing the electrical network.

        Returns
        -------
        A : sparse.csr_matrix
            The matrix A of the linear system A * x = b.
        """
        adm_imag_diag = sparse.diags(np.imag(adm_matr.diagonal()))
        real_diag_block, imag_diag_block = self._transform_to_diagonal_block_matrices_without_diagonal(adm_matr)
        real_diag_block = real_diag_block @ self._full_off_diag_to_antisym_upper_tri
        imag_diag_block = imag_diag_block @ self._full_off_diag_to_sym_upper_tri
        return sparse.hstack((-adm_imag_diag, -imag_diag_block, real_diag_block), 'csr')

    def create_jabr_constraints(self) -> Tuple[List[sparse.lil_matrix], List[sparse.lil_matrix]]:
        """
        Creates the socp constraints that relax the conventional OPF.

        Returns
        -------
        matrices : Tuple[List[sparse.lil_matrix], List[sparse.lil_matrix]]
            Returns the matrices `E_i` and the vectors `g_i`, describing the socp constraints
            `||E_i * x|| + g_i^T * x <= 0` for `i = 1, ..., q`,
            where x is the variable vector representing `_data`.
        """
        # initialize lists and calculate offsets
        size = self.dim + self._nof_unique_edges * 2
        matrix_list = [sparse.lil_matrix((3, size), dtype=float) for _ in range(self._nof_unique_edges)]
        vector_list = [sparse.lil_matrix((size, 1), dtype=float) for _ in range(self._nof_unique_edges)]

        # define the matrices and vectors one by one (there has to be a better way...)
        for i in range(self.dim, self._complex_size):
            # preliminary
            row = self._data.row[i]
            col = self._data.col[i]
            # matrix
            matrix = matrix_list[i-self.dim]
            matrix[0, row] = 0.5
            matrix[0, col] = -0.5
            matrix[1, i] = 1
            matrix[2, i+self._offset_complex_to_real] = 1
            # vector
            vector = vector_list[i-self.dim]
            vector[row, 0] = 0.5
            vector[col, 0] = 0.5

        # return matrices and vectors
        return matrix_list, vector_list

    def create_line_apparent_power_constraints(self,
                                               max_apparent_powers: np.ndarray,
                                               y_ff: np.ndarray,
                                               y_ft: np.ndarray,
                                               y_tf: np.ndarray,
                                               y_tt: np.ndarray) ->\
            Tuple[List[sparse.lil_matrix], List[float]]:
        """
        Creates socp constraints that limit apparent power flow on the electrical lines.

        Parameters
        ----------
        max_apparent_powers : np.ndarray
            Upper bounds on apparent power per line.
        y_ff : np.ndarray
            Admittances `y_ff` as defined by PYPOWER.
        y_ft : np.ndarray
            Admittances `y_ft` as defined by PYPOWER.
        y_tf : np.ndarray
            Admittances `y_tf` as defined by PYPOWER.
        y_tt : np.ndarray
            Admittances `y_tt` as defined by PYPOWER.

        Returns
        -------
        matrices : Tuple[List[sparse.lil_matrix], List[float]]
            Returns the matrices `E_i` and the scalars `d_i`, describing the socp constraints
            `||E_i * x|| + d_i <= 0 for i = 1, ..., q`,
            where x is the variable vector representing `_data`.
        """
        # create two diagonal matrices (one real and one imaginary) for each admittance vector
        real_y_ff = sparse.diags(np.real(y_ff))
        real_y_ft = sparse.diags(np.real(y_ft))
        real_y_tf = sparse.diags(np.real(y_tf))
        real_y_tt = sparse.diags(np.real(y_tt))
        imag_y_ff = sparse.diags(np.imag(y_ff))
        imag_y_ft = sparse.diags(np.imag(y_ft))
        imag_y_tf = sparse.diags(np.imag(y_tf))
        imag_y_tt = sparse.diags(np.imag(y_tt))

        # first and second row of ft constraint for each line, hence we obtain two matrices
        socp_ft_first_rows: sparse.csr_matrix = (real_y_ff @ self._lines_to_diag_ff +
                                                 real_y_ft @ self._lines_to_real_off_diag_ft +
                                                 imag_y_ft @ self._lines_to_imag_off_diag_ft)
        socp_ft_second_rows: sparse.csr_matrix = (-imag_y_ff @ self._lines_to_diag_ff +
                                                  -imag_y_ft @ self._lines_to_real_off_diag_ft +
                                                  real_y_ft @ self._lines_to_imag_off_diag_ft)

        # first and second row of tf constraint for each line, hence we obtain two matrices
        socp_tf_first_rows: sparse.csr_matrix = (real_y_tt @ self._lines_to_diag_tt +
                                                 real_y_tf @ self._lines_to_real_off_diag_tf +
                                                 imag_y_tf @ self._lines_to_imag_off_diag_tf)
        socp_tf_second_rows: sparse.csr_matrix = (-imag_y_tt @ self._lines_to_diag_tt +
                                                  -imag_y_tf @ self._lines_to_real_off_diag_tf +
                                                  real_y_tf @ self._lines_to_imag_off_diag_tf)

        # initialize constraints
        matrix_list = []
        scalar_list = []

        # compose constraints
        for i in range(self._nof_lines):
            # check whether line is unconstrained
            max_power = max_apparent_powers[i]
            if math.isnan(max_power) or math.isinf(max_power) or max_power == 0:
                continue
            # ft
            matrix_list.append(sparse.vstack((socp_ft_first_rows.getrow(i),
                                              socp_ft_second_rows.getrow(i)), format='lil'))
            scalar_list.append(max_power)
            # tf
            matrix_list.append(sparse.vstack((socp_tf_first_rows.getrow(i),
                                              socp_tf_second_rows.getrow(i)), format='lil'))
            scalar_list.append(max_power)

        # noinspection PyTypeChecker
        return matrix_list, scalar_list

    def create_line_current_constraints(self,
                                        max_currents: np.ndarray,
                                        y_ff: np.ndarray,
                                        y_ft: np.ndarray,
                                        y_tf: np.ndarray,
                                        y_tt: np.ndarray) ->\
            Tuple[sparse.csr_matrix, np.ndarray]:
        """
        Creates linear constraints that limit current flow on the electrical lines.

        Parameters
        ----------
        max_currents : np.ndarray
            Upper bounds on current per line.
        y_ff : np.ndarray
            Admittances `y_ff` as defined by PYPOWER.
        y_ft : np.ndarray
            Admittances `y_ft` as defined by PYPOWER.
        y_tf : np.ndarray
            Admittances `y_tf` as defined by PYPOWER.
        y_tt : np.ndarray
            Admittances `y_tt` as defined by PYPOWER.

        Returns
        -------
        matrices : Tuple[sparse.csr_matrix, np.ndarray]
            Returns the matrices `A` and the vector `b`, describing the linear constraints
            'A * x = b', where x is the variable vector representing `_data`.
        """
        # transform admittance vectors to diagonal matrices for better readability
        y_ff: sparse.dia_matrix = sparse.diags(y_ff)
        y_ft: sparse.dia_matrix = sparse.diags(y_ft)
        y_tf: sparse.dia_matrix = sparse.diags(y_tf)
        y_tt: sparse.dia_matrix = sparse.diags(y_tt)

        # calculate the matrices with  |•|^2 applied to each entry
        y_ff_sq = np.real(y_ff @ y_ff.conj())
        y_ft_sq = np.real(y_ft @ y_ft.conj())
        y_tf_sq = np.real(y_tf @ y_tf.conj())
        y_tt_sq = np.real(y_tt @ y_tt.conj())

        # calculate more auxiliary matrices
        y_fft = y_ff @ y_ft.conj()
        real_y_fft = np.real(y_fft)
        imag_y_fft = np.imag(y_fft)
        y_ttf = y_tt @ y_tf.conj()
        real_y_ttf = np.real(y_ttf)
        imag_y_ttf = np.imag(y_ttf)

        # compose matrix
        ft_constraints = (y_ff_sq @ self._lines_to_diag_ff
                          + y_ft_sq @ self._lines_to_diag_tt
                          + 2 * (real_y_fft @ self._lines_to_real_off_diag_ft
                                 - imag_y_fft @ self._lines_to_imag_off_diag_ft))
        tf_constraints = (y_tt_sq @ self._lines_to_diag_tt
                          + y_tf_sq @ self._lines_to_diag_ff
                          + 2 * (real_y_ttf @ self._lines_to_real_off_diag_tf
                                 - imag_y_ttf @ self._lines_to_imag_off_diag_tf))
        matrix = sparse.vstack((ft_constraints, tf_constraints), format='csr')

        # compose rhs
        max_currents_sq = np.square(max_currents)
        rhs = np.concatenate((max_currents_sq, max_currents_sq))

        return matrix, rhs
