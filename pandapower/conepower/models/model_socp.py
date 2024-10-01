from typing import Tuple

import numpy as np
from cvxopt import solvers
from cvxopt import spmatrix as cvxmatrix
from cvxopt import matrix as cvxvector
from scipy import sparse

from pandapower.conepower.model_components.constraints.constraints_linear import LinearConstraints
from pandapower.conepower.model_components.constraints.constraints_socp import SocpConstraints
from pandapower.conepower.model_components.costs.quadratic_cost import QuadraticCost
from pandapower.conepower.models.model_jabr import ModelJabr
from pandapower.conepower.types.variable_type import VariableType


class ModelSocp:
    cost: QuadraticCost
    linear_equality_constraints: LinearConstraints
    linear_inequality_constraints: LinearConstraints
    nof_variables: int
    socp_constraints: SocpConstraints
    values: np.ndarray

    def __init__(self, nof_variables: int):
        self.nof_variables = nof_variables

    def _box_to_linear_constraints(self, jabr: ModelJabr):

        # initialize
        nof_box_constraints = (jabr.variable_sets[VariableType.PG].size +
                               jabr.variable_sets[VariableType.QG].size +
                               jabr.variable_sets[VariableType.CJJ].size)

        # determine the variables for which the lower bound does not equal the upper bound
        mask = np.concatenate(((jabr.variable_sets[VariableType.PG].lower_bounds !=
                                jabr.variable_sets[VariableType.PG].upper_bounds),
                               (jabr.variable_sets[VariableType.QG].lower_bounds !=
                                jabr.variable_sets[VariableType.QG].upper_bounds),
                               (jabr.variable_sets[VariableType.CJJ].lower_bounds !=
                                jabr.variable_sets[VariableType.CJJ].upper_bounds)))

        # upper bounds
        ub_matrix = sparse.lil_matrix((nof_box_constraints, self.nof_variables), dtype=np.float64)
        ub_matrix.setdiag(1)
        ub_vector = np.concatenate((jabr.variable_sets[VariableType.PG].upper_bounds,
                                    jabr.variable_sets[VariableType.QG].upper_bounds,
                                    jabr.variable_sets[VariableType.CJJ].upper_bounds))

        # lower bounds
        lb_matrix = sparse.lil_matrix((nof_box_constraints, self.nof_variables), dtype=np.float64)
        lb_matrix.setdiag(-1)
        lb_vector = np.concatenate((jabr.variable_sets[VariableType.PG].lower_bounds,
                                    jabr.variable_sets[VariableType.QG].lower_bounds,
                                    jabr.variable_sets[VariableType.CJJ].lower_bounds))

        # combine
        matrix = sparse.vstack((ub_matrix[mask, :],
                                lb_matrix[mask, :]), 'csr')
        vector = np.concatenate((ub_vector[mask], -lb_vector[mask]))
        self.linear_inequality_constraints = LinearConstraints(matrix, vector)

        # equalities
        mask_inv = np.invert(mask)
        if not mask_inv.any():
            return
        self.linear_equality_constraints += LinearConstraints(ub_matrix[mask_inv, :], ub_vector[mask_inv])

    @staticmethod
    def _sparse_matrix_to_cvxopt(matrix) -> cvxmatrix:
        matrix = matrix.tocoo()
        return cvxmatrix(matrix.data.tolist(),
                         matrix.row.tolist(),
                         matrix.col.tolist(),
                         size=matrix.shape)

    @staticmethod
    def _dense_vector_to_cvxopt(vector: np.ndarray) -> cvxvector:
        return cvxvector(vector)

    @classmethod
    def from_jabr(cls, jabr: ModelJabr):
        # initialize
        socp = cls(jabr.nof_variables)

        # initial values
        socp.values = jabr.values

        # quadratic cost
        socp.cost = jabr.cost

        # linear equality constraints
        socp.linear_equality_constraints = jabr.power_flow_equalities

        # linear inequality constraints
        socp._box_to_linear_constraints(jabr)

        # socp constraints
        socp.socp_constraints = jabr.jabr_constraints + jabr.line_constraints

        return socp

    def solve(self) -> Tuple[bool, float, np.ndarray]:
        # linear part of objective function
        c = self._dense_vector_to_cvxopt(self.cost.linear_vector)

        # inequalities
        g_ineq, h_ineq, dim_ineq = self.linear_inequality_constraints.to_cone_formulation()

        # socp
        g_socp, h_socp, dims_socp = self.socp_constraints.to_cone_formulation()

        # combine cone constraints
        g = self._sparse_matrix_to_cvxopt(sparse.vstack((g_ineq, g_socp), format='coo'))
        h = self._dense_vector_to_cvxopt(np.concatenate((h_ineq, h_socp)))
        dims = {
            'l': dim_ineq,
            'q': dims_socp,
            's': []
        }

        # equalities
        a_eq, b_eq, _ = self.linear_equality_constraints.to_cone_formulation()
        a = self._sparse_matrix_to_cvxopt(a_eq)
        b = self._dense_vector_to_cvxopt(b_eq)

        # initial values
        # TODO: Initial values für Slack und Socp werden auch genötigt.

        if self.cost.is_linear():
            sol = solvers.conelp(c=c,
                                 G=g, h=h, dims=dims,
                                 A=a, b=b)
        else:
            # multiply quadratic matrix by two, since cvxopt expects f(x) = (1/2) * x^T * P * x + q^T * x
            p = self._sparse_matrix_to_cvxopt(self.cost.quadratic_matrix) * 2
            sol = solvers.coneqp(P=p, q=c,
                                 G=g, h=h, dims=dims,
                                 A=a, b=b)

        success = sol['status'] == 'optimal'
        objective = sol['primal objective']
        resulting_values = np.array(sol['x']).flatten()
        return success, objective, resulting_values
