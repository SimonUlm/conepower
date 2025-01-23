from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from scipy import sparse


class Constraints(ABC):
    """
    Represents either linear, socp, or sdp constraints.
    """
    nof_constraints: int = 0

    def __init__(self, nof_constraints):
        self.nof_constraints = nof_constraints

    @abstractmethod
    def __add__(self, other):
        pass

    def __iadd__(self, other):
        return self + other

    @abstractmethod
    def prepend_variable(self):
        """
        Extends the equational system that represents the constraints by one more variable.
        The variable does not affect the constraints.

        Example:
            Let `y`, `z` the variables and `3y + 5z = 0` the constraint, i.e. `(y, z) * (3, 5)^T = 0`.
            This method prepends a new variable `x` to the system,
            such that the constraint is defined by `(x, y, z) * (0, 3, 5)^T = 0`.
        """
        pass

    @abstractmethod
    def scaled(self):
        """
        Scales the equational system that represents the constraints,
        such that every line of the system contributes equally to the constraints.
        """
        pass

    @abstractmethod
    def to_cone_formulation(self) -> Tuple[sparse.coo_matrix, np.ndarray, int]:
        """
        Transforms all constraints (linear, socp, sdp) into the form of general cone constraints as defined by CVXOPT.
        """
        pass

    def is_empty(self) -> bool:
        """
        Checks whether no constraints have been specified within this object.
        """
        return self.nof_constraints == 0
