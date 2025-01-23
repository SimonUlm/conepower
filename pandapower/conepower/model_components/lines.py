import numpy as np

from pandapower.conepower.types.line_constraint_type import LineConstraintType


class Lines:
    """
    Represents the electrical lines of the network, including their orientation and respective limits.
    """
    buses_from: np.ndarray
    buses_to: np.ndarray
    constraint_type: LineConstraintType
    max_line_flows: np.ndarray
    nof_lines: int

    def __init__(self,
                 buses_from: np.ndarray,
                 buses_to: np.ndarray,
                 max_apparent_powers: np.ndarray,
                 constraint_type: LineConstraintType):
        self.nof_lines = max_apparent_powers.size
        assert buses_from.size == self.nof_lines
        assert buses_to.size == self.nof_lines
        self.buses_from = buses_from
        self.buses_to = buses_to
        self.max_line_flows = max_apparent_powers
        assert LineConstraintType is not LineConstraintType.UNKNOWN
        self.constraint_type = constraint_type
