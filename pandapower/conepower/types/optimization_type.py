from enum import Enum


class OptimizationType(Enum):
    """
    Specifies the type of the optimization problem.
    """
    UNKNOWN = 0
    SDP = 1
    SOCP = 2
