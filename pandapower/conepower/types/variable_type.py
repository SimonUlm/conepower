from enum import Enum


class VariableType(Enum):
    """
    Specifies the variable type.

    Attributes
    ----------
    UMAG
        A variable representing voltage magnitudes.
    UANG
        A variable representing voltage phase angles.
    PG
        A variable representing active power injections.
    QG
        A variable representing reactive power injections.
    CJJ
        A variable representing the diagonal elements of a Hermitian submatrix.
    CJK
        A variable representing the real parts of the off-diagonal elements of a Hermitian submatrix.
    SJK
        A variable representing the imaginary parts of the off-diagonal elements of a Hermitian submatrix.
    """
    UNKNOWN = 0
    UMAG = 1
    UANG = 2
    PG = 3
    QG = 4
    CJJ = 5
    CJK = 6
    SJK = 7

    @classmethod
    def from_str(cls, string):
        if string == 'Vm':
            return cls.UMAG
        elif string == 'Va':
            return cls.UANG
        elif string == 'Pg':
            return cls.PG
        elif string == 'Qg':
            return cls.QG

        return cls.UNKNOWN
