from enum import Enum


class RelaxationType(Enum):
    """
    Specifies the relaxation strategy.
    """
    UNKNOWN = 0
    JABR = 1

    @classmethod
    def from_str(cls, string):
        if string == "jabr":
            return cls.JABR

        return cls.UNKNOWN
