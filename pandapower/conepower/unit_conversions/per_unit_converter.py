import numpy as np


class PerUnitConverter:
    """
    Converter that converts quantities from and to the p.u. system.
    """
    _ref_power_mva: float
    _ref_voltage_kv: float

    def __init__(self, ref_power_mva: float, ref_voltage_kv: float = None):
        self._ref_power_mva = ref_power_mva
        self._ref_voltage_kv = ref_voltage_kv

    def from_linear_generator_cost(self, cost_per_mw: np.ndarray[float]) -> np.ndarray[float]:
        """
        Converts €/MW to €/p.u.
        """
        return cost_per_mw * self._ref_power_mva

    def from_quadratic_generator_cost(self, cost_per_mw_sq: np.ndarray[float]) -> np.ndarray[float]:
        """
        Converts €/(MW^2) to €/(p.u.^2)
        """
        return cost_per_mw_sq * self._ref_power_mva ** 2

    def from_power(self, power_mva: np.ndarray[float]) -> np.ndarray[float]:
        """
        Converts MW to p.u.
        """
        return power_mva / self._ref_power_mva

    def to_power(self, power_pu: np.ndarray[float]) -> np.ndarray[float]:
        """
        Converts p.u. to MW.
        """
        return power_pu * self._ref_power_mva
