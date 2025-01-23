import warnings
from typing import Dict, Self

import numpy as np
from scipy import sparse

from pandapower.conepower.model_components.admittances import Admittances
from pandapower.conepower.model_components.costs.generator_cost import GeneratorCost
from pandapower.conepower.model_components.lines import Lines
from pandapower.conepower.model_components.vector_variable import VariableSet
from pandapower.conepower.types.line_constraint_type import LineConstraintType
from pandapower.conepower.types.variable_type import VariableType
from pandapower.conepower.unit_conversions.per_unit_converter import PerUnitConverter

from pandapower.pypower.idx_brch import F_BUS, T_BUS, RATE_A
from pandapower.pypower.idx_cost import MODEL, STARTUP, SHUTDOWN, NCOST, COST, POLYNOMIAL
from pandapower.pypower.idx_gen import GEN_STATUS
from pandapower.pypower.makeSbus import _get_Cg, _get_Sload
from pandapower.pypower.makeYbus import branch_vectors, makeYbus
from pandapower.pypower.opf_model import opf_model


class ModelOpf:
    """
    Represents the Steady-State Single Time-Step Optimal Power Flow (OPF) problem in its original form.

    Note that the only cost function currently supported is a function of the active power injections.

    Attributes
    ----------
    active_generator_cost : GeneratorCost
        The cost of the active power injections of the generators.
    admittances : Admittances
        The admittances of the system that define the electrical network.
    generator_connection_matrix : sparse.csr_matrix
        A matrix that relates the generators to their respective connecting nodes.
    lines : Lines
        A list of all electrical lines in the network, including their respective limits.
    loads_active : np.ndarray
        The active power consumptions per node.
    loads_reactive : np.ndarray
        The reactive power consumptions per node.
    nof_unique_edges : int
        The number of edges in the network when disregarding duplicate lines.
        Since more than one electrical line can occur per edge,
        the value of `_nof_unique_edges` is generally smaller equal the number of `lines`.
    nof_nodes : int
        The number of nodes in the network.
    nof_variables : int
        The number of scalar-valued variables of the problem.
    values : np.ndarray
        The variable vector.
    variable_sets : Dict[VariableType, VariableSet]
        The sets the variables can be grouped into.
        For OPF these are:
        `PG` (Active Power Injections),
        `QG` (Reactive Power Injections),
        `UMAG` (Voltage Magnitudes),
        `UANG` (Voltage Phase Angles).
    """
    active_generator_cost: GeneratorCost
    admittances: Admittances
    generator_connection_matrix: sparse.csr_matrix
    lines: Lines
    loads_active: np.ndarray
    loads_reactive: np.ndarray
    nof_unique_edges: int
    nof_nodes: int
    nof_variables: int
    values: np.ndarray
    variable_sets: Dict[VariableType, VariableSet]

    def __init__(self):
        self.nof_unique_edges = 0
        self.nof_nodes = 0
        self.nof_variables = 0
        self.variable_sets = {}

    @classmethod
    def from_om(cls, om: opf_model, line_constraint_type: LineConstraintType = LineConstraintType.CURRENT) -> Self:
        """
        Gathers the information from the PYPOWER model and creates an instance of the ModelOpf class.

        Parameters
        ----------
        om : opf_model
            The mathematical model as defined by PYPOWER.
        line_constraint_type : LineConstraintType, optional
            The type of line constraints that are to be enforced. Default is current constraints.

        Returns
        -------
        model : ModelOpf
            An instance of the ModelOpf class.

        Raises
        ------
        ValueError
            If the cost function is not supported.
        """
        # initialize
        model = cls()
        model.nof_variables = om.var['N']

        # variables
        model.values = np.empty(model.nof_variables)
        for set_name in om.var['order']:
            starting_index = om.var['idx']['i1'][set_name]
            ending_index = om.var['idx']['iN'][set_name]
            var_type = VariableType.from_str(set_name)
            var_set = VariableSet(om.var['data']['v0'][set_name],
                                  om.var['data']['vl'][set_name],
                                  om.var['data']['vu'][set_name],
                                  model.values[starting_index:ending_index])
            model.variable_sets[var_type] = var_set

        # admittance matrix and number of nodes and edges
        base_mva, bus, gen, branch = \
            om.ppc["baseMVA"], om.ppc["bus"], om.ppc["gen"], om.ppc["branch"]
        y_tt, y_ff, y_ft, y_tf = branch_vectors(branch, branch.shape[0])
        y_bus, _, _ = makeYbus(base_mva, bus, branch)
        model.admittances = Admittances(y_bus, y_ff, y_ft, y_tf, y_tt)

        # create converter to convert everything into p.u.
        converter = PerUnitConverter(base_mva)

        # get number of nodes and a list of all lines
        model.nof_nodes = bus.shape[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            buses_from = branch[:, F_BUS].astype(int)
            buses_to = branch[:, T_BUS].astype(int)
            max_flows = converter.from_power(branch[:, RATE_A].astype(float))
        model.lines = Lines(buses_from, buses_to, max_flows, line_constraint_type)

        # calculate unique number of edges in case there exist more than one line between two nodes
        edges = np.column_stack((model.lines.buses_from, model.lines.buses_to))
        unique_edges = np.unique(np.sort(edges, axis=1), axis=0)
        model.nof_unique_edges = unique_edges.shape[0]
        assert model.lines.nof_lines >= model.nof_unique_edges

        # loads
        loads = converter.from_power(_get_Sload(bus, None))
        model.loads_active = np.real(loads)
        model.loads_reactive = np.imag(loads)

        # generator connection matrix
        on = np.flatnonzero(gen[:, GEN_STATUS] > 0)
        gen_on = gen[on, :]
        model.generator_connection_matrix = _get_Cg(gen_on, bus)

        # generator cost
        gen_cost = om.ppc['gencost']
        nof_generators = model.variable_sets[VariableType.PG].size
        assert gen_cost.shape[0] == nof_generators
        if np.any(gen_cost[:, MODEL] != POLYNOMIAL):
            raise ValueError("Only polynomial cost functions are supported.")
        if np.any(gen_cost[:, STARTUP] != 0):
            raise ValueError("Generator startup costs are not supported.")
        if np.any(gen_cost[:, SHUTDOWN] != 0):
            raise ValueError("Generator shutdown costs are not supported.")
        # filter out empty cost functions
        mask = gen_cost[:, NCOST] > 0
        # define quadratic cost
        if np.all(gen_cost[mask, NCOST] == 3):
            quadratic_coefficients = converter.from_quadratic_generator_cost(gen_cost[:, COST].astype(float))
            linear_coefficients = converter.from_linear_generator_cost(gen_cost[:, COST + 1].astype(float))
            constants = gen_cost[:, COST + 2].astype(float)
        elif np.all(gen_cost[mask, NCOST] == 2):
            quadratic_coefficients = np.zeros(nof_generators, dtype=float)
            linear_coefficients = converter.from_linear_generator_cost(gen_cost[:, COST].astype(float))
            constants = gen_cost[:, COST + 1].astype(float)
        else:
            raise ValueError("Only linear and quadratic cost functions are supported.")
        model.active_generator_cost = GeneratorCost(quadratic_coefficients=quadratic_coefficients,
                                                    linear_coefficients=linear_coefficients,
                                                    constants=constants)

        return model
