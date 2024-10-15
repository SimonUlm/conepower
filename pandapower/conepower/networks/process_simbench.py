import copy
from enum import Enum

import pandas as pd

import pandapower as pp
from pandapower.create import create_empty_network


REF_BUS_GEN_LIMIT = 10000
REF_BUS_VM = 1.0


class StudyCase(Enum):
    UNDEFINED = 0
    HIGH_LOAD_HIGH_PV = 1
    LOW_LOAD_HIGH_PV = 2

    def to_str(self) -> str:
        if self is self.HIGH_LOAD_HIGH_PV:
            return 'hPV'
        elif self is self.LOW_LOAD_HIGH_PV:
            return 'lPV'
        else:
            assert False


def _sgen_to_gen(sgen: pd.DataFrame, gen: pd.DataFrame) -> pd.DataFrame:
    common_columns = sgen.columns.intersection(gen.columns)
    gen[common_columns] = sgen[common_columns]
    return gen.assign(vm_pu=1.0, slack=False)


def lv_grid_to_opf(net: pp.pandapowerNet, study_case: StudyCase = StudyCase.HIGH_LOAD_HIGH_PV) -> pp.pandapowerNet:
    # temp
    #study_case = StudyCase.LOW_LOAD_HIGH_PV

    # extract data
    bus = net['bus'].copy()
    bus_geodata = net['bus_geodata'].copy()
    ext_grid = net['ext_grid'].copy()
    gen = net['gen'].copy()
    line = net['line'].copy()
    load = net['load'].copy()
    loadcases = net['loadcases'].copy()
    profiles = copy.deepcopy(net['profiles'])
    sgen = net['sgen'].copy()
    std_types = net['std_types'].copy()
    switch = net['switch'].copy()
    trafo = net['trafo'].copy()

    # extract load case, TODO: Profiles verwenden, um weitere Fälle zu erstellen
    p_gen_scaling = loadcases.loc[study_case.to_str()]['PV_p']
    p_load_scaling = loadcases.loc[study_case.to_str()]['pload']
    q_load_scaling = loadcases.loc[study_case.to_str()]['qload']

    # modify data
    ext_grid = ext_grid.assign(vm_pu=REF_BUS_VM,
                               max_p_mw=REF_BUS_GEN_LIMIT,
                               min_p_mw=-REF_BUS_GEN_LIMIT,
                               max_q_mvar=REF_BUS_GEN_LIMIT,
                               min_q_mvar=-REF_BUS_GEN_LIMIT)
    assert ext_grid.shape[0] == 1
    # gen
    gen = _sgen_to_gen(sgen, gen)
    gen['max_p_mw'] = gen['p_mw'] = gen['sn_mva'] * p_gen_scaling
    gen['min_p_mw'] = gen['sn_mva'] * 0.2
    gen['max_q_mvar'] = gen['sn_mva'] * 1.1 * 0.44
    gen['min_q_mvar'] = gen['sn_mva'] * 1.1 * (-0.44)
    # load
    load['p_mw'] *= p_load_scaling
    load['q_mvar'] *= q_load_scaling
    # trafo
    # trafo = trafo.assign(pfe_kw=0.0)
    # TODO: Überlege, ob man das wieder rausnehmen sollte, das beeinflusst die charging conductance!
    #       Evtl. sollte man einen verlustfreien Transformer annehmen.

    # create new net and add cost
    opf_net = create_empty_network()
    opf_net['bus'] = bus
    opf_net['bus_geodata'] = bus_geodata
    opf_net['ext_grid'] = ext_grid
    opf_net['gen'] = gen
    opf_net['line'] = line
    opf_net['load'] = load
    opf_net['std_types'] = std_types
    opf_net['switch'] = switch
    opf_net['trafo'] = trafo
    pp.create_poly_cost(opf_net,
                        element=0,
                        et='ext_grid',
                        cp1_eur_per_mw=1)

    return opf_net
