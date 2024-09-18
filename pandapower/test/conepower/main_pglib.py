import logging

import pandapower as pp
from pandapower.pypower.opf import opf
from pandapower.converter.matpower.from_mpc import _m2ppc
from pandapower.pypower.ppoption import ppoption
from pandapower.conepower.conv_opf import conv_opf


LOGGING_LEVEL = logging.INFO
FREQUENCY = 50
REFERENCE_POWER = 100  # MVar
REFERENCE_VOLTAGE = 400  # kV


def main():
    # setup logging
    logging.basicConfig(filename="/Users/simonakis/src/pandapower/pandapower/test/conepower/logs/main_pglib.log",
                        filemode='w',
                        level=LOGGING_LEVEL)

    # setup path and file
    path = '/Users/simonakis/src/pglib-opf/'
    file = path + 'pglib_opf_case3_lmbd.m'

    # run opf
    ppc_before = _m2ppc(file, 'mpc')
    net = pp.converter.from_mpc(file)
    ppc_after = pp.converter.to_ppc(net, init='flat')
    pass
    ppopt = ppoption()
    ppopt['INIT'] = 'flat'
    #results = opf(ppc_before, ppopt)
    conv_opf(ppc_before, ppopt, 'jabr', True)
    pass

    # run convex opf
    #pp.runconvopp(net, verbose=True, check_connectivity=True, suppress_warnings=False, enforce_equalities=True)


if __name__ == "__main__":
    main()
