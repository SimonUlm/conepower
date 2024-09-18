import logging
import math

import pandapower as pp
import pandapower.converter as converter
from pandapower.pypower.opf_args import opf_args2
from pandapower.pypower.opf_model import opf_model
from pandapower.pypower.opf_setup import opf_setup
from pandapower.pypower.ppoption import ppoption

LOGGING_LEVEL = logging.INFO
FREQUENCY = 50
REFERENCE_POWER = 100  # MVar
REFERENCE_VOLTAGE = 400  # kV


def _pu2mvar(pu: float) -> float:
    return pu * REFERENCE_POWER


def _pu2ohm(pu: float) -> float:
    return pu * (REFERENCE_VOLTAGE ** 2 / REFERENCE_POWER)


def _bpu2c(b: float) -> float:
    return b * ((10 ** 9) / (2 * math.pi * FREQUENCY)) / (REFERENCE_VOLTAGE ** 2 / REFERENCE_POWER)
    # why 2*pi?


def _b2pu2c(b2: float) -> float:
    return _bpu2c(b2 * 2)


def _create_three_bus_network() -> (pp.pandapowerNet, int):
    # create empty net
    net = pp.create_empty_network(f_hz=FREQUENCY, sn_mva=REFERENCE_POWER)

    # create buses
    b1 = pp.create_bus(net, vn_kv=REFERENCE_VOLTAGE, max_vm_pu=2, min_vm_pu=0)
    b2 = pp.create_bus(net, vn_kv=REFERENCE_VOLTAGE, max_vm_pu=2, min_vm_pu=0)
    b3 = pp.create_bus(net, vn_kv=REFERENCE_VOLTAGE, max_vm_pu=2, min_vm_pu=0)

    # create bus elements
    ext = pp.create_ext_grid(net, bus=b1,
                             vm_pu=1.4,
                             max_p_mw=_pu2mvar(10),
                             min_p_mw=_pu2mvar(0),
                             max_q_mvar=_pu2mvar(10),
                             min_q_mvar=_pu2mvar(-10))
    pp.create_load(net, bus=b2,
                   p_mw=_pu2mvar(0.70),
                   q_mvar=_pu2mvar(0.02))
    pp.create_load(net, bus=b3,
                   p_mw=_pu2mvar(0.65),
                   q_mvar=_pu2mvar(0.02))

    # create branch elements
    pp.create_line_from_parameters(net, from_bus=b1, to_bus=b2,
                                   length_km=1,
                                   r_ohm_per_km=_pu2ohm(0.10),
                                   x_ohm_per_km=_pu2ohm(0.50),
                                   c_nf_per_km=_b2pu2c(0.01),
                                   max_i_ka=float('nan'))
    pp.create_line_from_parameters(net, from_bus=b2, to_bus=b3,
                                   length_km=1,
                                   r_ohm_per_km=_pu2ohm(0.02),
                                   x_ohm_per_km=_pu2ohm(0.20),
                                   c_nf_per_km=_b2pu2c(0.01),
                                   max_i_ka=float('nan'))

    return net, ext


def _create_four_bus_network() -> (pp.pandapowerNet, int):
    # create empty net
    net = pp.create_empty_network(f_hz=FREQUENCY, sn_mva=REFERENCE_POWER)

    # create buses
    b1 = pp.create_bus(net, vn_kv=REFERENCE_VOLTAGE, max_vm_pu=2, min_vm_pu=0)
    b2 = pp.create_bus(net, vn_kv=REFERENCE_VOLTAGE, max_vm_pu=2, min_vm_pu=0)
    b3 = pp.create_bus(net, vn_kv=REFERENCE_VOLTAGE, max_vm_pu=2, min_vm_pu=0)
    b4 = pp.create_bus(net, vn_kv=REFERENCE_VOLTAGE, max_vm_pu=2, min_vm_pu=0)

    # create bus elements
    ext = pp.create_ext_grid(net, bus=b1,
                             vm_pu=1,
                             max_p_mw=_pu2mvar(10),
                             min_p_mw=_pu2mvar(0),
                             max_q_mvar=_pu2mvar(10),
                             min_q_mvar=_pu2mvar(-10))
    pp.create_load(net, bus=b2,
                   p_mw=_pu2mvar(0.90),
                   q_mvar=_pu2mvar(0.02))
    pp.create_load(net, bus=b3,
                   p_mw=_pu2mvar(0.60),
                   q_mvar=_pu2mvar(0.02))
    pp.create_load(net, bus=b4,
                   p_mw=_pu2mvar(0.90),
                   q_mvar=_pu2mvar(0.02))

    # create branch elements
    pp.create_line_from_parameters(net, from_bus=b1, to_bus=b2,
                                   length_km=1,
                                   r_ohm_per_km=_pu2ohm(0.10),
                                   x_ohm_per_km=_pu2ohm(0.10),
                                   c_nf_per_km=_b2pu2c(0.03),
                                   max_i_ka=0.3,
                                   max_loading_percent=100)
    pp.create_line_from_parameters(net, from_bus=b2, to_bus=b3,
                                   length_km=1,
                                   r_ohm_per_km=_pu2ohm(0.01),
                                   x_ohm_per_km=_pu2ohm(0.10),
                                   c_nf_per_km=_b2pu2c(0.01),
                                   max_i_ka=0.2,
                                   max_loading_percent=100)
    pp.create_line_from_parameters(net, from_bus=b4, to_bus=b1,
                                   length_km=1,
                                   r_ohm_per_km=_pu2ohm(0.01),
                                   x_ohm_per_km=_pu2ohm(0.20),
                                   c_nf_per_km=_b2pu2c(0.01),
                                   max_i_ka=0.2,
                                   max_loading_percent=100)

    return net, ext


def main():
    # setup logging
    logging.basicConfig(filename="/Users/simonakis/src/pandapower/pandapower/test/conepower/logs/main_jabr.log",
                        filemode='w',
                        level=LOGGING_LEVEL)

    # create network
    net, ext = _create_four_bus_network()

    # create cost function
    pp.create_poly_cost(net, element=ext, et="ext_grid", cp1_eur_per_mw=1)

    # inspect ppc structure
    ppc = converter.to_ppc(net,
                           calculate_voltage_angles=True,  # unclear what this option actually changes
                           trafo_model='pi',  # this is actually not recommended
                           switch_rx_ratio=2,  # only relevant for switches
                           check_connectivity=True,
                           voltage_depend_loads=False,  # not considered
                           init='flat',
                           mode='opf',
                           take_slack_vm_limits=True)  # ???

    # inspect mathematical object
    ppopt = ppoption(VERBOSE=True, PF_DC=False, INIT='flat')
    ppc, ppopt = opf_args2(ppc, ppopt)
    om = opf_setup(ppc, ppopt)

    pass

    # run opf
    #pp.runopp(net, verbose=True, calculate_voltage_angles=True)

    # run convex opf
    pp.runconvopp(net, verbose=True, check_connectivity=True, suppress_warnings=False, enforce_equalities=True)

    '''Fragen:
    verwenden die für den Referenzbus tatsächlich keine Gleichheitsnb sondern Ungleichheit??? -> Antwort: Ja, aber warum?'''


if __name__ == "__main__":
    main()
