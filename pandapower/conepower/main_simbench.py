import pandapower as pp
from pandapower.conepower.networks.process_simbench import StudyCase, lv_grid_to_opf


# net0 = pp.from_json("simbench_networks_original/1-LV-rural2--0-no_sw.json")
# net1 = pp.from_json("simbench_networks_original/1-LV-rural2--1-no_sw.json")
# net2 = pp.from_json("simbench_networks_original/1-LV-rural2--2-no_sw.json")
net0 = pp.from_json("simbench_networks_original/1-LV-urban6--0-no_sw.json")
net1 = pp.from_json("simbench_networks_original/1-LV-urban6--1-no_sw.json")
net2 = pp.from_json("simbench_networks_original/1-LV-urban6--2-no_sw.json")
opf_net = lv_grid_to_opf(net2)
ppc = pp.converter.to_ppc(opf_net, trafo_model='pi', init='flat', mode='opf')
# TODO: Warum ist shunt susceptance negative beim Trafo??? Gehört das so?
# TODO: Schaue, dass die Tap Einstellung passt. Die kommt mir komisch vor.
pass
if False:
    pp.runopp(opf_net, verbose=True, calculate_voltage_angles=False)
else:
    pp.runconvopp(opf_net,
                  verbose=True,
                  calculate_voltage_angles=True,
                  suppress_warnings=False)

pass
