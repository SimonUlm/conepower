import os

import pytest

import pandapower as pp


FOLDER = os.path.join(pp.pp_dir, 'test', 'conepower', 'testfiles')


def calculate_relaxation_gap(filename, ac_cost):
    file = os.path.join(FOLDER, filename)
    net = pp.converter.from_mpc(file)
    pp.runconvopp(net, check_connectivity=True, suppress_warnings=False, enforce_ext_grid_vm=False)
    return round((ac_cost - net["res_cost"]) / ac_cost, 4)


def test_pglib_opf_case3_lmbd():
    # relevant (line constraint error)
    filename = "pglib_opf_case3_lmbd.m"
    ac_cost = 5812.64
    expected_gap = 0.0132
    assert calculate_relaxation_gap(filename, ac_cost) == expected_gap


def test_pglib_opf_case5_pjm():
    # remove
    filename = "pglib_opf_case5_pjm.m"
    ac_cost = 17552
    expected_gap = 0.1454 # result taken from powermodels, since result from paper does not match
    assert calculate_relaxation_gap(filename, ac_cost) == expected_gap


def test_pglib_opf_case14_ieee():
    filename = "pglib_opf_case14_ieee.m"
    ac_cost = 2178.1
    expected_gap = 0.0011
    assert calculate_relaxation_gap(filename, ac_cost) == expected_gap


def test_pglib_opf_case24_ieee_rts():
    # FAIL (REASON UNKNOWN, MAYBE BECAUSE OF MULTIPLE GENERATORS)
    filename = "pglib_opf_case24_ieee_rts.m"
    ac_cost = 63352
    expected_gap = 0.0002
    assert calculate_relaxation_gap(filename, ac_cost) == expected_gap


@pytest.mark.skip
def test_pglib_opf_case30_as():
    # DOES NOT CONVERGE, BUT WOULD CONVERGE AND PASS WITH RELTOL OF 1e-6
    filename = "pglib_opf_case30_as.m"
    ac_cost = 803.13
    expected_gap = 0.0006
    assert calculate_relaxation_gap(filename, ac_cost) == expected_gap


def test_pglib_opf_case30_ieee():
    filename = "pglib_opf_case30_ieee.m"
    ac_cost = 8208.5
    expected_gap = 0.1884
    assert calculate_relaxation_gap(filename, ac_cost) == expected_gap


def test_pglib_opf_case39_epri():
    # remove
    filename = "pglib_opf_case39_epri.m"
    ac_cost = 138420
    expected_gap = 0.0055 # result taken from powermodels, since result from paper does not match
    assert calculate_relaxation_gap(filename, ac_cost) == expected_gap


def test_pglib_opf_case57_ieee():
    # relevant (allocation error due to duplicate lines)
    filename = "pglib_opf_case57_ieee.m"
    ac_cost = 37589
    expected_gap = 0.0016
    assert calculate_relaxation_gap(filename, ac_cost) == expected_gap


def test_pglib_opf_case73_ieee_rts():
    # FAIL (REASON UNKNOWN, MAYBE BECAUSE OF MULTIPLE GENERATORS)
    filename = "pglib_opf_case73_ieee_rts.m"
    ac_cost = 189760
    expected_gap = 0.0004
    assert calculate_relaxation_gap(filename, ac_cost) == expected_gap

def test_pglib_opf_case89_pegase():
    # WEIRD CVXOPT FAILURE, BUT WOULD CONVERGE AND PASS WITH RELTOL OF 1e-6
    filename = "pglib_opf_case89_pegase.m"
    ac_cost = 107290
    expected_gap = 0.0075
    assert calculate_relaxation_gap(filename, ac_cost) == expected_gap


def test_pglib_opf_case118_ieee():
    # FAIL (rounding error)
    filename = "pglib_opf_case118_ieee.m"
    ac_cost = 97214
    expected_gap = 0.0091
    assert calculate_relaxation_gap(filename, ac_cost) == expected_gap


def test_pglib_opf_case162_ieee_dtc():
    # COMPLETE FAILURE (WRONG RESULT AND TOO MANY ITERATIONS FOR 1e-9)
    filename = "pglib_opf_case162_ieee_dtc.m"
    ac_cost = 108080
    expected_gap = 0.0595
    assert calculate_relaxation_gap(filename, ac_cost) == expected_gap


def test_pglib_opf_case179_goc():
    filename = "pglib_opf_case179_goc.m"
    ac_cost = 754270
    expected_gap = 0.0016
    assert calculate_relaxation_gap(filename, ac_cost) == expected_gap


def test_pglib_opf_case240_pserc():
    # FAIL (CLOSE BUT TOO MUCH FOR A SIMPLE ROUNDING ERROR, SOLUTION TOO LOW, ALSO MULTIPLE GENERATORS)
    filename = "pglib_opf_case240_pserc.m"
    ac_cost = 3329700
    expected_gap = 0.0278
    assert calculate_relaxation_gap(filename, ac_cost) == expected_gap


def test_pglib_opf_case300_ieee():
    # FAIL (REASON UNKNOWN, MAYBE BECAUSE OF MULTIPLE GENERATORS)
    filename = "pglib_opf_case240_pserc.m"
    ac_cost = 565220
    expected_gap = 0.0263
    assert calculate_relaxation_gap(filename, ac_cost) == expected_gap
