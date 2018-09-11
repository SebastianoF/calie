"""
Test module for the aux_functions.py module
"""
from numpy.testing import assert_equal
import numpy as np


from VECtorsToolkit.tools.auxiliary.angles import mod_pipi


''' test for mod_pipi '''


def test_mod_pipi_plain():
    value = 1
    output = mod_pipi(value)
    assert_equal(output, value)


def test_mod_pipi_on_high_extreme():
    inp = np.pi
    expected_output = np.pi
    output = mod_pipi(inp)
    assert_equal(output, expected_output)


def test_mod_pipi_on_low_extreme():
    inp = - np.pi
    expected_output = np.pi
    output = mod_pipi(inp)
    assert_equal(output, expected_output)


def test_mod_pipi_for_greater_pi():
    additive_const = 2
    inp = np.pi + additive_const
    expected_output = - np.pi + additive_const
    output = mod_pipi(inp)
    assert_equal(output, expected_output)


def test_mod_pipi_for_smaller_pi():
    additive_const = 2
    inp = - (np.pi + additive_const)
    expected_output = np.pi - additive_const
    output = mod_pipi(inp)
    assert_equal(output, expected_output)


if __name__ == '__main__':
    test_mod_pipi_plain()
    test_mod_pipi_on_high_extreme()
    test_mod_pipi_on_low_extreme()
    test_mod_pipi_for_greater_pi()
    test_mod_pipi_for_smaller_pi()
