"""
Test module for the aux_functions.py module
"""
from numpy.testing import assert_equal
import numpy as np

from VECtorsToolkit.tools.auxiliary.angles import mod_pipi


''' test for mod_pipi '''


# noinspection PyTypeChecker
def test_mod_pipi_plain():
    alpha = 1
    output = mod_pipi(alpha)
    assert_equal(output, alpha)


# noinspection PyTypeChecker
def test_mod_pipi_on_high_extreme():
    alpha = np.pi
    expected_output = np.pi
    output = mod_pipi(alpha)
    assert_equal(output, expected_output)


# noinspection PyTypeChecker
def test_mod_pipi_on_low_extreme():
    alpha = - np.pi
    expected_output = np.pi
    output = mod_pipi(alpha)
    assert_equal(output, expected_output)


# noinspection PyTypeChecker
def test_mod_pipi_for_greater_pi():
    additive_const = 2
    alpha = np.pi + additive_const
    expected_output = - np.pi + additive_const
    output = mod_pipi(alpha)
    assert_equal(output, expected_output)


# noinspection PyTypeChecker
def test_mod_pipi_for_smaller_pi():
    additive_const = 2
    alpha = - (np.pi + additive_const)
    expected_output = np.pi - additive_const
    output = mod_pipi(alpha)
    assert_equal(output, expected_output)


# noinspection PyTypeChecker
def test_mod_pipi_for_negative_angle():
    alpha = - (3/2.) * np.pi
    expected_output = alpha + 2 * np.pi
    output = mod_pipi(alpha)
    assert_equal(output, expected_output)


# noinspection PyTypeChecker
def test_mod_pipi_for_smaller_minus_2pi():
    alpha = - (5/2.) * np.pi
    expected_output = alpha % (-2 * np.pi)
    output = mod_pipi(alpha)
    assert_equal(output, expected_output)


if __name__ == '__main__':

    test_mod_pipi_plain()
    test_mod_pipi_on_high_extreme()
    test_mod_pipi_on_low_extreme()
    test_mod_pipi_for_greater_pi()
    test_mod_pipi_for_smaller_pi()
    test_mod_pipi_for_negative_angle()
    test_mod_pipi_for_smaller_minus_2pi()
