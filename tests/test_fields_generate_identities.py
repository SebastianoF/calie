import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_raises

from VECtorsToolkit.tools.fields.generate_identities import vf_identity_lagrangian, vf_identity_eulerian, \
    vf_identity_lagrangian_like, vf_identity_eulerian_like


''' test vf_identity_lagrangian '''


def test_vf_identity_lagrangian_bad_input():
    with assert_raises(IOError):
        vf_identity_lagrangian((50, 50, 1, 1))


def test_vf_identity_lagrangian_ok_2d():
    assert_array_equal(vf_identity_lagrangian((50, 50)), np.zeros((50, 50, 1, 1, 2)))


def test_vf_identity_lagrangian_ok_3d():
    assert_array_equal(vf_identity_lagrangian((50, 49, 48)), np.zeros((50, 49, 48, 1, 3)))


def test_vf_identity_lagrangian_ok_3d_timepoints():
    assert_array_equal(vf_identity_lagrangian((50, 49, 48), t=3), np.zeros((50, 49, 48, 3, 3)))


def test_vf_identity_eulerian_like_ok():
    pass

    # assert_array_equal(vf_identity_eulerian(vf_identity_eulerian((4, 4)), expected_vf)


if __name__ == '__main__':
    test_vf_identity_lagrangian_ok_2d()
    test_vf_identity_lagrangian_ok_3d()
    test_vf_identity_lagrangian_ok_3d_timepoints()
    test_vf_identity_eulerian_like_ok()