import numpy as np
from numpy.testing import assert_array_equal, assert_raises

from VECtorsToolkit.tools.fields.generate_vf import generate_random, generate_from_matrix, \
    generate_from_projective_matrix


''' test generate_random '''


def test_generate_random_wrong_timepoints():
    with assert_raises(IndexError):
        generate_random((10, 11), t=2)


def test_generate_random_wrong_omega():
    with assert_raises(IndexError):
        generate_random((10, 11, 12, 12), t=2)


def test_generate_random_test_shape_2d():
    vf = generate_random((10, 11))
    assert_array_equal(vf.shape, (10, 11, 1, 1, 2))


def test_generate_random_test_shape_3d():
    vf = generate_random((10, 11, 9))
    assert_array_equal(vf.shape, (10, 11, 9, 1, 3))


''' test generate_from_matrix '''


def test_generate_from_matrix_wrong_timepoints():
    with assert_raises(IndexError):
        generate_from_matrix((10, 11), np.eye(4), t=2)


def test_generate_from_matrix_wrong_structure():
    with assert_raises(IOError):
        generate_from_matrix((10, 11), np.eye(4), t=1, structure='spam')


def test_generate_from_matrix_incompatible_matrix_omega_2d():
    with assert_raises(IOError):
        generate_from_matrix((10, 10), np.eye(4), t=1)


def test_generate_from_matrix_incompatible_matrix_omega_3d():
    with assert_raises(IOError):
        generate_from_matrix((10, 10, 12), np.eye(5), t=1)


def test_generate_from_matrix_from_algebra_element():
    pass


def test_generate_from_matrix_from_group_element():
    pass


''' test generate_from_projective_matrix '''


def test_generate_from_projective_matrix_wrong_timepoints():
    pass


def test_generate_from_projective_matrix_wrong_structure():
    pass


def test_generate_from_projective_matrix_from_algebra_element():
    pass


def test_generate_from_projective_matrix_from_group_element():
    pass






if __name__ == '__main__':
    test_generate_random_wrong_timepoints()
    test_generate_random_wrong_omega()
    test_generate_random_test_shape_2d()
    test_generate_random_test_shape_3d()

    test_generate_from_matrix_wrong_structure()
    test_generate_from_matrix_wrong_timepoints()
    test_generate_from_matrix_incompatible_matrix_omega_2d()
    test_generate_from_matrix_incompatible_matrix_omega_3d()
    test_generate_from_matrix_from_algebra_element()
    test_generate_from_matrix_from_group_element()

    test_generate_from_projective_matrix_wrong_structure()
    test_generate_from_projective_matrix_from_algebra_element()
    test_generate_from_projective_matrix_from_group_element()
