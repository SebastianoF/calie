import numpy as np
from numpy.testing import assert_array_equal, assert_raises

from calie.fields import generate as gen


''' test generate_random '''


def test_generate_random_wrong_timepoints():
    with assert_raises(IndexError):
        gen.generate_random((10, 11), t=2)


def test_generate_random_wrong_omega():
    with assert_raises(IndexError):
        gen.generate_random((10, 11, 12, 12), t=2)


def test_generate_random_test_shape_2d():
    vf = gen.generate_random((10, 11))
    assert_array_equal(vf.shape, (10, 11, 1, 1, 2))


def test_generate_random_test_shape_3d():
    vf = gen.generate_random((10, 11, 9))
    assert_array_equal(vf.shape, (10, 11, 9, 1, 3))


''' test generate_from_matrix '''


def test_generate_from_matrix_wrong_timepoints():
    with assert_raises(IndexError):
        gen.generate_from_matrix((10, 11), np.eye(4), t=2)


def test_generate_from_matrix_wrong_structure():
    with assert_raises(IOError):
        gen.generate_from_matrix((10, 11), np.eye(4), t=1, structure='spam')


def test_generate_from_matrix_incompatible_matrix_omega_2d():
    with assert_raises(IOError):
        gen.generate_from_matrix((10, 10), np.eye(4), t=1)


def test_generate_from_matrix_incompatible_matrix_omega_3d():
    with assert_raises(IOError):
        gen.generate_from_matrix((10, 10, 12), np.eye(5), t=1)


def test_generate_from_matrix_from_algebra_element_2d():
    theta, tx, ty = np.pi / 8, 5, 5
    a1 = [0, -theta, tx]
    a2 = [theta, 0, ty]
    a3 = [0, 0, 0]
    m = np.array([a1, a2, a3])

    vf = gen.generate_from_matrix((10, 10), m, t=1, structure='algebra')

    vf_expected = np.zeros((10, 10, 1, 1, 2))

    for x in range(10):
        for y in range(10):
            vf_expected[x, y, 0, 0, :] = m.dot(np.array([x, y, 1]))[:2]

    assert_array_equal(vf, vf_expected)


def test_generate_from_matrix_from_group_element_2d():
    theta, tx, ty = np.pi / 10, 5, 5
    a1 = [np.cos(theta), -np.sin(theta), tx]
    a2 = [np.sin(theta), np.cos(theta), ty]
    a3 = [0, 0, 1]
    m = np.array([a1, a2, a3])

    vf = gen.generate_from_matrix((10, 10), m, t=1, structure='group')

    vf_expected = np.zeros((10, 10, 1, 1, 2))

    # move to lagrangian coordinates to compute the ground truth:
    m = m - np.eye(3)

    for x in range(10):
        for y in range(10):
            vf_expected[x, y, 0, 0, :] = m.dot(np.array([x, y, 1]))[:2]

    assert_array_equal(vf, vf_expected)


def test_generate_from_matrix_from_algebra_element_3d_2d_matrix():
    theta, tx, ty = np.pi / 8, 5, 5
    a1 = [0, -theta, tx]
    a2 = [theta, 0, ty]
    a3 = [0, 0, 0]
    m = np.array([a1, a2, a3])

    vf = gen.generate_from_matrix((10, 10, 5), m, t=1, structure='algebra')

    vf_expected = np.zeros((10, 10, 5, 1, 3))

    for x in range(10):
        for y in range(10):
            vf_expected[x, y, 0, 0, :] = list(m.dot(np.array([x, y, 1]))[:2]) + [0]

    for z in range(1, 5):
        vf_expected[..., z, 0, :] = vf_expected[..., 0, 0, :]

    assert_array_equal(vf, vf_expected)


def test_generate_from_matrix_from_group_element_3d_2d_matrix():
    theta, tx, ty = np.pi / 10, 5, 5
    a1 = [np.cos(theta), -np.sin(theta), tx]
    a2 = [np.sin(theta), np.cos(theta), ty]
    a3 = [0, 0, 1]
    m = np.array([a1, a2, a3])

    vf = gen.generate_from_matrix((10, 10, 5), m, t=1, structure='group')

    vf_expected = np.zeros((10, 10, 5, 1, 3))

    # move to lagrangian coordinates to compute the ground truth:
    m = m - np.eye(3)

    for x in range(10):
        for y in range(10):
            vf_expected[x, y, 0, 0, :] = list(m.dot(np.array([x, y, 1]))[:2]) + [0]

    for z in range(1, 5):
        vf_expected[..., z, 0, :] = vf_expected[..., 0, 0, :]

    assert_array_equal(vf, vf_expected)


def test_generate_from_matrix_from_algebra_element_3d_3d_matrix():
    r = np.random.randn(3, 3)
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = np.array([3, 4, 5])

    vf = gen.generate_from_matrix((10, 10, 5), m, t=1, structure='algebra')
    vf_expected = np.zeros((10, 10, 5, 1, 3))

    for x in range(10):
        for y in range(10):
            for z in range(5):
                vf_expected[x, y, z, 0, :] = list(m.dot(np.array([x, y, z, 1])))[:3]

    assert_array_equal(vf, vf_expected)


def test_generate_from_matrix_from_group_element_3d_3d_matrix():
    r = np.random.randn(3, 3)
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = np.array([3, 4, 5])

    vf = gen.generate_from_matrix((10, 10, 5), m, t=1, structure='group')
    vf_expected = np.zeros((10, 10, 5, 1, 3))

    # move to lagrangian coordinates to compute the ground truth:
    m = m - np.eye(4)

    for x in range(10):
        for y in range(10):
            for z in range(5):
                vf_expected[x, y, z, 0, :] = list(m.dot(np.array([x, y, z, 1])))[:3]

    assert_array_equal(vf, vf_expected)


''' test generate_from_projective_matrix '''


def test_generate_from_projective_matrix_wrong_timepoints():
    with assert_raises(IndexError):
        gen.generate_from_projective_matrix((10, 11), np.eye(4), t=2)


def test_generate_from_projective_matrix_wrong_structure():
    with assert_raises(IOError):
        gen.generate_from_projective_matrix((10, 11), np.eye(4), t=1, structure='spam')


def test_generate_from_projective_matrix_with_algebra_element():
    pass


def test_generate_from_projective_matrix_with_group_element():
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
    test_generate_from_matrix_from_algebra_element_2d()
    test_generate_from_matrix_from_group_element_2d()
    test_generate_from_matrix_from_algebra_element_3d_2d_matrix()
    test_generate_from_matrix_from_group_element_3d_2d_matrix()
    test_generate_from_matrix_from_algebra_element_3d_3d_matrix()
    test_generate_from_matrix_from_group_element_3d_3d_matrix()

    test_generate_from_projective_matrix_wrong_timepoints()
    test_generate_from_projective_matrix_wrong_structure()
    test_generate_from_projective_matrix_with_algebra_element()
    test_generate_from_projective_matrix_with_group_element()
