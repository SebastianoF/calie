import os

import nibabel as nib
import numpy as np
from numpy.testing import assert_array_equal, assert_raises

from VECtorsToolkit.fields.generate_identities import id_lagrangian, id_eulerian, \
    id_lagrangian_like, id_eulerian_like, id_matrices, from_nib_to_omega, \
    id_lagrangian_like_image, id_eulerian_like_image
from .decorators_tools import create_and_erase_temporary_folder_with_a_dummy_nifti_image, pfo_tmp_test

''' test vf_identity_lagrangian '''


def test_vf_identity_lagrangian_bad_input():
    with assert_raises(IOError):
        id_lagrangian((50, 50, 1, 1))


def test_vf_identity_lagrangian_ok_2d():
    assert_array_equal(id_lagrangian((50, 50)), np.zeros((50, 50, 1, 1, 2)))


def test_vf_identity_lagrangian_ok_3d():
    assert_array_equal(id_lagrangian((50, 49, 48)), np.zeros((50, 49, 48, 1, 3)))


def test_vf_identity_lagrangian_ok_3d_timepoints():
    assert_array_equal(id_lagrangian((50, 49, 48), t=3), np.zeros((50, 49, 48, 3, 3)))


''' test vf_identity_eulerian '''


def test_vf_identity_eulerian_wrong_input():
    with assert_raises(IOError):
        id_eulerian((3, 3, 3, 3, 3))


def test_vf_identity_eulerian_ok_2d():
    expected_vf = np.zeros((10, 9, 1, 1, 2))
    for x in range(10):
        for y in range(9):
            expected_vf[x, y, 0, 0, :] = [x, y]

    assert_array_equal(id_eulerian((10, 9)), expected_vf)


def test_vf_identity_eulerian_ok_3d():
    expected_vf = np.zeros((10, 9, 8, 1, 3))
    for x in range(10):
        for y in range(9):
            for z in range(8):
                expected_vf[x, y, z, 0, :] = [x, y, z]

    assert_array_equal(id_eulerian((10, 9, 8)), expected_vf)


''' test identity_lagrangian_like '''


def test_vf_identity_lagrangian_like_wrong_input():
    with assert_raises(IOError):
        id_lagrangian_like(np.ones([4, 4, 4, 4]))


def test_vf_identity_lagrangian_like_ok_2d():
    assert_array_equal(id_lagrangian_like(np.ones((10, 9, 1, 1, 2))), np.zeros((10, 9, 1, 1, 2)))


def test_vf_identity_lagrangian_like_ok_3d():
    assert_array_equal(id_lagrangian_like(np.ones((10, 9, 8, 1, 3))), np.zeros((10, 9, 8, 1, 3)))


''' test identity_eulerian_like '''


def test_vf_identity_eulerian_like_wrong_input():
    with assert_raises(IOError):
        id_eulerian_like(np.zeros([3, 3, 3, 3, 7]))


def test_vf_identity_eulerian_like_ok_2d():
    expected_vf = np.zeros((10, 9, 1, 1, 2))
    for x in range(10):
        for y in range(9):
            expected_vf[x, y, 0, 0, :] = [x, y]

    assert_array_equal(id_eulerian_like(np.ones((10, 9, 1, 1, 2))), expected_vf)


def test_vf_identity_eulerian_like_ok_3d():
    expected_vf = np.zeros((10, 9, 8, 1, 3))
    for x in range(10):
        for y in range(9):
            for z in range(8):
                expected_vf[x, y, z, 0, :] = [x, y, z]

    assert_array_equal(id_eulerian_like(np.ones((10, 9, 8, 1, 3))), expected_vf)


''' test vf_identity_matrices '''


def test_vf_identity_matrices_wrong_input():
    with assert_raises(IOError):
        id_matrices((5, 4, 3, 1))


def test_vf_identity_matrices_test_shape():
    omega = (5, 7)
    expected_shape = (5, 7, 1, 1, 4)
    vf_id = id_matrices(omega)
    assert_array_equal(vf_id.shape, expected_shape)

    omega = (5, 6, 7)
    expected_shape = (5, 6, 7, 1, 9)
    vf_id = id_matrices(omega)
    assert_array_equal(vf_id.shape, expected_shape)

    omega = (5, 7)
    expected_shape = (5, 7, 1, 4, 4)
    vf_id = id_matrices(omega, t=4)
    assert_array_equal(vf_id.shape, expected_shape)

    omega = (5, 6, 7)
    expected_shape = (5, 6, 7, 12, 9)
    vf_id = id_matrices(omega, t=12)
    assert_array_equal(vf_id.shape, expected_shape)


def test_vf_identity_matrices_test_values():
    omega = (5, 4, 3)
    vf_id = id_matrices(omega, t=2)
    assert_array_equal(vf_id[1, 1, 1, 0, :], np.eye(3).flatten())
    assert_array_equal(vf_id[0, 0, 1, 0, :], np.eye(3).flatten())
    assert_array_equal(vf_id[1, 3, 2, 1, :], np.eye(3).flatten())


''' test from_image_to_omega'''


def test_from_image_to_omega_by_non_existing_path():
    with assert_raises(IOError):
        from_nib_to_omega('z_spam_folder')


@create_and_erase_temporary_folder_with_a_dummy_nifti_image
def test_from_image_to_omega_by_path():
    pfi_im = os.path.join(pfo_tmp_test, 'dummy_image.nii.gz')
    expected_omega = (30, 29, 28)
    obtained_omega = from_nib_to_omega(pfi_im)
    assert_array_equal(obtained_omega, expected_omega)


def test_from_image_to_omega_by_nifti():
    expected_omega = (30, 29, 28)
    nib_im = nib.Nifti1Image(np.ones(expected_omega), affine=np.eye(4))
    obtained_omega = from_nib_to_omega(nib_im)
    assert_array_equal(obtained_omega, expected_omega)


''' test vf_identity_lagrangian_like_image '''


def test_vf_identity_lagrangian_like_image():
    omega = (30, 29, 28)
    nib_im = nib.Nifti1Image(np.ones(omega), affine=np.eye(4))
    id_lagrangian = id_lagrangian_like_image(nib_im)
    assert_array_equal(id_lagrangian, np.zeros(list(omega) + [1, 3]))


''' test vf_identity_eulerian_like_image '''


def test_vf_identity_eulerian_like_image():
    omega = (30, 29, 28)
    nib_im = nib.Nifti1Image(np.ones(omega), affine=np.eye(4))
    id_eulerian = id_eulerian_like_image(nib_im)
    expected_id_eulerian = np.zeros(list(omega) + [1, 3])
    for x in range(omega[0]):
        for y in range(omega[1]):
            for z in range(omega[2]):
                expected_id_eulerian[x, y, z, 0, :] = np.array([x, y, z])

    assert_array_equal(id_eulerian, expected_id_eulerian)


if __name__ == '__main__':
    test_vf_identity_lagrangian_ok_2d()
    test_vf_identity_lagrangian_ok_3d()
    test_vf_identity_lagrangian_ok_3d_timepoints()

    test_vf_identity_eulerian_wrong_input()
    test_vf_identity_eulerian_ok_2d()
    test_vf_identity_eulerian_ok_3d()

    test_vf_identity_lagrangian_like_wrong_input()
    test_vf_identity_lagrangian_like_ok_2d()
    test_vf_identity_lagrangian_like_ok_3d()

    test_vf_identity_eulerian_like_wrong_input()
    test_vf_identity_eulerian_like_ok_2d()
    test_vf_identity_eulerian_like_ok_3d()

    test_vf_identity_matrices_wrong_input()
    test_vf_identity_matrices_test_shape()
    test_vf_identity_matrices_test_values()

    test_from_image_to_omega_by_non_existing_path()
    test_from_image_to_omega_by_path()
    test_from_image_to_omega_by_nifti()

    test_vf_identity_lagrangian_like_image()

    test_vf_identity_eulerian_like_image()
