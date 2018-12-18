import os
import nibabel as nib
import numpy as np
from numpy.testing import assert_array_equal, assert_raises, assert_equal, assert_almost_equal

from VECtorsToolkit.fields import queries as qr

from .decorators_tools import create_and_erase_temporary_folder_with_a_dummy_nifti_image, pfo_tmp_test

''' test check_omega '''


def test_check_omega_type():
    with assert_raises(IOError):
        qr.check_omega((10, 10, 10.2))


def test_check_omega_wrong_dimension4():
    with assert_raises(IOError):
        qr.check_omega((10, 10, 10, 1))


def test_check_omega_wrong_dimension1():
    with assert_raises(IOError):
        qr.check_omega((10, ))


def test_check_omega_ok():
    assert qr.check_omega((10, 11, 12)) == 3
    assert qr.check_omega((10, 11)) == 2


''' test check_is_vf '''


def test_check_is_vf_wrong_input():
    with assert_raises(IOError):
        qr.check_is_vf([1, 2, 3])


def test_check_is_vf_wrong_input_len():
    with assert_raises(IOError):
        qr.check_is_vf(np.array([1, 2, 3]))


def test_check_is_vf_mismatch_omega_last_dimension():
    with assert_raises(IOError):
        qr.check_is_vf(np.zeros([10, 10, 10, 1, 7]))


def test_check_is_vf_ok():
    assert qr.check_is_vf(np.zeros([10, 10, 10, 1, 3])) == 3
    assert qr.check_is_vf(np.zeros([10, 10, 10, 1, 9])) == 3
    assert qr.check_is_vf(np.zeros([10, 10, 1, 1, 2])) == 2
    assert qr.check_is_vf(np.zeros([10, 10, 1, 1, 4])) == 2


''' test get_omega '''


def test_get_omega_from_vf_wrong_input():
    with assert_raises(IOError):
        qr.get_omega_from_vf(np.zeros([10, 10, 10, 1, 2]))


def test_get_omega_from_vf_3d():
    assert_array_equal(qr.get_omega_from_vf(np.zeros([10, 10, 10, 1, 3])), [10, 10, 10])


def test_get_omega_from_vf_2d():
    assert_array_equal(qr.get_omega_from_vf(np.zeros([10, 10, 1, 1, 2])), [10, 10])


''' test from_image_to_omega'''


def test_from_image_to_omega_by_non_existing_path():
    with assert_raises(IOError):
        qr.from_nib_to_omega('z_spam_folder')


@create_and_erase_temporary_folder_with_a_dummy_nifti_image
def test_from_image_to_omega_by_path():
    pfi_im = os.path.join(pfo_tmp_test, 'dummy_image.nii.gz')
    expected_omega = (30, 29, 28)
    obtained_omega = qr.from_nib_to_omega(pfi_im)
    assert_array_equal(obtained_omega, expected_omega)


def test_from_image_to_omega_by_nifti():
    expected_omega = (30, 29, 28)
    nib_im = nib.Nifti1Image(np.ones(expected_omega), affine=np.eye(4))
    obtained_omega = qr.from_nib_to_omega(nib_im)
    assert_array_equal(obtained_omega, expected_omega)


''' test vf_shape_from_omega_and_timepoints '''

def test_vf_shape_from_omega_and_timepoints():
    assert_array_equal(qr.vf_shape_from_omega_and_timepoints([10, 10], 3), (10, 10, 1, 3, 2))


''' test vf_norm '''


def test_vf_norm_zeros():
    vf = np.zeros([10, 10, 10, 1, 3])
    assert_equal(qr.vf_norm(vf), 0)


def test_vf_norm_ones():
    vf = np.ones([10, 10, 10, 1, 3])
    assert_almost_equal(qr.vf_norm(vf, passe_partout_size=0, normalized=False), 10**3 * np.sqrt(3))


def test_vf_norm_ones_normalised():
    vf = np.ones([10, 10, 10, 1, 3])
    assert_almost_equal(qr.vf_norm(vf, passe_partout_size=0, normalized=True), np.sqrt(3))


if __name__ == '__main__':
    test_check_omega_type()
    test_check_omega_wrong_dimension4()
    test_check_omega_wrong_dimension1()
    test_check_omega_ok()

    test_check_is_vf_wrong_input()
    test_check_is_vf_wrong_input_len()
    test_check_is_vf_mismatch_omega_last_dimension()
    test_check_is_vf_ok()

    test_get_omega_from_vf_wrong_input()
    test_get_omega_from_vf_3d()
    test_get_omega_from_vf_2d()

    test_vf_shape_from_omega_and_timepoints()

    test_vf_norm_zeros()
    test_vf_norm_ones()
    test_vf_norm_ones_normalised()
