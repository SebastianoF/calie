import os
from os.path import join as jph
import numpy as np
import nibabel as nib


# PATH MANAGER


test_dir = os.path.dirname(os.path.abspath(__file__))
pfo_tmp_test = jph(test_dir, 'z_tmp_test')


# DECORATORS


def create_and_erase_temporary_folder(test_func):
    def wrap(*args, **kwargs):
        # 1) Before: create folder
        os.system('mkdir {}'.format(pfo_tmp_test))
        # 2) Run test
        test_func(*args, **kwargs)
        # 3) After: delete folder and its content
        os.system('rm -r {}'.format(pfo_tmp_test))

    return wrap


def create_and_erase_temporary_folder_with_a_dummy_nifti_image(test_func):
    def wrap(*args, **kwargs):
        # 1) Before: create folder
        os.system('mkdir {}'.format(pfo_tmp_test))
        nib_im = nib.Nifti1Image(np.zeros((30, 29, 28)), affine=np.eye(4))
        nib.save(nib_im, jph(pfo_tmp_test, 'dummy_image.nii.gz'))
        # 2) Run test
        test_func(*args, **kwargs)
        # 3) After: delete folder and its content
        os.system('rm -r {}'.format(pfo_tmp_test))

    return wrap