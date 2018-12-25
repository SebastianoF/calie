import os
from os.path import join as jph

from matplotlib import pyplot
import numpy as np
import nibabel as nib
import scipy
import nilabels as nis

from benchmarking.b_path_manager import pfo_brainweb, pfo_output_A1


if __name__ == '__main__':
    control = {'prepare_data'    : True,
               'elaborate'       : True,
               'elaborate_steps' : True,
               'make_video'      : True}

    params = {'show_results' : True}

    subject_id = 'BW38'
    labels_brain = [2, 3]
    y_slice = 118
    x_lim = [40, 22]

    if control['prepare_data']:

        pfi_input_T1W = jph(pfo_brainweb, 'A_nifti', subject_id, '{}_T1W.nii.gz'.format(subject_id))
        pfi_input_crisp = jph(pfo_brainweb, 'A_nifti', subject_id, '{}_CRISP.nii.gz'.format(subject_id))
        assert os.path.exists(pfi_input_T1W), pfi_input_T1W
        assert os.path.exists(pfi_input_crisp), pfi_input_crisp

        # get skull strip mask
        pfi_brain_tissue_mask = jph(pfo_output_A1, '{}_brain_tissue.nii.gz'.format(subject_id))
        nis_app = nis.App()
        nis_app.manipulate_labels.assign_all_other_labels_the_same_value(
            pfi_input_crisp, pfi_brain_tissue_mask, labels_brain, 0
        )
        nis_app.manipulate_labels.relabel(
            pfi_brain_tissue_mask, pfi_brain_tissue_mask, labels_brain, [1, ] * len(labels_brain)
        )

        # skull strip
        pfi_skull_stripped = jph(pfo_output_A1, '{}_T1W_brain.nii.gz'.format(subject_id))
        nis_app.math.prod(pfi_brain_tissue_mask, pfi_input_T1W, pfi_skull_stripped)

        # get a slice in PNG
        im_skullstripped = nib.load(pfi_skull_stripped)
        pfi_coronal_slice = jph(pfo_output_A1, '{}_coronal.jpg'.format(subject_id))
        scipy.misc.toimage(
            im_skullstripped.get_data()[x_lim[0]:x_lim[1], y_slice, :].T
        ).save(pfi_coronal_slice)

    else:
        # check data had been prepared
        pfi_brain_tissue_mask = jph(pfo_output_A1, '{}_brain_tissue.nii.gz'.format(subject_id))
        pfi_skull_stripped = jph(pfo_output_A1, '{}_T1W_brain.nii.gz'.format(subject_id))
        pfi_coronal_slice = jph(pfo_output_A1, '{}_coronal.jpg'.format(subject_id))
        assert os.path.exists(pfi_brain_tissue_mask)
        assert os.path.exists(pfi_skull_stripped)

    if params['show_results']:
        coronal_slice = scipy.ndimage.imread(pfi_coronal_slice)
        pyplot.imshow(coronal_slice, origin='lower', cmap='Greys', interpolation='bilinear')
        pyplot.show()

    if control['elaborate']:
        # generate rotational vector field same dimension of the given image, centered at the image centre

        # resample it

        # visualise it in the triptych
        pass

    else:
        # check data had been elaborated
        pass

    if control['elaborate_steps']:

        # Same as elaborate with subsequent steps.
        pass

    if control['make_video']:
        pass