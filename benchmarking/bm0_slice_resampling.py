import os
from os.path import join as jph

from matplotlib import pylab
import numpy as np
import nibabel as nib
import scipy
import nilabels as nis

from VECtorsToolkit.fields import generate as gen
from VECtorsToolkit.transformations import se2
from VECtorsToolkit.visualisations.fields import triptych
from VECtorsToolkit.fields import compose as cp
from VECtorsToolkit.operations import lie_exp
from benchmarking.b_path_manager import pfo_brainweb, pfo_output_A1


if __name__ == '__main__':
    control = {'prepare_data'    : False,
               'elaborate'       : True,
               'elaborate_steps' : True,
               'make_video'      : True}

    params = {'show_results' : True,
              'deformation_model' : 'gauss'}

    subject_id = 'BW38'
    labels_brain = [2, 3]
    y_slice = 118
    x_lim = [40, -40]

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
        im_skull_stripped = nib.load(pfi_skull_stripped)
        pfi_coronal_slice = jph(pfo_output_A1, '{}_coronal.jpg'.format(subject_id))
        scipy.misc.toimage(
            im_skull_stripped.get_data()[x_lim[0]:x_lim[1], y_slice, :].T
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
        # pylab.imshow(coronal_slice, origin='lower', cmap='Greys', interpolation='bilinear')
        # pylab.show(block=False)

    if control['elaborate']:
        # generate rotational vector field same dimension of the given image, centered at the image centre
        coronal_slice = scipy.ndimage.imread(pfi_coronal_slice)
        omega = coronal_slice.shape

        passepartout = 4
        sio = 3

        # -> transformation model <- #
        if params['deformation_model'] == 'rotation':

            x_c, y_c = [c / 2 for c in omega]

            theta = np.pi / 12

            tx = (1 - np.cos(theta)) * x_c + np.sin(theta) * y_c
            ty = -np.sin(theta) * x_c + (1 - np.cos(theta)) * y_c

            m_0 = se2.Se2G(theta, tx, ty)
            dm_0 = se2.se2g_log(m_0)

            print(m_0.get_matrix)
            print('')
            print(dm_0.get_matrix)

            sampling_svf = (15, 15)

            # Generate subsequent vector fields
            sdisp_0 = gen.generate_from_matrix(list(omega), m_0.get_matrix, structure='group')
            svf_0 = gen.generate_from_matrix(list(omega), dm_0.get_matrix, structure='algebra')

        if params['deformation_model'] == 'gauss':

            sampling_svf = (5, 5)

            svf_0 = gen.generate_random(omega, 1, (20, 4))

            l_exp = lie_exp.LieExp()
            sdisp_0 = l_exp.scaling_and_squaring(svf_0)

        else:
            raise IOError

        # resample it
        coronal_slice_resampled = cp.scalar_dot_lagrangian(coronal_slice, sdisp_0)

        pylab.imshow(coronal_slice_resampled, origin='lower', cmap='Greys', interpolation='bilinear')
        pylab.show(block=True)

        # coronal_slice_resampled = coronal_slice

        # fields_at_the_window.see_field(svf_0)
        # pylab.show()

        # visualise it in the triptych
        triptych.triptych_image_quiver_image(coronal_slice, svf_0, coronal_slice_resampled, sampling_svf=sampling_svf,
                                             fig_tag=2, h_slice=0)

    if params['show_results']:
        pylab.show()

    else:
        # check data had been elaborated
        pass

    if control['elaborate_steps']:

        # Same as elaborate with subsequent steps.
        pass

    if control['make_video']:
        pass