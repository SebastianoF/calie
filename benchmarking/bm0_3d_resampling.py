import os
from os.path import join as jph

from matplotlib import pylab
import numpy as np
import nibabel as nib
import scipy
import nilabels as nis
from nilabels.tools.aux_methods import utils_nib
import pickle

from calie.fields import generate as gen
from calie.transformations import se2
from calie.visualisations.fields import triptych
from calie.fields import compose as cp
from calie.fields import coordinate as coord
from calie.operations import lie_exp
from calie.transformations import linear
from benchmarking.b_path_manager import pfo_brainweb, pfo_output_A1_3d


if __name__ == '__main__':

    # As bm0_slice_resampling with a 3d volume instead.

    # ----------------------------------------------------
    # ----------  SET UP ---------------------------------
    # ----------------------------------------------------

    # controller and parameters

    control = {'prepare_data'    : True,
               'get_parts'       : True,
               'show_results'    : True,
               'make_video'      : True}

    params = {'deformation_model'    : 'translation',
              'integrate_with_scipy' : False,
              'integration_side'     : 'coronal',  # Can be axial, sagittal, coronal
              'field_of_view_side'   : (0.15, 0.10, 0.25)}

    # more parameters and initialisations:

    np.random.seed(42)

    subject_id = 'BW38'
    labels_brain_to_keep = [2, 3]  # WM and GM

    h_dist = 18

    sio = 3
    num_steps_integrations = 10

    l_exp = lie_exp.LieExp()

    # path to file (pfi) to stuff to save:

    pfi_brain_tissue_mask = jph(pfo_output_A1_3d, '{}_brain_tissue.nii.gz'.format(subject_id))
    pfi_skull_stripped = jph(pfo_output_A1_3d, '{}_T1W_brain.nii.gz'.format(subject_id))
    pfi_slab = jph(pfo_output_A1_3d, '{}_slab.nii.gz'.format(subject_id))
    pfi_int_curves = jph(pfo_output_A1_3d, 'int_curves_new.pickle')
    pfi_svf0 = jph(pfo_output_A1_3d, 'svf_0.nii.gz')

    if params['deformation_model'] in {'translation'}:
        sampling_svf = (12, 12, 12)
    elif params['deformation_model'] in {'rotation', 'linear'} :
        sampling_svf = (10, 10, 10)
    elif params['deformation_model'] == 'gauss':
        sampling_svf = (5, 5, 5)
    else:
        raise IOError

    # ----------------------------------------------------
    # ----------  START ----------------------------------
    # ----------------------------------------------------

    if control['prepare_data']:

        pfi_input_T1W = jph(pfo_brainweb, 'A_nifti', subject_id, '{}_T1W.nii.gz'.format(subject_id))
        pfi_input_crisp = jph(pfo_brainweb, 'A_nifti', subject_id, '{}_CRISP.nii.gz'.format(subject_id))
        assert os.path.exists(pfi_input_T1W), pfi_input_T1W
        assert os.path.exists(pfi_input_crisp), pfi_input_crisp

        # get mask with only the selected label_brain
        nis_app = nis.App()
        nis_app.manipulate_labels.assign_all_other_labels_the_same_value(
            pfi_input_crisp, pfi_brain_tissue_mask, labels_brain_to_keep, 0
        )
        nis_app.manipulate_labels.relabel(
            pfi_brain_tissue_mask, pfi_brain_tissue_mask, labels_brain_to_keep, [1, ] * len(labels_brain_to_keep)
        )

        # skull strip
        nis_app.math.prod(pfi_brain_tissue_mask, pfi_input_T1W, pfi_skull_stripped)

        # get a slab as a nibabel image
        im_slab = nib.load(pfi_skull_stripped)
        sh = im_slab.shape
        centre = [int(c/2) for c in sh]
        # noinspection PyTypeChecker
        x_lim, y_lim, z_lim = [[int(centre[j] - params['field_of_view_side'][j] * centre[j]),
                                int(centre[j] + params['field_of_view_side'][j] * centre[j])] for j in range(3)]
        new_data = im_slab.get_data()[x_lim[0]:x_lim[1], y_lim[0]:y_lim[1], z_lim[0]:z_lim[1]]

        im_slab = utils_nib.set_new_data(im_slab, new_data)
        nib.save(im_slab, pfi_slab)

    else:
        # check data had been prepared
        assert os.path.exists(pfi_brain_tissue_mask)
        assert os.path.exists(pfi_skull_stripped)

    if control['get_parts']:
        # Parts are: vector field, list of resampled images and integral curves for increasing steps

        # --- generate rotational vector field same dimension of the given image, centered at the image centre
        im_slab = nib.load(pfi_slab)
        omega = im_slab.shape

        print('Shape of the input image: {}'.format(omega))

        # -> transformation model <- #
        if params['deformation_model'] == 'translation':
            svf_0 = np.zeros(list(omega) + [1, 3])
            svf_0[..., 0] = 0
            svf_0[..., 1] = 12
            svf_0[..., 2] = 0

            # identify origin - set the origin corner to zero and the opposite and the right to ones:
            # svf_0[0:4, 0:4, 0:4, 0, :] = np.zeros_like(svf_0[0:4, 0:4, 0:4, 0, :])
            # svf_0[-4:, -4:, -4:, 0, :] = np.ones_like(svf_0[-4:, -4:, -4:, 0, :])
            # svf_0[-4:, 0:4, 0:4, 0, :] = 2 * np.ones_like(svf_0[-4:, 0:4, 0:4, 0, :])

        elif params['deformation_model'] == 'rotation':

            x_c, y_c = [c / 2 for c in omega]

            theta = np.pi / 8

            tx = (1 - np.cos(theta)) * x_c + np.sin(theta) * y_c
            ty = -np.sin(theta) * x_c + (1 - np.cos(theta)) * y_c

            m_0 = se2.Se2G(theta, tx, ty)
            dm_0 = se2.se2g_log(m_0)

            print(m_0.get_matrix)
            print('')
            print(dm_0.get_matrix)

            # Generate subsequent vector fields
            sdisp_0 = gen.generate_from_matrix(list(omega), m_0.get_matrix, structure='group')
            svf_0 = gen.generate_from_matrix(list(omega), dm_0.get_matrix, structure='algebra')

        elif params['deformation_model'] == 'linear':

            taste = 2
            beta = 0.8

            x_c, y_c, z_c = [c / 2 for c in omega]

            dm = beta * linear.randomgen_linear_by_taste(1, taste, (x_c, y_c))
            svf_0 = gen.generate_from_matrix(omega, dm, structure='algebra')
            sdisp_0 = l_exp.gss_aei(svf_0)

        elif params['deformation_model'] == 'gauss':

            sampling_svf = (5, 5)
            svf_0 = gen.generate_random(omega, 1, (20, 4))
            sdisp_0 = l_exp.scaling_and_squaring(svf_0)

        else:
            raise IOError

        # save svf as nifti image:
        im_svf0 = utils_nib.set_new_data(im_slab, svf_0)
        nib.save(im_svf0, pfi_svf0)
        print('shape of vector field: {}'.format(im_svf0.shape))

        # --- get integral curves and save ---

        # TODO visualisation of integral curves in 2d projection for the given integration_side param.
        # int_curves = []
        #
        # for i in range(sampling_svf[0], omega[0] - sampling_svf[0], sampling_svf[0]):
        #     for j in range(sampling_svf[1], omega[1] - sampling_svf[1], sampling_svf[1]):
        #         int_curves.append(np.array([[i, j]]))
        #
        # for st in range(num_steps_integrations):
        #     print('integrating step {}/{}'.format(st+1, num_steps_integrations))
        #     alpha = (st + 1) / float(num_steps_integrations)
        #     sdisp_0 = l_exp.gss_aei(alpha * svf_0)
        #
        #     sdisp_0 = coord.lagrangian_to_eulerian(sdisp_0)
        #
        #     ind_ij = 0
        #     for i in range(sampling_svf[0], omega[0] - sampling_svf[0], sampling_svf[0]):
        #         for j in range(sampling_svf[1], omega[1] - sampling_svf[1], sampling_svf[1]):
        #             int_curves[ind_ij] = np.vstack([int_curves[ind_ij], sdisp_0[i, j, h_dist, 0, :][:2]])
        #             ind_ij += 1
        #
        # with open(pfi_int_curves, 'wb') as f:
        #     pickle.dump(int_curves, f)

        # get resampled images and save:
        for st in range(num_steps_integrations):
            alpha = -1 * (st + 1) / float(num_steps_integrations)
            sdisp_0 = l_exp.gss_aei(alpha * svf_0)
            slab_resampled_st = cp.scalar_dot_lagrangian(im_slab.get_data(), sdisp_0)

            pfi_slab_resampled_st = jph(
                pfo_output_A1_3d, '{}_slab_step_{}.nii.gz'.format(subject_id, st+1)
            )
            im_slab = utils_nib.set_new_data(im_slab, slab_resampled_st)

            nib.save(im_slab, pfi_slab_resampled_st)

    else:
        assert os.path.exists(pfi_slab)
        assert os.path.exists(pfi_svf0)
        for st in range(num_steps_integrations):
            assert os.path.exists(jph(pfo_output_A1_3d, '{}_slab_step_{}.nii.gz'.format(subject_id, st+1)))

    # ----------------------------------------------------
    # ---------- SHOW ------------------------------------
    # ----------------------------------------------------

    # if control['show_results']:
    #     pylab.close('all')
    #
    #     # load slab
    #     im_slab = nib.load(pfi_slab)
    #     # load svf
    #     svf_0 = nib.load(pfi_svf0)
    #     # load latest transformed
    #     pfi_slab_resampled_last = jph(
    #         pfo_output_A1_3d, '{}_slab_step_{}.nii.gz'.format(subject_id, num_steps_integrations)
    #     )
    #     im_slab_resampled = nib.load(pfi_slab_resampled_last)
    #     # load integral curves
    #     with open(pfi_int_curves, 'rb') as f:
    #         int_curves = pickle.load(f)
    #
    #     # visualise it in the triptych
    #     triptych.volume_quiver_volume(im_slab.get_data(),
    #                                   svf_0,
    #                                   im_slab_resampled.get_data(),
    #                                   sampling_svf=sampling_svf,
    #                                   fig_tag=2, h_slice=0, integral_curves=int_curves)
    #
    #     pylab.show(block=True)
    #
    # if control['make_video']:
    #     # load slice
    #     im_slab = nib.load(pfi_slab)
    #     # load svf
    #     with open(pfi_svf0, 'rb') as f:
    #         svf_0 = pickle.load(f)
    #     # load integral curves
    #     with open(pfi_int_curves, 'rb') as f:
    #         int_curves = pickle.load(f)
    #
    #     # -- Produce images --
    #     for st in range(num_steps_integrations):
    #
    #         pylab.close('all')
    #
    #         pfi_slab_resampled = jph(pfo_output_A1_3d, '{}_slab_step_{}.nii.gz'.format(subject_id, st + 1))
    #         im_slab_resampled = nib.load(pfi_slab_resampled)
    #
    #         int_curves_step = [ic[:st+1, :] for ic in int_curves]
    #
    #         triptych.volume_quiver_volume(im_slab.get_data(),
    #                                       svf_0,
    #                                       im_slab_resampled.get_data(),
    #                                       sampling_svf=sampling_svf,
    #                                       fig_tag=2,
    #                                       h_slice=0,
    #                                       integral_curves=int_curves_step)
    #
    #         pylab.savefig(
    #             jph(pfo_output_A1_3d, 'final_{}_sj_{}_step_{}.jpg'.format(
    #                 params['deformation_model'], subject_id, st+1)
    #                 )
    #         )

        # -- produce video --
        # manually with: ffmpeg -r 2 -i final_sj_BW38_step_%01d.jpg -vcodec gif -y movie.gif
