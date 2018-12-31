import os
from os.path import join as jph

from matplotlib import pylab
import numpy as np
import nibabel as nib
import scipy
import nilabels as nis
import pickle

from VECtorsToolkit.fields import generate as gen
from VECtorsToolkit.transformations import se2
from VECtorsToolkit.visualisations.fields import triptych
from VECtorsToolkit.fields import compose as cp
from VECtorsToolkit.fields import coordinate as coord
from VECtorsToolkit.operations import lie_exp
from VECtorsToolkit.transformations import linear
from benchmarking.b_path_manager import pfo_brainweb, pfo_output_A1


if __name__ == '__main__':

    # TODO BUG yet unsolved coordinates problem when resampling with a scalar field

    # ----------------------------------------------------
    # ----------  SET UP ---------------------------------
    # ----------------------------------------------------

    # controller and parameters

    control = {'prepare_data'    : False,
               'get_parts'       : True,
               'show_results'    : True,
               'make_video'      : True}

    params = {'deformation_model'    : 'linear',
              'integrate_with_scipy' : False}

    # more parameters and initialisations:

    np.random.seed(42)

    subject_id = 'BW38'
    labels_brain_to_keep = [2, 3]  # WM and GM
    y_slice = 118
    x_lim = [40, -40]

    sio = 3
    num_steps_integrations = 10

    taste = ''  # for linear transformations

    l_exp = lie_exp.LieExp()

    # path to file (pfi) to stuff to save:

    pfi_brain_tissue_mask = jph(pfo_output_A1, '{}_brain_tissue.nii.gz'.format(subject_id))
    pfi_skull_stripped = jph(pfo_output_A1, '{}_T1W_brain.nii.gz'.format(subject_id))
    pfi_coronal_slice = jph(pfo_output_A1, '{}_coronal.jpg'.format(subject_id))
    pfi_int_curves = jph(pfo_output_A1, 'int_curves_new.pickle')
    pfi_svf0 = jph(pfo_output_A1, 'svf_0.pickle')

    if params['deformation_model'] in {'translation', 'rotation', 'linear'} :
        sampling_svf = (20, 20)
    elif params['deformation_model'] == 'gauss':
        sampling_svf = (10, 10)
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

        # get a slice in PNG
        im_skull_stripped = nib.load(pfi_skull_stripped)
        scipy.misc.toimage(
            im_skull_stripped.get_data()[x_lim[0]:x_lim[1], y_slice, :].T
        ).save(pfi_coronal_slice)

    else:
        # check data had been prepared
        assert os.path.exists(pfi_brain_tissue_mask)
        assert os.path.exists(pfi_skull_stripped)

    if control['get_parts']:
        # Parts are: vector field, list of resampled images and integral curves for increasing steps

        # --- generate rotational vector field same dimension of the given image, centered at the image centre

        coronal_slice = scipy.ndimage.imread(pfi_coronal_slice)
        omega = coronal_slice.shape

        # -> transformation model <- #
        if params['deformation_model'] == 'translation':
            svf_0 = np.zeros(list(omega)[::-1] + [1, 1, 2])
            svf_0[..., 0] = 10
            svf_0[..., 1] = 20

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
            sdisp_0 = gen.generate_from_matrix(list(omega)[::-1], m_0.get_matrix, structure='group')
            svf_0 = gen.generate_from_matrix(list(omega)[::-1], dm_0.get_matrix, structure='algebra')

        elif params['deformation_model'] == 'linear':

            taste = 1
            beta = 0.5

            x_c, y_c = [c / 2 for c in omega]

            dm = beta * linear.randomgen_linear_by_taste(1, taste, (x_c, y_c))
            svf_0 = gen.generate_from_matrix(omega[::-1], dm, structure='algebra')
            sdisp_0 = l_exp.gss_aei(svf_0)

        elif params['deformation_model'] == 'gauss':

            svf_0 = gen.generate_random(omega[::-1], 1, (20, 2))
            # svf_0 = svf_0[0:omega[0], 1:omega[1], ...]
            sdisp_0 = l_exp.scaling_and_squaring(svf_0)

        else:
            raise IOError

        # save svf:
        with open(pfi_svf0, 'wb') as f:
            pickle.dump(svf_0, f)

        # --- get integral curves and save ---

        if params['integrate_with_scipy']:
            # The first method is a very slow point-wise one.
            # Compare it with the second one to see how quicker it can be to use the flows.
            t0, t1 = 0, 1
            dt = (t1 - t0) / float(num_steps_integrations)

            r = scipy.integrate.ode(
                lambda t, x: list(cp.one_point_interpolation(svf_0, point=x, method='linear'))
            ).set_integrator('dopri5', method='bdf', max_step=dt)

            int_curves = []

            for i in range(sampling_svf[0], omega[0] - sampling_svf[0], sampling_svf[0]):
                for j in range(sampling_svf[1], omega[1] - sampling_svf[1], sampling_svf[1]):

                    print('Integrating vf at the point {} between {} and {}, step size {}'.format((i, j), t0, t1, dt))

                    Y, T = [], []
                    r.set_initial_value([i, j], t0).set_f_params()  # initial conditions are point on the grid
                    Y.append([i, j])
                    while r.successful() and r.t + dt < t1:
                        r.integrate(r.t + dt)
                        Y.append(r.y)

                    S = np.array(np.real(Y))
                    int_curves += [S]

            with open(pfi_int_curves, 'wb') as f:
                pickle.dump(int_curves, f)

        else:
            # Second method, way faster! It is based on the algebraic
            # manipulation of the input svf_0 and on the flow field.

            int_curves = []

            for i in range(sampling_svf[0], omega[0] - sampling_svf[0], sampling_svf[0]):
                for j in range(sampling_svf[1], omega[1] - sampling_svf[1], sampling_svf[1]):
                    int_curves.append(np.array([[i, j]]))

            for st in range(num_steps_integrations):
                print('integrating step {}/{}'.format(st+1, num_steps_integrations))
                alpha = (st + 1) / float(num_steps_integrations)
                sdisp_0 = l_exp.gss_aei(alpha * svf_0)

                sdisp_0 = coord.lagrangian_to_eulerian(sdisp_0)

                ind_ij = 0
                for i in range(sampling_svf[0], omega[0] - sampling_svf[0], sampling_svf[0]):
                    for j in range(sampling_svf[1], omega[1] - sampling_svf[1], sampling_svf[1]):
                        int_curves[ind_ij] = np.vstack([int_curves[ind_ij], sdisp_0[i, j, 0, 0, :]])
                        ind_ij += 1

            with open(pfi_int_curves, 'wb') as f:
                pickle.dump(int_curves, f)

        # get resampled images and save:
        for st in range(num_steps_integrations):
            alpha = -1 * (st + 1) / float(num_steps_integrations)
            sdisp_0 = l_exp.gss_aei(alpha * svf_0)
            coronal_slice_resampled_st = cp.scalar_dot_lagrangian(coronal_slice, sdisp_0)

            # coronal_slice_resampled_st = coronal_slice_resampled_st

            pfi_coronal_slice_resampled_st = jph(pfo_output_A1, '{}_coronal_step_{}.jpg'.format(subject_id, st+1))
            scipy.misc.toimage(coronal_slice_resampled_st).save(pfi_coronal_slice_resampled_st)

    else:
        assert os.path.exists(pfi_coronal_slice)
        assert os.path.exists(pfi_svf0)
        for st in range(num_steps_integrations):
            assert os.path.exists(jph(pfo_output_A1, '{}_coronal_step_{}.jpg'.format(subject_id, st+1)))

    # ----------------------------------------------------
    # ---------- SHOW ------------------------------------
    # ----------------------------------------------------

    if control['show_results']:
        pylab.close('all')

        # load coronal slice
        coronal_slice = scipy.ndimage.imread(pfi_coronal_slice)
        # load svf
        with open(pfi_svf0, 'rb') as f:
            svf_0 = pickle.load(f)
        # load latest transformed
        pfi_coronal_slice_resampled_last = jph(
            pfo_output_A1, '{}_coronal_step_{}.jpg'.format(subject_id, num_steps_integrations)
        )
        coronal_slice_resampled = scipy.ndimage.imread(pfi_coronal_slice_resampled_last)
        # load integral curves
        with open(pfi_int_curves, 'rb') as f:
            int_curves = pickle.load(f)

        # visualise it in the triptych
        triptych.image_quiver_image(coronal_slice, svf_0, coronal_slice_resampled, sampling_svf=sampling_svf,
                                    fig_tag=2, h_slice=0, integral_curves=int_curves)

        pylab.show(block=True)

    if control['make_video']:
        # load coronal slice
        coronal_slice = scipy.ndimage.imread(pfi_coronal_slice)
        # load svf
        with open(pfi_svf0, 'rb') as f:
            svf_0 = pickle.load(f)
        # load integral curves
        with open(pfi_int_curves, 'rb') as f:
            int_curves = pickle.load(f)

        # -- Produce images --
        for st in range(num_steps_integrations):

            pylab.close('all')

            pfi_coronal_slice_resampled = jph(
                pfo_output_A1, '{}_coronal_step_{}.jpg'.format(subject_id, st + 1)
            )
            coronal_slice_resampled = scipy.ndimage.imread(pfi_coronal_slice_resampled)

            int_curves_step = [ic[:st+1, :] for ic in int_curves]

            triptych.image_quiver_image(coronal_slice, svf_0, coronal_slice_resampled,
                                        sampling_svf=sampling_svf,
                                        fig_tag=2, h_slice=0, integral_curves=int_curves_step, show_overlay=False)

            pylab.savefig(
                jph(pfo_output_A1, 'final_{}_sj_{}_step_{}.jpg'.format(
                    params['deformation_model'] + str(taste), subject_id, st+1)
                    )
            )

        # -- produce video --
        '''
        # manually with: 
        ffmpeg -r 2 -i final_sj_BW38_step_%01d.jpg -vcodec gif -y movie.gif
        # or with:
        ffmpeg -r 3 -f concat -safe 0 -i myfile.txt -y output.gif
        # where myfile.txt contains something like:
        file 'final_translation_sj_BW38_step_1.jpg'
        file 'final_translation_sj_BW38_step_2.jpg'
        file 'final_translation_sj_BW38_step_3.jpg'
        file 'final_translation_sj_BW38_step_4.jpg'
        file 'final_translation_sj_BW38_step_5.jpg'
        file 'final_translation_sj_BW38_step_6.jpg'
        file 'final_translation_sj_BW38_step_7.jpg'
        file 'final_translation_sj_BW38_step_8.jpg'
        file 'final_translation_sj_BW38_step_9.jpg'
        file 'final_translation_sj_BW38_step_10.jpg'
        file 'final_rotation_sj_BW38_step_1.jpg'
        file 'final_rotation_sj_BW38_step_2.jpg'
        file 'final_rotation_sj_BW38_step_3.jpg'
        file 'final_rotation_sj_BW38_step_4.jpg'
        file 'final_rotation_sj_BW38_step_5.jpg'
        file 'final_rotation_sj_BW38_step_6.jpg'
        file 'final_rotation_sj_BW38_step_7.jpg'
        file 'final_rotation_sj_BW38_step_8.jpg'
        file 'final_rotation_sj_BW38_step_9.jpg'
        file 'final_rotation_sj_BW38_step_10.jpg'
        file 'final_linear5_sj_BW38_step_1.jpg'
        file 'final_linear5_sj_BW38_step_2.jpg'
        file 'final_linear5_sj_BW38_step_3.jpg'
        file 'final_linear5_sj_BW38_step_4.jpg'
        file 'final_linear5_sj_BW38_step_5.jpg'
        file 'final_linear5_sj_BW38_step_6.jpg'
        file 'final_linear5_sj_BW38_step_7.jpg'
        file 'final_linear5_sj_BW38_step_8.jpg'
        file 'final_linear5_sj_BW38_step_9.jpg'
        file 'final_linear5_sj_BW38_step_10.jpg' 
        '''
