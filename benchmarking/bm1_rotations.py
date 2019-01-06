import os
import pickle
import time
from os.path import join as jph

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy.core.cache import clear_cache

from VECtorsToolkit.transformations import se2
from VECtorsToolkit.fields import generate as gen
from VECtorsToolkit.operations import lie_exp
from VECtorsToolkit.fields import queries as qr

from benchmarking.a_main_controller import methods
from benchmarking.b_path_manager import pfo_output_A4_SE2

"""
Module for the computation of one 2d SVF generated with matrix of se2_a.
It compares the exponential computation with different methods for a number of
steps defined by the user.
"""


if __name__ == '__main__':

    # controller

    control = {'generate_dataset' : True,
               'compute_exps'     : True,
               'show_graphs'      : True}

    # parameters:

    x_1, y_1, z_1 = 20, 20, 5

    if z_1 == 1:
        omega = (x_1, y_1)
    else:
        omega = (x_1, y_1, z_1)

    kind = 'SE2'
    number = 'multiple'
    tag = '_' + str(1)

    passepartout = 5
    max_angle = np.pi / 8
    centre_delta = (5, 5, 5)
    interval_theta = (- max_angle, max_angle)
    epsilon = np.pi / 12
    interval_center = (int(omega[0] / 2 - centre_delta[0]), int(omega[0] / 2 + centre_delta[0]),
                       int(omega[1] / 2 - centre_delta[1]), int(omega[1] / 2 + centre_delta[1]))

    random_seed = 0

    s_i_o = 3  # spline interpolation order

    N = 50

    # Path manager

    print("\nPath to results folder {}\n".format(pfo_output_A4_SE2))

    ########################
    #   Generate dataset   #
    ########################

    if control['generate_dataset']:

        if random_seed > 0:
            np.random.seed(random_seed)

        print('----------------------------------------------------------')
        print('Generating dataset SE2! filename: se2_<s>_<algebra/group>.npy j = 1,...,N ')
        print('----------------------------------------------------------')

        for s in range(N):  # sample

            # generate matrices
            m_0 = se2.se2g_randomgen_custom_center(interval_theta=interval_theta, interval_center=interval_center,
                                                   epsilon_zero_avoidance=epsilon)
            dm_0 = se2.se2g_log(m_0)

            # Generate SVF
            svf1 = gen.generate_from_matrix(omega, dm_0.get_matrix, t=1, structure='algebra')
            flow1_ground = gen.generate_from_matrix(omega, m_0.get_matrix, t=1, structure='group')

            pfi_svf0 = jph(pfo_output_A4_SE2, 'se2_{}_algebra.npy'.format(s))
            pfi_flow = jph(pfo_output_A4_SE2, 'se2_{}_group.npy'.format(s))

            print('Sampling ' + str(s + 1) + '/' + str(N) + ' .')
            print('theta, tx, ty =    ' + str(m_0.get))
            print('dtheta, dtx, dty = ' + str(dm_0.get))


            np.save(pfi_svf0, svf1)
            np.save(pfi_flow, flow1_ground)

            print('svf saved in {}'.format(pfi_svf0))
            print('flow saved in {}'.format(pfi_flow))

        print('\n------------------------------------------')
        print('Data computed and saved in external files!')
        print('------------------------------------------')

    else:

        for s in range(N):
            pfi_svf0 = jph(pfo_output_A4_SE2, 'se2_{}_algebra.npy'.format(s + 1))
            pfi_flow = jph(pfo_output_A4_SE2, 'se2_{}_group.npy'.format(s + 1))
            assert os.path.exists(pfi_svf0), pfi_svf0
            assert os.path.exists(pfi_flow), pfi_flow

    if control['compute_exps']:

        for exp_method in
        errors = np.zeros([num_method_considered, N])  # Row: method, col: sampling
        res_time = np.zeros([num_method_considered, N])  # Row: method, col: sampling

        for s in range(N):  # sample
            pass

        for m in range(num_method_considered):  # method
            if names_method_considered[m] == 'vode' or names_method_considered[m] == 'lsoda':
                start = time.time()
                disp_computed = lie_exponential_scipy(svf_0, integrator=names_method_considered[m],
                                                      max_steps=steps_methods_considered[m])
                res_time[m] = (time.time() - start)

            else:
                start = time.time()
                disp_computed = lie_exponential(svf_0, algorithm=names_method_considered[m], s_i_o=s_i_o,
                                                input_num_steps=steps_methods_considered[m])
                res_time[m, s] = (time.time() - start)

            # compute error:
            errors[m, s] = vf_norm(disp_computed - disp_ground, passe_partout_size=pp, normalized=True)

            results_by_column = [[met, err, tim]
                                 for met, err, tim
                                 in zip(names_method_considered, list(errors[:, s]), list(res_time[:, s]))]

            print('--------------------')
            print('Sampling ' + str(s + 1) + '/' + str(N) + ' .')
            print('--------------------')
            print('theta, tx, ty =    ' + str(m_0.get))
            print('dtheta, dtx, dty = ' + str(dm_0.get))
            print('--------------------')
            print('--------------------')

    ### Save data to folder ###
    np.save(pfi_array_errors_output, errors)
    np.save(pfi_array_comp_time_output, res_time)

    with open(pfi_transformation_parameters, 'wb') as f:
        pickle.dump(parameters, f)

    with open(pfi_numerical_method_table, 'wb') as f:
        pickle.dump(methods, f)

    else:
        pass


    ###############################
    # Plot parameters and methods #
    ###############################

    if verbose:
        print('Error-bar and time for multiple se2 generated SVF')
        print
        '-------------------------------------------------'

        print
        '\nParameters of the transformation se2:'
        print
        'Number of samples = ' + str(parameters[3])
        print
        'domain = ' + str(parameters[:3])
        print
        'interval theta = ' + str(parameters[4:6])
        print
        'Omega, interval tx, ty = ' + str(parameters[6:])

        print
        '\n'
        print
        'Methods and parameters:'
        print
        tabulate(methods,
                 headers=['name', 'compute (True/False)', 'num_steps'])
        print
        '\n'

        print
        'List of the methods considered:'
        print
        names_method_considered
        print
        'List of the steps of the methods considered'
        print
        steps_methods_considered

    ################################
    # Visualization and statistics #
    ################################

    mean_errors = np.mean(errors, axis=1)
    mean_res_time = np.mean(res_time, axis=1)

    print
    mean_errors
    print
    len(mean_errors)

    results_by_column = [[met, err, tim]
                         for met, err, tim in zip(names_method_considered, list(mean_errors), list(mean_res_time))]

    print
    '\n'
    print
    'Results and computational time:'
    print
    tabulate(results_by_column,
             headers=['method', 'mean error', 'mean comp. time (sec)'])
    print
    '\n END'

    # plot results
    if plot_results:

        reordered_errors_for_plot = []
        reordered_times_for_plot = []
        for m in range(errors.shape[0]):
            reordered_errors_for_plot += [list(errors[m, :])]
            reordered_times_for_plot += [list(res_time[m, :])]

        # BOXPLOT custom

        plot_custom_boxplot(input_data=reordered_errors_for_plot,
                            input_names=names_method_considered,
                            fig_tag=11,
                            input_titles=('Error exponential map for multiple SE2-generated svf', 'field'),
                            kind='multiple_SE2',
                            window_title_input='bar_plot_multiple_se2',
                            additional_field=None,
                            log_scale=False,
                            input_parameters=parameters,

                            annotate_mean=True,
                            add_extra_annotation=mean_res_time)

        # SCATTER-PLOT custom

        plot_custom_cluster(reordered_errors_for_plot, reordered_times_for_plot,
                            fig_tag=22,
                            clusters_labels=names_method_considered,
                            clusters_colors=colour_methods_considered,
                            clusters_markers=markers_methods_considered)

        plt.show()

    ### Save figures in external folder ###

    if save_external:
        os.system('mkdir -p {}'.format(pfo_notes_figures))
        # Save table csv
        # np.savetxt(pfi_csv_table_errors_output, errors, delimiter=" & ")
        # np.savetxt(pfi_csv_table_comp_time_output, errors, delimiter=" & ")
        # Save image:
        plt.savefig(pfi_figure_output, format='pdf', dpi=400)
        print
        'Figure ' + fin_figure_output + ' saved in the external folder ' + str(pfi_figure_output)

