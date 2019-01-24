import os
import time
from os.path import join as jph
from collections import OrderedDict
from pprint import pprint

import tabulate
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sympy.core.cache import clear_cache

from calie.transformations import pgl2
from calie.fields import generate as gen
from calie.fields import queries as qr
from calie.visualisations.fields.fields_at_the_window import see_field
from calie.operations import lie_exp
from calie.visualisations.fields import fields_at_the_window

from benchmarking.a_main_controller import methods, spline_interpolation_order, steps
from benchmarking.b_path_manager import pfo_output_A4_HOM

"""
Module for the computation of a dataset of 2d non-linear generated with homographies.
It compares computational time and error for the exponential computation with different methods and time steps
for a dataset of homographies.
"""

if __name__ == '__main__':

    clear_cache()

    # controller

    control = {'generate_dataset' : False,
               'compute_exps'     : False,
               'get_statistics'   : False,
               'show_graphs'      : True}

    verbose = 1

    # parameters:

    params = OrderedDict()

    x_1, y_1, z_1 = 50, 50, 1
    if z_1 == 1:
        omega = (x_1, y_1)
    else:
        omega = (x_1, y_1, z_1)

    # Homographies parameters

    d = len(omega)
    scale_factor = 1. / (np.max(omega) * 3)
    special = False
    sigma = 2
    hom_attributes = [d, scale_factor, sigma, special]
    projective_centre = np.array([25, 25])

    params.update({'experiment id'   : 'ex1'})
    params.update({'omega'           : omega})
    params.update({'dim'             : len(omega)})
    params.update({'passepartout'    : 4})
    params.update({'sio'             : spline_interpolation_order})
    params.update({'random_seed'     : 5})
    params.update({'num_samples'     : 50})
    params.update({'steps'           : steps})

    # Path manager

    print("\nPath to results folder {}\n".format(pfo_output_A4_HOM))

    ########################
    #   Generate dataset   #
    ########################

    if control['generate_dataset']:

        if params['random_seed'] > 0:
            np.random.seed(params['random_seed'])

        print('--------------------------------------------------------------------------')
        print('Generating dataset HOM! filename: linear-<s>-<algebra/group>.npy j = 1,...,N ')
        print('--------------------------------------------------------------------------')

        print('Error-bar and time for one hom generated SVF')
        print('---------------------------------------------')
        print('\nParameters of the transformation hom:')
        print('domain            = {}'.format(omega))
        print('center            = {}'.format(None))
        print('scale factor      = {}'.format(hom_attributes[1]))
        print('sigma             = {}'.format(hom_attributes[2]))
        print('special           = {}'.format(hom_attributes[3]))
        print('projective centre = {}'.format(projective_centre))

        for s in range(params['num_samples']):  # sample s

            h_a, h_g = pgl2.get_random_hom_matrices(d=hom_attributes[0],
                                                    scale_factor=hom_attributes[1],
                                                    sigma=hom_attributes[2],
                                                    special=hom_attributes[3],
                                                    projective_center=projective_centre)

            svf1 = gen.generate_from_projective_matrix(omega, h_a, structure='algebra')
            flow1_ground = gen.generate_from_projective_matrix(omega, h_g, structure='group')

            # Generate corresponding SVF and flow

            print('\nSampling ' + str(s + 1) + '/' + str(params['num_samples']) + '.')
            print('hom_a = ')
            print(h_a)
            print('hom_g = ')
            print(h_g)

            pfi_svf1 = jph(pfo_output_A4_HOM, 'hom-{}-algebra.npy'.format(s + 1))
            pfi_flow = jph(pfo_output_A4_HOM, 'hom-{}-group.npy'.format(s + 1))

            np.save(pfi_svf1, svf1)
            np.save(pfi_flow, flow1_ground)

            print('svf saved in {}'.format(pfi_svf1))
            print('flow saved in {}'.format(pfi_flow))

        print('\n------------------------------------------')
        print('Data computed and saved in external files!')
        print('------------------------------------------')

    else:

        for s in range(params['num_samples']):
            pfi_svf0 = jph(pfo_output_A4_HOM, 'hom-{}-algebra.npy'.format(s + 1))
            pfi_flow = jph(pfo_output_A4_HOM, 'hom-{}-group.npy'.format(s + 1))
            assert os.path.exists(pfi_svf0), pfi_svf0
            assert os.path.exists(pfi_flow), pfi_flow

    ############################
    #   Compute exponentials   #
    ############################

    print('--------------------------------------------------------------------------')
    print('Compute exponentials HOM! filename: hom-<method>-STEPS_<steps>.csv        ')
    print('--------------------------------------------------------------------------')

    if control['compute_exps']:

        for method_name in [k for k in methods.keys() if methods[k][1]]:

            # matrices for tabulation:
            tab_errors    = np.zeros([params['num_samples'], len(params['steps'])])
            tab_comp_time = np.zeros([params['num_samples'], len(params['steps'])])

            for st in params['steps']:

                print('\n Computing method {} for steps {}'.format(method_name, st))

                exp_method = methods[method_name][0]
                sub_method = methods[method_name][6]

                # initialise pandas df
                df_time_error = pd.DataFrame(columns=['subject', 'time (sec)', 'error (mm)'],
                                             index=range(params['num_samples']))

                for s in range(params['num_samples']):

                    pfi_svf1 = jph(pfo_output_A4_HOM, 'hom-{}-algebra.npy'.format(s + 1))
                    pfi_flow = jph(pfo_output_A4_HOM, 'hom-{}-group.npy'.format(s + 1))

                    svf1         = np.load(pfi_svf1)
                    flow1_ground = np.load(pfi_flow)

                    if methods[method_name][6]:
                        raise IOError('TODO for point-wise methods differentiate vode, lsoda')

                    # compute exponetial with time:
                    start = time.time()
                    flow1_computed = exp_method(svf1, input_num_steps=st)
                    stop = (time.time() - start)

                    # compute error:
                    error = qr.norm(flow1_ground - flow1_computed, passe_partout_size=params['passepartout'], normalized=True)

                    df_time_error['subject'][s] = 'sj{}'.format(s+1)
                    df_time_error['time (sec)'][s] = stop
                    df_time_error['error (mm)'][s] = error

                # save pandas df in csv
                pfi_df_time_error = jph(pfo_output_A4_HOM, 'hom-{}-steps-{}.csv'.format(method_name, st))
                df_time_error.to_csv(pfi_df_time_error)

                # print something if you fancy:
                if verbose == 2:
                    print(df_time_error)
                if verbose == 1:

                    tab_errors[:, params['steps'].index(st)]    = df_time_error['error (mm)'].values
                    tab_comp_time[:, params['steps'].index(st)] = df_time_error['time (sec)'].values

                    print('\n')
                    print('Errors:')
                    print(tabulate.tabulate(tab_errors, headers=['steps {}'.format(s) for s in params['steps']]))
                    print('Computational time:')
                    print(tabulate.tabulate(tab_comp_time, headers=['steps {}'.format(s) for s in params['steps']]))
                    print('\n')

    else:
        # assert pandas dataframes exists
        for method_name in [k for k in methods.keys() if methods[k][1]]:
            for st in params['steps']:
                pfi_df_time_error = jph(pfo_output_A4_HOM, 'hom-{}-steps-{}.csv'.format(method_name, st))
                assert os.path.exists(pfi_df_time_error), pfi_df_time_error

    ##################
    # get statistics #
    ##################

    if control['get_statistics']:

        # for each method get mean and std indexed by num-steps.
        # | steps | mu_error | sigma_error | mu_time | sigma_error |
        # in a file called hom-stats-<method>.csv

        for method_name in [k for k in methods.keys() if methods[k][1]]:

            print('------------------------------------------------')
            print('\n Statistics for method {} '.format(method_name))

            df_mean_std = pd.DataFrame(columns=['steps', 'mu_time', 'std_time', 'mu_error', 'std_error'],
                                       index=range(len(params['steps'])))

            for st_index, st in enumerate(params['steps']):

                print('\n Steps {}'.format(st))

                pfi_df_time_error = jph(pfo_output_A4_HOM, 'hom-{}-steps-{}.csv'.format(method_name, st))
                df_time_error = pd.read_csv(pfi_df_time_error)

                df_mean_std['steps'][st_index] = st

                df_mean_std['mu_time'][st_index]  = df_time_error['time (sec)'].mean()
                df_mean_std['std_time'][st_index] = df_time_error['time (sec)'].std()

                df_mean_std['mu_error'][st_index]  = df_time_error['error (mm)'].mean()
                df_mean_std['std_error'][st_index] = df_time_error['error (mm)'].std()

            pfi_df_mean_std = jph(pfo_output_A4_HOM, 'hom-stats-{}.csv'.format(method_name))
            df_mean_std.to_csv(pfi_df_mean_std)

    else:

        for method_name in [k for k in methods.keys() if methods[k][1]][:1]:
            pfi_df_mean_std = jph(pfo_output_A4_HOM, 'hom-stats-{}.csv'.format(method_name))
            assert os.path.exists(pfi_df_mean_std), pfi_df_mean_std

    ###############
    # show graphs #
    ###############

    if control['show_graphs']:

        font_top = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': 14}
        font_bl = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        legend_prop = {'size': 12}

        sns.set_style()

        fig, ax = plt.subplots(figsize=(7, 7))

        fig.canvas.set_window_title('hom_times_vs_errors.pdf')

        for method_name in [k for k in methods.keys() if methods[k][1]]:

            pfi_df_mean_std = jph(pfo_output_A4_HOM, 'hom-stats-{}.csv'.format(method_name))
            df_mean_std = pd.read_csv(pfi_df_mean_std)

            ax.plot(df_mean_std['mu_time'].values,
                    df_mean_std['mu_error'].values,
                    label=method_name,
                    color=methods[method_name][3],
                    linestyle=methods[method_name][4],
                    marker=methods[method_name][5])

            for i in df_mean_std.index:
                el = mpatches.Ellipse((df_mean_std['mu_time'][i], df_mean_std['mu_error'][i]),
                                      df_mean_std['std_time'][i], df_mean_std['std_error'][i],
                                      angle=0,
                                      alpha=0.2,
                                      color=methods[method_name][3],
                                      linewidth=None)
                ax.add_artist(el)

        ax.set_title('Time error for HOM(2)', fontdict=font_top)
        ax.legend(loc='upper right', shadow=True, prop=legend_prop)

        ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax.set_axisbelow(True)

        ax.set_xlabel('Time (sec)', fontdict=font_bl, labelpad=5)
        ax.set_ylabel('Error (mm)', fontdict=font_bl, labelpad=5)
        ax.set_xscale('log', nonposx="mask")
        ax.set_yscale('log', nonposy="mask")

        pfi_figure_time_vs_error = jph(pfo_output_A4_HOM, 'graph_time_vs_error.pdf')
        plt.savefig(pfi_figure_time_vs_error, dpi=150)

        plt.show(block=True)
