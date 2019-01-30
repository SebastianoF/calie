"""
Module to compute the
+ Inverse Consistency (IC)
+ Scalar Associativity (SA)
+ Stepwise Error (SE)
for the selected data set, among the ones generated in the previous experiments bm1_ to bm6_ .
These three measures are not computed against any ground truth flow field and are therefore
unbiased (or better less biased) than the measure of errors in the previous experiments.

"""
import os
from os.path import join as jph
import tabulate
import pandas as pd
import numpy as np
import seaborn as sns
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sympy.core.cache import clear_cache

from nilabels.tools.aux_methods.utils import print_and_run
from nilabels.tools.aux_methods.utils_nib import set_new_data

from calie.fields import queries as qr
from calie.fields import compose as cp
from calie.fields import coordinate as coord

from benchmarking.a_main_controller import methods, spline_interpolation_order, num_samples, bw_subjects, \
    ad_subjects
from benchmarking.b_path_manager import pfo_output_A4_SE2, pfo_output_A4_GL2, pfo_output_A4_HOM, \
    pfo_output_A4_GAU, pfo_output_A4_BW, pfo_output_A4_AD, pfo_output_A5_3T


def three_assessments_collector(control, verbose):

    # ----------------------- #
    # Retrieve data set paths
    # ----------------------- #

    if control['svf_dataset'].lower() in {'rotation', 'rotations'}:
        pfi_svf_list = [jph(pfo_output_A4_SE2, 'se2-{}-algebra.npy'.format(s + 1)) for s in range(num_samples)]

    elif control['svf_dataset'].lower() in {'linear'}:
        pfi_svf_list = [jph(pfo_output_A4_GL2, 'gl2-{}-algebra.npy'.format(s + 1)) for s in range(num_samples)]

    elif control['svf_dataset'].lower() in {'homography', 'homographies'}:
        pfi_svf_list = [jph(pfo_output_A4_HOM, 'hom-{}-algebra.npy'.format(s + 1)) for s in range(num_samples)]

    elif control['svf_dataset'].lower() in {'gauss'}:
        pfi_svf_list = [jph(pfo_output_A4_GAU, 'gau-{}-algebra.npy'.format(s + 1)) for s in range(num_samples)]

    elif control['svf_dataset'].lower() in {'brainweb'}:
        pfi_svf_list = [jph(pfo_output_A4_BW, 'bw-{}-algebra.npy'.format(sj)) for sj in bw_subjects]

    elif control['svf_dataset'].lower() in {'adni'}:
        pfi_svf_list = [jph(pfo_output_A4_AD, 'ad-{}-algebra.npy'.format(sj)) for sj in ad_subjects]
    else:
        raise IOError('Svf data set not given'.format(control['svf_dataset']))

    for pfi in pfi_svf_list:
        assert os.path.exists(pfi)

    # --------------------------------------- #
    # Select number of steps for each method
    # --------------------------------------- #

    if control['computation'] == 'IC':
        steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]
    elif control['computation'] == 'SA':
        steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]
    elif control['computation'] == 'SE':
        steps = range(1, 30)

    else:
        raise IOError('Input control computation {} not defined.'.format(control['computation']))

    if control['collect']:

        print('---------------------------------------------------------------------------')
        print('Test {} for dataset {} '.format(control['computation'], control['svf_dataset']))
        print('---------------------------------------------------------------------------')

        for pfi_svf in pfi_svf_list:
            sj_id = os.path.basename(pfi_svf).split('-')[:2]
            sj_id = sj_id[0] + '-' + sj_id[1]

            method_names = [k for k in methods.keys() if methods[k][1]]

            df_steps_measures = pd.DataFrame(columns=method_names, index=steps)
            svf1 = np.load(pfi_svf)

            for met in method_names:
                print('Computing method {}.'.format(met))
                exp_method = methods[met][0]

                for st in steps:

                    if control['computation'] == 'IC':
                        exp_st_svf1     = exp_method(svf1, input_num_steps=st)
                        exp_st_neg_svf1 = exp_method(-1 * svf1, input_num_steps=st)
                        error = 0.5 * (qr.norm(cp.lagrangian_dot_lagrangian(exp_st_svf1, exp_st_neg_svf1)) +
                                       qr.norm(cp.lagrangian_dot_lagrangian(exp_st_neg_svf1, exp_st_svf1)))

                    elif control['computation'] == 'SA':
                        a, b, c = 0.3, 0.3, 0.4
                        exp_st_a_svf1 = exp_method(a * svf1, input_num_steps=st)
                        exp_st_b_svf1 = exp_method(b * svf1, input_num_steps=st)
                        exp_st_c_svf1 = exp_method(c * svf1, input_num_steps=st)
                        error = qr.norm(cp.lagrangian_dot_lagrangian(cp.lagrangian_dot_lagrangian(exp_st_a_svf1, exp_st_b_svf1), exp_st_c_svf1))

                    elif control['computation'] == 'SE':
                        exp_st_svf1          = exp_method(svf1, input_num_steps=st)
                        exp_st_plus_one_svf1 = exp_method(svf1, input_num_steps=st+1)
                        error = qr.norm(exp_st_svf1 - exp_st_plus_one_svf1)

                    else:
                        raise IOError('Input control computation {} not defined.'.format(control['computation']))

                    df_steps_measures['measure'][st] = error

                print(df_steps_measures)

            fin_output = 'test_{}_{}.csv'.format(control['computation'], sj_id)
            df_steps_measures.to_csv(jph(pfo_output_A5_3T, fin_output))
            print('Test result saved in:')
            print(fin_output)
            print('\n')

    else:
        # assert pandas data-frame exists.
        for pfi_svf in pfi_svf_list:
            sj_id = os.path.basename(pfi_svf).split('-')[:2]
            sj_id = sj_id[0] + '-' + sj_id[1]
            fin_output = 'test_{}_{}.csv'.format(control['computation'], sj_id)
            assert os.path.exists(jph(pfo_output_A5_3T, fin_output)), jph(pfo_output_A5_3T, fin_output)

    ##################
    # get statistics #
    ##################

    if control['get_statistics']:

        print('------------------------------------------------')
        print('\n Statistics for computation {} '.format(control['computation']))

        method_names = [k for k in methods.keys() if methods[k][1]]

        df_means_methods = pd.DataFrame(columns=method_names, index=steps)
        df_stds_methods  = pd.DataFrame(columns=method_names, index=steps)

        for pfi_svf in pfi_svf_list:
            sj_id = os.path.basename(pfi_svf).split('-')[:2]
            sj_id = sj_id[0] + '-' + sj_id[1]



        pass

    ###############
    # show graphs #
    ###############

    if control['show_graphs']:

        font_top = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': 14}
        font_bl = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        legend_prop = {'size': 12}

        sns.set_style()

        fig, ax = plt.subplots(figsize=(7, 7))

        fig.canvas.set_window_title('{}_{}.pdf'.format(control['svf_dataset'], control['computation']))

        for method_name in [k for k in methods.keys() if methods[k][1]]:

            pfi_df_mean_std = jph(pfo_output_A4_SE2, 'se2-stats-{}.csv'.format(method_name))
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

        ax.set_title('Time error for SE(2)', fontdict=font_top)
        ax.legend(loc='upper right', shadow=True, prop=legend_prop)

        ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax.set_axisbelow(True)

        ax.set_xlabel('Time (sec)', fontdict=font_bl, labelpad=5)
        ax.set_ylabel('Error (mm)', fontdict=font_bl, labelpad=5)
        ax.set_xscale('log', nonposx="mask")
        ax.set_yscale('log', nonposy="mask")

        pfi_figure_time_vs_error = jph(pfo_output_A4_SE2, 'graph_time_vs_error.pdf')
        plt.savefig(pfi_figure_time_vs_error, dpi=150)

        plt.show(block=True)


if __name__ == '__main__':

    clear_cache()

    # controller

    control_ = {'svf_dataset'   : 'rotation',  # can be rotation, linear, homography, gauss, brainweb, adni
                'computation'   : 'IC',  # can be IC, SA, SE
                'collect'       : True,
                'show_graphs'   : True}

    verbose_ = 1
    three_assessments_collector(control_, verbose_)
