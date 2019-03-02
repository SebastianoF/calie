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
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sympy.core.cache import clear_cache

from calie.fields import queries as qr
from calie.fields import compose as cp

from benchmarking.a_main_controller import methods, num_samples, bw_subjects, ad_subjects
from benchmarking.b_path_manager import pfo_output_A4_SE2, pfo_output_A4_GL2, pfo_output_A4_HOM, \
    pfo_output_A4_GAU, pfo_output_A4_BW, pfo_output_A4_AD, pfo_output_A5_3T


def three_assessments_collector(control):

    # ----------------------- #
    # Retrieve data set paths
    # ----------------------- #

    if control['svf_dataset'].lower() in {'rotation', 'rotations'}:
        pfi_svf_list = [jph(pfo_output_A4_SE2, 'se2-{}-algebra.npy'.format(s + 1)) for s in range(num_samples)]

    elif control['svf_dataset'].lower() in {'linear'}:
        pfi_svf_list = [jph(pfo_output_A4_GL2, 'gl2-{}-algebra.npy'.format(s + 1)) for s in range(num_samples)]

    elif control['svf_dataset'].lower() in {'homography', 'homographies'}:
        pfi_svf_list = [jph(pfo_output_A4_HOM, 'hom-{}-algebra.npy'.format(s + 1)) for s in range(12)]  # TODO

    elif control['svf_dataset'].lower() in {'gauss'}:
        pfi_svf_list = [jph(pfo_output_A4_GAU, 'gau-{}-algebra.npy'.format(s + 1)) for s in range(num_samples)]

    elif control['svf_dataset'].lower() in {'brainweb'}:
        pfi_svf_list = [jph(pfo_output_A4_BW, 'bw-{}-algebra.npy'.format(sj)) for sj in bw_subjects[1:]]

    elif control['svf_dataset'].lower() in {'adni'}:
        pfi_svf_list = [jph(pfo_output_A4_AD, 'ad-{}-algebra.npy'.format(sj)) for sj in ad_subjects]
    else:
        raise IOError('Svf data set not given'.format(control['svf_dataset']))

    for pfi in pfi_svf_list:
        assert os.path.exists(pfi), pfi

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

            print('Computation for subject {}.'.format(sj_id))

            method_names = [k for k in methods.keys() if methods[k][1]]

            df_steps_measures = pd.DataFrame(columns=method_names, index=steps)
            svf1 = np.load(pfi_svf)

            for met in method_names:
                print(' --> Computing method {}.'.format(met))
                exp_method = methods[met][0]

                for st in steps:
                    print(' ---> step {}'.format(st))

                    if control['computation'] == 'IC':
                        exp_st_svf1     = exp_method(svf1, input_num_steps=st)
                        exp_st_neg_svf1 = exp_method(-1 * svf1, input_num_steps=st)
                        error = 0.5 * (qr.norm(cp.lagrangian_dot_lagrangian(exp_st_svf1, exp_st_neg_svf1), normalized=True) +
                                       qr.norm(cp.lagrangian_dot_lagrangian(exp_st_neg_svf1, exp_st_svf1), normalized=True))

                    elif control['computation'] == 'SA':
                        a, b, c = 0.3, 0.3, 0.4
                        exp_st_a_svf1 = exp_method(a * svf1, input_num_steps=st)
                        exp_st_b_svf1 = exp_method(b * svf1, input_num_steps=st)
                        exp_st_c_svf1 = exp_method(c * svf1, input_num_steps=st)
                        error = qr.norm(cp.lagrangian_dot_lagrangian(cp.lagrangian_dot_lagrangian(exp_st_a_svf1, exp_st_b_svf1), exp_st_c_svf1), normalized=True)

                    elif control['computation'] == 'SE':
                        exp_st_svf1          = exp_method(svf1, input_num_steps=st)
                        exp_st_plus_one_svf1 = exp_method(svf1, input_num_steps=st+1)
                        error = qr.norm(exp_st_svf1 - exp_st_plus_one_svf1, normalized=True)

                    else:
                        raise IOError('Input control computation {} not defined.'.format(control['computation']))

                    df_steps_measures[met][st] = error

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

        print('---------------------------------------------------------------------------')
        print('Get statistics for {}, dataset {}. '.format(control['computation'], control['svf_dataset']))
        print('---------------------------------------------------------------------------')

        # for each method get mean and std indexed by num-steps.
        # | steps | mu_error | sigma_error | mu_time | sigma_error |
        # in a file called stats-<computation>-<method>.csv

        for method_name in [k for k in methods.keys() if methods[k][1]]:

            print('\n Statistics for method {} \n'.format(method_name))

            # for each method stack all the measurements in a single matrix STEPS x SVFs

            steps_times_subjects = np.nan * np.ones([len(steps), len(pfi_svf_list)])

            for pfi_svf_index, pfi_svf in enumerate(pfi_svf_list):

                sj_id = os.path.basename(pfi_svf).split('-')[:2]
                sj_id = sj_id[0] + '-' + sj_id[1]
                fin_test = 'test_{}_{}.csv'.format(control['computation'], sj_id)
                df_steps_measures = pd.read_csv(jph(pfo_output_A5_3T, fin_test))

                steps_times_subjects[:, pfi_svf_index] = df_steps_measures[method_name].as_matrix()

            df_mean_std = pd.DataFrame(columns=['steps', 'mu_error', 'std_error'], index=range(len(steps)))

            df_mean_std['steps']     = steps
            df_mean_std['mu_error']  = np.mean(steps_times_subjects, axis=1)
            df_mean_std['std_error'] = np.std(steps_times_subjects, axis=1)

            print(df_mean_std)

            pfi_df_mean_std = jph(pfo_output_A5_3T, 'stats-3T-{}-{}-{}.csv'.format(
                control['svf_dataset'], control['computation'], method_name)
            )

            df_mean_std.to_csv(jph(pfi_df_mean_std))

    else:

        for method_name in [k for k in methods.keys() if methods[k][1]][:1]:

            pfi_df_mean_std = jph(pfo_output_A5_3T, 'stats-3T-{}-{}-{}.csv'.format(
                control['svf_dataset'], control['computation'], method_name)
            )
            assert os.path.exists(pfi_df_mean_std), pfi_df_mean_std

    ###############
    # show graphs #
    ###############

    if control['show_graphs']:

        print('---------------------------------------------------------------------------')
        print('Showing graphs for {}, dataset {}. '.format(control['computation'], control['svf_dataset']))
        print('---------------------------------------------------------------------------')

        font_top = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': 14}
        font_bl = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        legend_prop = {'size': 11}

        sns.set_style()

        fig, ax = plt.subplots(figsize=(11, 6))

        fig.canvas.set_window_title('{}_{}.pdf'.format(control['svf_dataset'], control['computation']))

        for method_name in [k for k in methods.keys() if methods[k][1]]:

            pfi_df_mean_std = jph(pfo_output_A5_3T, 'stats-3T-{}-{}-{}.csv'.format(
                control['svf_dataset'], control['computation'], method_name)
            )

            df_mean_std = pd.read_csv(pfi_df_mean_std)

            if method_name in ['gss_ei', 'gss_ei_mod', 'gss_aei', 'gss_rk4', 'euler_aei']:
                method_name_bypass = method_name + ' *'
            elif method_name in ['scaling_and_squaring']:
                method_name_bypass = 'ss'
            else:
                method_name_bypass = method_name

            ax.plot(df_mean_std['steps'].values,
                    df_mean_std['mu_error'].values,
                    label=method_name_bypass,
                    color=methods[method_name][3],
                    linestyle=methods[method_name][4],
                    marker=methods[method_name][5])

            plt.errorbar(df_mean_std['steps'].values, df_mean_std['mu_error'].values, df_mean_std['std_error'].values,
                         linestyle='None', marker='None', color=methods[method_name][3], alpha=0.5, elinewidth=0.8)

        ax.set_title('Experiment {} for {}'.format(control['computation'], control['svf_dataset']), fontdict=font_top)
        ax.legend(loc='upper right', shadow=True, prop=legend_prop)

        ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax.set_axisbelow(True)

        ax.set_xlabel('Steps', fontdict=font_bl, labelpad=5)
        ax.set_ylabel('Error (mm)', fontdict=font_bl, labelpad=5)
        # ax.set_xscale('log', nonposx="mask")
        # ax.set_yscale('log', nonposy="mask")

        pfi_figure_time_vs_error = jph(pfo_output_A5_3T, 'three_experiments_{}_{}.pdf'.format(control['computation'], control['svf_dataset']))
        plt.savefig(pfi_figure_time_vs_error, dpi=150)

        plt.show(block=True)


if __name__ == '__main__':

    clear_cache()

    # Show all the columns when displaying methods.
    pd.set_option('display.max_columns', 50)

    # controller Brainweb

    # control_ = {'svf_dataset'    : 'brainweb',  # can be rotation, linear, homography, gauss, brainweb, adni
    #             'computation'    : 'IC',  # can be IC, SA, SE
    #             'collect'        : False,
    #             'get_statistics' : True,
    #             'show_graphs'    : True}
    #
    # three_assessments_collector(control_)
    #
    # #
    #
    # control_ = {'svf_dataset'    : 'brainweb',  # can be rotation, linear, homography, gauss, brainweb, adni
    #             'computation'    : 'SA',  # can be IC, SA, SE
    #             'collect'        : False,
    #             'get_statistics' : True,
    #             'show_graphs'    : True}
    #
    # three_assessments_collector(control_)
    #
    # #
    #
    # control_ = {'svf_dataset'    : 'brainweb',  # can be rotation, linear, homography, gauss, brainweb, adni
    #             'computation'    : 'SE',  # can be IC, SA, SE
    #             'collect'        : False,
    #             'get_statistics' : True,
    #             'show_graphs'    : True}
    #
    # three_assessments_collector(control_)

    '''
    # linear
    
    control_ = {'svf_dataset'    : 'linear',  # can be rotation, linear, homography, gauss, brainweb, adni
                'computation'    : 'IC',  # can be IC, SA, SE
                'collect'        : False,
                'get_statistics' : False,
                'show_graphs'    : True}

    three_assessments_collector(control_)
    
    #
    
    control_ = {'svf_dataset'    : 'linear',  # can be rotation, linear, homography, gauss, brainweb, adni
                'computation'    : 'SA',  # can be IC, SA, SE
                'collect'        : False,
                'get_statistics' : False,
                'show_graphs'    : True}

    three_assessments_collector(control_)
    
    #
    
    control_ = {'svf_dataset'    : 'linear',  # can be rotation, linear, homography, gauss, brainweb, adni
                'computation'    : 'SE',  # can be IC, SA, SE
                'collect'        : False,
                'get_statistics' : False,
                'show_graphs'    : True}

    three_assessments_collector(control_)
    '''

    # Homography
    
    control_ = {'svf_dataset'    : 'homography',  # can be rotation, linear, homography, gauss, brainweb, adni
                'computation'    : 'IC',  # can be IC, SA, SE
                'collect'        : False,
                'get_statistics' : False,
                'show_graphs'    : True}

    three_assessments_collector(control_)
    
    #
    
    # control_ = {'svf_dataset'    : 'homography',  # can be rotation, linear, homography, gauss, brainweb, adni
    #             'computation'    : 'SA',  # can be IC, SA, SE
    #             'collect'        : False,
    #             'get_statistics' : False,
    #             'show_graphs'    : True}
    #
    # three_assessments_collector(control_)
    #
    # #
    #
    # control_ = {'svf_dataset'    : 'homography',  # can be rotation, linear, homography, gauss, brainweb, adni
    #             'computation'    : 'SE',  # can be IC, SA, SE
    #             'collect'        : False,
    #             'get_statistics' : False,
    #             'show_graphs'    : True}
    #
    # three_assessments_collector(control_)
    #
    '''
    # gauss
    
    control_ = {'svf_dataset'    : 'gauss',  # can be rotation, linear, homography, gauss, brainweb, adni
                'computation'    : 'IC',  # can be IC, SA, SE
                'collect'        : False,
                'get_statistics' : False,
                'show_graphs'    : True}

    three_assessments_collector(control_)
    
    #
    
    control_ = {'svf_dataset'    : 'gauss',  # can be rotation, linear, homography, gauss, brainweb, adni
                'computation'    : 'SA',  # can be IC, SA, SE
                'collect'        : False,
                'get_statistics' : False,
                'show_graphs'    : True}

    three_assessments_collector(control_)
    
    #
    
    control_ = {'svf_dataset'    : 'gauss',  # can be rotation, linear, homography, gauss, brainweb, adni
                'computation'    : 'SE',  # can be IC, SA, SE
                'collect'        : False,
                'get_statistics' : False,
                'show_graphs'    : True}

    three_assessments_collector(control_)
    
    
    '''