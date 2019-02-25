import os
import time
from os.path import join as jph
from collections import OrderedDict

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sympy.core.cache import clear_cache

from nilabels.tools.aux_methods.utils import print_and_run

from benchmarking.b_path_manager import pfo_output_A4_AD, pfo_adni
from benchmarking.a_main_controller import ad_subjects, ad_subjects_first_time_point, ad_subjects_second_time_point

"""
Module to assess if there is any improvements in computing the lie exponential with approximated exponential 
integrators within NiftyReg.
THIS IS BASED ON THE DATA GENERATED IN THE PREVIOUS ADNI EXPERIMENT.
"""

if __name__ == '__main__':

    clear_cache()

    # controller

    control = {'generate_dataset'  : True,
               'show_graphs'       : True}

    verbose = 1

    add_OMP = True

    pfo_nifty_reg_app_standard = '/Users/sebastiano/a_data/TData/wbir_16/Code/install{}/niftyreg_install/reg-apps'.format('_omp' if add_OMP else '')
    pfo_nifty_reg_app_ei       = '/Users/sebastiano/a_data/TData/wbir_16/Code/Install{}/niftyreg_install_aei/reg-apps'.format('_omp' if add_OMP else '')

    if add_OMP:
        pfo_output_A4_AD = pfo_output_A4_AD + '_omp'

    # parameters:

    params = OrderedDict()

    params.update({'experiment id'      : 'ex1'})
    params.update({'subjects_FirstTP'   : ad_subjects_first_time_point})
    params.update({'subjects_SecondTP'  : ad_subjects_second_time_point})

    assert len(params['subjects_FirstTP']) == len(params['subjects_SecondTP'])

    # list of subjects id both time-points:
    subjects = ad_subjects

    # Path manager

    print("\nPath to results folder {}\n".format(pfo_output_A4_AD))

    ####################
    # Generate dataset #
    ####################

    if control['generate_dataset']:

        print('---------------------------------------------------------------------------')
        print('Generating dataset COMPARISON NIFTYREG COMPUTATIONAL TIME!')
        print(' filename:  time-comparison-sj<>.csv')
        print(' USING OMP : {}'.format(add_OMP))
        print('---------------------------------------------------------------------------')

        experiments = ['standard', 'diff', 'diff ei']

        df_computational_time = pd.DataFrame(columns=experiments, index=subjects)

        for sj_name, sj_first_tp, sj_second_tp in zip(subjects, params['subjects_FirstTP'], params['subjects_SecondTP']):

            print('--------------------------------------------------------------------------------------------------')
            print('Non-rigid registration to target, sj first tp {}, second tp {}\n'.format(sj_first_tp, sj_second_tp))

            pfi_T1W_fixed_tp1 = jph(pfo_adni, 'FirstTP', 'MaskedT1_{}.nii.gz'.format(sj_first_tp))
            pfi_mask_fixed_tp1 = jph(pfo_adni, 'FirstTP', '{}_GIF_B1.nii.gz'.format(sj_first_tp))

            assert os.path.exists(pfi_T1W_fixed_tp1), pfi_T1W_fixed_tp1
            assert os.path.exists(pfi_mask_fixed_tp1), pfi_mask_fixed_tp1

            pfi_moving_on_target_warp_aff = jph(pfo_output_A4_AD,
                                                '{}_on_{}_warp_aff.nii.gz'.format(sj_second_tp, sj_first_tp))
            pfi_moving_on_target_mask_aff = jph(pfo_output_A4_AD,
                                                '{}_on_{}_mask_warp_aff.nii.gz'.format(sj_second_tp, sj_first_tp))

            assert os.path.exists(pfi_moving_on_target_warp_aff), pfi_moving_on_target_warp_aff
            assert os.path.exists(pfi_moving_on_target_mask_aff), pfi_moving_on_target_mask_aff

            pfi_cpp_time_comp = jph(pfo_output_A4_AD, 'time_comp_{}_on_{}_cpp.nii.gz'.format(sj_second_tp, sj_first_tp))
            pfi_moving_on_target_nrig_time_comp = jph(pfo_output_A4_AD,
                                            'time_comp_{}_on_{}_warp_nrig.nii.gz'.format(sj_second_tp, sj_first_tp))

            cmd_standard = '{0}/reg_f3d -ref {1} -rmask {2} -flo {3} -fmask {4} -cpp {5} -res {6} -omp 8 '.format(
                pfo_nifty_reg_app_standard,
                pfi_T1W_fixed_tp1, pfi_mask_fixed_tp1, pfi_moving_on_target_warp_aff, pfi_moving_on_target_mask_aff,
                pfi_cpp_time_comp, pfi_moving_on_target_nrig_time_comp
            )

            cmd_diff     = '{0}/reg_f3d -ref {1} -rmask {2} -flo {3} -fmask {4} -cpp {5} -res {6} -vel  -omp 8 '.format(
                pfo_nifty_reg_app_standard,
                pfi_T1W_fixed_tp1, pfi_mask_fixed_tp1, pfi_moving_on_target_warp_aff, pfi_moving_on_target_mask_aff,
                pfi_cpp_time_comp, pfi_moving_on_target_nrig_time_comp
            )

            cmd_diff_ei  = '{0}/reg_f3d -ref {1} -rmask {2} -flo {3} -fmask {4} -cpp {5} -res {6} -vel  -omp 8 '.format(
                pfo_nifty_reg_app_ei,
                pfi_T1W_fixed_tp1, pfi_mask_fixed_tp1, pfi_moving_on_target_warp_aff, pfi_moving_on_target_mask_aff,
                pfi_cpp_time_comp, pfi_moving_on_target_nrig_time_comp
            )

            # standard

            start = time.time()
            print_and_run(cmd_standard)
            stop_standard = (time.time() - start)

            df_computational_time['standard'][sj_name] = stop_standard

            # diff

            start = time.time()
            print_and_run(cmd_diff)
            stop_diff = (time.time() - start)

            df_computational_time['diff'][sj_name] = stop_diff

            # diff ei

            start = time.time()
            print_and_run(cmd_diff_ei)
            stop_diff_ei = (time.time() - start)

            df_computational_time['diff ei'][sj_name] = stop_diff

            print(df_computational_time)

        pfi_df_time_comparison = jph(pfo_output_A4_AD, 'time_comparison.csv')
        df_computational_time.to_csv(pfi_df_time_comparison)

    else:
        pfi_df_time_comparison = jph(pfo_output_A4_AD, 'time_comparison.csv')
        assert os.path.exists(pfi_df_time_comparison)

    ###############
    # show graphs #
    ###############

    if control['show_graphs']:

        font_top = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': 14}
        font_bl = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        legend_prop = {'size': 12}

        # sns.set_style()
        #
        # fig, ax = plt.subplots(figsize=(7, 7))
        #
        # fig.canvas.set_window_title('ad_times_vs_errors.pdf')
        #
        # for method_name in [k for k in methods.keys() if methods[k][1]]:
        #
        #     pfi_df_mean_std = jph(pfo_output_A4_AD, 'ad-stats-{}.csv'.format(method_name))
        #     df_mean_std = pd.read_csv(pfi_df_mean_std)
        #
        #     ax.plot(df_mean_std['mu_time'].values,
        #             df_mean_std['mu_error'].values,
        #             label=method_name,
        #             color=methods[method_name][3],
        #             linestyle=methods[method_name][4],
        #             marker=methods[method_name][5])
        #
        #     for i in df_mean_std.index:
        #         el = mpatches.Ellipse((df_mean_std['mu_time'][i], df_mean_std['mu_error'][i]),
        #                               df_mean_std['std_time'][i], df_mean_std['std_error'][i],
        #                               angle=0,
        #                               alpha=0.2,
        #                               color=methods[method_name][3],
        #                               linewidth=None)
        #         ax.add_artist(el)
        #
        # ax.set_title('Time error for SE(2)', fontdict=font_top)
        # ax.legend(loc='upper right', shadow=True, prop=legend_prop)
        #
        # ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        # ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        # ax.set_axisbelow(True)
        #
        # ax.set_xlabel('Time (sec)', fontdict=font_bl, labelpad=5)
        # ax.set_ylabel('Error (mm)', fontdict=font_bl, labelpad=5)
        # ax.set_xscale('log', nonposx="mask")
        # ax.set_yscale('log', nonposy="mask")
        #
        # pfi_figure_time_vs_error = jph(pfo_output_A4_AD, 'graph_time_vs_error.pdf')
        # plt.savefig(pfi_figure_time_vs_error, dpi=150)
        #
        # plt.show(block=True)
