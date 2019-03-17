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

    control = {'generate_dataset'  : False,
               'elaborate_output'  : True,
               'show_graphs'       : True}

    verbose = 1

    add_OMP = True

    pfo_nifty_reg_app_standard = '/Users/sebastiano/a_data/TData/wbir_16/Code/install{}/niftyreg_install/reg-apps'.format('_omp' if add_OMP else '')
    pfo_nifty_reg_app_ei       = '/Users/sebastiano/a_data/TData/wbir_16/Code/Install{}/niftyreg_install_aei/reg-apps'.format('_omp' if add_OMP else '')

    if add_OMP:
        pfo_output_A4_AD = pfo_output_A4_AD + '_omp'

    # parameters:

    experiments = ['standard', 'diff', 'diff aei']

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

            pfi_niftyreg_output_standard = jph(pfo_output_A4_AD, 'time_comparison_standard_sj_{}.txt'.format(sj_name))
            cmd_standard = '{0}/reg_f3d -ref {1} -rmask {2} -flo {3} -fmask {4} -cpp {5} -res {6} -omp 8 -ln 1 -maxit 400 > {7}'.format(  # -ln 1 -maxit 500
                pfo_nifty_reg_app_standard,
                pfi_T1W_fixed_tp1, pfi_mask_fixed_tp1, pfi_moving_on_target_warp_aff, pfi_moving_on_target_mask_aff,
                pfi_cpp_time_comp, pfi_moving_on_target_nrig_time_comp,
                pfi_niftyreg_output_standard
            )

            pfi_niftyreg_output_diff = jph(pfo_output_A4_AD, 'time_comparison_diff_sj_{}.txt'.format(sj_name))
            cmd_diff     = '{0}/reg_f3d -ref {1} -rmask {2} -flo {3} -fmask {4} -cpp {5} -res {6} -vel -omp 8 -ln 1 -maxit 400 > {7}  '.format(
                pfo_nifty_reg_app_standard,
                pfi_T1W_fixed_tp1, pfi_mask_fixed_tp1, pfi_moving_on_target_warp_aff, pfi_moving_on_target_mask_aff,
                pfi_cpp_time_comp, pfi_moving_on_target_nrig_time_comp,
                pfi_niftyreg_output_diff
            )

            pfi_niftyreg_output_aei = jph(pfo_output_A4_AD, 'time_comparison_diff_aei_sj_{}.txt'.format(sj_name))
            cmd_diff_ei  = '{0}/reg_f3d -ref {1} -rmask {2} -flo {3} -fmask {4} -cpp {5} -res {6} -vel  -omp 8 -ln 1 -maxit 400  > {7}  '.format(
                pfo_nifty_reg_app_ei,
                pfi_T1W_fixed_tp1, pfi_mask_fixed_tp1, pfi_moving_on_target_warp_aff, pfi_moving_on_target_mask_aff,
                pfi_cpp_time_comp, pfi_moving_on_target_nrig_time_comp,
                pfi_niftyreg_output_aei
            )

            # standard

            start = time.time()
            print_and_run(cmd_standard)
            stop_standard = (time.time() - start)

            df_computational_time['standard'][sj_name] = stop_standard
            print(stop_standard)

            # diff

            start = time.time()
            print_and_run(cmd_diff)
            stop_diff = (time.time() - start)

            df_computational_time['diff'][sj_name] = stop_diff
            print(stop_diff)

            # diff ei

            start = time.time()
            print_and_run(cmd_diff_ei)
            stop_diff_ei = (time.time() - start)

            df_computational_time['diff aei'][sj_name] = stop_diff_ei
            print(stop_diff_ei)

            print(df_computational_time)

        pfi_df_time_comparison = jph(pfo_output_A4_AD, 'time_comparison.csv')
        df_computational_time.to_csv(pfi_df_time_comparison)

    else:
        pfi_df_time_comparison = jph(pfo_output_A4_AD, 'time_comparison.csv')
        assert os.path.exists(pfi_df_time_comparison)

    ####################
    # elaborate output #
    ####################
    if control['elaborate_output']:

        print('---------------------------------------------------------------------------')
        print('  Elaborating dataset dataset COMPARISON NIFTYREG COMPUTATIONAL TIME!      ')
        print('---------------------------------------------------------------------------')

        pfi_df_time_comparison = jph(pfo_output_A4_AD, 'time_comparison.csv')
        print('computational time table')
        df_computational_time = pd.read_csv(pfi_df_time_comparison)
        print(df_computational_time.to_latex())

        print()

        df_had_converged_before_limit = df_computational_time.copy()
        df_had_converged_before_limit[:] = True

        df_main = pd.DataFrame(columns=['subject', 'experiment', 'objective function'])

        for sj_name_idx, sj_name in enumerate(subjects):
            for exper in experiments:

                pfi_niftireg_output = jph(pfo_output_A4_AD, 'time_comparison_{}_sj_{}.txt'.format(exper.replace(' ', '_'), sj_name))
                print('\n\n Elaborating {}\n\n'.format(pfi_niftireg_output))

                assert os.path.exists(pfi_niftireg_output)

                df_local = pd.DataFrame(index=range(401), columns=['experiment', 'objective function'])

                f = open(pfi_niftireg_output, 'r')
                for line in f.readlines():

                    if 'Initial objective function' in line:
                        val = line.replace('Initial objective function:', '').replace('[NiftyReg F3D]', '').replace('[NiftyReg F3D2]', '').split('=')[0].strip()
                        of_value = float(val)
                        step = 0
                        df_local['objective function'][step] = of_value

                    if 'Current objective function' in line:
                        vals = line.replace('[NiftyReg F3D]', '').replace('[NiftyReg F3D2]', '').replace('Current objective function', '').split('=')[0].strip().replace('[', '').replace(']', '').split(':')
                        of_value = float(vals[1].strip())
                        step = int(vals[0].strip())
                        df_local['objective function'][step] = of_value

                    if 'WARNING' in line:
                        df_had_converged_before_limit[exper][sj_name_idx] = False


                f.close()
                df_local['subject'] = sj_name
                df_local['experiment'] = exper
                df_local['objective function'] = df_local['objective function'].astype(float).interpolate(method='polynomial', order=3)

                df_main = df_main.append(df_local)

    #     pfi_df_main_experiments_values = jph(pfo_output_A4_AD, 'steps_time_dataframe.csv')
    #     df_main.to_csv(pfi_df_main_experiments_values)
    #
    #     # print(df_main)
    #     print(df_had_converged_before_limit)
    #
    # ###############
    # # show graphs #
    # ###############
    #
    # if control['show_graphs']:
    #     print('---------------------------------------------------------------------------')
    #     print('  Showing graphs      ')
    #     print('---------------------------------------------------------------------------')
    #
    #     pfi_df_main_experiments_values = jph(pfo_output_A4_AD, 'steps_time_dataframe.csv')
    #     df_main = pd.read_csv(pfi_df_main_experiments_values)
    #
        df_main['timepoint'] = df_main.index

        font_top = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': 14}
        font_bl = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        legend_prop = {'size': 11}

        sns.set_style('darkgrid')

        plt.subplots(figsize=(11, 6))

        ax = sns.lineplot(x="timepoint", y="objective function", hue="experiment", data=df_main)

        ax.set_title('Objective function per step', fontdict=font_top)

        ax.set_xlabel('Steps', fontdict=font_bl, labelpad=5)
        ax.set_ylabel('Cost Function', fontdict=font_bl, labelpad=5)

        pfi_figure_time_vs_error = jph(pfo_output_A4_AD, 'steps_cost_functions.pdf')
        plt.savefig(pfi_figure_time_vs_error, dpi=150)

        plt.show(block=True)
