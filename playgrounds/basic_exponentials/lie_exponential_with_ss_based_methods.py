"""
Module to see the integration of an svf generated with a matrix in se2.
Only the output of scaling and squaring-based methods are compared with the ground truth.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from VECtorsToolkit.operations import lie_exponential
from VECtorsToolkit.transformations import se2
from VECtorsToolkit.visualisations.fields import fields_comparisons

from VECtorsToolkit.fields import generate as gen
from VECtorsToolkit.fields import queries as qr

if __name__ == '__main__':

    # -> controller <- #

    domain = (41, 41)

    x_c = 20
    y_c = 20
    theta = np.pi / 12

    tx   = (1 - np.cos(theta)) * x_c + np.sin(theta) * y_c
    ty   = -np.sin(theta) * x_c + (1 - np.cos(theta)) * y_c

    passepartout = 4
    sio = 3  # spline interpolation order

    res_time = np.zeros(9)

    # -> model <- #

    m_0 = se2.Se2G(theta, tx, ty)
    dm_0 = se2.se2g_log(m_0)

    print(m_0.get_matrix)
    print('')
    print(dm_0.get_matrix)

    # Generate subsequent vector fields

    sdisp_0 = gen.generate_from_matrix(domain, m_0.get_matrix, structure='group')
    svf_0 = gen.generate_from_matrix(domain, dm_0.get_matrix, structure='algebra')

    # Compute exponential with different available methods:

    start = time.time()
    sdisp_euler = lie_exponential.lie_exponential(svf_0, algorithm='euler', s_i_o=sio)
    res_time[0] = (time.time() - start)

    start = time.time()
    sdisp_midpoint = lie_exponential.lie_exponential(svf_0, algorithm='midpoint', s_i_o=sio)
    res_time[1] = (time.time() - start)

    start = time.time()
    sdisp_ss      = lie_exponential.lie_exponential(svf_0, algorithm='ss', s_i_o=sio)
    res_time[2] = (time.time() - start)

    start = time.time()
    sdisp_ss_pa   = lie_exponential.lie_exponential(svf_0, algorithm='gss_ei', s_i_o=sio)
    res_time[3] = (time.time() - start)

    start = time.time()
    sdisp_ss_pa_m = lie_exponential.lie_exponential(svf_0, algorithm='gss_aei', s_i_o=sio)
    res_time[4] = (time.time() - start)

    start = time.time()
    sdisp_trap_eu = lie_exponential.lie_exponential(svf_0, algorithm='trapezoid_euler', s_i_o=sio)
    res_time[5] = (time.time() - start)

    start = time.time()
    sdisp_gss_trap_eu = lie_exponential.lie_exponential(svf_0, algorithm='gss_trapezoid_euler', s_i_o=sio)
    res_time[6] = (time.time() - start)

    start = time.time()
    sdisp_trap_mid = lie_exponential.lie_exponential(svf_0, algorithm='trapezoid_midpoint', s_i_o=sio)
    res_time[7] = (time.time() - start)

    start = time.time()
    sdisp_gss_trap_mid = lie_exponential.lie_exponential(svf_0, algorithm='gss_trapezoid_midpoint', s_i_o=sio)
    res_time[8] = (time.time() - start)
    # -> view <- #

    print('--------------------')
    print("Norm of the svf:")
    print(qr.vf_norm(svf_0, passe_partout_size=passepartout))

    print('--------------------')
    print("Norm of the displacement field:")
    print(qr.vf_norm(sdisp_0, passe_partout_size=passepartout))

    print('--------------------')
    print("Norm of the errors:")
    print('--------------------')

    error_norm_euler         = qr.vf_norm(sdisp_euler - sdisp_0, passe_partout_size=passepartout)
    error_norm_midpoint      = qr.vf_norm(sdisp_midpoint - sdisp_0, passe_partout_size=passepartout)
    error_norm_ss            = qr.vf_norm(sdisp_ss - sdisp_0, passe_partout_size=passepartout)
    error_norm_ss_ei         = qr.vf_norm(sdisp_ss_pa - sdisp_0, passe_partout_size=passepartout)
    error_norm_ss_aei        = qr.vf_norm(sdisp_ss_pa_m - sdisp_0, passe_partout_size=passepartout)
    error_norm_trap_eu       = qr.vf_norm(sdisp_trap_eu - sdisp_0, passe_partout_size=passepartout)
    error_norm_gss_trap_eu   = qr.vf_norm(sdisp_gss_trap_eu - sdisp_0, passe_partout_size=passepartout)
    error_norm_trap_mid      = qr.vf_norm(sdisp_trap_mid - sdisp_0, passe_partout_size=passepartout)
    error_norm_gss_trap_mid  = qr.vf_norm(sdisp_gss_trap_mid - sdisp_0, passe_partout_size=passepartout)

    print('|euler - disp|        = {}'.format(str(error_norm_euler)))
    print('|midpoint - disp|     = {}'.format(str(error_norm_midpoint)))
    print('|ss - disp|           = {}'.format(str(error_norm_ss)))
    print('|ss_ei - disp|        = {}'.format(str(error_norm_ss_ei)))
    print('|ss_aei - disp|       = {}'.format(str(error_norm_ss_aei)))
    print('|trap_eu - disp|      = {}'.format(str(error_norm_trap_eu)))
    print('|gss_trap_eu - disp|  = {}'.format(str(error_norm_gss_trap_eu)))
    print('|trap_mid - disp|     = {}'.format(str(error_norm_trap_mid)))
    print('|gss_trap_mid - disp| = {}'.format(str(error_norm_gss_trap_mid)))

    print('--------------------')
    print("Computational Times: ")
    print('--------------------')

    print(' time euler        = {}'.format(str(res_time[0])))
    print(' time midpoint     = {}'.format(str(res_time[1])))
    print(' time ss           = {}'.format(str(res_time[2])))
    print(' time ss_ei        = {}'.format(str(res_time[3])))
    print(' time ss_aei       = {}'.format(str(res_time[4])))
    print(' time trap_eu      = {}'.format(str(res_time[5])))
    print(' time gss_trap_eu  = {}'.format(str(res_time[6])))
    print(' time trap_mid     = {}'.format(str(res_time[7])))
    print(' time gss_trap_mid = {}'.format(str(res_time[8])))

    fields_list = [svf_0, sdisp_0, sdisp_ss,   sdisp_ss_pa,   sdisp_ss_pa_m, sdisp_trap_eu, sdisp_gss_trap_eu,
                   sdisp_trap_mid, sdisp_gss_trap_mid]

    title_input_l = ['Sfv Input',
                     'Ground Output',
                     'Euler',
                     'Midpoint',
                     'Scaling and Squaring',
                     'Gen Scal. and Sq. exp int',
                     'Gen Scal. and Sq. approx exp int',
                     'Trapezoid-Euler method',
                     'Gen Scal. and Sq. trapezoid-Euler',
                     'Trapezoid-Midpoint method',
                     'Gen Scal. and Sq. trapezoid-Midopint'
                     ]

    list_fields_of_field = [[svf_0], [sdisp_0]]
    list_colors = ['r', 'b']
    for third_field in fields_list[2:]:
        list_fields_of_field += [[svf_0, sdisp_0, third_field]]
        list_colors += ['r', 'b', 'm']

        fields_comparisons.see_n_fields_special(list_fields_of_field,
                                                fig_tag=50,
                                                row_fig=2,
                                                col_fig=5,
                                                input_figsize=(10, 5),
                                                colors_input=list_colors,
                                                titles_input=title_input_l,
                                                sample=(1, 1),
                                                zoom_input=[0 + passepartout, domain[0] - passepartout,
                                                            0 + passepartout, domain[1] - passepartout],
                                                window_title_input='matrix, random generated',
                                                legend_on=False)

    plt.tight_layout()
    plt.show()

    # assert error_norm_ss < 0.09
    # assert error_norm_ss_ei < 0.09
    # assert error_norm_ss_aei < 0.09

