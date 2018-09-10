"""
Module to see the integration of an svf generated with a matrix in se2.
Only the output of scaling and squaring-based methods are compared with the ground truth.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from VECtorsToolkit.tools.fields.generate_vf import generate_from_matrix
from VECtorsToolkit.tools.fields.queries import vf_norm
from VECtorsToolkit.tools.local_operations.lie_exponential import lie_exponential
from VECtorsToolkit.tools.transformations.se2_a import se2_g
from VECtorsToolkit.tools.visualisations.fields.fields_comparisons import see_n_fields_special

if __name__ == '__main__':

    # -> controller <- #

    domain = (21, 21)

    x_c = 10
    y_c = 10
    theta = np.pi / 8

    tx   = (1 - np.cos(theta)) * x_c + np.sin(theta) * y_c
    ty   = -np.sin(theta) * x_c + (1 - np.cos(theta)) * y_c

    passepartout = 4
    spline_interpolation_order = 3

    res_time = np.zeros(4)

    # -> model <- #

    m_0 = se2_g.se2_g(theta, tx, ty)
    dm_0 = se2_g.se2_g_log(m_0)

    print(m_0.get_matrix)
    print('')
    print(dm_0.get_matrix)

    # Generate subsequent vector fields

    sdisp_0 = generate_from_matrix(domain, m_0.get_matrix, structure='group')
    svf_0 = generate_from_matrix(domain, dm_0.get_matrix, structure='algebra')

    # Compute exponential with different available methods:

    start = time.time()
    sdisp_ss      = lie_exponential(svf_0, algorithm='ss', s_i_o=spline_interpolation_order)
    res_time[0] = (time.time() - start)

    start = time.time()
    sdisp_ss_pa   = lie_exponential(svf_0, algorithm='gss_ei', s_i_o=spline_interpolation_order)
    res_time[1] = (time.time() - start)

    start = time.time()
    sdisp_ss_pa_m = lie_exponential(svf_0, algorithm='gss_aei', s_i_o=spline_interpolation_order)
    res_time[2] = (time.time() - start)

    # -> view <- #

    print('--------------------')
    print("Norm of the svf:")
    print(vf_norm(svf_0, passe_partout_size=passepartout))

    print('--------------------')
    print("Norm of the displacement field:")
    print(vf_norm(sdisp_0, passe_partout_size=passepartout))

    print('--------------------')
    print("Norm of the errors:")
    print('--------------------')

    error_norm_ss     = vf_norm(sdisp_ss - sdisp_0, passe_partout_size=passepartout)
    error_norm_ss_ei  = vf_norm(sdisp_ss_pa - sdisp_0, passe_partout_size=passepartout)
    error_norm_ss_aei = vf_norm(sdisp_ss_pa_m - sdisp_0, passe_partout_size=passepartout)

    print('|ss - disp|       = {}'.format(str(error_norm_ss)))
    print('|ss_ei - disp|    = {}'.format(str(error_norm_ss_ei)))
    print('|ss_aei - disp|   = {}'.format(str(error_norm_ss_aei)))

    print('--------------------')
    print("Computational Times: ")
    print('--------------------')

    print(' time ss       = {}'.format(str(res_time[0])))
    print(' time ss_ei    = {}'.format(str(res_time[1])))
    print(' time ss_aei   = {}'.format(str(res_time[2])))

    fields_list = [svf_0, sdisp_0, sdisp_ss,   sdisp_ss_pa,   sdisp_ss_pa_m]

    title_input_l = ['Sfv Input',
                     'Ground Output',
                     'Scaling and Squaring',
                     'Generalised Scal. and Sq. exp int',
                     'Generalised Scal. and Sq. approx exp int']

    list_fields_of_field = [[svf_0], [sdisp_0]]
    list_colors = ['r', 'b']
    for third_field in fields_list[2:]:
        list_fields_of_field += [[svf_0, sdisp_0, third_field]]
        list_colors += ['r', 'b', 'm']

    see_n_fields_special(list_fields_of_field,
                         fig_tag=50,
                         row_fig=2,
                         col_fig=3,
                         input_figsize=(10, 7),
                         colors_input=list_colors,
                         titles_input=title_input_l,
                         sample=(1, 1),
                         zoom_input=[0 + passepartout, domain[0] - passepartout,
                                     0 + passepartout, domain[1] - passepartout],
                         window_title_input='matrix, random generated',
                         legend_on=False)

    plt.show()

    assert error_norm_ss < 0.09
    assert error_norm_ss_ei < 0.09
    assert error_norm_ss_aei < 0.09

