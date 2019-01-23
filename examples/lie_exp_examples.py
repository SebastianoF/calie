"""
Module to see the integration of an svf generated with a matrix in se2.
Only the output of scaling and squaring-based methods are compared with the ground truth.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from calie.operations import lie_exp
from calie.transformations import se2
from calie.visualisations.fields import fields_comparisons

from calie.fields import generate as gen
from calie.fields import queries as qr

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

    l_exp = lie_exp.LieExp()
    l_exp.s_i_o = sio

    methods = [l_exp.euler,
               l_exp.midpoint,
               l_exp.scaling_and_squaring,
               l_exp.gss_ei,
               l_exp.gss_aei,
               l_exp.trapeziod_euler,
               l_exp.gss_trapezoid_euler,
               l_exp.trapezoid_midpoint,
               l_exp.gss_trapezoid_midpoint]

    res_time = np.zeros(len(methods))
    res_err = np.zeros(len(methods))
    fields_list = []

    for met_id, met in enumerate(methods):

        start = time.time()
        sdisp_num = met(svf_0)
        res_time[met_id] = (time.time() - start)

        res_err[met_id] = qr.norm(sdisp_num - sdisp_0, passe_partout_size=passepartout)
        print(res_err[met_id])
        fields_list.append(sdisp_num)

    print('--------------------')
    print("Norm of the svf:")
    print(qr.norm(svf_0, passe_partout_size=passepartout))

    print('--------------------')
    print("Norm of the displacement field:")
    print(qr.norm(sdisp_0, passe_partout_size=passepartout))

    print('--------------------')
    print("Norm of the errors:")
    print('--------------------')

    for met_id, met in enumerate(methods):
        print('|{0:>22} - disp|        = {1}'.format(met.__name__, res_err[met_id]))

    print('--------------------')
    print("Computational Times: ")
    print('--------------------')
    for met_id, met in enumerate(methods):
        print('time {0:>22}  = {1}'.format(met.__name__, res_time[met_id]))

    title_input_l = ['Sfv Input', 'Ground Output'] + [met.__name__ for met in methods]

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
