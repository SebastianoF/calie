"""
Module to test the scaling and squaring-based methods.
Some tests are visual test only
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from VECtorsToolkit.operations.lie_exponential import lie_exponential
from VECtorsToolkit.transformations.se2 import Se2G, se2g_log
from VECtorsToolkit.visualisations.fields.fields_comparisons import see_n_fields_special

from VECtorsToolkit.fields.generate import generate_from_matrix
from VECtorsToolkit.fields.queries import vf_norm


def test_visual_assessment_method_one_se2(show=False):
    """
    :param show: to add the visualisation of a figure.

    This test is for visual assessment. Please put show to True.

    Aimed to test the prototyping of the computation of the exponential map
    with some methods.

    (Nothing is saved in external folder.)
    """

    ##############
    # controller #
    ##############

    domain = (20, 20)

    x_c = 10
    y_c = 10
    theta = np.pi / 8

    tx   = (1 - np.cos(theta)) * x_c + np.sin(theta) * y_c
    ty   = -np.sin(theta) * x_c + (1 - np.cos(theta)) * y_c

    passepartout = 5
    spline_interpolation_order = 3

    methods_list = ['ss',
                    'gss_ei',
                    'gss_ei_mod',
                    'gss_aei',
                    'midpoint',
                    'series',
                    'euler',
                    'euler_aei',
                    'euler_mod',
                    'heun',
                    'heun_mod',
                    'rk4',
                    'gss_rk4',
                    ]

    # -----
    # model
    # -----

    m_0 = Se2G(theta, tx, ty)
    dm_0 = se2g_log(m_0)

    # -- generate subsequent vector fields

    svf_0   = generate_from_matrix(domain, dm_0.get_matrix, structure='algebra')
    sdisp_0 = generate_from_matrix(domain, m_0.get_matrix, structure='group')

    # -- compute exponential with different available methods:

    sdisp_list = []
    res_time = np.zeros(len(methods_list))

    for num_met, met in enumerate(methods_list):
        start = time.time()
        sdisp_list.append(lie_exponential(svf_0, algorithm=met, s_i_o=spline_interpolation_order, input_num_steps=10))
        res_time[num_met] = (time.time() - start)

    # ----
    # view
    # ----

    print('--------------------')
    print('Norm of the svf: ')
    print(vf_norm(svf_0, passe_partout_size=4))

    print('--------------------')
    print("Norm of the displacement field:")
    print(vf_norm(sdisp_0, passe_partout_size=4))

    print('--------------------')
    print('Norm of the errors: ')
    print('--------------------')

    for num_met in range(len(methods_list)):
        err = vf_norm(sdisp_list[num_met] - sdisp_0, passe_partout_size=passepartout)
        print('|{0:>12} - disp|  = {1}'.format(methods_list[num_met], err))

        if methods_list[num_met] == 'euler':
            assert err < 3
        else:
            assert err < 0.5

    print('---------------------')
    print('Computational Times: ')
    print('---------------------')

    if show:
        title_input_l = ['Sfv Input', 'Ground Output'] + methods_list
        fields_list = [svf_0, sdisp_0] + sdisp_list

        list_fields_of_field = [[svf_0], [sdisp_0]]
        list_colors = ['r', 'b']
        for third_field in fields_list[2:]:
            list_fields_of_field += [[svf_0, sdisp_0, third_field]]
            list_colors += ['r', 'b', 'm']

        see_n_fields_special(list_fields_of_field, fig_tag=50,
                             row_fig=3,
                             col_fig=5,
                             input_figsize=(14, 7),
                             colors_input=list_colors,
                             titles_input=title_input_l,
                             sample=(1, 1),
                             zoom_input=[0, 20, 0, 20],
                             window_title_input='matrix generated svf')

        plt.show()

if __name__ == '__main__':
    test_visual_assessment_method_one_se2(True)
