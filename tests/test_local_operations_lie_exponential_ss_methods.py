import time

import matplotlib.pyplot as plt
import numpy as np
from VECtorsToolkit.operations import lie_exp
from VECtorsToolkit.transformations import se2
from VECtorsToolkit.visualisations.fields import fields_comparisons

from VECtorsToolkit.fields import generate as gen
from VECtorsToolkit.fields import queries as qr


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

    l_exp = lie_exp.LieExp()
    l_exp.s_i_o = spline_interpolation_order

    methods_list = [l_exp.scaling_and_squaring,
                    l_exp.gss_ei,
                    l_exp.gss_ei_mod,
                    l_exp.gss_aei,
                    l_exp.midpoint,
                    l_exp.series,
                    l_exp.euler,
                    l_exp.euler_aei,
                    l_exp.euler_mod,
                    l_exp.heun,
                    l_exp.heun_mod,
                    l_exp.rk4,
                    l_exp.gss_rk4,
                    l_exp.trapeziod_euler,
                    l_exp.trapzoid_midpoint,
                    l_exp.gss_trapezoid_euler,
                    l_exp.gss_trapezoid_midpoint
                    ]

    # -----
    # model
    # -----

    m_0 = se2.Se2G(theta, tx, ty)
    dm_0 = se2.se2g_log(m_0)

    # -- generate subsequent vector fields

    svf_0   = gen.generate_from_matrix(domain, dm_0.get_matrix, structure='algebra')
    sdisp_0 = gen.generate_from_matrix(domain, m_0.get_matrix, structure='group')

    # -- compute exponential with different available methods:

    sdisp_list = []
    res_time = np.zeros(len(methods_list))

    for num_met, met in enumerate(methods_list):
        start = time.time()
        sdisp_list.append(met(svf_0, input_num_steps=10))
        res_time[num_met] = (time.time() - start)

    # ----
    # view
    # ----

    print('--------------------')
    print('Norm of the svf: ')
    print(qr.norm(svf_0, passe_partout_size=4))

    print('--------------------')
    print("Norm of the displacement field:")
    print(qr.norm(sdisp_0, passe_partout_size=4))

    print('--------------------')
    print('Norm of the errors: ')
    print('--------------------')

    for num_met in range(len(methods_list)):
        err = qr.norm(sdisp_list[num_met] - sdisp_0, passe_partout_size=passepartout)
        print('|{0:>22} - disp|  = {1}'.format(methods_list[num_met].__name__, err))

        if methods_list[num_met].__name__ == 'euler':
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

        fields_comparisons.see_n_fields_special(list_fields_of_field,
                                                fig_tag=50,
                                                row_fig=5,
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
