"""
Module to test the scaling and squaring-based methods.
Some tests are visual test only
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from VECtorsToolkit.tools.transformations.se2_g import se2_g, se2_g_log
from VECtorsToolkit.tools.visualisations.fields_comparisons import see_n_fields_special
from VECtorsToolkit.tools.fields.generate_vf import generate_from_matrix
from VECtorsToolkit.tools.local_operations.exponential import lie_exponential
from VECtorsToolkit.tools.fields.queries import vf_norm


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

    res_time = np.zeros(4)

    # -----
    # model
    # -----

    m_0 = se2_g(theta, tx, ty)
    dm_0 = se2_g_log(m_0)

    # -- generate subsequent vector fields

    svf_0   = generate_from_matrix(domain, dm_0.get_matrix, structure='algebra')
    sdisp_0 = generate_from_matrix(domain, m_0.get_matrix - np.eye(3), structure='group')

    # -- compute exponential with different available methods:

    start = time.time()
    sdisp_ss      = lie_exponential(svf_0, algorithm='ss', s_i_o=spline_interpolation_order)
    res_time[0] = (time.time() - start)

    start = time.time()
    sdisp_gss_ei   = lie_exponential(svf_0, algorithm='gss_ei', s_i_o=spline_interpolation_order)
    res_time[1] = (time.time() - start)

    start = time.time()
    sdisp_gss_aei = lie_exponential(svf_0, algorithm='gss_aei', s_i_o=spline_interpolation_order)
    res_time[2] = (time.time() - start)

    # ----
    # view
    # ----

    print('--------------------')
    print("Norm of the svf:")
    print(vf_norm(svf_0, passe_partout_size=4))

    print('--------------------')
    print("Norm of the displacement field:")
    print(vf_norm(sdisp_0, passe_partout_size=4))

    print('--------------------')
    print("Norm of the errors:")
    print('--------------------')

    err_ss     = vf_norm(sdisp_ss - sdisp_0, passe_partout_size=passepartout)
    err_ss_ei  = vf_norm(sdisp_gss_ei - sdisp_0, passe_partout_size=passepartout)
    err_ss_aei = vf_norm(sdisp_gss_aei - sdisp_0, passe_partout_size=passepartout)

    print('|ss - disp|        = ' + str(err_ss))
    print('|ss_ei - disp|     = ' + str(err_ss_ei))
    print('|ss_aei - disp|   = ' + str(err_ss_aei))

    print('--------------------')
    print("Computational Times: ")
    print('--------------------')

    print(' ss        = ' + str(res_time[0]))
    print(' ss_pa     = ' + str(res_time[1]))
    print(' ss_pa_m   = ' + str(res_time[2]))

    fields_list = [svf_0, sdisp_0, sdisp_ss,   sdisp_gss_ei,   sdisp_gss_aei]

    if 1:
        title_input_l = ['Sfv Input',
                         'Ground Output',
                         'Scaling and Squaring',
                         'gss ei',
                         'gss aei']

        list_fields_of_field = [[svf_0], [sdisp_0]]
        list_colors = ['r', 'b']
        for third_field in fields_list[2:]:
            list_fields_of_field += [[svf_0, sdisp_0, third_field]]
            list_colors += ['r', 'b', 'm']

        see_n_fields_special(list_fields_of_field, fig_tag=50,
                             row_fig=2,
                             col_fig=3,
                             input_figsize=(14, 7),
                             colors_input=list_colors,
                             titles_input=title_input_l,
                             sample=(1, 1),
                             zoom_input=[0, 20, 0, 20],
                             window_title_input='matrix, random generated',
                             legend_on=False)

    if show:
        plt.show()

    assert err_ss < 0.05
    assert err_ss_ei < 0.05
    assert err_ss_aei < 0.05

test_visual_assessment_method_one_se2(True)
