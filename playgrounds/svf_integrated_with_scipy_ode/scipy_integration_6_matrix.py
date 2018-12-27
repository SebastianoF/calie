"""
Integration with scipy for one matrix-generated svf.
"""
import matplotlib.pyplot as plt
import numpy as np

from VECtorsToolkit.operations import lie_exp
from VECtorsToolkit.transformations import se2
from VECtorsToolkit.visualisations.fields import fields_comparisons
from VECtorsToolkit.visualisations.fields import fields_and_integral_curves

from VECtorsToolkit.fields import generate as gen
from VECtorsToolkit.fields import queries as qr

if __name__ == '__main__':

    omega = (20, 20)
    see_graphs = True
    use_also_scipy = True

    x_c = 10
    y_c = 10
    theta = np.pi/8

    tx   = (1 - np.cos(theta)) * x_c + np.sin(theta) * y_c
    ty   = -np.sin(theta) * x_c + (1 - np.cos(theta)) * y_c
    m_0 = se2.Se2G(theta, tx, ty)
    dm_0 = se2.se2g_log(m_0)

    passepartout = 3

    # Initialize the displacement field that will be computed using the integral curves.

    methods_vode = ['bdf', 'adams']
    steps_scipy  = 7  # , 15, 20, 25, 30]

    print(dm_0.get_matrix)
    print(m_0.get_matrix)

    # Get the ground vector fields:

    # svf
    svf_0 = gen.generate_from_matrix(omega, dm_0.get_matrix, structure='algebra')

    # displacement, I am subtracting the id to have a displacement and not a deformation.
    sdisp_0 = gen.generate_from_matrix(omega, m_0.get_matrix, structure='group')

    print(type(svf_0))
    print(type(sdisp_0))

    spline_interpolation_order = 3

    #
    l_exp = lie_exp.LieExp()
    l_exp.s_i_o = spline_interpolation_order
    sdisp_ss      = l_exp.scaling_and_squaring(svf_0)
    sdisp_gss_ei  = l_exp.gss_ei(svf_0)
    sdisp_euler   = l_exp.euler(svf_0)
    sdisp_mid_p   = l_exp.midpoint(svf_0)
    sdisp_euler_m = l_exp.euler_mod(svf_0)
    sdisp_rk4     = l_exp.rk4(svf_0)
    #

    if use_also_scipy:
        print('--------------------')
        print('Number of steps for scipy method : ' + str(steps_scipy))
        print('--------------------')
        disp_scipy_out  = l_exp.scipy_pointwise(svf_0, method=methods_vode[1], max_steps=steps_scipy,
                                                verbose=False, passepartout=passepartout,
                                                return_integral_curves=True)

        disp_scipy   = disp_scipy_out[0]
        integral_curves = disp_scipy_out[1]

        print(type(integral_curves))
        print(type(integral_curves[0]))

        error = qr.norm(disp_scipy - sdisp_0, passe_partout_size=3)

    print(type(sdisp_ss))
    print(type(sdisp_gss_ei))
    print(type(sdisp_euler))
    print(type(sdisp_euler_m))
    print(type(sdisp_rk4))

    print('--------------------')
    print("Norm of the svf:")
    print(qr.norm(svf_0, passe_partout_size=4))

    print('--------------------')
    print("Norm of the displacement field:")
    print(qr.norm(sdisp_0, passe_partout_size=4))

    print('--------------------')
    print("Norm of the errors:")
    print('--------------------')
    print('|ss - disp|        = ' + str(qr.norm(sdisp_ss - sdisp_0, passe_partout_size=passepartout)))
    print('|gss_ei - disp|    = ' + str(qr.norm(sdisp_gss_ei - sdisp_0, passe_partout_size=passepartout)))
    print('|euler - disp|     = ' + str(qr.norm(sdisp_euler - sdisp_0, passe_partout_size=passepartout)))
    print('|midpoint - disp|  = ' + str(qr.norm(sdisp_mid_p - sdisp_0, passe_partout_size=passepartout)))
    print('|euler_mod - disp| = ' + str(qr.norm(sdisp_euler_m - sdisp_0, passe_partout_size=passepartout)))
    print('|rk4 - disp|       = ' + str(qr.norm(sdisp_rk4 - sdisp_0, passe_partout_size=passepartout)))

    if use_also_scipy:
        print('----------  Error scipy ----------------------')
        print('|vode - disp| = ' + str(error))
        print('--------------------------------\n\n')

    if see_graphs:
        title_input_l = ['Sfv Input',
                         'Ground Output',
                         'Vode integrator']

        list_fields_of_field = [[svf_0], [sdisp_0],
                                [svf_0, sdisp_0, sdisp_ss],
                                # [svf_0, sdisp_0, sdisp_gss_ei],
                                # [svf_0, sdisp_0, sdisp_euler],
                                # [svf_0, sdisp_0, sdisp_mid_p],
                                # [svf_0, sdisp_0, sdisp_euler_m],
                                # [svf_0, sdisp_0, sdisp_rk4]
                                ]
        list_colors = ['r', 'b', 'r', 'b', 'm', 'r', 'b', 'r', 'b', 'm']

        fields_comparisons.see_n_fields_special(list_fields_of_field,
                                                fig_tag=50,
                                                row_fig=1, col_fig=3,
                                                colors_input=list_colors,
                                                titles_input=title_input_l,
                                                zoom_input=[0, 20, 0, 20], sample=(1, 1),
                                                window_title_input='matrix, random generated',
                                                legend_on=False)

    if see_graphs and use_also_scipy:
        fields_list_0 = [svf_0, sdisp_0, disp_scipy]

        fields_and_integral_curves.see_overlay_of_n_fields_and_flow(fields_list_0,
                                                                    integral_curves,
                                                                    fig_tag=14,
                                                                    input_color=['r', 'b', 'm'],
                                                                    input_label=None,
                                                                    see_tips=False,
                                                                    list_of_alpha_for_obj=[0.8, 0.5, 0.5],
                                                                    alpha_for_integral_curves=0.5)

    plt.show()
