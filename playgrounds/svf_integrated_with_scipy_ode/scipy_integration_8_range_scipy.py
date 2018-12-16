"""
Try BDF with steps < 7 where the error is still stable.

"""
import time

import matplotlib.pyplot as plt
import numpy as np
from VECtorsToolkit.tools.operations.lie_exponential import lie_exponential_scipy
from VECtorsToolkit.tools.transformations.se2_a import se2_g
from VECtorsToolkit.tools.visualisations.fields.fields_comparisons import see_n_fields_special

from VECtorsToolkit.fields import generate_from_matrix
from VECtorsToolkit.fields import vf_norm

if __name__ == '__main__':

    # -> Controller <- #

    omega = (14, 14)  # Matrix coordinates: x = -Y, y = X
    passepartout = 3

    interval_theta = (- np.pi / 4, np.pi / 4)
    epsilon = 0.01

    N = 5

    verbose            = True  # moderate amount of information
    plot_em_all        = False  # to be True only for emergency debugging!

    # Parameter choices for exponential scipy

    max_steps      = range(2, 15, 1)
    methods_vode   = ['bdf', 'adams']  # consider only adams, since they provides almost the same results!
    interp_methods = ['linear', 'cubic']
    verbose_exp    = False

    # -> Data structure to store data <- #

    # 4d matrix x:steps, y:integrators or methods, z:interpolation, t:error
    main_errors = np.zeros([len(max_steps), len(methods_vode), len(interp_methods), N])
    main_computational_time = np.zeros([len(max_steps), len(methods_vode), len(interp_methods), N])

    if verbose:
        print('Main matrices for storage created:')
        print(main_errors.shape)
        print(main_computational_time.shape)

    # -> Loops: <- #

    for i in range(N):

        # -> Compute random matrices of transformations:
        m_0 = se2_g.randomgen_custom_center(interval_theta=interval_theta, omega=omega, epsilon_zero_avoidance=epsilon)
        dm_0 = se2_g.se2_g_log(m_0)

        print('Matrices to generate svf and disp ground truth created:')
        print(dm_0.get_matrix)
        print(m_0.get_matrix)

        # -> generate subsequent vector fields out of the matrices <- #
        svf_0 = generate_from_matrix(omega, dm_0.get_matrix, structure='algebra')
        sdisp_0 = generate_from_matrix(omega, m_0.get_matrix, structure='group')

        for interp_method_i, interp_method in enumerate(interp_methods):
            for method_i, method in enumerate(methods_vode):
                for max_step_i, max_step in enumerate(max_steps):

                    print('------- Beginning of a new cycle for matrix {} out of {} -------'.format(str(i+1), str(N)))
                    print('interp method     : {}'.format(str(interp_methods[interp_method_i])))
                    print('vode method       : vode')
                    print('step              : {}'.format(str(max_steps[max_step_i])))
                    print('---------------------------------------------------------------')

                    # -> compute exponential with different available methods: <- #

                    start = time.time()

                    sdisp_scipy = lie_exponential_scipy(svf_0,
                                                        integrator='vode',
                                                        method=method,
                                                        max_steps=max_step,
                                                        interpolation_method=interp_method,
                                                        verbose=verbose_exp,
                                                        passepartout=passepartout,
                                                        return_integral_curves=False)

                    operation_time = (time.time() - start)
                    error = vf_norm(sdisp_scipy - sdisp_0, passe_partout_size=passepartout)

                    print('----------  Error  and Computational Time  ----')
                    print('|vode - disp| = {} voxel'.format(str(error)))
                    print('Comp Time     = {} sec.'.format(str(operation_time)))
                    print('-----------------------------------------------')

                    # store errors and computational time in appropriate matrices:
                    # 4d matrix x: steps, y: integrators and methods, z: interpolation, t: error/computational time
                    main_errors[max_step_i, method_i, interp_method_i, i] = error
                    main_computational_time[max_step_i, method_i, interp_method_i, i] = operation_time

                    # Keep it false!!! add an if, if there is some data that does not work
                    fields_list_0 = [svf_0, sdisp_0, sdisp_scipy]

                    list_fields_of_field = [[svf_0], [sdisp_0], [svf_0, sdisp_0, sdisp_scipy]]
                    list_colors = ['r', 'b', 'r', 'b', 'm']

                    see_n_fields_special(list_fields_of_field,
                                         fig_tag=i,
                                         row_fig=1, col_fig=3,
                                         colors_input=list_colors,
                                         zoom_input=[0, 20, 0, 20], sample=(1, 1),
                                         window_title_input='matrix, random generated',
                                         legend_on=False)

                    plt.show()
