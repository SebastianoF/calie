"""
Parameters of the scipy integrator test and performance evaluation.
"""
import time
import matplotlib.pyplot as plt
import numpy as np

from VECtorsToolkit.operations.lie_exp import LieExp
from VECtorsToolkit.transformations import se2
from VECtorsToolkit.visualisations.fields.fields_comparisons import see_n_fields_special

from VECtorsToolkit.fields import generate as gen
from VECtorsToolkit.fields import queries as qr


if __name__ == '__main__':

    # --  Controller -- #

    omega = (14, 14)  # Matrix coordinates: x = -Y, y = X
    passepartout = 3

    interval_theta = (- np.pi / 4, np.pi / 4)
    epsilon = 0.01

    N = 5

    # -> Parameter choices for exponential Scipy.

    max_steps      = range(5, 45, 5)
    integrators    = ['vode', 'lsoda', 'dopri5', 'dop853']
    methods_vode   = ['bdf', 'adams']  # consider only adams, since they provides almost the same results!
    interp_methods = ['linear']
    verbose_exp    = False

    # -> Data structure to store data

    # 4d matrix x:steps, y:integrators and methods, z:interpolation, t:error
    main_errors = np.zeros([len(max_steps), len(integrators), len(interp_methods), N])
    main_computational_time = np.zeros([len(max_steps), len(integrators), len(interp_methods), N])

    print('Main matrices for storage created:')
    print(main_errors.shape)
    print(main_computational_time.shape)

    for i in range(N):

        # -> Compute random matrices of transformations
        m_0 = se2.se2g_randomgen_custom_center(interval_theta=interval_theta, epsilon_zero_avoidance=epsilon)
        dm_0 = se2.se2g_log(m_0)

        print('Matrices to generate svf and disp ground truth created:')
        print(dm_0.get_matrix)
        print(m_0.get_matrix)

        svf_0   = gen.generate_from_matrix(omega, dm_0.get_matrix, structure='algebra')
        sdisp_0 = gen.generate_from_matrix(omega, m_0.get_matrix, structure='group')

        print('Svf and disp ground truth created:')
        print(type(svf_0))
        print(type(sdisp_0))

        for interp_method_i in range(len(interp_methods)):
            for integrator_i in range(len(integrators)):
                for max_step_i in range(len(max_steps)):

                    print('------ Beginning of a new cycle for matrix {} out of {} --------'.format(str(i+1), str(N)))
                    print('interp method     : {}'.format(str(interp_methods[interp_method_i])))
                    print('integrator method : {}'.format(str(integrators[integrator_i])))
                    print('step              : {}'.format(str(max_steps[max_step_i])))
                    print('---------------------------------------------------------------')

                    # -> compute exponential with different available methods:
                    start = time.time()

                    l_exp = LieExp()
                    sdisp_scipy = l_exp.scipy_pointwise(svf_0,
                                                        integrator=integrators[integrator_i],
                                                        method='adams',
                                                        max_steps=max_steps[max_step_i],
                                                        interpolation_method=interp_methods[interp_method_i],
                                                        verbose=verbose_exp,
                                                        passepartout=passepartout,
                                                        return_integral_curves=False)

                    operation_time = (time.time() - start)
                    error = qr.norm(sdisp_scipy - sdisp_0, passe_partout_size=passepartout)

                    print('----------  Error  ----------------------------')
                    print('|vode - disp| = {}'.format(str(error)))
                    print('----------  Computational Time  ---------------')
                    print('{} sec.'.format(str(operation_time)))
                    print('-----------------------------------------------\n')

                    # store errors and computational time in appropriate matrices:
                    # 4d matrix x:steps, y:integrators and methods, z:interpolation, t:error/computational time
                    main_errors[max_step_i, integrator_i, interp_method_i, i] = error
                    main_computational_time[max_step_i, integrator_i, interp_method_i, i] = operation_time

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
