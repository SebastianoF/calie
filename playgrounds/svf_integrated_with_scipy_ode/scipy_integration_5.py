"""
Integral vector field is computed out of the integral curves.
Structure of the bi-points shown in the previous module is here stored in a new vector field.
"""
import copy

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import ode

from VECtorsToolkit.tools.fields.composition import one_point_interpolation
from VECtorsToolkit.tools.fields.generate_identities import vf_identity_eulerian_like
from VECtorsToolkit.tools.fields.generate_vf import generate_random
from VECtorsToolkit.tools.local_operations.lie_exponential import lie_exponential
from VECtorsToolkit.tools.visualisations.fields.fields_comparisons import see_2_fields_separate_and_overlay


# Auxiliary vector fields function


def vf(t, x):
    global svf_0
    return list(one_point_interpolation(svf_0, point=x, method='cubic'))

if __name__ == '__main__':

    # Initialize the random field with the function input:
    svf_0  = generate_random(omega=(20, 20), parameters=(4, 2))

    # Initialize the displacement field that will be computed using the integral curves.
    disp_0 = vf_identity_eulerian_like(svf_0)

    t0, t1 = 0, 1
    steps = 10.
    dt = (t1 - t0) / steps

    r = ode(vf).set_integrator('dopri5', method='bdf', max_step=dt)

    # Start getting the figure:

    fig = plt.figure(num=1)
    ax = fig.add_subplot(111)

    # Plot vector field
    id_field = vf_identity_eulerian_like(svf_0)

    input_field_copy = copy.deepcopy(svf_0)

    ax.quiver(id_field[..., 0, 0, 0],
               id_field[..., 0, 0, 1],
               input_field_copy[..., 0, 0, 0],
               input_field_copy[..., 0, 0, 1], color='r', alpha=0.9,
               linewidths=0.01, width=0.05, scale=1, scale_units='xy', units='xy', angles='xy')

    print('Beginning of the integral curves computations')

    # Plot integral curves

    passepartout = 4

    for i in range(0 + passepartout, disp_0.shape[0] - passepartout + 1):
        for j in range(0 + passepartout, disp_0.shape[0] - passepartout + 1):  # cycle on the point of the grid.

            Y, T = [], []
            r.set_initial_value([i, j], t0).set_f_params()  # initial conditions are point on the grid
            print('Integrating vf at the point {} between {} and {}, step size {}'.format((i, j), t0, t1, dt))
            while r.successful() and r.t + dt < t1:
                r.integrate(r.t+dt)
                Y.append(r.y)

            S = np.array(np.real(Y))

            disp_0[i, j, 0, 0, :] = S[S.shape[0]-1, :]

            ax.plot(S[:, 0], S[:, 1], color='b', lw=1)

            ax.plot(i, j, 'go', alpha=0.5)
            ax.plot(S[S.shape[0]-1, 0], S[S.shape[0]-1, 1], 'bo', alpha=0.5)
            if S.shape[0] < steps-2:  # the first step is not directly considered
                print("--------")
                print("Warning!")  # steps jumped for the point
                print("--------")

    print('End of the integral curves computations')

    plt.xlim([0, 20])
    plt.ylim([0, 20])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.grid()

    see_2_fields_separate_and_overlay(disp_0, svf_0,
                                      fig_tag=2,
                                      title_input_0='disp',
                                      title_input_1='svf',
                                      title_input_both='overlay')

    # Initialize the displacement field that will be computed using the integral curves.
    disp_1 = lie_exponential(svf_0)
    
    see_2_fields_separate_and_overlay(disp_1, svf_0,
                                      fig_tag=3,
                                      title_input_0='disp',
                                      title_input_1='svf',
                                      title_input_both='overlay',
                                      window_title_input='embedded')

    print('Showing outcome: ...')
    plt.show()
