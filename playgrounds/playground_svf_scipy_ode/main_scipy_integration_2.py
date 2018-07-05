"""
Integral curve starting from 5 different points, dropped into a random generated vector field.
"""

import numpy as np
import matplotlib.pyplot as plt
import copy

from scipy.integrate import ode

from VECtorsToolkit.tools.fields.generate_identities import vf_identity_lagrangian_like
from VECtorsToolkit.tools.fields.generate_vf import generate_random
from VECtorsToolkit.tools.fields.resampling import one_point_interpolation


# Auxiliary vector field function


def vf(t, x):
    global field_0
    return list(one_point_interpolation(field_0, point=x, method='cubic'))


if __name__ == '__main__':

    # Initialize the field with the function input:
    field_0 = generate_random(omega=(20, 20), parameters=(2, 2))

    t0, t1, dt = 0, 20., 0.1
    ic = [[10, 4], [10, 7], [10, 10], [5, 7], [5, 10]]
    colors = ['r', 'b', 'g', 'm', 'c']

    r = ode(vf).set_integrator('vode', method='bdf', max_step=dt)

    fig = plt.figure(num=1)
    ax = fig.add_subplot(111)

    # Plot vector field
    id_field = vf_identity_lagrangian_like(field_0)

    input_field_copy = copy.deepcopy(field_0)

    ax.quiver(id_field[..., 0, 0, 0],
              id_field[..., 0, 0, 1],
              input_field_copy[..., 0, 0, 0],
              input_field_copy[..., 0, 0, 1],
              linewidths=0.01, width=0.03, scale=1, scale_units='xy', units='xy', angles='xy', )

    print('Beginning of the integral curves computations')

    # Plot integral curves
    for k in range(len(ic)):
        print('Integrating vf at the point {} between {} and {}, step size {}'.format(tuple(ic[k]), t0, t1, dt))
        Y, T = [], []
        r.set_initial_value(ic[k], t0).set_f_params()
        while r.successful() and r.t + dt < t1:
            r.integrate(r.t + dt)
            Y.append(r.y)

        S = np.array(np.real(Y))
        ax.plot(S[:, 0], S[:, 1], color=colors[k], lw=1.25)

    print('End of the integral curves computations')

    plt.xlim([0, 20])
    plt.ylim([0, 20])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.grid()
    print('Opening figure:')
    plt.show()
