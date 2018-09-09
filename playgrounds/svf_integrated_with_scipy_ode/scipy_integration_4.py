"""
Integral vector field from integral curves.
Emphasis on the initial and on the final point of the integral curve in transparency.
Initial and final points of each integral curve at each point of the grid is called bi-point.
"""
import numpy as np
import matplotlib.pyplot as plt
import copy

from scipy.integrate import ode

from VECtorsToolkit.tools.fields.generate_identities import vf_identity_eulerian_like
from VECtorsToolkit.tools.fields.generate_vf import generate_random
from VECtorsToolkit.tools.fields.composition import one_point_interpolation


# Auxiliary vector field function:


def vf(t, x):
    global field_0
    return list(one_point_interpolation(field_0, point=x, method='cubic'))

if __name__ == '__main__':

    # Initialize the field with the function input:
    field_0 = generate_random(omega=(20, 20), parameters=(2, 2))

    t0, t1 = 0, 1
    steps = 20.
    dt = (t1 - t0) / steps
    ic = [[i, j] for i in range(2, 18) for j in range(2, 18)]
    colors = ['b'] * len(ic)

    r = ode(vf).set_integrator('vode', method='bdf', max_step=dt)

    fig = plt.figure(num=1)
    ax = fig.add_subplot(111)

    # Plot vector field
    id_field = vf_identity_eulerian_like(field_0)

    input_field_copy = copy.deepcopy(field_0)

    ax.quiver(id_field[..., 0, 0, 0],
               id_field[..., 0, 0, 1],
               input_field_copy[..., 0, 0, 0],
               input_field_copy[..., 0, 0, 1], color='r', alpha=0.9,
               linewidths=0.01, width=0.05, scale=1, scale_units='xy', units='xy', angles='xy', )

    print('Beginning of the integral curves computations')

    # Plot integral curves
    for k in range(len(ic)):
        Y, T = [], []
        r.set_initial_value(ic[k], t0).set_f_params()
        while r.successful() and r.t + dt < t1:
            r.integrate(r.t+dt)
            Y.append(r.y)

        S = np.array(np.real(Y))
        ax.plot(S[:, 0], S[:, 1], color=colors[k], lw=1)
        print('final point of ' + str(ic[k]) + ':')
        print(ic[k])
        print('size of S : ' + str(S.shape[0]))
        # ic and S are the searched bi-points.
        ax.plot(ic[k][0], ic[k][1], 'go', alpha=0.5)
        ax.plot(S[S.shape[0]-1, 0], S[S.shape[0]-1, 1], 'bo', alpha=0.5)
        if S.shape[0] < steps-2:
            print("Warning!")  # steps jumped for the point
            print("--------")

    print('End of the integral curves computations')

    plt.xlim([0, 20])
    plt.ylim([0, 20])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.grid()
    plt.show()
