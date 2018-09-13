import copy

import numpy as np
from scipy.linalg import expm
from scipy.misc import factorial as fact
from scipy import integrate


from VECtorsToolkit.tools.fields.queries import check_is_vf, get_omega_from_vf
from VECtorsToolkit.tools.fields.composition import one_point_interpolation, \
    lagrangian_dot_lagrangian
from VECtorsToolkit.tools.local_operations.jacobians import compute_jacobian, iterative_jacobian_product, \
    jacobian_product
from VECtorsToolkit.tools.auxiliary.matrices import matrix_vector_field_product


def lie_exponential(input_vf, algorithm='ss', s_i_o=3, input_num_steps=None, pix_dims=None):
    """
    Compute the exponential of this velocity field using the
    scaling and squaring approach.

    GIGO, SIRO: we assume that the input vector field is in the tangent space.
    This code design allows to compute twice the exponential of the same field, even if not  formally correct
    from a theoretical point of view.

    Scaling and squaring:
    (1) -- Scaling step:
    divides data time .

    (2) -- Squaring step:
    Do the squaring step to perform the integration
    The exponential is num_steps times recursive composition of
    the field with itself, which is equivalent to integration over
    the unit interval.

    Generalised scaling and squaring

    Euler method

    Midpoint method

    Euler modified

    Trapezoidal method

    Runge Kutta method

    -> These method has been rewritten externally as an external function in utils exp_svf
    :param input_vf: input vector field. Must be an svf.
    :param algorithm: algorithm name
    :param s_i_o: spline interpolation order
    :param input_num_steps: num steps of the algorithm
    :param : It returns a displacement, element of the class disp.
    :param pix_dims: conversion of pixel-mm for each dimension, from matrix to mm.
    """
    d = check_is_vf(input_vf)
    omega = get_omega_from_vf(input_vf)

    vf = copy.deepcopy(input_vf)
    phi = np.zeros_like(vf)

    ''' automatic computation of the optimal number of steps: '''
    if input_num_steps is None:
        norm = np.linalg.norm(vf, axis=d - 1)
        max_norm = np.max(norm[:])

        if max_norm < 0:
            raise ValueError('Maximum norm is invalid.')
        if max_norm == 0:
            return phi

        if pix_dims is None:
            num_steps = max([0, np.ceil(np.log2(max_norm / 0.5)).astype('int')]) + 3
        else:
            min_size = np.min(pix_dims[pix_dims > 0])
            num_steps = max([0, np.ceil(np.log2(max_norm / (min_size / 2))).astype('int')]) + 3

    # Automatic step selector:
    elif input_num_steps is 'test_method':
        norm_vf = np.linalg.norm(vf, axis=vf.ndim - 1)
        max_norm = np.max(norm_vf)
        toll = 1e-3
        k = 10
        while max_norm / fact(k) > toll:
            k += 1
        num_steps = k
        print('automatic steps selector for series method: ' + str(k))

    else:
        num_steps = input_num_steps

    ''' Collection of numerical method: '''

    if algorithm == 'ss':  # scaling and squaring:

        # (1)
        init = 1 << num_steps  # equivalent to 1 * pow(2, num_steps)
        phi = vf / float(init)

        # (2)
        for _ in range(0, num_steps):
            phi = lagrangian_dot_lagrangian(phi, phi, s_i_o=s_i_o)

    elif algorithm == 'gss_ei':  # Scaling and squaring exponential integrators

        # (1)
        init = 1 << num_steps
        phi = vf / init

        # (1.5)
        jv = compute_jacobian(phi)

        if d == 2:

            v_matrix = np.array([0.0] * 3 * 3).reshape([3, 3])

            for x in range(phi.shape[0]):
                for y in range(phi.shape[1]):
                    # skew symmetric part
                    v_matrix[0:2, 0:2] = jv[x, y, 0, 0, :].reshape([2, 2])
                    # translational part
                    v_matrix[0, 2], v_matrix[1, 2] = phi[x, y, 0, 0, 0:2]  # + \
                    # jv[x, y, 0, 0, :].reshape([2, 2]).dot([x, y])

                    # translational part of the exp is the answer:
                    phi[x, y, 0, 0, :] = expm(v_matrix)[0:2, 2]

        elif d == 3:

            v_matrix = np.array([0.0] * 4 * 4).reshape([4, 4])

            for x in range(phi.shape[0]):
                for y in range(phi.shape[1]):
                    for z in range(phi.shape[2]):

                        # skew symmetric part
                        v_matrix[0:3, 0:3] = jv[x, y, z, 0, :].reshape([3, 3])

                        # translation part
                        v_matrix[0, 3], v_matrix[1, 3], v_matrix[2, 3] = phi[x, y, z, 0, 0:3]

                        phi[x, y, z, 0, :] = expm(v_matrix)[0:3, 3]

        # (2)
        for _ in range(0, num_steps):
            phi = lagrangian_dot_lagrangian(phi, phi, s_i_o=s_i_o)

    elif algorithm == 'gss_ei_mod':  # Affine scaling and squaring exponential integrators modified

        # (1) copy the reduced v in phi, future solution of the ODE.
        init = 1 << num_steps
        phi = vf / init

        # (1.5)
        jv = compute_jacobian(phi)

        if d == 2:

            for x in range(phi.shape[0]):
                for y in range(phi.shape[1]):

                    j = jv[x, y, 0, 0, :].reshape([2, 2])
                    tr = phi[x, y, 0, 0, 0:2]
                    j_tr = j.dot(tr)
                    phi[x, y, 0, 0, :] = tr + 0.5 * j_tr  # + 1/6. * J.dot(J_tr)

        elif d == 3:

            for x in range(phi.shape[0]):
                for y in range(phi.shape[1]):
                    for z in range(phi.shape[2]):

                        j = jv[x, y, z, 0, :].reshape([3, 3])
                        tr = phi[x, y, z, 0, 0:3]
                        j_tr = j.dot(tr)
                        phi[x, y, z, 0, :] = tr + 0.5 * j_tr  # + 1/6. * j.dot(j_tr)

        # (2)
        for _ in range(0, num_steps):
            phi = lagrangian_dot_lagrangian(phi, phi, s_i_o=s_i_o)

    elif algorithm == 'gss_aei':  # scaling and squaring approximated exponential integrators

        # (1)
        if num_steps == 0:
            phi = vf
        else:
            init = 1 << num_steps
            phi = vf / init

        # (1.5)  phi = 1 + v + 0.5jac*v
        jv = np.squeeze(compute_jacobian(phi))
        v_sq = np.squeeze(phi)
        jv_prod_v = matrix_vector_field_product(jv, v_sq).reshape(list(omega) + [1]*(4 - d) + [d])

        phi += 0.5 * jv_prod_v

        # (2)
        for _ in range(0, num_steps):
            phi = lagrangian_dot_lagrangian(phi, phi, s_i_o=s_i_o)

    elif algorithm == 'midpoint':

        if num_steps == 0:
            h = 1.0
        else:
            h = 1.0 / num_steps

        for _ in range(num_steps):
            phi_tilda = phi + (h / 2) * lagrangian_dot_lagrangian(vf, phi, s_i_o=s_i_o, add_right=False)
            phi += h * lagrangian_dot_lagrangian(vf, phi_tilda, s_i_o=s_i_o, add_right=False)

    elif algorithm == 'series':  # Series method

        phi = np.copy(vf)  # final output is phi.

        for k in range(2, num_steps):
            jac_v = iterative_jacobian_product(vf, k)
            phi = phi[...] + jac_v[...] / fact(k)

    elif algorithm == 'series_mod':  # Series method  -- jacobian computed in the improper way

        jac_v = copy.deepcopy(vf)
        phi = np.copy(vf)  # final output is phi.

        for k in range(1, input_num_steps):
            jac_v = jacobian_product(jac_v, vf)
            phi = phi[...] + jac_v[...] / fact(k)

    elif algorithm == 'euler':  # Euler method

        if num_steps == 0:
            h = 1.0
        else:
            h = 1.0 / num_steps

        for _ in range(num_steps):
            phi += h * lagrangian_dot_lagrangian(vf, phi, s_i_o=s_i_o, add_right=False)

    elif algorithm == 'euler_aei':  # Euler approximated exponential integrator

        vf = vf / num_steps

        jv = np.squeeze(compute_jacobian(vf))
        v_sq = np.squeeze(vf)
        jv_prod_v = matrix_vector_field_product(jv, v_sq).reshape(list(omega) + [1]*(4 - d) + [d])

        vf += 0.5 * jv_prod_v

        for _ in range(num_steps):
            phi = lagrangian_dot_lagrangian(vf, phi, s_i_o=s_i_o)

    elif algorithm == 'euler_mod':  # Euler modified

        if num_steps == 0:
            h = 1.0
        else:
            h = 1.0 / num_steps

        for _ in range(num_steps):

            phi_tilda = phi + h * lagrangian_dot_lagrangian(vf, phi, s_i_o=s_i_o, add_right=False)

            phi += (h/2) * (lagrangian_dot_lagrangian(vf, phi, s_i_o=s_i_o, add_right=False) +
                            lagrangian_dot_lagrangian(vf, phi_tilda, s_i_o=s_i_o, add_right=False))

    elif algorithm == 'heun':  # Heun method

        if num_steps == 0:
            h = 1.0
        else:
            h = 1.0 / num_steps

        for i in range(num_steps):

            psi_1 = phi + h * (2. / 3) * lagrangian_dot_lagrangian(vf, phi, s_i_o=s_i_o, add_right=False)

            psi_2 = lagrangian_dot_lagrangian(vf, phi, s_i_o=s_i_o, add_right=False) + \
                                              3 * lagrangian_dot_lagrangian(vf, psi_1, s_i_o=s_i_o, add_right=False)

            phi += (h / 4) * psi_2

    elif algorithm == 'heun_mod':  # Heun modified

        if num_steps == 0:
            h = 1.0
        else:
            h = 1.0 / num_steps
        for i in range(num_steps):

            psi_1 = phi + (h / 3) * lagrangian_dot_lagrangian(vf, phi, s_i_o=s_i_o, add_right=False)

            psi_2 = phi + h * (2. / 3) * lagrangian_dot_lagrangian(vf, psi_1, s_i_o=s_i_o, add_right=False)

            psi_3 = lagrangian_dot_lagrangian(vf, phi, s_i_o=s_i_o, add_right=False) + \
                                              3 * lagrangian_dot_lagrangian(vf, psi_2, s_i_o=s_i_o, add_right=False)

            phi += (h / 4) * psi_3

    elif algorithm == 'rk4':  # Runge Kutta 4

        if num_steps == 0:
            h = 1.0
        else:
            h = 1.0 / num_steps

        for _ in range(num_steps):

            r_1 = h * lagrangian_dot_lagrangian(vf, phi, s_i_o=s_i_o, add_right=False)
            psi_1 = phi + .5 * r_1

            r_2 = h  * lagrangian_dot_lagrangian(vf, psi_1, s_i_o=s_i_o, add_right=False)

            psi_2 = phi + .5 * r_2
            r_3 = h * lagrangian_dot_lagrangian(vf, psi_2, s_i_o=s_i_o, add_right=False)

            psi_3 = phi + r_3
            r_4 = h  * lagrangian_dot_lagrangian(vf, psi_3, s_i_o=s_i_o, add_right=False)

            phi += (1. / 6) * (r_1 + 2 * r_2 + 2 * r_3 + r_4)

    elif algorithm == 'gss_rk4':  # Generalized scaling and squaring with runge kutta

        norm = np.linalg.norm(vf, axis=vf.ndim - 1)
        max_norm = np.max(norm[:])

        if max_norm < 0:
            raise ValueError('Maximum norm is invalid.')
        if max_norm == 0:
            return phi

        if num_steps == 0:
            h = 1.0
        else:
            h = 1.0 / num_steps

        # (1)
        init = 1 << num_steps
        vf = vf / init

        for _ in range(num_steps):

            r_1   = h * lagrangian_dot_lagrangian(vf, phi, s_i_o=s_i_o, add_right=False)

            psi_1 = phi + .5 * r_1
            r_2   = h  * lagrangian_dot_lagrangian(vf, psi_1, s_i_o=s_i_o, add_right=False)

            psi_2 = phi + .5 * r_2
            r_3   = h  * lagrangian_dot_lagrangian(vf, psi_2, s_i_o=s_i_o, add_right=False)

            psi_3 = phi + r_3
            r_4 = h  * lagrangian_dot_lagrangian(vf, psi_3, s_i_o=s_i_o, add_right=False)

            phi += (1. / 6) * (r_1 + 2 * r_2 + 2 * r_3 + r_4)

        # (2)
        for _ in range(num_steps):
            phi = lagrangian_dot_lagrangian(phi, phi, s_i_o=s_i_o, add_right=True)

    else:
        raise TypeError('Error: wrong algorithm name. You inserted {}'.format(algorithm))

    return phi


def lie_exponential_scipy(input_vf,
                          integrator='vode',
                          method='bdf',
                          max_steps=7,
                          interpolation_method='cubic',
                          passepartout=3,
                          return_integral_curves=False,
                          verbose=False):
    """
    Compute the exponential of this velocity field using scipy libraries.
    :param input_vf:
    :param integrator: vode, zvode, lsoda, dopri5, dopri853, see scipy documentations:
        http://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.integrate.ode.html
    :param method: adams (non-stiff) or bdf (stiff)
    :param interpolation_method:
    :param max_steps: maximum steps of integrations. If less steps are required, it prints a warning message.
    :param passepartout:
    :param return_integral_curves:
    :param verbose:
    :param : It returns a displacement phi, element of the class disp, as
                phi(x) = psi(x) - x
            where psi is the chosen integrator.

    """
    d = check_is_vf(input_vf)
    assert d == 2  # only for 2 dim images at the moment.

    omega = get_omega_from_vf(input_vf)

    vf = copy.deepcopy(input_vf)
    phi = np.zeros_like(vf)

    flows_collector = []

    # transform v in a function suitable for ode library :
    def vf_function(t, x):
        return list(one_point_interpolation(vf, point=x, method=interpolation_method))

    t0, t_n = 0, 1
    dt = (t_n - t0)/float(max_steps)

    r = integrate.ode(vf_function).set_integrator(integrator, method=method, max_step=max_steps)

    for i in range(0 + passepartout, omega[0] - passepartout + 1):
        for j in range(0 + passepartout, omega[1] - passepartout + 1):  # cycle on the point of the grid.

            y = []
            r.set_initial_value([i, j], t0).set_f_params()  # initial conditions are point on the grid
            while r.successful() and r.t + dt < t_n:
                r.integrate(r.t+dt)
                y.append(r.y)

            # flow of the svf at the point [i,j]
            fl = np.array(np.real(y))

            # subtract id on the run to return a displacement.
            if fl.shape[0] > 0:
                phi[i, j, 0, 0, :] = fl[fl.shape[0]-1, :] - np.array([i, j])
            else:
                phi[i, j, 0, 0, :] = - np.array([i, j])

            if verbose:
                print('Integral curve at grid point ' + str([i, j]) + ' is computed.')

            # In some cases as critical points, or when too closed to the closure of the domain
            # the number of steps can be reduced by the algorithm.
            if fl.shape[0] < max_steps - 2 and verbose:  # the first step is not directly considered
                print("--------")
                print("Warning!") # steps jumped for the point
                print("--------")

            if return_integral_curves:
                flows_collector += [fl]

    if return_integral_curves:
        return phi, flows_collector
    else:
        return phi
