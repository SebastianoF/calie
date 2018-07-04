import numpy as np
import copy
from scipy.misc import factorial as fact
from scipy.linalg import expm
import nibabel as nib

from scipy.integrate import ode

from src.tools.auxiliary.sanity_checks import check_is_vector_field


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

    Polyaffine scaling and squaring

    Euler method

    Midpoint method

    Euler modified

    Trapezoidal method

    Runge Kutta method

    -> These method has been rewritten externally as an external function in utils exp_svf

    :param algorithm: algorithm name
    :param s_i_o: spline interpolation order
    :param input_num_steps: num steps of the algorithm
    :param : It returns a displacement, element of the class disp.
    :param pix_dims: conversion of pixel-mm for each dimension, from matrix to mm.
    """
    d = check_is_vector_field(input_vf)

    vf = copy.deepcopy(input_vf)
    phi = copy.deepcopy(np.zeros_like(vf))

    ''' automatic computation of the optimal number of steps: '''
    if input_num_steps is None:
        norm = np.linalg.norm(vf, axis=d - 1)
        max_norm = np.max(norm[:])

        if max_norm < 0:
            raise ValueError('Maximum norm is invalid.')
        if max_norm == 0:
            return phi

        if pix_dims is None:
            num_steps = max(0, np.ceil(np.log2(max_norm / 0.5)).astype('int')) + 3
        else:
            min_size = np.min(pix_dims[pix_dims > 0])
            num_steps = max(0, np.ceil(np.log2(max_norm / (min_size / 2))).astype('int')) + 3
    else:
        num_steps = input_num_steps

    ''' Collection of numerical method: '''

    # scaling and squaring:
    if algorithm == 'ss':

        # (1)
        init = 1 << num_steps  # equivalent to 1 * pow(2, num_steps)
        phi = vf / init




    # create and test the composition in basic_vector_fields
    # Never use displacemen or deformation. Use only diffeomorphisms.
    # Lagrangian coordinates or Euclidean Coordinates

    # REFACTOR COMPOSITION!!







        # (2)
        for _ in range(0, num_steps):
            phi = SDISP.composition(phi, phi, s_i_o=s_i_o)

    # Scaling and squaring exponential integrators
    elif algorithm == 'gss_ei':

        # (1)
        init = 1 << num_steps
        phi.field = v.field / init

        # (1.5)
        jv = SDISP.compute_jacobian(phi)

        if v.dim == 2:

            v_matrix = np.array([0.0] * 3 * 3).reshape([3, 3])

            for x in range(phi.field.shape[0]):
                for y in range(phi.field.shape[1]):
                    # skew symmetric part
                    v_matrix[0:2, 0:2] = jv.field[x, y, 0, 0, :].reshape([2, 2])
                    # translation part
                    v_matrix[0, 2], v_matrix[1, 2] = phi.field[x, y, 0, 0, 0:2] #+ \
                                                     #jv.field[x, y, 0, 0, :].reshape([2, 2]).dot([x, y])

                    # translational part of the exp is the answer:
                    phi.field[x, y, 0, 0, :] = expm(v_matrix)[0:2, 2]

        elif v.dim == 3:

            v_matrix = np.array([0.0] * 4 * 4).reshape([4, 4])

            for x in range(phi.field.shape[0]):
                for y in range(phi.field.shape[1]):
                    for z in range(phi.field.shape[2]):

                        # skew symmetric part
                        v_matrix[0:3, 0:3] = jv.field[x, y, z, 0, :].reshape([3, 3])

                        # translation part
                        v_matrix[0, 3], v_matrix[1, 3], v_matrix[2, 3] = phi.field[x, y, z, 0, 0:3]

                        phi.field[x, y, z, 0, :] = expm(v_matrix)[0:3, 3]

        else:
            raise TypeError("Problem in the number of dimensions!")

        # (2)
        for _ in range(0, num_steps):
            phi = SDISP.composition(phi, phi, s_i_o=s_i_o)

    # Affine scaling and squaring exponential integrators modified.
    elif algorithm == 'gss_ei_mod':

        # (1) copy the reduced v in phi, future solution of the ODE.
        init = 1 << num_steps
        phi.field = v.field / init

        # (1.5)
        jv = SDISP.compute_jacobian(phi)

        if v.dim == 2:

            for x in range(phi.shape[0]):
                for y in range(phi.shape[1]):

                    j = jv.field[x, y, 0, 0, :].reshape([2, 2])
                    tr = phi.field[x, y, 0, 0, 0:2]
                    j_tr = j.dot(tr)
                    phi.field[x, y, 0, 0, :] = tr + 0.5 * j_tr  # + 1/6. * J.dot(J_tr)

        elif v.dim == 3:

            for x in range(phi.field.shape[0]):
                for y in range(phi.field.shape[1]):
                    for z in range(phi.field.shape[2]):

                        j = jv.field[x, y, z, 0, :].reshape([3, 3])
                        tr = phi.field[x, y, z, 0, 0:3]
                        j_tr = j.dot(tr)
                        phi.field[x, y, z, 0, :] = tr + 0.5 * j_tr  # + 1/6. * j.dot(j_tr)

        else:
            raise TypeError("Problem in dimensions number!")

        # (2)
        for _ in range(0, num_steps):
            phi = SDISP.composition(phi, phi, s_i_o=s_i_o)

    # scaling and squaring approximated exponential integrators
    elif algorithm == 'gss_aei':

        # (1)
        if num_steps == 0:
            phi.field = v.field
        else:
            init = 1 << num_steps
            phi.field = v.field / init

        # (1.5)  phi = 1 + v + 0.5jac*v
        jv = np.squeeze(SDISP.compute_jacobian(phi).field)
        v_sq = np.squeeze(phi.field)
        jv_prod_v = matrix_vector_field_product(jv, v_sq).reshape(list(v.vol_ext) + [1]*(4-v.dim) + [v.dim])

        phi.field += 0.5*jv_prod_v

        # (2)
        for _ in range(0, num_steps):
            phi = SDISP.composition(phi, phi, s_i_o=s_i_o)

    elif algorithm == 'midpoint':

        if input_num_steps is None:
            num_steps = 10
        else:
            num_steps = input_num_steps

        if num_steps == 0:
            h = 1.0
        else:
            h = 1.0 / num_steps
        for i in range(num_steps):

            phi_pos = SDISP.deformation_from_displacement(phi)
            phi_tilda = SDISP.from_array(phi.field)
            phi_tilda.field = phi.field + (h / 2) * SDISP.compose_with_deformation_field(v, phi_pos,
                                                                                         s_i_o=s_i_o).field

            phi_tilda_pos = SDISP.deformation_from_displacement(phi_tilda)

            phi.field += h * SDISP.compose_with_deformation_field(v, phi_tilda_pos, s_i_o=s_i_o).field

    # Series method
    elif algorithm == 'series':

        # Automatic step selector:
        if input_num_steps is None:
            norm = np.linalg.norm(v.field, axis=v.field.ndim - 1)
            max_norm = np.max(norm[:])
            toll = 1e-3
            k = 10
            while max_norm / fact(k) >  toll:
                k += 1
            num_steps = k
            print 'automatic steps selector for series method: ' + str(k)

        else:
            num_steps = input_num_steps

        phi.field = v.field[...]  # final output is phi.

        for k in range(2, num_steps):
            jac_v = SVF.iterative_jacobian_product(v, k)
            phi.field = phi.field[...] + jac_v.field[...] / fact(k)

    # Series method
    elif algorithm == 'series_mod':  # jacobian computed in the improper way

        jac_v = copy.deepcopy(v)
        phi.field = v.field[...]  # final output is phi.

        for k in range(1, input_num_steps):
            jac_v = SVF.jacobian_product(jac_v, v)
            phi.field = phi.field[...] + jac_v.field[...] / fact(k)

    # Euler method
    elif algorithm == 'euler':

        if input_num_steps is None:
            num_steps = 10
        else:
            num_steps = input_num_steps
        if num_steps == 0:
            h = 1.0
        else:
            h = 1.0 / num_steps
        for i in range(num_steps):
            phi_def = SDISP.deformation_from_displacement(phi)
            phi.field += h * SDISP.compose_with_deformation_field(v, phi_def, s_i_o=s_i_o).field

    # Euler approximated exponential integrator
    elif algorithm == 'euler_aei':

        v.field = v.field / num_steps

        jv = np.squeeze(SDISP.compute_jacobian(v).field)
        v_sq = np.squeeze(v.field)
        jv_prod_v = matrix_vector_field_product(jv, v_sq).reshape(list(v.vol_ext) + [1]*(4-v.dim) + [v.dim])

        v.field += 0.5*jv_prod_v

        for _ in range(num_steps):
            phi = SDISP.composition(v, phi, s_i_o=s_i_o)

    # Euler modified
    elif algorithm == 'euler_mod':

        if input_num_steps is None:
            num_steps = 10
        else:
            num_steps = input_num_steps

        if num_steps == 0:
            h = 1.0
        else:
            h = 1.0 / num_steps

        for i in range(num_steps):
            # Code can be optimized if we allow vector operations for displacement field, but not deformation!

            phi_def = SDISP.deformation_from_displacement(phi)

            psi_1 = SDISP.from_array(phi.field)
            psi_2 = SDISP.from_array(phi.field)

            psi_1.field = phi.field + h * SDISP.compose_with_deformation_field(v, phi_def, s_i_o=s_i_o).field
            psi_1_def   = SDISP.deformation_from_displacement(psi_1)

            psi_2.field = SDISP.compose_with_deformation_field(v, phi_def, s_i_o=s_i_o).field + \
                    SDISP.compose_with_deformation_field(v, psi_1_def, s_i_o=s_i_o).field

            phi.field += (h/2) * psi_2.field

    # Heun
    elif algorithm == 'heun':

        if input_num_steps is None:
            num_steps = 10
        else:
            num_steps = input_num_steps
        if num_steps == 0:
            h = 1.0
        else:
           h = 1.0 / num_steps

        for i in range(num_steps):
            phi_def = SDISP.deformation_from_displacement(phi)

            psi_1 = SDISP.from_array(phi.field)
            psi_2 = SDISP.from_array(phi.field)

            psi_1.field = phi.field + h * (2. / 3) * SDISP.compose_with_deformation_field(v, phi_def,
                                                                                          s_i_o=s_i_o).field
            psi_1_def   = SDISP.deformation_from_displacement(psi_1)

            psi_2.field = SDISP.compose_with_deformation_field(v, phi_def, s_i_o=s_i_o).field + \
                          3 * SDISP.compose_with_deformation_field(v, psi_1_def, s_i_o=s_i_o).field

            phi.field += (h / 4) * psi_2.field

    # Heun modified
    elif algorithm == 'heun_mod':

        if input_num_steps is None:
            num_steps = 10
        else:
            num_steps = input_num_steps

        if num_steps == 0:
            h = 1.0
        else:
            h = 1.0 / num_steps
        for i in range(num_steps):
            phi_def = SDISP.deformation_from_displacement(phi)

            psi_1 = SDISP.from_array(phi.field)
            psi_2 = SDISP.from_array(phi.field)
            psi_3 = SDISP.from_array(phi.field)

            psi_1.field = phi.field + (h / 3) * SDISP.compose_with_deformation_field(v, phi_def, s_i_o=s_i_o).field
            psi_1_def   = SDISP.deformation_from_displacement(psi_1)

            psi_2.field = phi.field + h * (2. / 3) * SDISP.compose_with_deformation_field(v, psi_1_def,
                                                                                          s_i_o=s_i_o).field
            psi_2_def   = SDISP.deformation_from_displacement(psi_2)

            psi_3.field = SDISP.compose_with_deformation_field(v, phi_def, s_i_o=s_i_o).field + \
                          3 * SDISP.compose_with_deformation_field(v, psi_2_def, s_i_o=s_i_o).field

            phi.field += (h / 4) * psi_3.field

    # Runge Kutta 4
    elif algorithm == 'rk4':

        if input_num_steps is None:
            num_steps = 10
        else:
            num_steps = input_num_steps

        if num_steps == 0:
            h = 1.0
        else:
            h = 1.0 / num_steps

        for i in range(num_steps):

            phi_def = SDISP.deformation_from_displacement(phi)

            r_1 = SDISP.from_array(phi.field)
            r_2 = SDISP.from_array(phi.field)
            r_3 = SDISP.from_array(phi.field)
            r_4 = SDISP.from_array(phi.field)

            psi_1 = SDISP.from_array(phi.field)
            psi_2 = SDISP.from_array(phi.field)
            psi_3 = SDISP.from_array(phi.field)

            r_1.field   = h * SDISP.compose_with_deformation_field(v, phi_def, s_i_o=s_i_o).field

            psi_1.field = phi.field + .5 * r_1.field
            psi_1_def  = SDISP.deformation_from_displacement(psi_1)
            r_2.field   = h  * SDISP.compose_with_deformation_field(v, psi_1_def, s_i_o=s_i_o).field

            psi_2.field = phi.field + .5 * r_2.field
            psi_2_def  = SDISP.deformation_from_displacement(psi_2)
            r_3.field   = h  * SDISP.compose_with_deformation_field(v, psi_2_def, s_i_o=s_i_o).field

            psi_3.field = phi.field + r_3.field
            psi_3_def  = SDISP.deformation_from_displacement(psi_3)
            r_4.field = h  * SDISP.compose_with_deformation_field(v, psi_3_def, s_i_o=s_i_o).field

            phi.field += (1. / 6) * (r_1.field + 2 * r_2.field + 2 * r_3.field + r_4.field)

    # Generalized scaling and squaring with runge kutta
    elif algorithm == 'gss_rk4':

        norm = np.linalg.norm(v.field, axis=v.field.ndim - 1)
        max_norm = np.max(norm[:])

        if max_norm < 0:
            raise ValueError('Maximum norm is invalid.')
        if max_norm == 0:
            return phi

        if input_num_steps is None:
            # automatic computation of the optimal number of steps:
            pix_dims = np.asarray(v.zooms)
            min_size = np.min(pix_dims[pix_dims > 0])
            num_steps = max(0, np.ceil(np.log2(max_norm / (min_size / 2))).astype('int')) + 2  # adaptative method.
        else:
            num_steps = input_num_steps

        # (1)
        init = 1 << num_steps  # equivalent to 1 * pow(2, num_steps)
        v.field = v.field / init  # LET IT LIKE THAT! No augmented assignment!!

        # rk steps:
        input_num_steps_rk4 = 7
        h = 1.0 / input_num_steps_rk4

        for i in range(input_num_steps_rk4):

            phi_def = SDISP.deformation_from_displacement(phi)

            r_1 = SDISP.from_array(phi.field)
            r_2 = SDISP.from_array(phi.field)
            r_3 = SDISP.from_array(phi.field)
            r_4 = SDISP.from_array(phi.field)

            psi_1 = SDISP.from_array(phi.field)
            psi_2 = SDISP.from_array(phi.field)
            psi_3 = SDISP.from_array(phi.field)

            r_1.field   = h * SDISP.compose_with_deformation_field(v, phi_def, s_i_o=s_i_o).field

            psi_1.field = phi.field + .5 * r_1.field
            psi_1_def  = SDISP.deformation_from_displacement(psi_1)
            r_2.field   = h  * SDISP.compose_with_deformation_field(v, psi_1_def, s_i_o=s_i_o).field

            psi_2.field = phi.field + .5 * r_2.field
            psi_2_def  = SDISP.deformation_from_displacement(psi_2)
            r_3.field   = h  * SDISP.compose_with_deformation_field(v, psi_2_def, s_i_o=s_i_o).field

            psi_3.field = phi.field + r_3.field
            psi_3_def  = SDISP.deformation_from_displacement(psi_3)
            r_4.field = h  * SDISP.compose_with_deformation_field(v, psi_3_def, s_i_o=s_i_o).field

            phi.field += (1. / 6) * (r_1.field + 2 * r_2.field + 2 * r_3.field + r_4.field)

        # (2)
        for _ in range(num_steps):
            phi = SDISP.composition(phi, phi, s_i_o=s_i_o)

    else:
        raise TypeError('Error: wrong algorithm name. You inserted ' + algorithm)

    return phi


