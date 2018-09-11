import numpy as np
import scipy.ndimage.filters as fil

from VECtorsToolkit.tools.fields.queries import check_omega, vf_shape_from_omega_and_timepoints
from VECtorsToolkit.tools.fields.coordinates import vf_affine_to_homogeneous, vf_homogeneous_to_affine
from VECtorsToolkit.tools.fields.generate_identities import vf_identity_eulerian


def generate_random(omega, t=1, parameters=(5, 2)):
    """
    Return a random vector field v in Lagrangian coordinates.
    :param omega: domain of the vector field
    :param t: time not yet implemented.
    :param parameters: (sigma initial randomness, sigma of the gaussian filter).
    :return:
    """
    if t > 1:  # TODO upgrade. correct tests afterwards.
        raise IndexError('Random generator not defined (yet) for multiple time points')

    v_shape = vf_shape_from_omega_and_timepoints(omega, t)

    sigma_init, sigma_gaussian_filter = parameters
    vf = np.random.normal(0, sigma_init, v_shape)

    for i in range(v_shape[-1]):
        vf[..., 0, i] = fil.gaussian_filter(vf[..., 0, i], sigma_gaussian_filter)

    return vf


def generate_from_matrix(omega, input_matrix, t=1, structure='algebra'):
    """
    :param omega: domain of the vector field.
    :param input_matrix: matrix generating the transformation of the vector field representing elements form groups
    SE(3), SE(2) or algebras so(3) and so(2).
    :param t: timepoints.
    :param structure: can be 'algebra' or 'group'.
    :return: vector field with the given input parameters.
    """
    if t > 1:  # TODO
        raise IndexError('Random generator not defined (yet) for multiple time points')

    d = check_omega(omega)
    v_shape = vf_shape_from_omega_and_timepoints(omega, t)
    vf = np.zeros(v_shape)

    if structure == 'algebra':
        pass
    elif structure == 'group':
        input_matrix = input_matrix - np.eye(d + 1)
    else:
        raise IOError

    if d == 2:

        if not np.alltrue(input_matrix.shape == (3, 3)):
            raise IOError('Omega dimension not compatible with the matrix dimension')

        vf = vf_affine_to_homogeneous(vf_identity_eulerian(omega))

        x_intervals, y_intervals = omega
        for i in range(x_intervals):
            for j in range(y_intervals):
                vf[i, j, 0, 0, :] = input_matrix.dot(vf[i, j, 0, 0, :])

        vf = vf_homogeneous_to_affine(vf)

    elif d == 3:
        # TODO after se3_a and se3_g
        # If the matrix provides a 2d rototranslation, we consider the rotation axis perpendicular to the plane z=0.
        # this must be improved for 3d rotations in the space.
        x_intervals, y_intervals, z_intervals = omega

        if np.alltrue(input_matrix.shape == (3, 3)):

            # Create the slice at the ground of the domain (x,y,z) , z = 0, as a 2d rotation:
            base_slice = vf_affine_to_homogeneous(vf_identity_eulerian(list(omega[:2]) + [1, 1, 2]))

            for i in range(x_intervals):
                for j in range(y_intervals):
                    base_slice[i, j, 0, 0, :] = input_matrix.dot(base_slice[i, j, 0, 0, :])

            # Copy the slice at the ground on all the others:
            for k in range(z_intervals):
                vf[..., k, 0, :2] = base_slice[..., 0, 0, :2]

        # If the matrix is 2d the rotation axis is perpendicular to the plane z=0.
        elif np.alltrue(input_matrix.shape == (4, 4)):

            vf = vf_affine_to_homogeneous(vf_identity_eulerian(v_shape))

            for i in range(x_intervals):
                for j in range(y_intervals):
                    for k in range(y_intervals):
                        vf[i, j, k, 0, :] = input_matrix.dot(vf[i, j, k, 0, :])

            vf = vf_homogeneous_to_affine(vf)
        else:
            raise IOError('Wrong input matrix shape. Must be 3x3 or 4x4.')

    return vf


def generate_from_projective_matrix(omega, input_homography, t=1, structure='algebra'):
    """

    :param omega: domain of the vector field.
    :param input_homography: matrix representing an element form the homography group.
    :param t: number of timepoints.
    :param structure: can be 'algebra' or 'group'.
    :return: vector field with the given input parameters.
    """

    if t > 1:  # TODO
        raise IndexError('Random generator not defined (yet) for multiple time points')

    d = check_omega(omega)
    v_shape = vf_shape_from_omega_and_timepoints(omega, t)
    vf = np.zeros(v_shape)
    idd = vf_affine_to_homogeneous(vf_identity_eulerian(v_shape))

    if structure == 'algebra':
        if d == 2:
            x_intervals, y_intervals = omega
            for i in range(x_intervals):
                for j in range(y_intervals):
                    vf[i, j, 0, 0, 0] = input_homography[0, :].dot(idd[i, j, 0, 0, :]) - \
                                        i * input_homography[2, :].dot(idd[i, j, 0, 0, :])
                    vf[i, j, 0, 0, 1] = input_homography[1, :].dot(idd[i, j, 0, 0, :]) - \
                                        j * input_homography[2, :].dot(idd[i, j, 0, 0, :])
        elif d == 3:
            x_intervals, y_intervals, z_intervals = omega
            for i in range(x_intervals):
                for j in range(y_intervals):
                    for k in range(z_intervals):
                        vf[i, j, k, 0, 0] = input_homography[0, :].dot(idd[i, j, k, 0, :]) - \
                                            i * input_homography[3, :].dot(idd[i, j, k, 0, :])
                        vf[i, j, k, 0, 1] = input_homography[1, :].dot(idd[i, j, k, 0, :]) - \
                                            j * input_homography[3, :].dot(idd[i, j, k, 0, :])
                        vf[i, j, k, 0, 2] = input_homography[2, :].dot(idd[i, j, k, 0, :]) - \
                                            k * input_homography[3, :].dot(idd[i, j, k, 0, :])
    elif structure == 'group':
        if d == 2:
            x_intervals, y_intervals = omega
            for i in range(x_intervals):
                for j in range(y_intervals):

                    s = input_homography.dot(np.array([i, j, 1]))[:]
                    if abs(s[2]) > 1e-5:
                        # subtract the id point-wise to have the result in displacement coordinates
                        vf[i, j, 0, 0, :] = (s[0:2] / float(s[2])) - np.array([i, j])

        elif d == 3:
            x_intervals, y_intervals, z_intervals = omega
            for i in range(x_intervals):
                for j in range(y_intervals):
                    for k in range(z_intervals):

                        s = input_homography.dot(np.array([i, j, k, 1]))[:]
                        if abs(s[3]) > 1e-5:
                            vf[i, j, k, 0, :] = (s[0:3] / float(s[3])) - np.array([i, j, k])

    else:
        raise IOError("structure can be only 'algebra' or 'group' corresponding to the algebraic structure.")

    return vf
