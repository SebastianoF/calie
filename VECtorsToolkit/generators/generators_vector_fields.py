import os
import numpy as np
import scipy.ndimage.filters as fil
import nibabel as nib

from src.tools.auxiliary.sanity_checks import check_omega, get_v_shape_from_omega

from src.tools.auxiliary.sanity_checks import get_omega_from_vf


#########################
# Change coordinates #
#########################


def change_coordinates_vf_to_homogeneous(input_v):
    """
    Adds the homogeneous coordinates to the given vector field.

    input_v(x,y,z,0,:) = (vx, vy, vz)

    output_v(x,y,z,0,:) = (vx, vy, vz, 1)

    :param input_v:
    """

    slice_shape = list(input_v.shape)
    slice_shape[4] = 1

    return np.append(input_v, np.ones(slice_shape), axis=4)


def change_coordinates_vf_to_affine(input_v):
    """
    Removes the homogeneous coordinates to the given homogeneous field.
    It changes the provided data structure instead of creating a new one.
    :param input_v:
    :return: field in affine coordinates.

    NOTE: it works only for vector fields, so for fields with one more coordinate.
    If the given vector field was already affine nothing happen.
    """
    return input_v[..., :-1]


def change_coordinates_vf_from_eulerian_to_lagrangian(input_vf_eul):
    # from a vector field in eulerian returns the equivalent in lagrangian
    return input_vf_eul - generate_identity_lagrangian_like(input_vf_eul)


def change_coordinates_vf_from_lagrangian_to_eulerian(input_vf_lag):
    # from a vector field in lagrangian returns the equivalent in eulerian
    return input_vf_lag + generate_identity_lagrangian_like(input_vf_lag)


############################
# Vector Fields Generators #
############################


def generate_identity_lagrangian(omega, t=1):

    d = len(omega)
    v_shape = get_v_shape_from_omega(omega, t=t)
    id_vf = np.zeros(v_shape)

    if d == 2:
        x = range(v_shape[0])
        y = range(v_shape[1])
        gx, gy = np.meshgrid(x, y, indexing='ij')

        id_vf[..., 0, :, 0] = np.repeat(gx, t).reshape(omega + [t])
        id_vf[..., 0, :, 1] = np.repeat(gy, t).reshape(omega + [t])

    elif d == 3:
        x = range(v_shape[0])
        y = range(v_shape[1])
        z = range(v_shape[2])
        gx, gy, gz = np.meshgrid(x, y, z, indexing='ij')

        id_vf[..., :, 0] = np.repeat(gx, t).reshape(omega + [t])
        id_vf[..., :, 1] = np.repeat(gy, t).reshape(omega + [t])
        id_vf[..., :, 2] = np.repeat(gz, t).reshape(omega + [t])

    else:
        raise IOError("Dimensions allowed: 2, 3")

    return id_vf


def generate_identity_lagrangian_like(input_vf):
    """
    :param input_vf: input vector field.
    :return: corresponding grid position, i.e. the identity vector field sampled in the input_vf grid matrix
    in Lagrangian coordinates.
    """
    time_points_vf = input_vf.shape[3]
    omega = get_omega_from_vf(input_vf)
    return generate_identity_lagrangian(omega, t=time_points_vf)


def generate_identity_eulerian_like(input_vf):
    """
    :param input_vf: input vector field.
    :return: corresponding identity grid position in Eulerian coordinates
    """
    return np.zeros_like(input_vf)



def generate_id_matrix(omega):
    """
    From a omega of dimension dim =2,3, it returns the identity field
    that at each point of the omega has the (row mayor) vectorized identity
    matrix. Utilised in the matrix_manipulation module.
    :param omega: a squared or cubed omega
    :return:
    """
    dim = len(omega)
    if dim not in [2, 3]:
        assert IOError

    shape = list(omega) + [1] * (4 - dim) + [dim**2]
    flat_id = np.eye(dim).reshape(1, dim**2)
    return np.repeat(flat_id, np.prod(omega)).reshape(shape, order='F')


def generate_random(omega, t=1, parameters=(5,2)):
    """
    Return a random vector field v.
    :param omega: domain of the vector field
    :param t: time not yet implemented.
    :param parameters:
    :return:
    """
    v_shape = get_v_shape_from_omega(omega, t)

    if t > 1:
        raise IndexError('Random generator not defined (yet) for multiple time points')

    sigma_init, sigma_gaussian_filter = parameters
    v = np.random.normal(0, sigma_init, v_shape)

    for i in range(v_shape[-1]):
        v[..., 0, i] = fil.gaussian_filter(v[..., 0, i], sigma_gaussian_filter)

    return v


def generate_from_matrix(omega, input_matrix, t=1, structure='algebra'):

    d = len(omega)
    v_shape = get_v_shape_from_omega(omega, t)

    if structure == 'algebra':
        # Need to subtract the id somewhere...!
        pass
    elif structure == 'group':
        pass
    else:
        raise IOError

    if t > 1:
        raise IndexError('Random generator not defined (yet) for multiple time points')

    if d == 2:

        v = change_coordinates_vf_to_homogeneous(generate_identity_lagrangian(v_shape))

        x_intervals, y_intervals = omega
        for i in range(x_intervals):
            for j in range(y_intervals):
                v[i, j, 0, 0, :] = input_matrix.dot(v[i, j, 0, 0, :])

        v = change_coordinates_vf_to_affine(v)

    elif d == 3:

        x_intervals, y_intervals, z_intervals = omega

        # If the matrix provides a 2d rototranslation, we consider the rotation axis perpendicular to the plane z=0.
        if np.alltrue(input_matrix.shape == (3, 3)):

            v = np.zeros(v_shape)

            # Create the slice at the ground of the domain (x,y,z) , z = 0, as a 2d rotation:
            base_slice = change_coordinates_vf_to_homogeneous(generate_identity_lagrangian(list(omega[:2]) + [1, 1, 2]))

            for i in range(x_intervals):
                for j in range(y_intervals):
                    base_slice[i, j, 0, 0, :] = input_matrix.dot(base_slice[i, j, 0, 0, :])

            # Copy the slice at the ground on all the others:
            for k in range(z_intervals):
                v[..., k, 0, :2] = base_slice[..., 0, 0, :2]

        # If the matrix is 2d the rotation axis is perpendicular to the plane z=0.
        elif np.alltrue(input_matrix.shape == (4, 4)):

            v = change_coordinates_vf_to_homogeneous(generate_identity_lagrangian(v_shape))

            for i in range(x_intervals):
                for j in range(y_intervals):
                    for k in range(y_intervals):
                        v[i, j, k, 0, :] = input_matrix.dot(v[i, j, k, 0, :])

            v = change_coordinates_vf_to_affine(v)
        else:
            raise IOError('Wrong input parameter.')

    else:
        raise IOError("Dimensions allowed: 2 or 3")

    return v



def generate_from_projective_matrix(omega, input_homography, structure='algebra'):

    t = 1  # TODO t depends on the value of the matrix, if it is a stack of matrices (one for each time point)
           # or if it is a single matrix.

    if structure == 'algebra':


        d = check_omega(omega)
        v_shape = get_v_shape_from_omega(omega, t)

        if t > 1:
            raise IndexError('Random generator not defined (yet) for multiple time points')

        if d == 2:

            idd = generate_identity_lagrangian(v_shape)
            vf = np.zeros(v_shape)

            idd = change_coordinates_vf_to_homogeneous(idd)

            x_intervals, y_intervals = omega
            for i in range(x_intervals):
                for j in range(y_intervals):
                    vf[i, j, 0, 0, 0] = input_homography[0, :].dot(idd[i, j, 0, 0, :]) - \
                                        i * input_homography[2, :].dot(idd[i, j, 0, 0, :])
                    vf[i, j, 0, 0, 1] = input_homography[1, :].dot(idd[i, j, 0, 0, :]) - \
                                        j * input_homography[2, :].dot(idd[i, j, 0, 0, :])

        elif d == 3:

            idd = generate_identity_lagrangian(v_shape)
            vf = np.zeros(v_shape)

            idd = change_coordinates_vf_to_homogeneous(idd)

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

        else:
            raise IOError("Dimensions allowed: 2 or 3")

        return vf

    elif structure == 'group':

        d = check_omega(omega)
        v_shape = get_v_shape_from_omega(omega)

        # TODO Debug carefully!!

        if d == 2:

            vf = np.zeros(v_shape)

            x_intervals, y_intervals = omega
            for i in range(x_intervals):
                for j in range(y_intervals):

                    s = input_homography.dot(np.array([i, j, 1]))[:]
                    if abs(s[2]) > 1e-5:
                        # subtract the id to have the result in displacement coordinates
                        vf[i, j, 0, 0, :] = (s[0:2] / float(s[2])) - np.array([i, j])

        elif d == 3:

            vf = np.zeros(v_shape)

            x_intervals, y_intervals, z_intervals = omega
            for i in range(x_intervals):
                for j in range(y_intervals):
                    for k in range(z_intervals):

                        s = input_homography.dot(np.array([i, j, k, 1]))[:]
                        if abs(s[3]) > 1e-5:
                            vf[i, j, k, 0, :] = (s[0:3] / float(s[3])) - np.array([i, j, k])




        else:
            raise IOError("Dimensions allowed: 2 or 3")
        return vf
    else:
        raise IOError



def generate_from_image(input_nib_image):

    # input can be a path to a nifti, a matrix or a nibabel image.
    # In the future a simpleITK image.

    if isinstance(input_nib_image, str):
        msg = 'Input path {} does not exist.'.format(input_nib_image)
        assert os.path.exists(input_nib_image), msg
        input_nib_image = nib.load(input_nib_image)

    return input_nib_image.get_array()
