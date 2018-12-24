"""
Lagrangian coordinates: the vector field is represented from the perspective of the particle in it.
Eulerian coordinates: the vector field is represented from the origin of the coordinate system / matrix.

A confusing nomenclature sometimes used in medical imaging is:
Lagrangian -> displacement
Eulerian   -> deformation
"""
import numpy as np

from VECtorsToolkit.fields import queries as qr


def id_lagrangian(omega, t=1):
    """
    :param omega: discretized domain of the vector field
    :param t: number of timepoints
    :return: identity vector field of given domain and timepoints, in Lagrangian coordinates.
    """
    d = qr.check_omega(omega)
    vf_shape = list(omega) + [1] * (3 - d) + [t, d]
    return np.zeros(vf_shape)


def id_eulerian(omega, t=1):
    """
    :param omega: discretized domain of the vector field
    :param t: number of timepoints
    :return: identity vector field of given domain and timepoints, in Eulerian coordinates.
    """
    d = qr.check_omega(omega)
    omega = list(omega)
    v_shape = qr.shape_from_omega_and_timepoints(omega, t=t)
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

    return id_vf


def id_lagrangian_like(input_vf):
    """
    :param input_vf: input vector field.
    :return: corresponding identity grid position in Eulerian coordinates
    """
    qr.check_is_vf(input_vf)
    return np.zeros_like(input_vf)


def id_eulerian_like(input_vf):
    """
    :param input_vf: input vector field.
    :return: corresponding grid position, i.e. the identity vector field sampled in the input_vf grid matrix
    in Lagrangian coordinates.
    """
    qr.check_is_vf(input_vf)
    return id_eulerian(qr.get_omega(input_vf), t=input_vf.shape[3])


def id_matrices(omega, t=1):
    """
    From a omega of dimension dim =2,3, it returns the identity field
    that at each point of the omega has the (row mayor) vectorized identity matrix.
    :param omega: a squared or cubed omega
    :param t: timepoint
    :return: vector field with a vectorised identity matrix at each point.
    """
    d = qr.check_omega(omega)

    shape = list(omega) + [1] * (4 - d) + [d**2]
    shape[3] = t
    flat_id = np.eye(d).reshape(1, d**2)
    return np.repeat(flat_id, np.prod(list(omega) + [t])).reshape(shape, order='F')


def id_lagrangian_like_image(input_nib_image, t=1):
    """
    :param input_nib_image: nibabel image or path to a nifti image.
    :param t: additional timepoint
    :return: identity in lagrangian coordinate with same shape of the input image
    """
    return id_lagrangian(qr.nib_to_omega(input_nib_image), t=t)


def id_eulerian_like_image(input_nib_image, t=1):
    """
    :param input_nib_image: nibabel image or path to a nifti image.
    :param t: additional timepoint
    :return: identity in eulerian coordinate with same shape of the input image
    """
    return id_eulerian(qr.nib_to_omega(input_nib_image), t=t)
