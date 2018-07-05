"""
Sometimes:
Lagrangian -> displacement
Eulerian   -> deformation
"""
import numpy as np

from VECtorsToolkit.tools.fields.queries import vf_shape_from_omega_and_timepoints, get_omega


def vf_identity_lagrangian(omega, t=1):

    d = len(omega)
    v_shape = vf_shape_from_omega_and_timepoints(omega, t=t)
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


def vf_identity_eulerian(omega, t=1):
    d = len(omega)
    vf_shape = list(omega) + [1] * (3 - d) + [t, d]
    return np.zeros(vf_shape)


def vf_identity_lagrangian_like(input_vf):
    """
    :param input_vf: input vector field.
    :return: corresponding grid position, i.e. the identity vector field sampled in the input_vf grid matrix
    in Lagrangian coordinates.
    """
    return vf_identity_lagrangian(get_omega(input_vf), t=input_vf.shape[3])


def vf_identity_eulerian_like(input_vf):
    """
    :param input_vf: input vector field.
    :return: corresponding identity grid position in Eulerian coordinates
    """
    return np.zeros_like(input_vf)
