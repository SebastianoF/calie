import numpy as np

from VECtorsToolkit.fields.generate_identities import vf_identity_eulerian_like


def vf_affine_to_homogeneous(input_vf):
    """
    Adds the homogeneous coordinates to the given vector field.
    input_v(x,y,z,0,:) = (vx, vy, vz)
    output_v(x,y,z,0,:) = (vx, vy, vz, 1)
    :param input_vf: input vector field
    """
    slice_shape = list(input_vf.shape)
    slice_shape[4] = 1

    return np.append(input_vf, np.ones(slice_shape), axis=4)


def vf_homogeneous_to_affine(input_vf):
    """
    Removes the homogeneous coordinates to the given homogeneous field.
    It changes the provided data structure instead of creating a new one.
    :param input_vf:
    :return: field in affine coordinates.

    NOTE: it works only for vector fields, so for fields with one more coordinate.
    If the given vector field was already affine nothing happen.
    """
    return input_vf[..., :-1]


def vf_eulerian_to_lagrangian(input_vf_eul):
    return input_vf_eul - vf_identity_eulerian_like(input_vf_eul)


def vf_lagrangian_to_eulerian(input_vf_lag):
    return input_vf_lag + vf_identity_eulerian_like(input_vf_lag)

