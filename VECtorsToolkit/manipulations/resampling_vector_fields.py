import numpy as np
from scipy import ndimage

from src.tools.auxiliary.sanity_checks import get_omega_from_vf
from src.tools.auxiliary.matrices_manipulations import matrix_vector_field_product


def compose_eulerian_vf_with_lagrangian_vf(vf_left_eul, vf_right_lag,
                                           affine_left_right=None,
                                           spline_interpolation_order=2,
                                           mode='constant',
                                           cval=0.0,
                                           prefilter=True):

    omega_right = get_omega_from_vf(vf_right_lag)
    d = len(omega_right)

    if affine_left_right is not None:
        # affine matrices of the vector field from voxel space to real space. If defined the composition is
        # computed in the real space. If dealing with nifty images they are the affine transformations.
        A_l, A_r = affine_left_right
        # multiply each point of the vector field by the transformation matrix
        vf_right_lag = matrix_vector_field_product(np.linalg.inv(A_l).dot(A_r), vf_right_lag)

    coord = [vf_right_lag[..., i].reshape(omega_right, order='F') for i in range(d)]
    result = np.squeeze(np.zeros_like(vf_left_eul))

    for i in range(d):  # see if the for can be avoided with tests.

        ndimage.map_coordinates(np.squeeze(vf_left_eul[..., i]),
                                coord,
                                output=result[..., i],
                                order=spline_interpolation_order,
                                mode=mode,
                                cval=cval,
                                prefilter=prefilter)

    return result.reshape(vf_left_eul.shape)


def compose_scalar_field_with_lagrangian_vector_field(sf_left, vf_right_lag,
                                                      affine_left_right=None,
                                                      spline_interpolation_order=2,
                                                      mode='constant',
                                                      cval=0.0,
                                                      prefilter=True):

    omega_right = get_omega_from_vf(vf_right_lag)
    d = len(omega_right)

    if affine_left_right is not None:
        A_l, A_r = affine_left_right
        # multiply each point of the vector field by the transformation matrix
        vf_right_lag = matrix_vector_field_product(np.linalg.inv(A_l).dot(A_r), vf_right_lag)

    coord = [vf_right_lag[..., i].reshape(omega_right, order='F') for i in range(d)]
    result = np.zeros_like(sf_left)

    ndimage.map_coordinates(sf_left,
                            coord,
                            output=result,
                            order=spline_interpolation_order,
                            mode=mode,
                            cval=cval,
                            prefilter=prefilter)

    return result
