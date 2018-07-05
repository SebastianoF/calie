import numpy as np
from VECtorsToolkit.tools.auxiliary.matrices import matrix_vector_field_product
from scipy import ndimage
from scipy.interpolate import griddata

from VECtorsToolkit.tools.auxiliary.sanity_checks import get_omega_from_vf


def one_point_interpolation(input_vf, point, method='linear', as_float=True):
    """
    For the moment only 2d and in matrix coordinates.
    :param input_vf:
    :param point:
    :param method:
    :param as_float:
    :return:
    """
    # TODO make nd
    if not len(point) == 2:
        raise IOError("Input expected is a 2d point for a 2d field.")

    # rename for clarity
    x_p, y_p = point[0], point[1]

    # all of the grid point in 2 columns:
    points = np.array([[i * 1.0, j * 1.0] for i in range(input_vf.shape[0])
                       for j in range(input_vf.shape[1])])
    values_x = np.array([input_vf[i, j, 0, 0, 0]
                         for i in range(input_vf.shape[0])
                         for j in range(input_vf.shape[1])]).T
    values_y = np.array([input_vf[i, j, 0, 0, 1]
                         for i in range(input_vf.shape[0])
                         for j in range(input_vf.shape[1])]).T

    grid_x = griddata(points, values_x, (x_p, y_p), method=method)
    grid_y = griddata(points, values_y, (x_p, y_p), method=method)

    if as_float:
        v_at_point = (float(grid_x), float(grid_y))
    else:
        v_at_point = (grid_x, grid_y)
    return v_at_point


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
