import numpy as np
from VECtorsToolkit.tools.auxiliary.matrices import matrix_vector_field_product
from scipy import ndimage
from scipy.interpolate import griddata, Rbf

from VECtorsToolkit.tools.fields.queries import get_omega_from_vf
from VECtorsToolkit.tools.fields.coordinates import vf_eulerian_to_lagrangian, vf_lagrangian_to_eulerian


def one_point_interpolation(input_vf, point, method='linear', as_float=True):
    """
    For the moment only 2d and in matrix coordinates.
    :param input_vf:
    :param point:
    :param method:
    :param as_float:
    :return:
    """
    # TODO n-dim
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
        vf_at_point = (float(grid_x), float(grid_y))
    else:
        vf_at_point = (grid_x, grid_y)
    return vf_at_point


def one_point_interpolation_rdf(input_vf, point, epsilon=50, as_float=True):
    """
    For the moment only 2d and in matrix coordinates.
    Secondary method for the interpolation, with radial basis function:
    :param input_vf:
    :param point:
    :param epsilon: see Rdf documentation
    :param as_float:
    :return:
    """
    if not len(point) == 2:
        raise IOError("Input expected is a 2d point for a 2d field.")

    # rename for clarity
    x_p, y_p = point[0], point[1]

    sh = input_vf.shape
    x_grid, y_grid = np.mgrid[0:sh[0], 0:sh[1]]

    v_x = input_vf[x_grid, y_grid, 0, 0, 0]
    v_y = input_vf[x_grid, y_grid, 0, 0, 1]

    rbf_x_in = Rbf(x_grid, y_grid, v_x, epsilon=epsilon)
    rbf_y_in = Rbf(x_grid, y_grid, v_y, epsilon=epsilon)

    if as_float:
        vf_at_point = (float(rbf_x_in(x_p, y_p)), float(rbf_y_in(x_p, y_p)))
    else:
        vf_at_point = (rbf_x_in(x_p, y_p), rbf_y_in(x_p, y_p))
    return vf_at_point


# ---- CORE methods ---- #

def lagrangian_dot_eulerian(vf_left_lag, vf_right_eul,
                            affine_left_right=None,
                            spline_interpolation_order=2,
                            mode='constant',
                            cval=0.0,
                            prefilter=True,
                            add_right=True):

    omega_right = get_omega_from_vf(vf_right_eul)
    d = len(omega_right)

    if affine_left_right is not None:
        A_l, A_r = affine_left_right
        vf_right_eul = matrix_vector_field_product(np.linalg.inv(A_l).dot(A_r), vf_right_eul)

    coord = [vf_right_eul[..., i].reshape(omega_right, order='F') for i in range(d)]
    result = np.squeeze(np.zeros_like(vf_left_lag))

    for i in range(d):  # see if the for can be avoided with tests.

        ndimage.map_coordinates(np.squeeze(vf_left_lag[..., i]),
                                coord,
                                output=result[..., i],
                                order=spline_interpolation_order,
                                mode=mode,
                                cval=cval,
                                prefilter=prefilter)
    if add_right:  # option for the scaling and squaring.
        return result.reshape(vf_left_lag.shape) + vf_eulerian_to_lagrangian(vf_right_eul)
    else:
        return result.reshape(vf_left_lag.shape)


def scalar_dot_eulerian(sf_left, vf_right_eul,
                        affine_left_right=None,
                        spline_interpolation_order=2,
                        mode='constant',
                        cval=0.0,
                        prefilter=True):

    omega_right = get_omega_from_vf(vf_right_eul)
    d = len(omega_right)

    if affine_left_right is not None:
        A_l, A_r = affine_left_right
        # multiply each point of the vector field by the transformation matrix
        vf_right_eul = matrix_vector_field_product(np.linalg.inv(A_l).dot(A_r), vf_right_eul)

    coord = [vf_right_eul[..., i].reshape(omega_right, order='F') for i in range(d)]
    result = np.zeros_like(sf_left)

    ndimage.map_coordinates(sf_left,
                            coord,
                            output=result,
                            order=spline_interpolation_order,
                            mode=mode,
                            cval=cval,
                            prefilter=prefilter)

    return result


# ---- Derived methods ---- #


def lagrangian_dot_lagrangian(vf_left_lag, vf_right_lag,
                              affine_left_right=None,
                              spline_interpolation_order=2,
                              mode='constant',
                              cval=0.0,
                              prefilter=True,
                              add_right=True):

    vf_right_eul = vf_lagrangian_to_eulerian(vf_right_lag)

    return lagrangian_dot_eulerian(vf_left_lag, vf_right_eul,
                                   affine_left_right=affine_left_right,
                                   spline_interpolation_order=spline_interpolation_order,
                                   mode=mode,
                                   cval=cval,
                                   prefilter=prefilter,
                                   add_right=add_right)


def eulerian_dot_lagrangian(vf_left_eul, vf_right_lag,
                            affine_left_right=None,
                            spline_interpolation_order=2,
                            mode='constant',
                            cval=0.0,
                            prefilter=True,
                            add_right=True):

    vf_left_lag = vf_eulerian_to_lagrangian(vf_left_eul)
    vf_right_eul = vf_lagrangian_to_eulerian(vf_right_lag)

    return lagrangian_dot_eulerian(vf_left_lag, vf_right_eul,
                                   affine_left_right=affine_left_right,
                                   spline_interpolation_order=spline_interpolation_order,
                                   mode=mode,
                                   cval=cval,
                                   prefilter=prefilter,
                                   add_right=add_right)


def eulerian_dot_eulerian(vf_left_eul, vf_right_eul,
                          affine_left_right=None,
                          spline_interpolation_order=2,
                          mode='constant',
                          cval=0.0,
                          prefilter=True,
                          add_right=True):

    vf_left_lag = vf_eulerian_to_lagrangian(vf_left_eul)

    return lagrangian_dot_eulerian(vf_left_lag, vf_right_eul,
                                   affine_left_right=affine_left_right,
                                   spline_interpolation_order=spline_interpolation_order,
                                   mode=mode,
                                   cval=cval,
                                   prefilter=prefilter,
                                   add_right=add_right)


def scalar_dot_lagrangian(sf_left, vf_right_lag,
                          affine_left_right=None,
                          spline_interpolation_order=2,
                          mode='constant',
                          cval=0.0,
                          prefilter=True):

    vf_right_eul = vf_lagrangian_to_eulerian(vf_right_lag)

    return scalar_dot_eulerian(sf_left, vf_right_eul,
                               affine_left_right=affine_left_right,
                               spline_interpolation_order=spline_interpolation_order,
                               mode=mode,
                               cval=cval,
                               prefilter=prefilter)

#
# if __name__ == '__main__':
#
#     from VECtorsToolkit.tools.fields.generate_identities import vf_identity_lagrangian
#
#     def u(x, y):
#         return x, y
#
#     def v(x, y):
#         return 0.5, 0.5
#
#     def u_dot_v(x, y):
#         return 0.5, 0.5
#
#     def v_dot_u(x, y):
#         return 0.5, 0.5
#
#     omega = (6, 6)
#
#     svf_u = vf_identity_lagrangian(omega=omega)
#     svf_v = vf_identity_lagrangian(omega=omega)
#     svf_u_dot_v = vf_identity_lagrangian(omega=omega)
#     svf_v_dot_u = vf_identity_lagrangian(omega=omega)
#
#     for x in range(omega[0]):
#         for y in range(omega[1]):
#             svf_u[x, y, 0, 0, :] = u(x, y)
#             svf_v[x, y, 0, 0, :] = v(x, y)
#             svf_u_dot_v[x, y, 0, 0, :] = u_dot_v(x, y)
#             svf_v_dot_u[x, y, 0, 0, :] = v_dot_u(x, y)
#
#     svf_v_dot_u_numerical = lagrangian_dot_lagrangian(svf_v, svf_u, add_right=False)
#     svf_u_dot_v_numerical = lagrangian_dot_lagrangian(svf_u, svf_v, add_right=False)
