import numpy as np
from VECtorsToolkit.tools.aux.matrices import id_matrix_field, matrix_vector_field_product, \
    matrix_fields_product_iterative

from VECtorsToolkit.fields import check_is_vf


def initialise_jacobian(input_vf):
    d = check_is_vf(input_vf)
    sh = list(input_vf.shape)
    while len(sh) < 5:
        sh.extend([1])
    sh[-1] = d ** 2

    return np.zeros(sh, dtype=np.float64)


def compute_jacobian(input_vf, affine=np.eye(4), is_lagrangian=False):
    """
    :param input_vf: input vecgor field
    :param affine: The affine transformation optionally associated to the field.
    :param is_lagrangian: if the identity matrix should be added to each jacobian matrix
    See itk documentation:
    http://www.itk.org/Doxygen/html/classitk_1_1DisplacementFieldJacobianDeterminantFilter.html

    On the diagonal it possess the sample distances for each dimension.
    Jacobian matrix at each point of the grid is stored in a vector of size 9 in row major order.
    """
    # TODO It works only for svf (1 time point) - provisional - do with multiple time point
    d = check_is_vf(input_vf)

    jacobian = initialise_jacobian(input_vf)

    dims = []

    for i in range(d):
        dims.append(affine[i, i])

    output_data = jacobian.squeeze()

    for i in range(d):
        grad = np.gradient(np.squeeze(input_vf[..., i]), *dims)

        for j in range(d):
            output_data[..., i * d + j] = grad[j].squeeze()

    if is_lagrangian:
        jacobian += id_matrix_field(input_vf.shape[:d])

    return jacobian


def compute_jacobian_determinant(input_vf, is_lagrangian=False):
    """
    :param input_vf: The Field or children whose jacobian we need to compute.
    :param is_lagrangian: add the identity to the jacobian matrix before the computation of the
    jacobian determinant.
    If it is none, it is allocated.
    Jacobian matrix at each point of the grid is stored in a vector of size 9 in row major order.
    !! It works only for 1 time point - provisional !!
    """
    d = check_is_vf(input_vf)

    vf_jacobian = compute_jacobian(input_vf, is_lagrangian=is_lagrangian)

    sh = list(input_vf.shape)
    while len(sh) < 5:
        sh.extend([1])
    sh = sh[:-1]

    sh.extend([d, d])

    v = vf_jacobian.reshape(sh)
    return np.linalg.det(v)


def jacobian_product(vf_left, vf_right):
    """
    Compute the jacobian product between two 2d or 3d SVF self and right: J_{self}(right)
    the results is a second SVF.
    :param vf_left : svf
    :param vf_right: svf
    :return: J_{right}(left) : jacobian product between 2 svfs
    """
    left  = np.copy(vf_left)
    right = np.copy(vf_right)

    result_array = matrix_vector_field_product(compute_jacobian(right), left)

    return result_array


def iterative_jacobian_product(vf, n):
    """
    :param vf: input SVF
    :param n: number of iterations
    :return: a new SVF defined by the jacobian product J_v^(n-1) v
    """
    jv_field = compute_jacobian(vf)[...]
    v_field = vf[...]

    jv_n_prod_v = matrix_vector_field_product(matrix_fields_product_iterative(jv_field, n-1), v_field)

    return jv_n_prod_v


def lie_bracket(vf_left, vf_right):
    """
    Compute the Lie bracket of two velocity fields.

    Parameters:
    -----------
    :param vf_left: Left velocity field.
    :param vf_right: Right velocity field.
    Order of Lie bracket: [left,right] = Jac(left)*right - Jac(right)*left
    :return Return the resulting velocity field
    """

    return jacobian_product(vf_left, vf_right) - jacobian_product(vf_right, vf_left)
