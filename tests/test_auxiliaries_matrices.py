from random import uniform

import numpy as np
import scipy.linalg as lin
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_raises

from calie.transformations import se2
from calie.aux import matrices as mat #import bch_right_jacobian, matrix_vector_field_product, \
# matrix_fields_product, matrix_fields_product_iterative, id_matrix_field, split_the_time

''' test BCH right Jacobian '''


def test_bch_right_jacobian_all_zero():
    inp = [0, 0, 0]
    output = mat.bch_right_jacobian(inp)
    expected_output = np.identity(3)
    assert np.alltrue(output == expected_output)


def test_bch_right_jacobian_zero_angle():
    inp = [0, 2.6, 6.9]
    output = mat.bch_right_jacobian(inp)
    expected_output = np.identity(3)
    expected_output[1, 0] = 0.5 * inp[2]
    expected_output[2, 0] = - 0.5 * inp[1]
    assert np.alltrue(output == expected_output)


def test_bch_right_jacobian_zero_translation():
    theta = 0.2
    inp = [theta, 0, 0]
    output = mat.bch_right_jacobian(inp)
    expected_output = np.identity(3)
    factor = (theta * 0.5) / np.tan(theta * 0.5)
    expected_output[1, 1] = factor
    expected_output[2, 2] = factor
    expected_output[1, 2] = -0.5 * theta
    expected_output[2, 1] = 0.5 * theta
    assert np.alltrue(output == expected_output)


def test_bch_right_jacobian_little_rotation_zero_traslation():
    theta = abs(np.spacing(0))
    factor1 = theta / 12.0
    factor2 = 1 - (theta ** 2) / 12.0
    inp = [theta - 0.001 * np.spacing(0), 0, 0]
    output = mat.bch_right_jacobian(inp)
    expected_output = np.identity(3)
    expected_output[1, 0] = - factor1 * inp[1] + 0.5 * inp[2]
    expected_output[2, 0] = - 0.5 * inp[2] - factor1 * inp[1]
    expected_output[1, 1] = factor2
    expected_output[2, 2] = factor2
    assert np.alltrue(output == expected_output)


def test_bch_ground_random_input_comparing_matrices_step_1():
    any_angle_1 = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    any_tx_1 = uniform(-10, 10)
    any_ty_1 = uniform(-10, 10)
    any_angle_2 = uniform(-np.pi + abs(np.spacing(-np.pi)), np.pi)
    any_tx_2 = uniform(-10, 10)
    any_ty_2 = uniform(-10, 10)
    a = se2.Se2A(any_angle_1, any_tx_1, any_ty_1)
    da = se2.Se2A(any_angle_2, any_tx_2, any_ty_2)
    a_matrix = a.get_matrix
    da_matrix = da.get_matrix
    exp_exp_pade = lin.expm(a_matrix).real.dot(lin.expm(da_matrix).real).real
    exp_exp_my = se2.se2a_exp(a) * se2.se2a_exp(da)
    assert_array_almost_equal(exp_exp_pade, exp_exp_my.get_matrix)


''' test split the time '''


def test_split_the_time_incorrect_input_len():
    t = [0, 1, 2, 3]
    x = [0.1, 0.2, 0.3, 0.4, 0.5]
    with assert_raises(TypeError):
        mat.time_splitter(t, x)


def test_split_the_time_easy_array_1():
    t = [0, 2, 1, 3, 4]
    x = [0.1, 0.2, 0.3, 0.4, 0.5]
    x_split = mat.time_splitter(t, x, number_of_intervals=2)
    x_expected = [[0.1, 0.3], [0.2, 0.4, 0.5]]
    assert_array_equal(x_expected, x_split)


def test_split_the_time_easy_array_2():
    t = range(11)[::-1]
    x = range(11)[::-1]
    x_split = mat.time_splitter(t, x, number_of_intervals=5)
    x_expected = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9, 10]]
    assert_array_equal(x_expected, x_split)


def test_split_the_time_easy_array_3():
    t = range(11)[::-1]
    x = range(11)[::-1]
    x_split = mat.time_splitter(t, x, number_of_intervals=3, len_range=(2.1, 7))
    x_expected = [[3], [4, 5], [6, 7]]
    assert_array_equal(x_expected, x_split)


''' test matrix_vector_field_product '''


def test_matrix_vector_field_product_2d():

    def field_vector(t, x):
        t = float(t); x = [float(y) for y in x]
        return x[1], -1 * x[0]

    def field_matrix(t, x):
        t = float(t); x = [float(y) for y in x]
        return 0.5, 0.6, x[0], 0.8

    def field_ground_product(t, x):
        t = float(t); x = [float(y) for y in x]
        return 0.5*x[1] - 0.6*x[0], x[0]*x[1] - 0.8*x[0]

    v   = np.zeros([20, 20, 2])
    jac = np.zeros([20, 20, 4])

    ground_jac_v = np.zeros([20, 20, 2])

    for i in range(0, 20):
        for j in range(0, 20):

            v[i, j, :]   = field_vector(1, [i, j])
            jac[i, j, :] = field_matrix(1, [i, j])

            ground_jac_v[i, j, :] = field_ground_product(1, [i, j])

    jac_v = mat.matrix_vector_field_product(jac, v)

    assert_array_equal(jac_v.shape, np.array([20, 20, 2]))
    assert_array_equal(jac_v, ground_jac_v)


def test_matrix_vector_field_product_toy_example_3d():

    def field_vector(t, x):
        t = float(t); x = [float(y) for y in x]
        return 2*x[1], 3*x[0], x[0] - x[2]

    def field_matrix(t, x):
        t = float(t); x = [float(y) for y in x]
        return 0.5*x[0], 0.5*x[1], 0.5*x[2], \
               0.5,      x[0],     0.3,      \
               0.2*x[2], 3.0,      2.1*x[2]

    def field_ground_product(t, x):
        t = float(t); x = [float(y) for y in x]
        return 0.5*2*x[0]*x[1] + 0.5*3*x[0]*x[1] + 0.5*x[2]*(x[0] - x[2]), \
               0.5*2*x[1] + 3*x[0]*x[0] + 0.3*(x[0] - x[2]), \
               0.2*2*x[2]*x[1] + 3*3*x[0] + 2.1*x[2]*(x[0] - x[2])

    v   = np.zeros([20, 20, 20, 3])
    jac = np.zeros([20, 20, 20, 9])

    ground_jac_v = np.zeros([20, 20, 20, 3])

    for i in range(0, 20):
        for j in range(0, 20):
            for k in range(0, 20):
                v[i, j, k, :]   = field_vector(1, [i, j, k])
                jac[i, j, k, :] = field_matrix(1, [i, j, k])

                ground_jac_v[i, j, k, :] = field_ground_product(1, [i, j, k])

    jac_v = mat.matrix_vector_field_product(jac, v)

    assert_array_equal(jac_v.shape, np.array([20, 20, 20, 3]))
    assert_array_almost_equal(jac_v, ground_jac_v)


''' matrix_fields_product '''


def test_matrix_fields_product_2d_1():

    def field_matrix_a(t, x):
        t = float(t); x = [float(y) for y in x]
        return 2*x[0]*x[1], 3*x[0], \
               4,           x[1]

    def field_matrix_b(t, x):
        t = float(t); x = [float(y) for y in x]
        return x[0], 2, \
               x[0], 3*x[1]

    def product_a_b(t, x):
        t = float(t); x = [float(y) for y in x]
        return 2*(x[0]**2)*x[1] + 3*x[0]**2, 13*x[0]*x[1], \
               4*x[0] + x[0]*x[1],           8 + 3*x[1]**2

    def product_b_a(t, x):
        t = float(t); x = [float(y) for y in x]
        return 2*(x[0]**2)*x[1] + 8,  3*x[0]**2 + 2*x[1], \
               2*(x[0]**2)*x[1] + 12*x[1], 3*x[0]**2 + 3*x[1]**2

    m1   = np.zeros([20, 20, 4])
    m2   = np.zeros([20, 20, 4])

    ground_m1_times_m2 = np.zeros([20, 20, 4])
    ground_m2_times_m1 = np.zeros([20, 20, 4])

    for i in range(0, 20):
        for j in range(0, 20):

            m1[i, j, :] = field_matrix_a(1, [i, j])
            m2[i, j, :] = field_matrix_b(1, [i, j])

            ground_m1_times_m2[i, j, :] = product_a_b(1, [i, j])
            ground_m2_times_m1[i, j, :] = product_b_a(1, [i, j])

    computed_m1_times_m2 = mat.matrix_fields_product(m1, m2)
    computed_m2_times_m1 = mat.matrix_fields_product(m2, m1)

    assert_array_equal(computed_m1_times_m2.shape, ground_m1_times_m2.shape)
    assert_array_equal(computed_m2_times_m1.shape, ground_m2_times_m1.shape)

    assert_array_almost_equal(computed_m1_times_m2, ground_m1_times_m2)
    assert_array_almost_equal(computed_m2_times_m1, ground_m2_times_m1)


def test_matrix_fields_product_3d():

    def field_matrix_a(t, x):
        t = float(t); x = [float(y) for y in x]
        return 2*x[0]*x[2], 3*x[0], x[1], \
               4,           x[1],   0, \
               x[2],        1,      3

    def field_matrix_b(t, x):
        t = float(t); x = [float(y) for y in x]
        return x[0], 2,      x[2],\
               x[1], 3*x[0], 1, \
               2,    x[2],   1

    def product_a_b(t, x):
        t = float(t); x = [float(y) for y in x]
        return 2*(x[0]**2)*x[2] + 3*x[0]*x[1] + 2*x[1], 4*x[0]*x[2] + 9*x[0]**2 + x[1]*x[2], 2*x[0]*x[2]**2 + 3*x[0] + x[1], \
               4*x[0] + x[1]**2,                        8 + 3*x[0]*x[1],                     4*x[2] + x[1], \
               x[0]*x[2] + x[1] + 6,                    5*x[2] + 3*x[0],                     x[2]**2 + 4

    def product_b_a(t, x):
        t = float(t); x = [float(y) for y in x]
        return 2*(x[0]**2)*x[2] + 8 + x[2]**2,       3*x[0]**2 + 2*x[1] + x[2],       x[0]*x[1] + 3*x[2], \
               2*x[0]*x[1]*x[2] + 12*x[0] + x[2],    6*x[0]*x[1] + 1,                 x[1]**2 + 3, \
               4*x[0]*x[2] + 5*x[2],                 6*x[0] + x[1]*x[2] + 1,          2*x[1] + 3

    m1   = np.zeros([20, 20, 20, 9])
    m2   = np.zeros([20, 20, 20, 9])

    ground_m1_times_m2 = np.zeros([20, 20, 20, 9])
    ground_m2_times_m1 = np.zeros([20, 20, 20, 9])

    for i in range(0, 20):
        for j in range(0, 20):
            for k in range(0, 20):
                m1[i, j, k, :] = field_matrix_a(1, [i, j, k])
                m2[i, j, k, :] = field_matrix_b(1, [i, j, k])

                ground_m1_times_m2[i, j, k, :] = product_a_b(1, [i, j, k])
                ground_m2_times_m1[i, j, k, :] = product_b_a(1, [i, j, k])

    computed_m1_times_m2 = mat.matrix_fields_product(m1, m2)
    computed_m2_times_m1 = mat.matrix_fields_product(m2, m1)

    assert_array_equal(computed_m1_times_m2.shape, ground_m1_times_m2.shape)
    assert_array_equal(computed_m2_times_m1.shape, ground_m2_times_m1.shape)

    assert_array_almost_equal(computed_m1_times_m2, ground_m1_times_m2)
    assert_array_almost_equal(computed_m2_times_m1, ground_m2_times_m1)


test_matrix_fields_product_2d_1()
test_matrix_fields_product_3d()


''' matrix_fields_product_iterative '''


def test_matrix_fields_product_iterative_2d():

    def field_matrix_a(t, x):
        t = float(t); x = [float(y) for y in x]
        return 2,     3*x[0], \
               x[1],  -1

    def field_matrix_a_square(t, x):
        t = float(t); x = [float(y) for y in x]
        return 4 + 3*x[0]*x[1], 3*x[0], \
               x[1],            3*x[0]*x[1] + 1

    def field_matrix_a_cube(t, x):
        t = float(t); x = [float(y) for y in x]
        return 8 + 9*x[0]*x[1],         9*x[0] + 9*(x[0]**2)*x[1], \
               3*x[1] + 3*x[0]*x[1]**2, -1

    m1    = np.zeros([20, 20, 4])

    ground_m2 = np.zeros([20, 20, 4])
    ground_m3 = np.zeros([20, 20, 4])

    for i in range(0, 20):
        for j in range(0, 20):

            m1[i, j, :]  = field_matrix_a(1, [i, j])

            ground_m2[i, j, :] = field_matrix_a_square(1, [i, j])
            ground_m3[i, j, :] = field_matrix_a_cube(1, [i, j])

    computed_m1 = mat.matrix_fields_product_iterative(m1)
    computed_m2 = mat.matrix_fields_product_iterative(m1, 2)
    computed_m3 = mat.matrix_fields_product_iterative(m1, 3)

    assert_array_almost_equal(m1, computed_m1)
    assert_array_almost_equal(ground_m2, computed_m2)
    assert_array_almost_equal(ground_m3, computed_m3)


def test_matrix_fields_product_iterative_diag_matrix_2d():

    def field_matrix_a(t, x):
        t = float(t); x = [float(y) for y in x]
        return x[1],     0, \
               0,  1 + x[0]

    def field_matrix_a_pow_n(t, x, n):
        t = float(t); x = [float(y) for y in x]
        return x[1]**n, 0, \
               0,       (1 + x[0])**n

    m1    = np.zeros([20, 20, 4])
    ground_m1_pow_n = np.zeros([20, 20, 4])

    n = 5

    for i in range(0, 20):
        for j in range(0, 20):

            m1[i, j, :]  = field_matrix_a(1, [i, j])
            ground_m1_pow_n[i, j, :] = field_matrix_a_pow_n(1, [i, j], n)

    computed_m1 = mat.matrix_fields_product_iterative(m1)
    computed_m1_pow_n = mat.matrix_fields_product_iterative(m1, 5)

    assert_array_almost_equal(m1, computed_m1)
    assert_array_almost_equal(ground_m1_pow_n, computed_m1_pow_n)


def test_id_matrix_field_2d_and_3d():

    domain_2d = [13, 17]
    id_2d = mat.id_matrix_field(domain_2d)

    domain_3d = domain_2d + [23, ]
    id_3d = mat.id_matrix_field(domain_3d)

    flat_id_2d = np.eye(2).reshape(4)
    flat_id_3d = np.eye(3).reshape(9)

    for x in range(domain_2d[0]):
        for y in range(domain_2d[1]):
            assert_array_equal(id_2d[x, y, 0, 0, :], flat_id_2d)
            for z in range(domain_3d[2]):
                assert_array_equal(id_3d[x, y, z, 0, :], flat_id_3d)


if __name__ == '__main__':

    test_bch_right_jacobian_all_zero()
    test_bch_right_jacobian_zero_angle()
    test_bch_right_jacobian_zero_translation()
    test_bch_right_jacobian_little_rotation_zero_traslation()
    test_bch_ground_random_input_comparing_matrices_step_1()

    test_split_the_time_incorrect_input_len()
    test_split_the_time_easy_array_1()
    test_split_the_time_easy_array_2()
    test_split_the_time_easy_array_3()

    test_matrix_vector_field_product_2d()
    test_matrix_vector_field_product_toy_example_3d()

    test_matrix_fields_product_2d_1()
    test_matrix_fields_product_3d()

    test_matrix_fields_product_iterative_2d()
    test_matrix_fields_product_iterative_diag_matrix_2d()
    test_id_matrix_field_2d_and_3d()
