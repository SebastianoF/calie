"""
Test to perform the composition between vector fields.

Here can be found some hints to compare the error of the composition provided by the resampling.
This is actually very high, as soon as the field gets complicated.
"""
import matplotlib.pyplot as plt
import numpy as np
from VECtorsToolkit.tools.operations.lie_exponential import lie_exponential
from VECtorsToolkit.tools.visualisations.fields.fields_at_the_window import see_field
from numpy.testing import assert_array_almost_equal

from VECtorsToolkit.fields import generate_random
from VECtorsToolkit.fields import lagrangian_dot_lagrangian
from VECtorsToolkit.fields import vf_identity_lagrangian


# Lagrangian dot lagrangian

def test_2_easy_vector_fields(get_figures=False):

    dec = 15  # decimal for the error
    passe_partout = 0

    omega = (20, 20)

    svf_zeros = vf_identity_lagrangian(omega)
    svf_f     = vf_identity_lagrangian(omega)
    svf_f_inv = vf_identity_lagrangian(omega)

    def function_f(t, x):
        t = float(t)
        x = [float(y) for y in x]
        return np.array([0, 0.3])

    for x in range(20):
        for y in range(20):
            svf_f[x, y, 0, 0, :] = function_f(1, [x, y])
            svf_f_inv[x, y, 0, 0, :] = -1 * function_f(1, [x, y])

    f_o_f_inv = lagrangian_dot_lagrangian(svf_f, svf_f_inv, add_right=True)
    f_inv_o_f = lagrangian_dot_lagrangian(svf_f_inv, svf_f, add_right=True)

    assert_array_almost_equal(f_o_f_inv[10, 10, 0, 0, :], [.0, .0], decimal=dec)
    assert_array_almost_equal(f_inv_o_f[10, 10, 0, 0, :], [.0, .0], decimal=dec)

    # #results of a composition of 2
    assert_array_almost_equal(f_o_f_inv[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0, :],
                              svf_zeros[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0, :],
                              decimal=dec)
    assert_array_almost_equal(f_inv_o_f[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0, :],
                              svf_zeros[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0, :],
                              decimal=dec)

    if get_figures:
        see_field(svf_f)
        see_field(svf_f_inv, input_color='r', title_input='2 vector fields: f blue, g red')

        see_field(svf_f, fig_tag=2)
        see_field(svf_f_inv, fig_tag=2, input_color='r')
        see_field(f_o_f_inv, fig_tag=2, input_color='g', title_input='composition (f o f^(-1)) in green')

        see_field(svf_f, fig_tag=3)
        see_field(svf_f_inv, fig_tag=3, input_color='r')
        see_field(f_inv_o_f, fig_tag=3, input_color='g', title_input='composition (f^(-1) o f) in green')

    plt.show()


def test_2_less_easy_vector_fields(get_figures=False):

    dec = 1  # decimal for the error
    passe_partout = 3

    omega = (20, 20)

    svf_zeros = vf_identity_lagrangian(omega)
    svf_f     = vf_identity_lagrangian(omega)
    svf_f_inv = vf_identity_lagrangian(omega)

    def function_f(t, x):
        t = float(t)
        x = [float(y) for y in x]
        return np.array([np.sin(0.01*x[1]), (3*np.cos(x[0])) / (x[0] + 2)])

    for x in range(20):
        for y in range(20):
            svf_f[x, y, 0, 0, :] = function_f(1, [x, y])
            svf_f_inv[x, y, 0, 0, :] = -1 * function_f(1, [x, y])

    f_o_f_inv = lagrangian_dot_lagrangian(svf_f, svf_f_inv, add_right=True)
    f_inv_o_f = lagrangian_dot_lagrangian(svf_f_inv, svf_f, add_right=True)

    assert_array_almost_equal(f_o_f_inv[10, 10, 0, 0, :], [.0, .0], decimal=dec)
    assert_array_almost_equal(f_inv_o_f[10, 10, 0, 0, :], [.0, .0], decimal=dec)

    # results of a composition
    assert_array_almost_equal(f_o_f_inv[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0, :],
                              svf_zeros[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0, :],
                              decimal=dec)
    assert_array_almost_equal(f_inv_o_f[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0, :],
                              svf_zeros[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0, :],
                              decimal=dec)

    if get_figures:
        see_field(svf_f)
        see_field(svf_f_inv, input_color='r', title_input='2 vector fields: f blue, g red')

        see_field(svf_f, fig_tag=2)
        see_field(svf_f_inv, fig_tag=2)
        see_field(f_o_f_inv, fig_tag=2, input_color='g', title_input='composition (f o f^(-1)) in green')

        see_field(svf_f, fig_tag=3)
        see_field(svf_f_inv, fig_tag=3, input_color='r')
        see_field(f_inv_o_f, fig_tag=3, input_color='g', title_input='composition (f^(-1) o f) in green')

    plt.show()


def test_easy_composition_with_identity(get_figures=False):

    dec = 6
    passe_partout = 0

    omega = (10, 10)

    svf_zeros = vf_identity_lagrangian(omega)
    svf_f     = vf_identity_lagrangian(omega)
    svf_id    = vf_identity_lagrangian(omega)  # id in lagrangian coordinates is the zero field

    def function_f(t, x):
        t = float(t); x = [float(y) for y in x]
        return np.array([0, 0.3])

    for x in range(10):
        for y in range(10):
            svf_f[x, y, 0, 0, :] = function_f(1, [x, y])

    f_o_id = lagrangian_dot_lagrangian(svf_f, svf_id, add_right=True)
    id_o_f = lagrangian_dot_lagrangian(svf_id, svf_f, add_right=True)

    # sfv_0 is provided in Lagrangian coordinates!
    if get_figures:
        see_field(svf_f, fig_tag=21)
        see_field(svf_id, fig_tag=21, input_color='r', title_input='2 vector fields: f blue, g red')

        see_field(svf_f, fig_tag=22)
        see_field(svf_id, fig_tag=22, input_color='r')
        see_field(f_o_id, fig_tag=22, input_color='g', title_input='composition (f o id) in green')

        see_field(svf_f, fig_tag=23)
        see_field(svf_id, fig_tag=23, input_color='r')
        see_field(id_o_f, fig_tag=23, input_color='g', title_input='composition (id o f) in green')

    plt.show()

    # test if the compositions are still in lagrangian coordinates, as attributes and as shape
    assert_array_almost_equal(f_o_id[5, 5, 0, 0, :], function_f(1, [5, 5]), decimal=dec)
    assert_array_almost_equal(id_o_f[5, 5, 0, 0, :], function_f(1, [5, 5]), decimal=dec)

    # results of a composition of 2 lagrangian must be a lagrangian zero field
    assert_array_almost_equal(f_o_id[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0, :],
                              svf_zeros[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0, :],
                              decimal=dec)
    assert_array_almost_equal(id_o_f[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0, :],
                              svf_zeros[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0, :],
                              decimal=dec)


def test_less_easy_composition_with_identity(get_figures=False):

    dec = 0  # decimal for the error
    passe_partout = 4

    omega = (20, 25)

    svf_zeros = vf_identity_lagrangian(omega=omega)
    svf_f     = vf_identity_lagrangian(omega=omega)
    svf_id    = vf_identity_lagrangian(omega=omega)

    def function_f(t, x):
        t = float(t)
        x = [float(y) for y in x]
        return np.array([np.sin(0.01*x[1]), (3*np.cos(x[0])) / (x[0] + 2)])

    for x in range(20):
        for y in range(20):
            svf_f[x, y, 0, 0, :] = function_f(1, [x, y])

    f_o_id = lagrangian_dot_lagrangian(svf_f, svf_id, add_right=True)
    id_o_f = lagrangian_dot_lagrangian(svf_id, svf_f, add_right=True)

    # test if the compositions are still in lagrangian coordinates, as attributes and as shape
    assert_array_almost_equal(f_o_id[5, 5, 0, 0, :], function_f(1, [5, 5]), decimal=dec)
    assert_array_almost_equal(id_o_f[5, 5, 0, 0, :], function_f(1, [5, 5]), decimal=dec)

    # results of a composition of 2 lagrangian must be a lagrangian zero field

    assert_array_almost_equal(f_o_id[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0, :],
                              svf_zeros[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0, :],
                              decimal=dec)
    assert_array_almost_equal(id_o_f[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0, :],
                              svf_zeros[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0, :],
                              decimal=dec)

    if get_figures:
        see_field(svf_f, fig_tag=41)
        see_field(svf_id, fig_tag=41, input_color='r', title_input='2 vector fields: f blue, g red')

        see_field(svf_f, fig_tag=42)
        see_field(svf_id, fig_tag=42, input_color='r')
        see_field(f_o_id, fig_tag=42, input_color='g', title_input='composition (f o id) in green')

        see_field(svf_f, fig_tag=43)
        see_field(svf_id, fig_tag=43, input_color='r')
        see_field(id_o_f, fig_tag=43, input_color='g', title_input='composition (id o f) in green')

    plt.show()


def test_2_random_vector_fields_svf(get_figures=False):
    """
    The composition is not the identity since we are working in the tangent space.
    """
    dec = 3
    passe_partout = 5

    omega = (10, 10)

    svf_f     = vf_identity_lagrangian(omega=omega)
    svf_f_inv = np.copy(-1 * svf_f)

    f_o_f_inv = lagrangian_dot_lagrangian(svf_f, svf_f_inv, add_right=True)
    f_inv_o_f = lagrangian_dot_lagrangian(svf_f_inv, svf_f, add_right=True)
    svf_id = vf_identity_lagrangian(omega=omega)

    # # results of a composition of 2 lagrangian must be a lagrangian zero field
    assert_array_almost_equal(f_o_f_inv[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0, :],
                              svf_id[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0, :],
                              decimal=dec)
    assert_array_almost_equal(f_inv_o_f[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0, :],
                              svf_id[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0, :],
                              decimal=dec)

    if get_figures:
        see_field(svf_f, fig_tag=51)
        see_field(svf_f_inv, fig_tag=51, input_color='r', title_input='2 vector fields: f blue, g red')

        see_field(svf_f, fig_tag=52)
        see_field(svf_f_inv, fig_tag=52, input_color='r')
        see_field(f_o_f_inv, fig_tag=52, input_color='g', title_input='composition (f o f^(-1)) in green')

        see_field(svf_f, fig_tag=53)
        see_field(svf_f_inv, fig_tag=53, input_color='r')
        see_field(f_inv_o_f, fig_tag=53, input_color='g', title_input='composition (f^(-1) o f) in green')

    plt.show()


def test_2_random_vector_fields_as_deformations(get_figures=False):

    dec = 1
    passe_partout = 3

    omega = (15, 15)

    sigma_init = 4
    sigma_gaussian_filter = 2
    svf_zeros = vf_identity_lagrangian(omega)
    svf_0     = generate_random(omega, parameters=(sigma_init, sigma_gaussian_filter))

    sdisp_0 = lie_exponential(svf_0, algorithm='ss')
    sdisp_0_inv = lie_exponential(-1 * svf_0, algorithm='ss')

    f_o_f_inv = lagrangian_dot_lagrangian(sdisp_0, sdisp_0_inv, add_right=True)
    f_inv_o_f = lagrangian_dot_lagrangian(sdisp_0_inv, sdisp_0, add_right=True)

    # results of a composition of 2 lagrangian must be a lagrangian zero field
    assert_array_almost_equal(f_o_f_inv[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0, :],
                              svf_zeros[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0, :],
                              decimal=dec)
    assert_array_almost_equal(f_inv_o_f[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0, :],
                              svf_zeros[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0, :],
                              decimal=dec)

    if get_figures:
        see_field(sdisp_0, fig_tag=61)
        see_field(sdisp_0_inv, fig_tag=61, input_color='r', title_input='2 displacement fields: f blue, g red')

        see_field(sdisp_0, fig_tag=62)
        see_field(sdisp_0_inv, fig_tag=62, input_color='r')
        see_field(f_o_f_inv, fig_tag=62, input_color='g', title_input='composition (f o f^(-1)) in green')

        see_field(sdisp_0, fig_tag=63)
        see_field(sdisp_0_inv, fig_tag=63, input_color='r')
        see_field(f_inv_o_f, fig_tag=63, input_color='g', title_input='composition (f^(-1) o f) in green')

    plt.show()


def test_controlled_composition_of_two_closed_form_vector_fields_2d_1(get_figures=False):
    omega = (30, 30)
    dec = 1
    passe_partout = 10

    def u(x, y):
        return 0, 0

    def v(x, y):
        return 1, 1

    def u_dot_v(x, y):
        return 0, 0

    def v_dot_u(x, y):
        return 1, 1

    svf_u = vf_identity_lagrangian(omega=omega)
    svf_v = vf_identity_lagrangian(omega=omega)
    svf_u_dot_v = vf_identity_lagrangian(omega=omega)
    svf_v_dot_u = vf_identity_lagrangian(omega=omega)

    for x in range(omega[0]):
        for y in range(omega[1]):
            svf_u[x, y, 0, 0, :] = u(x, y)
            svf_v[x, y, 0, 0, :] = v(x, y)
            svf_u_dot_v[x, y, 0, 0, :] = u_dot_v(x, y)
            svf_v_dot_u[x, y, 0, 0, :] = v_dot_u(x, y)

    svf_u_dot_v_numerical = lagrangian_dot_lagrangian(svf_u, svf_v, add_right=False)
    svf_v_dot_u_numerical = lagrangian_dot_lagrangian(svf_v, svf_u, add_right=False)

    if get_figures:

        see_field(svf_u, fig_tag=62)
        see_field(svf_v, fig_tag=62, input_color='r')
        see_field(svf_u_dot_v, fig_tag=62, input_color='g', title_input='(u o v) closed form')

        see_field(svf_u, fig_tag=63)
        see_field(svf_v, fig_tag=63, input_color='r')
        see_field(svf_u_dot_v_numerical, fig_tag=63, input_color='g', title_input='(u o v) numerical')

        see_field(svf_u, fig_tag=64)
        see_field(svf_v, fig_tag=64, input_color='r')
        see_field(svf_v_dot_u, fig_tag=64, input_color='g', title_input='composition v o u closed form')

        see_field(svf_u, fig_tag=65)
        see_field(svf_v, fig_tag=65, input_color='r')
        see_field(svf_v_dot_u_numerical, fig_tag=65, input_color='g', title_input='composition v o u numerical')

        plt.show()

    assert_array_almost_equal(svf_u_dot_v[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0, :],
                              svf_u_dot_v_numerical[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0, :],
                              decimal=dec)
    assert_array_almost_equal(svf_v_dot_u[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0, :],
                              svf_v_dot_u_numerical[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0, :],
                              decimal=dec)

# def test_controlled_composition_of_two_closed_form_vector_fields_2d_2(get_figures=True):
#     passe_partout = 3
#     dec = 2
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
#     omega = (12, 12)
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
#
#     print('svf_v_dot_u[5, 5]')
#     print(svf_v_dot_u[5, 5])
#
#
#     print('svf_v_dot_u_numerical[5, 5]')
#     print(svf_v_dot_u_numerical[5, 5])
#
#
#     if get_figures:
#
#         see_field(svf_u, fig_tag=62)
#         see_field(svf_v, fig_tag=62, input_color='r')
#         see_field(svf_v_dot_u, fig_tag=62, input_color='g', title_input='composition (v o u) closed form')
#
#         see_field(svf_u, fig_tag=63)
#         see_field(svf_v, fig_tag=63, input_color='r')
#         see_field(svf_v_dot_u_numerical, fig_tag=63, input_color='g', title_input='composition v o u numerical')
#
#         see_field(svf_u, fig_tag=64)
#         see_field(svf_v, fig_tag=64, input_color='r')
#         see_field(svf_u_dot_v, fig_tag=64, input_color='g', title_input='composition u o v closed form')
#
#         see_field(svf_u, fig_tag=65)
#         see_field(svf_v, fig_tag=65, input_color='r')
#         see_field(svf_u_dot_v_numerical, fig_tag=65, input_color='g', title_input='composition u o v numerical')
#
#         plt.show()
#
#     assert_array_almost_equal(svf_u_dot_v[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0, :],
#                               svf_u_dot_v_numerical[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0,
#                               :], decimal=dec)
#     assert_array_almost_equal(svf_v_dot_u[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0, :],
#                               svf_v_dot_u_numerical[passe_partout:-passe_partout, passe_partout:-passe_partout, 0, 0,
#                               :], decimal=dec)
#
#
# def test_controlled_composition_of_two_closed_form_vector_fields_2d_3(get_figures=True):
#
#     def u(x, y):
#         return 0.3 , 0.3
#
#     def v(x, y):
#         return 0.7, 0.7
#
#     def u_dot_v(x, y):
#         return 0.3, 0.3
#
#     def v_dot_u(x, y):
#         return 0.7, 0.7
#
#     omega = (20, 20)
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
#
#     if get_figures:
#         see_field(svf_u, fig_tag=61)
#         see_field(svf_v, fig_tag=61, input_color='r', title_input='2 displacement fields: u blue, v red')
#
#         see_field(svf_u, fig_tag=62)
#         see_field(svf_v, fig_tag=62, input_color='r')
#         see_field(svf_v_dot_u, fig_tag=62, input_color='g', title_input='composition (v o u) closed form')
#
#         see_field(svf_u, fig_tag=63)
#         see_field(svf_v, fig_tag=63, input_color='r')
#         see_field(svf_v_dot_u_numerical, fig_tag=63, input_color='g', title_input='composition v o u numerical')
#
#         see_field(svf_u, fig_tag=64)
#         see_field(svf_v, fig_tag=64, input_color='r')
#         see_field(svf_u_dot_v, fig_tag=64, input_color='g', title_input='composition u o v closed form')
#
#         see_field(svf_u, fig_tag=65)
#         see_field(svf_v, fig_tag=65, input_color='r')
#         see_field(svf_u_dot_v_numerical, fig_tag=65, input_color='g', title_input='composition u o v numerical')
#
#         plt.show()


if __name__ == '__main__':

    see_graphs = False

    test_2_easy_vector_fields(see_graphs)
    test_2_less_easy_vector_fields(see_graphs)
    test_easy_composition_with_identity(see_graphs)
    test_less_easy_composition_with_identity(see_graphs)
    test_2_random_vector_fields_svf(see_graphs)
    test_2_random_vector_fields_as_deformations(see_graphs)

    test_controlled_composition_of_two_closed_form_vector_fields_2d_1(see_graphs)
    # test_controlled_composition_of_two_closed_form_vector_fields_2d_2(True)
    # test_controlled_composition_of_two_closed_form_vector_fields_2d_3(see_graphs)
