from random import uniform

import numpy as np
from numpy.testing import assert_array_almost_equal

from calie.transformations import se2


### confirm that exp and log are ones other inverse for matrices ###


def test_theory_inverse_exp_log():
    any_angle_1 = uniform(-np.pi + np.abs(np.spacing(-np.pi)), np.pi)
    any_tx_1 = uniform(-10, 10)
    any_ty_1 = uniform(-10, 10)
    a = se2.Se2G(any_angle_1, any_tx_1, any_ty_1)
    ans = se2.se2a_exp(se2.se2g_log(a))
    assert_array_almost_equal(a.get, ans.get)


def test_theory_inverse_log_exp():
    any_angle_1 = uniform(-np.pi + np.abs(np.spacing(-np.pi)), np.pi)
    any_tx_1 = uniform(-10, 10)
    any_ty_1 = uniform(-10, 10)
    a = se2.Se2A(any_angle_1, any_tx_1, any_ty_1)
    ans = se2.se2g_log(se2.se2a_exp(a))
    assert_array_almost_equal(a.get, ans.get)


def test_theory_inverse_log_exp_input_not_in_quotient():
    any_angle_1 = 2 * np.pi + uniform(-np.pi + np.abs(np.spacing(-np.pi)), np.pi)
    any_tx_1 = uniform(-10, 10)
    any_ty_1 = uniform(-10, 10)
    a = se2.Se2A(any_angle_1, any_tx_1, any_ty_1)
    ans = se2.se2g_log(se2.se2a_exp(a))
    assert_array_almost_equal(a.get, ans.get)


### Deformation and displacement Methods
#
#
# def test_deformation_and_displacement_passages_from_matrices():
#     # passing from deformation to displacement and vice versa using a matrix as
#     # ground truth.
#     domain = (16, 16)
#     theta, tx, ty = np.pi / 6, 2.5, -2
#     m_0 = se2_g.se2_g(theta, tx, ty)
#     dm_0 = se2_g.log(m_0)
#
#     print dm_0.get_matrix
#     print m_0.get_matrix
#
#     ### generate subsequent vector fields
#     svf_0   = SVF.generate_from_matrix(domain, dm_0.get_matrix, affine=np.eye(4))
#
#     # This provides the displacement since I am subtracting the id.
#     disp_0 = SDISP.generate_from_matrix(domain, m_0.get_matrix - np.eye(3), affine=np.eye(4))
#
#     # This provides the deformation since I am not subtracting the id
#     def_0 = SDISP.generate_from_matrix(domain, m_0.get_matrix, affine=np.eye(4))
#
#     # I generate the deformation from the displacement using deformation_from_displacement
#     def_generated_0 = SDISP.deformation_from_displacement(disp_0)
#
#     # I generate the displacement from the deformation using displacement_from_deformation
#     disp_generated_0 = SDISP.displacement_from_deformation(def_0)
#
#     assert_array_almost_equal(def_0.field, def_generated_0.field)
#     assert_array_almost_equal(disp_0.field, disp_generated_0.field)
#
#
#
# def test_invariance_under_linear_translations_for_projective_svf():
#
#     random_seed = 5
#
#     if random_seed > 0:
#         np.random.seed(random_seed)
#
#     domain = (10, 10)
#
#     # Algebra
#     # generate a 2 projective vector field up to a linear translation and verify that they are the same:
#     h_a = ProjectiveAlgebra.randomgen(d=2)
#
#     svf_h1 = SVF.generate_from_projective_matrix_algebra(input_vol_ext=domain, input_h=h_a.matrix)
#     svf_h2 = SVF.generate_from_projective_matrix_algebra(input_vol_ext=domain, input_h=h_a.matrix - 17*np.eye(3))
#
#     assert_array_almost_equal(svf_h1.field, svf_h2.field)
#
#     # Group
#     # generate a 2 projective vector field up to a scalar factor and verify that they are the same:
#     h_g = h_a.exponentiate()
#
#     # generate the corresponding disp
#     disp_h1 = SDISP.generate_from_projective_matrix_group(input_vol_ext=domain, input_exp_h=h_g.matrix)
#     disp_h2 = SDISP.generate_from_projective_matrix_group(input_vol_ext=domain, input_exp_h=5*h_g.matrix)
#
#     assert_array_almost_equal(disp_h1.field, disp_h2.field)
#
# test_invariance_under_linear_translations_for_projective_svf()
#
#
# test_deformation_and_displacement_passages_from_matrices()
# test_theory_inverse_exp_log()