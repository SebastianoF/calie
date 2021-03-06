"""
We consider a randomly generated svf v in the Lie algebra.
We then consider its inverse in the lie Algebra: -v

The composition in the Lie algebra does not exist. But we apply the numerical method anyway to see what may happen.
v dot (-v) and (-v) dot v does not return the approximated identity (in green).

Afterwards we compose exp(v) and exp(-v) to see the approximated identity with the correct composition (again in green).

"""
import matplotlib.pyplot as plt
import numpy as np

from calie.operations import lie_exp
from calie.visualisations.fields import fields_at_the_window

from calie.fields import generate as gen
from calie.fields import compose as cp

if __name__ == '__main__':

    # generate two vector fields
    omega = (20, 20)

    svf_v     = gen.generate_random(omega, parameters=(2, 2))
    svf_v_inv = np.copy(-1 * svf_v)

    # we wrongly perform the composition of stationary velocity fields. The outcome is not the identity.
    v_o_v_inv_alg = cp.lagrangian_dot_lagrangian(svf_v, svf_v_inv)
    v_inv_o_v_alg = cp.lagrangian_dot_lagrangian(svf_v_inv, svf_v)

    # we correctly perform the composition after exponentiating the SVF in the Lie group.
    # The outcome is the identity, as expected.

    l_exp = lie_exp.LieExp()

    disp_v = l_exp.scaling_and_squaring(svf_v)
    disp_v_inv = l_exp.scaling_and_squaring(svf_v_inv)

    v_o_v_inv_grp = cp.lagrangian_dot_lagrangian(disp_v, disp_v_inv)
    f_inv_o_f_grp = cp.lagrangian_dot_lagrangian(disp_v_inv, disp_v)

    # see svf map the svfs
    fields_at_the_window.see_field(svf_v, fig_tag=77)
    fields_at_the_window.see_field(svf_v_inv, fig_tag=77, input_color='r', title_input='2 SVF: v blue, -v red ')

    fields_at_the_window.see_2_fields(svf_v, svf_v, fig_tag=78, window_title_input='Improper composition of SVF')
    fields_at_the_window.see_2_fields(svf_v_inv, svf_v_inv, fig_tag=78, input_color='r')
    fields_at_the_window.see_2_fields(v_inv_o_v_alg, v_o_v_inv_alg, fig_tag=78, input_color='g',
                                      title_input_0='(-v o v)', title_input_1='(v o -v)')

    fields_at_the_window.see_2_fields(disp_v, disp_v, fig_tag=79,
                                      window_title_input='Properly computed composition of SVF after exp')
    fields_at_the_window.see_2_fields(disp_v_inv, disp_v_inv, fig_tag=79, input_color='r')
    fields_at_the_window.see_2_fields(f_inv_o_f_grp, v_o_v_inv_grp, fig_tag=79, input_color='g',
                                      title_input_0='(exp(-v) o exp(v))', title_input_1='(exp(v) o exp(-v))')

    plt.show()
