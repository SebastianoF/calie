"""
We consider a randomly generated svf v in the Lie algebra.
We then consider its inverse in the lie Algebra: -v

The composition in the Lie algebra does not exist. But we apply the numerical method anyway to see what may happen.
v dot (-v) and (-v) dot v does not return the approximated identity (in green).

Afterwards we compose exp(v) and exp(-v) to see the approximated identity with the correct composition (again in green).

"""
import numpy as np
import matplotlib.pyplot as plt

from VECtorsToolkit.tools.fields.composition import eulerian_dot_eulerian
from VECtorsToolkit.tools.fields.generate_vf import generate_random
from VECtorsToolkit.tools.visualisations.fields_at_the_window import see_field, see_2_fields
from VECtorsToolkit.tools.local_operations.exponential import lie_exponential


if __name__ == '__main__':

    # generate two vector fields
    omega = (20, 20)

    svf_v     = generate_random(omega, parameters=(2, 2))
    svf_v_inv = np.copy(-1 * svf_v)

    v_o_v_inv_alg = eulerian_dot_eulerian(svf_v, svf_v_inv)
    v_inv_o_v_alg = eulerian_dot_eulerian(svf_v_inv, svf_v)

    disp_v = lie_exponential(svf_v)
    disp_v_inv = lie_exponential(svf_v_inv)

    v_o_v_inv_grp = eulerian_dot_eulerian(disp_v, disp_v_inv)
    f_inv_o_f_grp = eulerian_dot_eulerian(disp_v_inv, disp_v)

    # see svf map the svfs
    see_field(svf_v, fig_tag=77)
    see_field(svf_v_inv, fig_tag=77, input_color='r', title_input='2 vector fields: f blue, g red')

    see_2_fields(svf_v, svf_v, fig_tag=78)
    see_2_fields(svf_v_inv, svf_v_inv, fig_tag=78, input_color='r')
    see_2_fields(v_inv_o_v_alg, v_o_v_inv_alg, fig_tag=78, input_color='g',
                 title_input_0='(v^(-1) o v)', title_input_1='(v o v^(-1))')

    see_2_fields(disp_v, disp_v, fig_tag=79)
    see_2_fields(disp_v_inv, disp_v_inv, fig_tag=79, input_color='r')
    see_2_fields(f_inv_o_f_grp, v_o_v_inv_grp, fig_tag=79, input_color='g',
                 title_input_0='(f^(-1) o f) inv after exp', title_input_1='(f o f^(-1)) inv after exp')

    plt.show()
